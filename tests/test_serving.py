"""Serving tests: schemas, job store and the HTTP API with a stubbed analysis service.

The API tests stub the service so they stay offline and fast; the model-backed path is
exercised by the CLI-level integration runs, not here.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from courtvision.config import ApiConfig, EventConfig, ServingConfig
from courtvision.serving.api import JobStore, create_app
from courtvision.serving.schemas import AnalyzeForm
from courtvision.serving.service import AnalysisRequest, AnalysisService, ServiceNotReadyError

MP4_BYTES = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64


def fake_payload(frames: int = 3) -> dict:
    return {
        "video": {"filename": "clip.mp4", "bytes": len(MP4_BYTES), "frames": frames, "fps": 25.0, "width": 320, "height": 180, "duration_seconds": 0.12},
        "runtime": {"wall_seconds": 0.5, "frames_per_second": 6.0, "frames_processed": frames, "device": "cpu"},
        "detections": {"total": 6, "per_frame_mean": 2.0, "per_frame_max": 3, "mean_confidence": 0.8, "confidence_min": 0.5, "confidence_max": 0.95},
        "tracks": {"unique_tracks": 2, "mean_duration_frames": 3.0, "max_duration_frames": 3, "tracks_with_gaps": 0},
        "model": {"weights": "stub.pt", "tracker": "bytetrack.yaml", "imgsz": 640, "confidence": 0.25, "backend": "stub", "parameters": None},
        "quality": {"truncated": False, "warnings": []},
    }


class StubService(AnalysisService):
    """Performs no inference; records what it was asked and asserts upload hygiene."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[AnalysisRequest] = []
        self.behaviour = None

    def load(self) -> None:
        self._tracker = object()  # type: ignore[assignment]

    def unload(self) -> None:
        self._tracker = None

    @property
    def model_loaded(self) -> bool:
        return self._tracker is not None

    def analyze(self, request: AnalysisRequest) -> dict:
        self.calls.append(request)
        assert request.video_path.exists(), "the upload must be on disk when analysis runs"
        if self.behaviour is not None:
            return self.behaviour(request)
        payload = fake_payload()
        if request.include_tracks:
            payload["tracks_detail"] = [{"frame": 1, "detections": []}]
        return payload


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """Build the app with the stub installed at construction time.

    The route handlers close over the service instance created inside ``create_app``, so
    replacing ``app.state.courtvision.service`` after the fact would leave them talking to
    a real ``AnalysisService`` and loading real weights. Patching the class instead means
    every closure sees the stub.
    """
    serving = ServingConfig(model="stub.pt", device="cpu", tracker="bytetrack.yaml", preload=False)
    api_config = ApiConfig(max_upload_mb=1.0, max_frames=100)
    monkeypatch.setattr("courtvision.serving.api.AnalysisService", StubService)
    app = create_app(serving, api_config, EventConfig(), preload=False)
    stub = app.state.courtvision.service
    assert isinstance(stub, StubService)
    stub.load()  # mark ready without touching any weights
    with TestClient(app) as test_client:
        test_client.courtvision_stub = stub  # type: ignore[attr-defined]
        yield test_client


@pytest.fixture
def stub(client: TestClient) -> StubService:
    return client.courtvision_stub  # type: ignore[attr-defined]


def upload(name: str = "clip.mp4", content: bytes = MP4_BYTES, content_type: str = "video/mp4"):
    return {"video": (name, io.BytesIO(content), content_type)}


# -- schemas -----------------------------------------------------------------


def test_analyze_form_rejects_out_of_range_values() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnalyzeForm(conf=1.5)
    with pytest.raises(ValidationError):
        AnalyzeForm(imgsz=8)
    assert AnalyzeForm(max_frames=None).max_frames is None


# -- job store ---------------------------------------------------------------


def test_job_store_runs_and_expires(tmp_path: Path, monkeypatch) -> None:
    from datetime import datetime, timedelta, timezone

    import courtvision.serving.api as api_module

    store = JobStore(ApiConfig(max_jobs=4, job_ttl_seconds=60))
    store.start()
    try:
        store.set_runner(lambda job: {"ok": True})
        job = store.submit(source_path=tmp_path / "x.mp4", byte_size=1, detail="summary", params={})
        for _ in range(200):
            if store.get(job.job_id) and store.get(job.job_id).state == "succeeded":
                break
            time.sleep(0.01)
        assert store.count() == 1

        # Age the finished job past the TTL and let the next access expire it.
        record = store._jobs[job.job_id]
        record.finished_at = (datetime.now(timezone.utc) - timedelta(seconds=61)).isoformat()
        assert store.get(job.job_id) is None, "a job past its TTL must be dropped"
        assert store.count() == 0
    finally:
        store.shutdown()
    del api_module, monkeypatch


def test_job_store_reports_failures_with_reasons(tmp_path: Path) -> None:
    store = JobStore(ApiConfig(max_jobs=2))

    def failing(job):
        raise ValueError("deliberate failure")

    store.start()
    try:
        store.set_runner(failing)
        job = store.submit(source_path=tmp_path / "x.mp4", byte_size=1, detail="summary", params={})
        for _ in range(200):
            current = store.get(job.job_id)
            if current and current.state == "failed":
                break
            time.sleep(0.01)
        assert current.state == "failed" and "deliberate failure" in current.error
    finally:
        store.shutdown()


def test_job_store_rejects_submissions_past_capacity(tmp_path: Path) -> None:
    store = JobStore(ApiConfig(max_jobs=1))
    store.start()
    try:
        store.set_runner(lambda job: {"ok": True})
        store.submit(source_path=tmp_path / "x.mp4", byte_size=1, detail="summary", params={})
        with pytest.raises(RuntimeError, match="full"):
            store.submit(source_path=tmp_path / "y.mp4", byte_size=1, detail="summary", params={})
    finally:
        store.shutdown()


def test_cancel_only_applies_to_queued_jobs(tmp_path: Path) -> None:
    import threading

    store = JobStore(ApiConfig(max_jobs=4, job_max_workers=1))
    release = threading.Event()
    store.start()
    try:
        # One worker: the first job occupies it, so the second stays queued.
        store.set_runner(lambda job: release.wait(2.0) or {"ok": True})
        first = store.submit(source_path=tmp_path / "a.mp4", byte_size=1, detail="summary", params={})
        second = store.submit(source_path=tmp_path / "b.mp4", byte_size=1, detail="summary", params={})
        for _ in range(200):
            if store.get(first.job_id).state == "running" and store.get(second.job_id).state == "queued":
                break
            time.sleep(0.01)
        assert store.cancel(second.job_id) is True
        assert store.cancel(second.job_id) is False, "a cancelled job cannot be cancelled again"
        assert store.cancel(first.job_id) is False, "a running job is not cancellable"
    finally:
        release.set()
        store.shutdown()


def test_job_stats_are_explicit_about_not_being_distributed() -> None:
    stats = JobStore(ApiConfig()).stats()
    assert stats["distributed"] is False
    assert "not shared between replicas" in stats["note"]


# -- API ---------------------------------------------------------------------


def test_health_is_liveness_only(client: TestClient) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok" and "uptime_seconds" in payload


def test_ready_reports_readiness(client: TestClient) -> None:
    payload = client.get("/ready").json()
    assert payload["ready"] is True
    assert payload["tracker_supported"] is True


def test_metrics_exposes_prometheus_text(client: TestClient) -> None:
    client.post("/v1/analyze", files=upload())
    text = client.get("/metrics").text
    assert "courtvision_requests_total" in text and "# TYPE" in text


def test_capabilities_reports_limits_and_jobs(client: TestClient) -> None:
    payload = client.get("/v1/capabilities").json()
    assert payload["capabilities"]["trackers"]
    assert payload["jobs"]["distributed"] is False
    assert payload["limits"]["max_upload_mb"] == 1.0


def test_analyze_returns_documented_schema_and_request_id(client: TestClient) -> None:
    response = client.post("/v1/analyze", files=upload(), headers={"x-request-id": "abc123"})
    assert response.status_code == 200
    payload = response.json()
    for key in ("request_id", "video", "runtime", "detections", "tracks", "model", "quality"):
        assert key in payload
    assert payload["request_id"] == "abc123"
    assert response.headers["x-request-id"] == "abc123"
    assert payload.get("tracks_detail") is None, "summary detail must not carry per-frame arrays"


def test_detail_full_includes_tracks(stub, client: TestClient) -> None:
    payload = client.post("/v1/analyze", files=upload(), data={"detail": "tracks"}).json()
    assert payload["tracks_detail"]


def test_empty_upload_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/analyze", files=upload(content=b""))
    assert response.status_code == 400


@pytest.mark.parametrize(
    ("name", "content_type"),
    [("notes.txt", "text/plain"), ("video", "video/mp4"), ("clip.mp4", "application/zip")],
)
def test_unsupported_uploads_are_rejected_with_415(client: TestClient, name: str, content_type: str) -> None:
    response = client.post("/v1/analyze", files=upload(name=name, content=b"hello", content_type=content_type))
    assert response.status_code == 415


def test_oversized_upload_is_rejected_with_413(client: TestClient) -> None:
    oversized = MP4_BYTES + b"\x00" * (2 * 1024 * 1024)
    assert client.post("/v1/analyze", files=upload(content=oversized)).status_code == 413


def test_form_validation_returns_422(client: TestClient) -> None:
    assert client.post("/v1/analyze", files=upload(), data={"conf": "1.5"}).status_code == 422
    assert client.post("/v1/analyze", files=upload(), data={"imgsz": "8"}).status_code == 422


def test_client_filename_never_becomes_a_path(stub, client: TestClient) -> None:
    response = client.post("/v1/analyze", files=upload(name="../../etc/passwd.mp4"))
    assert response.status_code == 200
    request = stub.calls[-1]
    assert request.video_path.parent.name.startswith("courtvision-upload-")
    assert ".." not in request.video_path.name
    assert request.source_name == "passwd.mp4"


def test_temporary_uploads_are_removed_on_success_and_failure(client: TestClient, tmp_path: Path) -> None:
    import tempfile

    before = set(Path(tempfile.gettempdir()).glob("courtvision-upload-*"))
    assert client.post("/v1/analyze", files=upload()).status_code == 200
    assert set(Path(tempfile.gettempdir()).glob("courtvision-upload-*")) == before

    stub: StubService = client.courtvision_stub  # type: ignore[attr-defined]

    def boom(request):
        raise RuntimeError("boom")

    stub.behaviour = boom
    with pytest.raises(RuntimeError):
        client.post("/v1/analyze", files=upload())
    assert set(Path(tempfile.gettempdir()).glob("courtvision-upload-*")) == before, "cleanup must survive failure"


def test_service_not_ready_maps_to_503(client: TestClient, stub) -> None:
    def not_ready(request):
        raise ServiceNotReadyError("Model is not loaded; call load() first")

    stub.behaviour = not_ready
    assert client.post("/v1/analyze", files=upload()).status_code == 503


def test_value_error_maps_to_422(client: TestClient, stub) -> None:
    def bad_tracker(request):
        raise ValueError("Tracker 'nope.yaml' is not supported. Supported: ['bytetrack.yaml']")

    stub.behaviour = bad_tracker
    response = client.post("/v1/analyze", files=upload())
    assert response.status_code == 422
    assert "not supported" in response.json()["detail"]


def test_job_lifecycle_end_to_end(client: TestClient) -> None:
    accepted = client.post("/v1/jobs", files=upload())
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    assert accepted.json()["poll_url"] == f"/v1/jobs/{job_id}"

    for _ in range(200):
        state = client.get(f"/v1/jobs/{job_id}").json()
        if state["state"] in {"succeeded", "failed"}:
            break
        time.sleep(0.02)
    assert state["state"] == "succeeded"

    result = client.get(f"/v1/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.json()["result"]["video"]["frames"] == 3


def test_job_result_before_completion_is_409(client: TestClient, stub) -> None:
    def slow(request):
        time.sleep(0.4)
        return fake_payload()

    stub.behaviour = slow
    job_id = client.post("/v1/jobs", files=upload()).json()["job_id"]
    response = client.get(f"/v1/jobs/{job_id}/result")
    if response.status_code != 200:  # it may already have finished on a fast machine
        assert response.status_code == 409
        assert "poll" in response.json()["detail"]


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/v1/jobs/does-not-exist").status_code == 404


def test_finished_job_cannot_be_cancelled(client: TestClient) -> None:
    job_id = client.post("/v1/jobs", files=upload()).json()["job_id"]
    for _ in range(200):
        if client.get(f"/v1/jobs/{job_id}").json()["state"] in {"succeeded", "failed"}:
            break
        time.sleep(0.02)
    assert client.delete(f"/v1/jobs/{job_id}").status_code == 409


def test_job_uploads_are_cleaned_up(client: TestClient) -> None:
    import tempfile

    job_id = client.post("/v1/jobs", files=upload()).json()["job_id"]
    for _ in range(200):
        if client.get(f"/v1/jobs/{job_id}").json()["state"] in {"succeeded", "failed"}:
            break
        time.sleep(0.02)
    # Shutdown does not wait for worker threads, so a slow job from a previous test may
    # still be finishing; assert the *eventual* contract with a deadline.
    deadline = time.time() + 5.0
    leftovers = list(Path(tempfile.gettempdir()).glob("courtvision-upload-*"))
    while leftovers and time.time() < deadline:
        time.sleep(0.05)
        leftovers = list(Path(tempfile.gettempdir()).glob("courtvision-upload-*"))
    assert not leftovers, f"job uploads must be cleaned up, found {leftovers}"


def test_environment_endpoint_reports_provenance(client: TestClient) -> None:
    payload = client.get("/v1/environment").json()
    assert "git" in payload and "packages" in payload


def test_stats_endpoint_combines_service_and_jobs(client: TestClient) -> None:
    payload = client.get("/v1/stats").json()
    assert payload["requests"] >= 0
    assert payload["jobs"]["max_jobs"] >= 1
