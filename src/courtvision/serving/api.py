"""FastAPI application.

Hardening decisions, each mapping to a failure mode that bites real services:

* **Blocking work is off the event loop.** The analysis routes are ``def`` (sync), so
  Starlette runs them in its threadpool; an ``async def`` handler doing 30 s of
  inference would stall every other request.
* **Uploads are validated before use**: extension allow-list, declared content type,
  and a size limit enforced *while streaming* so an oversized body is rejected rather
  than buffered.
* **The client filename never becomes a path.** The upload is written into a
  per-request temporary directory under a fixed name, and that directory is always
  removed — on success and on failure.
* **Per-request id in structured logs**, echoed in responses and errors.
* **Large payloads are opt-in** (``detail``), and event lists are capped.

The job store is **in-process only**: jobs are lost on restart and are not shared
between replicas. That is a deliberate scope decision, documented rather than hidden
behind a fake distributed layer.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .. import __version__
from ..config import ApiConfig, EventConfig, ServingConfig, load_config
from ..repro import environment_snapshot
from ..utils import get_logger
from .schemas import (
    AnalysisResponse,
    AnalyzeForm,
    ErrorResponse,
    HealthResponse,
    JobAccepted,
    JobResult,
    JobStatus,
    ReadinessResponse,
)
from .service import AnalysisRequest, AnalysisService, ServiceNotReadyError

LOGGER = get_logger("serving.api")
CHUNK_BYTES = 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    job_id: str
    state: str = "queued"
    submitted_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    result: dict | None = None
    error: str | None = None
    source_path: Path | None = None
    working_dir: Path | None = None
    byte_size: int = 0
    detail: str = "summary"
    params: dict = field(default_factory=dict)

    def to_status(self) -> dict:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "detail": self.detail,
        }


class JobStore:
    """Bounded, TTL-expired, thread-safe job registry backed by a thread pool."""

    def __init__(self, api: ApiConfig | None = None, *, runner=None) -> None:
        self.api = api or ApiConfig()
        self._runner = runner
        self._lock = threading.RLock()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._executor: ThreadPoolExecutor | None = None

    def start(self) -> None:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=max(1, self.api.job_max_workers), thread_name_prefix="courtvision-job")

    def shutdown(self, *, wait: bool = False) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=not wait)

    def set_runner(self, runner) -> None:
        self._runner = runner

    def submit(self, *, source_path: Path, byte_size: int, detail: str, params: dict, working_dir: Path | None = None) -> Job:
        with self._lock:
            self._expire_locked()
            if len(self._jobs) >= self.api.max_jobs:
                raise RuntimeError(
                    f"Job store is full ({self.api.max_jobs} jobs). Jobs expire after "
                    f"{self.api.job_ttl_seconds}s, or raise COURTVISION_API_MAX_JOBS."
                )
            job = Job(
                job_id=uuid.uuid4().hex[:16],
                source_path=source_path,
                working_dir=working_dir if working_dir is not None else source_path.parent,
                byte_size=byte_size,
                detail=detail,
                params=dict(params),
            )
            self._jobs[job.job_id] = job
            executor = self._executor
        if executor is None:
            raise RuntimeError("Job store has not been started")
        executor.submit(self._run, job.job_id)
        return job

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.state, job.started_at = "running", _now()
        try:
            if self._runner is None:
                raise RuntimeError("No job runner is configured")
            result = self._runner(job)
        except Exception as exc:
            LOGGER.warning("job %s failed: %s", job_id, exc)
            with self._lock:
                job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
        else:
            with self._lock:
                job.result = result
        finally:
            # Publish completion only after the upload is gone: a client that sees
            # "succeeded" must never race a still-present temporary file.
            self._cleanup(job_id)
            with self._lock:
                if job.state == "running":
                    job.state = "succeeded"
                job.finished_at = _now()

    def _cleanup(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            working_dir = job.working_dir if job else None
            path = job.source_path if job else None
        if working_dir is not None:
            shutil.rmtree(working_dir, ignore_errors=True)
        elif path is not None:
            with suppress(OSError):
                path.unlink(missing_ok=True)
        with self._lock:
            if job is not None:
                job.source_path = job.working_dir = None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._expire_locked()
            return self._jobs.get(job_id)

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != "queued":
                return False
            job.state, job.finished_at = "cancelled", _now()
            return True

    def _expire_locked(self) -> None:
        if self.api.job_ttl_seconds <= 0:
            return
        cutoff = time.time() - self.api.job_ttl_seconds
        for job_id in [
            job_id
            for job_id, job in self._jobs.items()
            if job.state in {"succeeded", "failed", "cancelled"} and job.finished_at and _timestamp(job.finished_at) < cutoff
        ]:
            self._jobs.pop(job_id, None)
            self._cleanup(job_id)

    def stats(self) -> dict:
        with self._lock:
            states: dict[str, int] = {}
            for job in self._jobs.values():
                states[job.state] = states.get(job.state, 0) + 1
            return {
                "jobs": len(self._jobs),
                "by_state": states,
                "max_jobs": self.api.max_jobs,
                "ttl_seconds": self.api.job_ttl_seconds,
                "workers": self.api.job_max_workers,
                "distributed": False,
                "note": (
                    "In-process store: jobs are lost on restart and are not shared between replicas. "
                    "The Airflow DAG is the documented path to real orchestration."
                ),
            }


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:  # pragma: no cover - defensive
        return time.time()


@dataclass
class ApplicationState:
    service: AnalysisService
    api: ApiConfig
    jobs: JobStore
    metrics: Any = None


def _analyze_form(
    tracker: Annotated[str | None, Form()] = None,
    conf: Annotated[float, Form(gt=0.0, le=1.0)] = 0.25,
    imgsz: Annotated[int, Form(ge=32, le=4096)] = 640,
    max_frames: Annotated[int | None, Form(ge=1)] = None,
    detail: Annotated[str, Form(pattern="^(summary|tracks|full)$")] = "summary",
    include_events: Annotated[bool, Form()] = True,
) -> AnalyzeForm:
    """Bind the multipart form fields into a validated :class:`AnalyzeForm`.

    This must read from ``Form``, not ``Depends()``: a Pydantic model used directly as a
    dependency binds from the **query string**, which would silently ignore the documented
    multipart parameters and accept ``conf=1.5`` at its default.
    """
    return AnalyzeForm(
        tracker=tracker,
        conf=conf,
        imgsz=imgsz,
        max_frames=max_frames,
        detail=detail,  # type: ignore[arg-type]
        include_events=include_events,
    )


def create_app(
    serving: ServingConfig | None = None,
    api_config: ApiConfig | None = None,
    events: EventConfig | None = None,
    *,
    preload: bool | None = None,
) -> FastAPI:
    """Build the API. Configuration comes from the config file/env unless supplied."""
    if serving is None or api_config is None or events is None:
        config = load_config()
        serving = serving or config.serving
        api_config = api_config or config.api
        events = events or config.events
    should_preload = serving.preload if preload is None else preload

    service = AnalysisService(serving, api_config, events)
    state = ApplicationState(service=service, api=api_config, jobs=JobStore(api_config))
    if serving.enable_metrics:
        from ..monitoring.metrics import REGISTRY

        state.metrics = REGISTRY

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        if should_preload:
            service.load()  # a broken checkpoint must fail the deploy, not a request
        state.jobs.start()
        state.jobs.set_runner(lambda job: _run_job(state, job))
        LOGGER.info("CourtVision API started", extra={"weights": serving.model, "device": serving.device, "preload": should_preload})
        try:
            yield
        finally:
            state.jobs.shutdown(wait=False)
            service.unload()
            LOGGER.info("CourtVision API stopped")

    app = FastAPI(
        title="CourtVision API",
        version=__version__,
        description=(
            "Sports video analytics: detection, multi-object tracking, trajectory and event analysis. "
            "Metrics are measured, never estimated; see /v1/capabilities."
        ),
        lifespan=lifespan,
    )
    app.state.courtvision = state

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        # Handlers and error responses read the id from request.state; without this the
        # response payload's request_id field would always be empty.
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["x-request-id"] = request_id
        LOGGER.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=str(exc.detail), request_id=request.headers.get("x-request-id")).model_dump(),
        )

    # -- operational endpoints --------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness only: the process is serving. Deliberately says nothing about the model."""
        return HealthResponse(status="ok", version=__version__, uptime_seconds=round(service.uptime_seconds, 3))

    @app.get("/ready", response_model=ReadinessResponse, tags=["ops"])
    def ready() -> ReadinessResponse:
        """Readiness, including why not ready. Route traffic on this, not on /health."""
        return ReadinessResponse(**service.readiness())

    @app.get("/metrics", tags=["ops"])
    def metrics() -> PlainTextResponse:
        if state.metrics is None:
            raise HTTPException(status_code=404, detail="Metrics are disabled (COURTVISION_SERVING_ENABLE_METRICS=false)")
        return PlainTextResponse(state.metrics.render_prometheus(), media_type="text/plain; version=0.0.4")

    # -- versioned API -----------------------------------------------------

    @app.get("/v1/capabilities", tags=["v1"])
    def capabilities() -> dict:
        return {
            "capabilities": service.capabilities(),
            "jobs": state.jobs.stats(),
            "limits": {
                "max_upload_mb": api_config.max_upload_mb,
                "allowed_extensions": list(api_config.allowed_extensions),
                "max_frames": api_config.max_frames,
                "max_event_list": api_config.max_event_list,
            },
        }

    @app.get("/v1/stats", tags=["v1"])
    def stats() -> dict:
        payload = service.stats.to_dict()
        payload["jobs"] = state.jobs.stats()
        payload["uptime_seconds"] = round(service.uptime_seconds, 3)
        return payload

    @app.post("/v1/analyze", response_model=AnalysisResponse, tags=["v1"])
    def analyze(
        request: Request,
        form: Annotated[AnalyzeForm, Depends(_analyze_form)],
        video: UploadFile = File(...),
    ) -> AnalysisResponse:
        """Analyse an uploaded video synchronously; use /v1/jobs for long clips."""
        _assert_ready(service)
        return AnalysisResponse(**_analyze_upload(state, video, form, request_id=getattr(request.state, "request_id", "")))

    @app.post("/v1/jobs", response_model=JobAccepted, status_code=202, tags=["v1"])
    def submit_job(
        form: Annotated[AnalyzeForm, Depends(_analyze_form)],
        video: UploadFile = File(...),
    ) -> JobAccepted:
        """Queue an analysis and return immediately with a job id."""
        _assert_ready(service)
        working_dir, path, size = _store_upload(state, video)
        try:
            job = state.jobs.submit(
                source_path=path,
                working_dir=working_dir,
                byte_size=size,
                detail=form.detail,
                params={
                    "tracker": form.tracker,
                    "conf": form.conf,
                    "imgsz": form.imgsz,
                    "max_frames": form.max_frames,
                    "include_events": form.include_events,
                    "source_name": _safe_name(video.filename),
                },
            )
        except RuntimeError as exc:
            shutil.rmtree(working_dir, ignore_errors=True)
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return JobAccepted(
            job_id=job.job_id,
            state=job.state,
            submitted_at=job.submitted_at,
            poll_url=f"/v1/jobs/{job.job_id}",
            result_url=f"/v1/jobs/{job.job_id}/result",
        )

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus, tags=["v1"])
    def job_status(job_id: str) -> JobStatus:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown or expired job {job_id}")
        return JobStatus(**job.to_status())

    @app.get("/v1/jobs/{job_id}/result", response_model=JobResult, tags=["v1"])
    def job_result(job_id: str) -> JobResult:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown or expired job {job_id}")
        if job.state == "failed":
            return JobResult(job_id=job.job_id, state=job.state, error=job.error)
        if job.state != "succeeded" or job.result is None:
            raise HTTPException(status_code=409, detail=f"Job {job_id} is {job.state}; poll /v1/jobs/{job_id}")
        return JobResult(job_id=job.job_id, state=job.state, result=AnalysisResponse(**job.result))

    @app.delete("/v1/jobs/{job_id}", tags=["v1"])
    def cancel_job(job_id: str) -> dict:
        if not state.jobs.cancel(job_id):
            raise HTTPException(status_code=409, detail=f"Job {job_id} is not cancellable (unknown, finished or running)")
        return {"job_id": job_id, "state": "cancelled"}

    @app.get("/v1/environment", tags=["v1"])
    def environment() -> dict:
        """Provenance of this deployment, for reproducing a reported number."""
        return environment_snapshot()

    return app


def _assert_ready(service: AnalysisService) -> None:
    if not service.model_loaded:
        try:
            service.load()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


def _safe_name(filename: str | None) -> str:
    """Reduce a client filename to a display name with no path semantics."""
    if not filename:
        return "upload.mp4"
    return Path(filename).name.replace("\\", "_")[:200] or "upload.mp4"


def _validate_upload(state: ApplicationState, upload: UploadFile) -> str:
    suffix = Path(_safe_name(upload.filename)).suffix.lower()
    if not suffix:
        raise HTTPException(
            status_code=415,
            detail=f"Upload has no extension, so it cannot be verified as video. Allowed: {list(state.api.allowed_extensions)}",
        )
    if suffix not in state.api.allowed_extensions:
        raise HTTPException(status_code=415, detail=f"Unsupported extension {suffix!r}. Allowed: {list(state.api.allowed_extensions)}")
    if upload.content_type and upload.content_type not in state.api.allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type {upload.content_type!r}. Allowed: {list(state.api.allowed_content_types)}",
        )
    return suffix


def _store_upload(state: ApplicationState, upload: UploadFile) -> tuple[Path, Path, int]:
    """Stream an upload into a per-request temp dir, enforcing the limit while streaming."""
    suffix = _validate_upload(state, upload)
    limit_bytes = int(state.api.max_upload_mb * 1024 * 1024)
    working_dir = Path(tempfile.mkdtemp(prefix="courtvision-upload-"))
    path = working_dir / f"input{suffix}"
    written = 0
    try:
        with path.open("wb") as handle:
            while True:
                chunk = upload.file.read(CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {state.api.max_upload_mb:g} MiB limit (COURTVISION_API_MAX_UPLOAD_MB)",
                    )
                handle.write(chunk)
    except HTTPException:
        shutil.rmtree(working_dir, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(working_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Could not store upload: {exc}") from exc
    finally:
        with suppress(OSError):  # pragma: no cover
            upload.file.close()

    if written == 0:
        shutil.rmtree(working_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Empty upload")
    return working_dir, path, written


def _analyze_upload(state: ApplicationState, upload: UploadFile, form: AnalyzeForm, *, request_id: str) -> dict:
    working_dir, path, size = _store_upload(state, upload)
    try:
        return _execute(
            state,
            AnalysisRequest(
                video_path=path,
                tracker=form.tracker,
                conf=form.conf,
                imgsz=form.imgsz,
                max_frames=form.max_frames or state.api.max_frames,
                include_tracks=form.detail in {"tracks", "full"},
                include_trajectories=form.detail == "full",
                include_events=form.include_events,
                source_name=_safe_name(upload.filename),
                byte_size=size,
            ),
            request_id=request_id,
        )
    finally:
        shutil.rmtree(working_dir, ignore_errors=True)  # always, success or failure


def _execute(state: ApplicationState, request: AnalysisRequest, *, request_id: str) -> dict:
    try:
        payload = state.service.analyze(request)
    except ServiceNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"Unreadable video: {exc}") from exc
    payload["request_id"] = request_id
    snapshot = environment_snapshot()
    payload["system"] = {
        "python_version": snapshot["python_version"],
        "packages": snapshot["packages"],
        "cuda_available": snapshot["hardware"].get("cuda_available", False),
        "gpu": snapshot["hardware"].get("gpu"),
    }
    if state.metrics is not None:
        state.metrics.record_analysis(
            frames=payload["video"]["frames"],
            detections=payload["detections"]["total"],
            unique_tracks=payload["tracks"]["unique_tracks"],
            latency_ms=payload["runtime"]["wall_seconds"] * 1000.0,
        )
    return payload


def _run_job(state: ApplicationState, job: Job) -> dict:
    if job.source_path is None:
        raise RuntimeError("Job source file is no longer available")
    params = job.params
    return _execute(
        state,
        AnalysisRequest(
            video_path=job.source_path,
            tracker=params.get("tracker"),
            conf=params.get("conf"),
            imgsz=params.get("imgsz"),
            max_frames=params.get("max_frames") or state.api.max_frames,
            include_tracks=job.detail in {"tracks", "full"},
            include_trajectories=job.detail == "full",
            include_events=bool(params.get("include_events", True)),
            source_name=params.get("source_name"),
            byte_size=job.byte_size,
        ),
        request_id=f"job-{job.job_id}",
    )


#: Module-level ASGI app for ``uvicorn courtvision.serving.api:app``.
app = create_app()
