"""Analysis service: model lifecycle, tracking, analytics and result assembly.

Rules this module enforces:

* **The model is loaded once** and loading is explicit, so startup fails loudly rather
  than the first request doing so.
* **Tracker state is reset per request** by the backend, so id assignment cannot depend
  on how many videos this process has already seen.
* **Blocking work never runs on the event loop.** :meth:`analyze` is synchronous and
  CPU-bound; the API layer runs it in a worker thread.
* **Frame count is bounded**, and truncation is reported in the response rather than
  being silent.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..analytics.events import detect_events, event_rate, summarize_events
from ..analytics.trajectories import analyze_trajectories, track_statistics
from ..config import ApiConfig, EventConfig, ServingConfig
from ..detection.ultralytics_detector import (
    available_trackers,
    describe_capabilities,
    tracker_is_supported,
    ultralytics_version,
)
from ..tracking.base import TrackingResult
from ..tracking.ultralytics_tracker import UltralyticsTracker
from ..utils import get_logger

LOGGER = get_logger("serving.service")


class ServiceNotReadyError(RuntimeError):
    """Raised when analysis is requested before the model is loaded."""


@dataclass
class AnalysisRequest:
    """One analysis job's parameters."""

    video_path: Path
    tracker: str | None = None
    conf: float | None = None
    imgsz: int | None = None
    max_frames: int | None = None
    include_tracks: bool = False
    include_trajectories: bool = False
    include_events: bool = True
    source_name: str | None = None
    byte_size: int | None = None


@dataclass
class ServiceStats:
    """Operational counters exposed at ``/v1/stats``."""

    requests: int = 0
    failures: int = 0
    frames_processed: int = 0
    total_wall_seconds: float = 0.0
    last_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "requests": self.requests,
            "failures": self.failures,
            "frames_processed": self.frames_processed,
            "total_wall_seconds": round(self.total_wall_seconds, 4),
            "mean_seconds_per_request": (round(self.total_wall_seconds / self.requests, 4) if self.requests else None),
            "last_error": self.last_error,
        }


class AnalysisService:
    """Owns the detector/tracker and turns videos into structured analytics."""

    def __init__(self, serving: ServingConfig | None = None, api: ApiConfig | None = None, events: EventConfig | None = None) -> None:
        self.serving = serving or ServingConfig()
        self.api = api or ApiConfig()
        self.events = events or EventConfig()
        self._tracker: UltralyticsTracker | None = None
        self._lock = threading.Lock()
        self._started_at = time.time()
        self.stats = ServiceStats()

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load the detector/tracker once. Safe to call repeatedly."""
        with self._lock:
            if self._tracker is not None:
                return
            if not tracker_is_supported(self.serving.tracker):
                raise ValueError(
                    f"Tracker {self.serving.tracker!r} is not supported by ultralytics "
                    f"{ultralytics_version()}. Supported: {available_trackers()}"
                )
            self._tracker = UltralyticsTracker(
                weights=self.serving.model,
                tracker=self.serving.tracker,
                device=self.serving.device,
                imgsz=self.serving.imgsz,
                conf=self.serving.conf,
            )
            _ = self._tracker.model  # touch it so a bad checkpoint fails here, not mid-request
            LOGGER.info(
                "model loaded",
                extra={"weights": self.serving.model, "device": self.serving.device, "tracker": self.serving.tracker},
            )

    def unload(self) -> None:
        with self._lock:
            if self._tracker is not None:
                self._tracker.close()
                self._tracker = None
            LOGGER.info("model unloaded")

    @property
    def model_loaded(self) -> bool:
        return self._tracker is not None

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    def readiness(self) -> dict:
        """Everything a readiness probe needs, including why it is not ready."""
        reasons: list[str] = []
        supported = tracker_is_supported(self.serving.tracker)
        if not supported:
            reasons.append(f"tracker {self.serving.tracker!r} is not shipped by the installed ultralytics")
        if not self.model_loaded:
            reasons.append("model is not loaded")
        return {
            "ready": not reasons,
            "model_loaded": self.model_loaded,
            "weights": self.serving.model,
            "device": self.serving.device,
            "tracker": self.serving.tracker,
            "tracker_supported": supported,
            "reasons": reasons,
        }

    def capabilities(self) -> dict:
        return {
            **describe_capabilities(),
            "serving": {
                "weights": self.serving.model,
                "tracker": self.serving.tracker,
                "device": self.serving.device,
                "imgsz": self.serving.imgsz,
                "conf": self.serving.conf,
            },
        }

    # -- analysis ----------------------------------------------------------

    def analyze(self, request: AnalysisRequest) -> dict:
        """Run the pipeline on one video and return the response payload."""
        if self._tracker is None:
            raise ServiceNotReadyError("Model is not loaded; call load() first")

        started = time.perf_counter()
        tracker_name = request.tracker or self.serving.tracker
        if not tracker_is_supported(tracker_name):
            raise ValueError(f"Tracker {tracker_name!r} is not supported. Supported: {available_trackers()}")
        conf = self.serving.conf if request.conf is None else request.conf
        imgsz = self.serving.imgsz if request.imgsz is None else request.imgsz

        backend = self._tracker
        previous = (backend.tracker, backend.conf, backend.imgsz)
        backend.tracker, backend.conf, backend.imgsz = tracker_name, conf, imgsz
        try:
            result = backend.track(request.video_path, max_frames=request.max_frames)
        finally:
            backend.tracker, backend.conf, backend.imgsz = previous

        truncated = bool(request.max_frames) and len(result.frames) >= request.max_frames
        payload = self._build_payload(request, result, conf=conf, imgsz=imgsz, truncated=truncated)
        elapsed = time.perf_counter() - started
        payload["runtime"]["wall_seconds"] = round(elapsed, 4)
        payload["runtime"]["frames_per_second"] = (
            round(len(result.frames) / elapsed, 4) if elapsed > 0 and result.frames else 0.0
        )
        with self._lock:
            self.stats.requests += 1
            self.stats.frames_processed += len(result.frames)
            self.stats.total_wall_seconds += elapsed
        return payload

    def _build_payload(self, request: AnalysisRequest, result: TrackingResult, *, conf: float, imgsz: int, truncated: bool) -> dict:
        frames = result.frames
        counts = result.detection_counts()
        confidences = result.confidences()
        tracks = result.tracks()
        durations = [len(items) for items in tracks.values()]
        warnings: list[str] = []
        if truncated:
            warnings.append(
                f"Analysis stopped after {len(frames)} frames (max_frames={request.max_frames}); metrics "
                "describe the processed prefix only."
            )
        if not frames:
            warnings.append("No frames were decoded from this source.")

        payload: dict[str, Any] = {
            "video": {
                "filename": request.source_name or Path(result.source).name,
                "bytes": int(request.byte_size or 0),
                "frames": len(frames),
                "fps": result.source_fps,
                "width": result.frame_size[0] if result.frame_size else None,
                "height": result.frame_size[1] if result.frame_size else None,
                "duration_seconds": (round(len(frames) / result.source_fps, 4) if result.source_fps and frames else None),
            },
            # wall_seconds/fps are filled by the caller once the wall clock is known.
            "runtime": {"wall_seconds": 0.0, "frames_per_second": 0.0, "frames_processed": len(frames), "device": result.meta.get("device", self.serving.device)},
            "detections": {
                "total": result.detections,
                "per_frame_mean": round(sum(counts) / len(counts), 4) if counts else 0.0,
                "per_frame_max": max(counts) if counts else 0,
                "mean_confidence": round(sum(confidences) / len(confidences), 6) if confidences else 0.0,
                "confidence_min": round(min(confidences), 6) if confidences else None,
                "confidence_max": round(max(confidences), 6) if confidences else None,
            },
            "tracks": {
                "unique_tracks": len(tracks),
                "mean_duration_frames": round(sum(durations) / len(durations), 4) if durations else 0.0,
                "max_duration_frames": max(durations) if durations else 0,
                "tracks_with_gaps": self._count_gapped_tracks(tracks),
            },
            "model": {
                "weights": self.serving.model,
                "tracker": result.tracker or self.serving.tracker,
                "imgsz": imgsz,
                "confidence": conf,
                "backend": f"ultralytics {ultralytics_version()}",
                "parameters": None,
            },
            "quality": {"truncated": truncated, "warnings": warnings},
        }

        if request.include_tracks:
            payload["tracks_detail"] = [
                {"frame": frame.frame, "detections": [item.to_dict() for item in frame.detections]} for frame in frames
            ]

        if request.include_trajectories or request.include_events:
            _trajectories, features = analyze_trajectories(
                result,
                min_track_length=max(1, min(self.events.min_track_length, 5)),
                smoothing_window=self.events.smoothing_window,
                zone_grid=self.events.zone_grid,
                stationary_speed_threshold=self.events.stationary_speed,
            )
            if request.include_trajectories:
                payload["trajectories"] = [feature.to_dict() for feature in features]
                payload["trajectory_statistics"] = track_statistics(features)
            if request.include_events:
                events = detect_events(features, self.events)
                payload["events"] = {
                    "summary": summarize_events(events),
                    "rate": event_rate(events, len(frames), result.source_fps),
                    "detector": "heuristic-thresholds",
                    "accuracy_note": (
                        "Event detection is deterministic threshold logic. No labelled temporal event "
                        "dataset is available here, so event accuracy is NOT MEASURED."
                    ),
                }
                payload["event_list"] = [event.to_dict() for event in events[: self.api.max_event_list]]
                if len(events) > self.api.max_event_list:
                    payload["quality"]["warnings"].append(
                        f"Event list truncated to {self.api.max_event_list} of {len(events)} events."
                    )
        return payload

    @staticmethod
    def _count_gapped_tracks(tracks: dict[int, list]) -> int:
        gapped = 0
        for detections in tracks.values():
            frames = sorted(item.frame for item in detections)
            if any(b - a > 1 for a, b in pairwise(frames)):
                gapped += 1
        return gapped
