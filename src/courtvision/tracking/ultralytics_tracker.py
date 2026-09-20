"""Ultralytics tracker backend.

Ultralytics keeps its tracker objects on the model instance. In 8.4.156 it resets them
when the source changes, and this was verified directly: two different clips tracked
back-to-back with one model produced ids ``[1,2,3,4,5]`` and ``[1,2,3,4,6,7]``, and
three repeats of the *same* clip produced identical id sets every time.

That reset is still performed explicitly here, for two reasons that matter for a
service: it guarantees id assignment cannot depend on how many videos this process has
already seen, and it releases the per-video capture handles and tracker buffers
instead of relying on the library's internal behaviour staying the same.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..detection.ultralytics_detector import (
    UltralyticsDetector,
    resolve_tracker_name,
    ultralytics_version,
    verify_model_identifier,
)
from ..repro import resolve_device
from ..tracking.base import Detection, FrameResult, TrackingResult
from ..utils import ensure_dir, write_csv, write_json, write_text

#: Tracker YAML keys CourtVision is willing to override.
TRACKER_OVERRIDE_KEYS = (
    "track_high_thresh",
    "track_low_thresh",
    "new_track_thresh",
    "match_thresh",
    "track_buffer",
    "fuse_score",
)


@dataclass
class UltralyticsTracker:
    """Multi-object tracking via Ultralytics, with explicit per-call state isolation."""

    weights: str = "yolo26n.pt"
    tracker: str = "bytetrack.yaml"
    device: str = "cpu"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    half: bool = False
    overrides: Mapping[str, float | int | bool] = field(default_factory=dict)
    name: str = "ultralytics-tracker"

    _model: Any = None
    _tracker_config_path: Path | None = None
    _temporary_dir: tempfile.TemporaryDirectory | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            verify_model_identifier(self.weights)
            resolve_tracker_name(self.tracker)
            self._model = YOLO(self.weights)
        return self._model

    def close(self) -> None:
        self.reset_state()
        self._model = None
        if self._temporary_dir is not None:
            self._temporary_dir.cleanup()
            self._temporary_dir = None
            self._tracker_config_path = None

    def __enter__(self) -> "UltralyticsTracker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def tracker_config(self) -> str:
        """Resolve the tracker argument, materialising overrides into a temp YAML.

        Writing a merged config keeps the effective configuration reproducible and
        inspectable instead of mutating library internals in memory.
        """
        if not self.overrides:
            return self.tracker
        if self._tracker_config_path is not None:
            return str(self._tracker_config_path)

        unknown = set(self.overrides) - set(TRACKER_OVERRIDE_KEYS)
        if unknown:
            raise ValueError(
                f"Unsupported tracker override(s) {sorted(unknown)}; supported: {list(TRACKER_OVERRIDE_KEYS)}"
            )
        base = self._load_base_tracker_config()
        base.update({key: value for key, value in self.overrides.items()})
        self._temporary_dir = tempfile.TemporaryDirectory(prefix="courtvision-tracker-")
        path = Path(self._temporary_dir.name) / f"{Path(self.tracker).stem}-override.yaml"
        write_text(path, yaml.safe_dump(base, sort_keys=False))
        self._tracker_config_path = path
        return str(path)

    def _load_base_tracker_config(self) -> dict:
        from ..detection.ultralytics_detector import tracker_config_dir

        shipped = Path(self.tracker)
        if not shipped.is_file():
            shipped = tracker_config_dir() / self.tracker
        if not shipped.is_file():
            raise FileNotFoundError(
                f"Tracker config {self.tracker!r} not found. Run `courtvision doctor` to list supported trackers."
            )
        loaded = yaml.safe_load(shipped.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Tracker config {shipped} is not a YAML mapping")
        return loaded

    def reset_state(self) -> None:
        """Drop tracker state so the next call cannot depend on this one."""
        predictor = getattr(self._model, "predictor", None)
        if predictor is None:
            return
        for tracker in getattr(predictor, "trackers", None) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
        for attribute in ("vid_path", "vid_cap", "vid_writer"):
            if hasattr(predictor, attribute):
                with suppress(AttributeError):  # pragma: no cover - property without setter
                    setattr(predictor, attribute, None)

    def track(
        self,
        source: str | Path,
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        max_frames: int | None = None,
    ) -> TrackingResult:
        """Track one video. Ids are stable inside the call and reset afterwards."""
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou
        imgsz = self.imgsz if imgsz is None else imgsz
        device = resolve_device(self.device)

        self.reset_state()
        started = time.perf_counter()
        try:
            results = self.model.track(
                source=str(source),
                tracker=self.tracker_config(),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=self.max_det,
                device=device,
                half=self.half and device != "cpu",
                stream=True,
                persist=True,  # stable ids within this video
                verbose=False,
            )
            frames: list[FrameResult] = []
            frame_size: tuple[int, int] | None = None
            source_fps: float | None = None
            for index, result in enumerate(results, start=1):
                if max_frames is not None and index > max_frames:
                    break
                if frame_size is None and getattr(result, "orig_shape", None):
                    height, width = result.orig_shape[:2]
                    frame_size = (int(width), int(height))
                if source_fps is None:
                    speed = getattr(result, "speed", None) or {}
                    if speed.get("fps"):
                        source_fps = float(speed["fps"])
                frames.append(
                    FrameResult(
                        frame=index,
                        detections=self._to_detections(result, index),
                        timings_ms=_timings(result),
                    )
                )
        finally:
            # Never let this video's state bleed into the next call, even on failure.
            self.reset_state()
        elapsed = time.perf_counter() - started

        return TrackingResult(
            source=str(source),
            frames=frames,
            frame_size=frame_size,
            source_fps=source_fps,
            tracker=Path(self.tracker).name,
            weights=self.weights,
            meta={
                "task": "track",
                "conf": conf,
                "iou": iou,
                "imgsz": imgsz,
                "device": device,
                "overrides": dict(self.overrides),
                "ultralytics_version": ultralytics_version(),
                "wall_seconds": round(elapsed, 6),
                "fps_wall": round(len(frames) / elapsed, 6) if elapsed > 0 and frames else None,
            },
        )

    def track_frames(
        self, frames: Sequence[np.ndarray], *, conf: float | None = None, imgsz: int | None = None
    ) -> TrackingResult:
        """Track an in-memory frame sequence with the same state isolation."""
        conf = self.conf if conf is None else conf
        imgsz = self.imgsz if imgsz is None else imgsz
        device = resolve_device(self.device)
        self.reset_state()
        try:
            results: list[FrameResult] = []
            for index, frame in enumerate(frames, start=1):
                produced = self.model.track(
                    source=frame,
                    tracker=self.tracker_config(),
                    conf=conf,
                    iou=self.iou,
                    imgsz=imgsz,
                    max_det=self.max_det,
                    device=device,
                    half=self.half and device != "cpu",
                    persist=True,
                    verbose=False,
                )
                result = produced[0] if isinstance(produced, (list, tuple)) else produced
                results.append(
                    FrameResult(
                        frame=index, detections=self._to_detections(result, index), timings_ms=_timings(result)
                    )
                )
        finally:
            self.reset_state()
        frame_size = None
        if frames:
            height, width = np.asarray(frames[0]).shape[:2]
            frame_size = (int(width), int(height))
        return TrackingResult(
            source="<frames>",
            frames=results,
            frame_size=frame_size,
            tracker=Path(self.tracker).name,
            weights=self.weights,
            meta={"task": "track", "conf": conf, "imgsz": imgsz, "device": device},
        )

    @staticmethod
    def _to_detections(result: Any, frame: int) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
        ids = boxes.id.int().cpu().numpy() if boxes.id is not None else None
        return [
            Detection(
                frame=frame,
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(score),
                class_id=int(class_id),
                track_id=int(ids[position]) if ids is not None else None,
            )
            for position, ((x1, y1, x2, y2), score, class_id) in enumerate(
                zip(xyxy, confidences, classes, strict=True)
            )
        ]

    def detector(self) -> UltralyticsDetector:
        """A detection-only view sharing these settings."""
        return UltralyticsDetector(
            weights=self.weights,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            half=self.half,
        )


def _timings(result: Any) -> dict[str, float]:
    speed = getattr(result, "speed", None)
    if not isinstance(speed, dict):
        return {}
    return {key: float(value) for key, value in speed.items() if isinstance(value, (int, float))}


def export_tracking_artifacts(
    result: TrackingResult, output_dir: str | Path, *, include_mot: bool = True, include_details: bool = True
) -> dict[str, Path]:
    """Persist a tracking run: MOT file, per-detection CSV and a summary."""
    from ..tracking.base import tracking_result_to_records

    output_dir = ensure_dir(output_dir)
    paths: dict[str, Path] = {}
    if include_mot:
        paths["mot"] = result.write_mot(output_dir / "predictions.txt")
    if include_details:
        paths["detections_csv"] = write_csv(output_dir / "detections.csv", tracking_result_to_records(result))
    paths["summary"] = write_json(output_dir / "summary.json", result.to_dict())
    return paths
