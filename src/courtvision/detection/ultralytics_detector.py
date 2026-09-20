"""Ultralytics detector backend and runtime capability discovery.

Everything this module claims about the installed package is *queried from* the
installed package, because model and tracker names change between releases.
``courtvision doctor`` prints exactly these answers, so a README claim can always be
checked against the runtime.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ..repro import resolve_device
from ..tracking.base import Detection, FrameResult, TrackingResult
from .base import DetectorInfo, ValidationResult


class ModelNotAvailableError(RuntimeError):
    """Raised when a checkpoint name is not resolvable by the installed Ultralytics."""


def ultralytics_version() -> str:
    try:
        return metadata.version("ultralytics")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def tracker_config_dir() -> Path:
    import ultralytics

    return Path(ultralytics.__file__).parent / "cfg" / "trackers"


def available_trackers() -> list[str]:
    """Tracker config names the installed release ships (a filesystem query)."""
    directory = tracker_config_dir()
    return sorted(path.name for path in directory.glob("*.yaml")) if directory.is_dir() else []


def tracker_is_supported(name: str) -> bool:
    if name in available_trackers():
        return True
    candidate = Path(name)
    return candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml"}


def downloadable_asset_names() -> list[str]:
    """Checkpoint names Ultralytics can fetch, from its own registry."""
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

        return sorted(GITHUB_ASSETS_NAMES)
    except (ImportError, AttributeError):  # pragma: no cover - registry moved
        return []


def model_is_available(name: str) -> bool:
    return Path(name).is_file() or name in downloadable_asset_names()


def verify_model_identifier(name: str) -> str:
    """Validate a checkpoint identifier, suggesting alternatives when unknown."""
    if model_is_available(name):
        return name
    assets = downloadable_asset_names()
    stem = Path(name).stem.lower()
    suggestions = sorted(asset for asset in assets if stem[:6] in asset.lower())[:8]
    if not suggestions and stem[:3]:
        suggestions = sorted(asset for asset in assets if stem[:3] in asset.lower())[:8]
    hint = f" Did you mean: {suggestions}?" if suggestions else ""
    raise ModelNotAvailableError(
        f"{name!r} is neither an existing file nor a checkpoint known to ultralytics "
        f"{ultralytics_version()}.{hint} Run `courtvision doctor` to list identifiers."
    )


def resolve_tracker_name(name: str) -> str:
    if tracker_is_supported(name):
        return name
    raise ModelNotAvailableError(
        f"Tracker {name!r} is not shipped by ultralytics {ultralytics_version()} and is not an "
        f"existing YAML file. Supported: {available_trackers()}"
    )


def yolo26_assets() -> list[str]:
    return sorted(asset for asset in downloadable_asset_names() if asset.startswith("yolo26"))


def describe_capabilities() -> dict[str, Any]:
    """Runtime capability report, queried rather than hardcoded."""
    report: dict[str, Any] = {
        "ultralytics_version": ultralytics_version(),
        "trackers": available_trackers(),
        "yolo26_assets": yolo26_assets(),
    }
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = torch.version.cuda
        report["device_count"] = torch.cuda.device_count()
        report["device_names"] = [
            torch.cuda.get_device_properties(index).name for index in range(torch.cuda.device_count())
        ]
    except ModuleNotFoundError:  # pragma: no cover
        report["torch_version"] = None
        report["cuda_available"] = False
    return report


@dataclass
class UltralyticsDetector:
    """Detection-only wrapper around an Ultralytics model.

    Uses ``predict`` rather than ``track`` so it never leaves tracking state behind,
    which is what makes detection and tracking evaluation independently meaningful.
    """

    weights: str = "yolo26n.pt"
    device: str = "cpu"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    half: bool = False
    name: str = "ultralytics-detector"

    _model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            verify_model_identifier(self.weights)
            self._model = YOLO(self.weights)
        return self._model

    def info(self) -> DetectorInfo:
        model = self.model
        names = getattr(model, "names", None) or {}
        if not isinstance(names, dict):
            names = {index: str(name) for index, name in enumerate(names)}
        try:
            parameters = int(sum(parameter.numel() for parameter in model.model.parameters()))
        except (AttributeError, TypeError):  # pragma: no cover
            parameters = None
        return DetectorInfo(
            backend=f"ultralytics {ultralytics_version()}",
            weights=self.weights,
            task=getattr(model, "task", None),
            class_names={int(key): str(value) for key, value in names.items()},
            device=self.device,
            imgsz=self.imgsz,
            parameters=parameters,
        )

    def detect(
        self,
        source: str | Path,
        *,
        conf: float | None = None,
        imgsz: int | None = None,
        max_frames: int | None = None,
    ) -> TrackingResult:
        device = resolve_device(self.device)
        results = self.model.predict(
            source=str(source),
            conf=self.conf if conf is None else conf,
            iou=self.iou,
            imgsz=self.imgsz if imgsz is None else imgsz,
            max_det=self.max_det,
            device=device,
            half=self.half and device != "cpu",
            stream=True,
            verbose=False,
        )
        frames: list[FrameResult] = []
        frame_size: tuple[int, int] | None = None
        for index, result in enumerate(results, start=1):
            if max_frames is not None and index > max_frames:
                break
            if frame_size is None and getattr(result, "orig_shape", None):
                height, width = result.orig_shape[:2]
                frame_size = (int(width), int(height))
            frames.append(
                FrameResult(frame=index, detections=self._to_detections(result, index), timings_ms=_timings(result))
            )
        return TrackingResult(
            source=str(source),
            frames=frames,
            frame_size=frame_size,
            weights=self.weights,
            meta={"task": "detect", "device": device, "imgsz": self.imgsz if imgsz is None else imgsz},
        )

    def detect_frames(
        self, frames: Sequence[np.ndarray], *, conf: float | None = None, imgsz: int | None = None
    ) -> TrackingResult:
        device = resolve_device(self.device)
        results = self.model.predict(
            source=list(frames),
            conf=self.conf if conf is None else conf,
            iou=self.iou,
            imgsz=self.imgsz if imgsz is None else imgsz,
            max_det=self.max_det,
            device=device,
            half=self.half and device != "cpu",
            verbose=False,
        )
        out = [
            FrameResult(frame=index, detections=self._to_detections(result, index), timings_ms=_timings(result))
            for index, result in enumerate(results, start=1)
        ]
        frame_size = None
        if frames:
            height, width = np.asarray(frames[0]).shape[:2]
            frame_size = (int(width), int(height))
        return TrackingResult(source="<frames>", frames=out, frame_size=frame_size, weights=self.weights)

    @staticmethod
    def _to_detections(result: Any, frame: int) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
        return [
            Detection(
                frame=frame,
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(score),
                class_id=int(class_id),
            )
            for (x1, y1, x2, y2), score, class_id in zip(xyxy, confidences, classes, strict=True)
        ]

    def validate(
        self, data: str | Path, *, split: str = "val", imgsz: int | None = None, batch: int = 8, plots: bool = False
    ) -> ValidationResult:
        """Run Ultralytics' own mAP evaluation (the reference implementation)."""
        metrics = self.model.val(
            data=str(data),
            split=split,
            imgsz=self.imgsz if imgsz is None else imgsz,
            batch=batch,
            device=resolve_device(self.device),
            plots=plots,
            verbose=False,
        )
        box = metrics.box
        per_class: dict[str, dict[str, float]] = {}
        names = getattr(self.model, "names", {}) or {}
        indices = getattr(box, "ap_class_index", None)
        if indices is not None:
            for position, class_index in enumerate(list(indices)):
                per_class[str(names.get(int(class_index), int(class_index)))] = {
                    "precision": _at(box.p, position),
                    "recall": _at(box.r, position),
                    "map50": _at(box.ap50, position),
                    "map50_95": _at(box.ap, position),
                }
        precision, recall = float(box.mp), float(box.mr)
        return ValidationResult(
            metrics={
                "map50_95": float(box.map),
                "map50": float(box.map50),
                "map75": float(box.map75),
                "precision": precision,
                "recall": recall,
                "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
            },
            per_class=per_class,
            speed_ms={key: float(value) for key, value in (getattr(metrics, "speed", {}) or {}).items()},
            save_dir=str(getattr(metrics, "save_dir", "")) or None,
        )

    def export(self, format: str = "onnx", **kwargs: Any) -> Path:
        return Path(str(self.model.export(format=format, imgsz=self.imgsz, **kwargs)))


def _timings(result: Any) -> dict[str, float]:
    speed = getattr(result, "speed", None)
    if not isinstance(speed, dict):
        return {}
    return {key: float(value) for key, value in speed.items() if isinstance(value, (int, float))}


def _at(values: Any, index: int) -> float:
    try:
        return float(np.asarray(values).reshape(-1)[index])
    except (IndexError, TypeError, ValueError):
        return 0.0


def timed(prediction_callable, *args, **kwargs) -> tuple[Any, float]:
    """Run a call and return ``(result, milliseconds)``."""
    started = time.perf_counter()
    result = prediction_callable(*args, **kwargs)
    return result, (time.perf_counter() - started) * 1000.0
