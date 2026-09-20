"""Detector interface types.

Detection is modelled separately from tracking so that mAP and MOT metrics can be
evaluated independently — which is what makes it possible to answer "is the tracking
bad, or is the detector feeding it bad boxes?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..tracking.base import TrackingResult


@dataclass
class DetectorInfo:
    """What the loaded model actually is, as reported by the backend."""

    backend: str
    weights: str
    task: str | None = None
    class_names: dict[int, str] = field(default_factory=dict)
    device: str | None = None
    imgsz: int | None = None
    parameters: int | None = None

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "weights": self.weights,
            "task": self.task,
            "class_names": {str(key): value for key, value in sorted(self.class_names.items())},
            "device": self.device,
            "imgsz": self.imgsz,
            "parameters": self.parameters,
        }


@dataclass
class ValidationResult:
    """Output of a detector validation run (mAP and friends)."""

    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    speed_ms: dict[str, float] = field(default_factory=dict)
    save_dir: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metrics": {key: _round(value) for key, value in sorted(self.metrics.items())},
            "per_class": self.per_class,
            "speed_ms": {key: _round(value) for key, value in self.speed_ms.items()},
            "save_dir": self.save_dir,
        }


def _round(value: Any, digits: int = 6) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


@runtime_checkable
class DetectorProtocol(Protocol):
    """Backend interface for detection."""

    name: str

    def info(self) -> DetectorInfo:  # pragma: no cover - protocol definition
        ...

    def detect(self, source: str, **kwargs) -> TrackingResult:  # pragma: no cover
        ...

    def validate(self, data: str, **kwargs) -> ValidationResult:  # pragma: no cover
        ...
