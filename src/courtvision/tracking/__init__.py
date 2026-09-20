"""Multi-object tracking backends and MOT-format export.

The Ultralytics-backed tracker is exposed through a module ``__getattr__`` (PEP 562) so
importing this package does not pull in the detector. Without that, importing
``courtvision.tracking.base`` would execute this module, which imports the tracker, which
imports ``courtvision.detection``, which imports ``courtvision.tracking.base`` again — a
circular import. Importing ``UltralyticsTracker`` from here still works exactly as before.
"""

from .base import (
    Detection,
    Detector,
    FrameResult,
    Tracker,
    TrackingResult,
    filter_detections,
    tracking_result_from_mot,
    tracking_result_to_records,
)

__all__ = [
    "TRACKER_OVERRIDE_KEYS",
    "Detection",
    "Detector",
    "FrameResult",
    "Tracker",
    "TrackingResult",
    "UltralyticsTracker",
    "export_tracking_artifacts",
    "filter_detections",
    "tracking_result_from_mot",
    "tracking_result_to_records",
]

_LAZY = {"UltralyticsTracker", "export_tracking_artifacts", "TRACKER_OVERRIDE_KEYS"}


def __getattr__(name: str):
    if name in _LAZY:
        from . import ultralytics_tracker

        return getattr(ultralytics_tracker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
