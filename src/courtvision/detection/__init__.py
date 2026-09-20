"""Detector backends behind a framework-agnostic interface."""

from .base import DetectorInfo, DetectorProtocol, ValidationResult
from .ultralytics_detector import (
    ModelNotAvailableError,
    UltralyticsDetector,
    available_trackers,
    describe_capabilities,
    downloadable_asset_names,
    model_is_available,
    resolve_tracker_name,
    tracker_config_dir,
    tracker_is_supported,
    ultralytics_version,
    verify_model_identifier,
    yolo26_assets,
)

__all__ = [
    "DetectorInfo",
    "DetectorProtocol",
    "ValidationResult",
    "ModelNotAvailableError",
    "UltralyticsDetector",
    "available_trackers",
    "describe_capabilities",
    "downloadable_asset_names",
    "model_is_available",
    "resolve_tracker_name",
    "tracker_config_dir",
    "tracker_is_supported",
    "ultralytics_version",
    "verify_model_identifier",
    "yolo26_assets",
]
