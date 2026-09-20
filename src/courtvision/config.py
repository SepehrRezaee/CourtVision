"""Centralised configuration: ``defaults < YAML < COURTVISION_* env < explicit``.

Unknown sections and keys are hard errors, so a typo cannot silently leave a setting
at its default. Explicit overrides are applied last so a CLI flag beats an inherited
environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

ENV_PREFIX = "COURTVISION_"
DEFAULT_CONFIG_PATH = Path("configs/base.yaml")


class ConfigError(ValueError):
    """Raised when a configuration document is invalid."""


@dataclass
class PathsConfig:
    raw_data_root: Path = Path("data/SportsMOT")
    data_root: Path = Path("data/sportsmot_yolo")
    reports_dir: Path = Path("reports")
    runs_dir: Path = Path("runs")

    def __post_init__(self) -> None:
        for name in ("raw_data_root", "data_root", "reports_dir", "runs_dir"):
            setattr(self, name, Path(getattr(self, name)))


@dataclass
class SplitConfig:
    seed: int = 42
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    stratify_by_sport: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_ratio < 1.0:
            raise ConfigError("val_ratio must be in [0, 1)")
        if not 0.0 <= self.test_ratio < 1.0:
            raise ConfigError("test_ratio must be in [0, 1)")
        if self.val_ratio + self.test_ratio >= 1.0:
            raise ConfigError("val_ratio + test_ratio must be < 1")


@dataclass
class DetectorConfig:
    model: str = "yolo26n.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    device: str = "cpu"
    half: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.conf <= 1.0:
            raise ConfigError("conf must be in (0, 1]")
        if not 0.0 < self.iou <= 1.0:
            raise ConfigError("iou must be in (0, 1]")
        if self.imgsz < 32:
            raise ConfigError("imgsz must be >= 32")


@dataclass
class TrainConfig:
    data: str = "data/sportsmot_yolo/dataset.yaml"
    model: str = "yolo26n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = -1
    device: str = "0"
    workers: int = 8
    seed: int = 42
    patience: int = 20
    optimizer: str = "auto"
    cos_lr: bool = True
    close_mosaic: int = 10
    name: str = "courtvision"
    augment: bool = True


@dataclass
class TrackerConfig:
    name: str = "bytetrack.yaml"
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300


@dataclass
class EventConfig:
    """Deterministic event thresholds. Units are **pixels** and **pixels per frame**."""

    zone_grid: tuple[int, int] = (3, 3)
    sprint_speed: float = 12.0
    sprint_min_frames: int = 5
    stationary_speed: float = 1.0
    stationary_min_frames: int = 25
    accel_threshold: float = 2.0
    decel_threshold: float = -2.0
    direction_change_deg: float = 90.0
    direction_change_min_speed: float = 2.0
    close_proximity_distance: float = 80.0
    proximity_min_frames: int = 3
    convergence_window: int = 5
    convergence_delta: float = 20.0
    min_track_length: int = 5
    smoothing_window: int = 5

    def __post_init__(self) -> None:
        if self.accel_threshold <= 0:
            raise ConfigError("accel_threshold must be > 0")
        if self.decel_threshold >= 0:
            raise ConfigError("decel_threshold must be < 0")
        if min(self.zone_grid) < 1:
            raise ConfigError("zone_grid must be positive")


@dataclass
class BenchmarkConfig:
    warmup_frames: int = 10
    max_frames: int = 200
    repeats: int = 1
    percentiles: tuple[float, ...] = (50.0, 95.0)

    def __post_init__(self) -> None:
        if self.warmup_frames < 0:
            raise ConfigError("warmup_frames must be >= 0")
        if self.max_frames <= 0:
            raise ConfigError("max_frames must be > 0")


@dataclass
class ApiConfig:
    max_upload_mb: float = 200.0
    allowed_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    allowed_content_types: tuple[str, ...] = (
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
        "application/octet-stream",
    )
    max_frames: int = 3000
    max_event_list: int = 500
    job_ttl_seconds: int = 3600
    max_jobs: int = 32
    job_max_workers: int = 1


@dataclass
class MonitoringConfig:
    reference_path: Path = Path("reports/monitoring/reference.json")
    bins: int = 20
    psi_warn: float = 0.1
    psi_alert: float = 0.25

    def __post_init__(self) -> None:
        self.reference_path = Path(self.reference_path)


@dataclass
class ServingConfig:
    model: str = "yolo26n.pt"
    device: str = "cpu"
    tracker: str = "bytetrack.yaml"
    imgsz: int = 640
    conf: float = 0.25
    preload: bool = True
    enable_metrics: bool = True


@dataclass
class CourtVisionConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    events: EventConfig = field(default_factory=EventConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    SECTIONS = (
        "paths",
        "split",
        "detector",
        "train",
        "tracker",
        "events",
        "benchmark",
        "api",
        "monitoring",
        "serving",
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CourtVisionConfig":
        data = dict(data or {})
        unknown = set(data) - set(cls.SECTIONS)
        if unknown:
            raise ConfigError(f"Unknown config section(s): {sorted(unknown)}. Valid: {list(cls.SECTIONS)}")
        kwargs: dict[str, Any] = {}
        for section in cls.SECTIONS:
            payload = data.get(section)
            if payload is None:
                continue
            if not isinstance(payload, Mapping):
                raise ConfigError(f"Config section {section!r} must be a mapping")
            section_cls = SECTION_TYPES[section]
            known = {f.name for f in fields(section_cls)}
            extra = set(payload) - known
            if extra:
                raise ConfigError(
                    f"Unknown key(s) in section {section!r}: {sorted(extra)}. Valid: {sorted(known)}"
                )
            values = dict(payload)
            for name in known:
                if isinstance(values.get(name), list) and _is_tuple_field(section_cls, name):
                    values[name] = tuple(values[name])
            kwargs[section] = section_cls(**values)
        return cls(**kwargs)

    def apply_env(self, env: Mapping[str, str] | None = None) -> "CourtVisionConfig":
        env = os.environ if env is None else env
        for section in self.SECTIONS:
            section_obj = getattr(self, section)
            for spec in fields(section_obj):
                key = f"{ENV_PREFIX}{section.upper()}_{spec.name.upper()}"
                if key in env:
                    setattr(section_obj, spec.name, _coerce(env[key], spec.type))
        return self


def _declared_name(declared: Any) -> str:
    """Field type as a comparable name; ``from __future__ import annotations`` makes these strings."""
    return declared.strip() if isinstance(declared, str) else getattr(declared, "__name__", str(declared))


def _is_tuple_field(section_cls: type, name: str) -> bool:
    declared = next(f.type for f in fields(section_cls) if f.name == name)
    return _declared_name(declared).startswith("tuple")


def _coerce(raw: str, declared: Any) -> Any:
    text = raw.strip()
    name = _declared_name(declared)
    if name == "bool":
        return text.lower() in {"1", "true", "yes", "on"}
    if name == "int":
        return int(text)
    if name == "float":
        return float(text)
    if name.startswith("tuple"):
        return tuple(part.strip() for part in text.split(",") if part.strip())
    if name == "Path":
        return Path(text)
    if name in {"str", "Any"} or not name:
        return text
    raise ConfigError(f"Cannot apply an environment override to a field of type {name!r}")


SECTION_TYPES: dict[str, type] = {
    "paths": PathsConfig,
    "split": SplitConfig,
    "detector": DetectorConfig,
    "train": TrainConfig,
    "tracker": TrackerConfig,
    "events": EventConfig,
    "benchmark": BenchmarkConfig,
    "api": ApiConfig,
    "monitoring": MonitoringConfig,
    "serving": ServingConfig,
}


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    use_env: bool = True,
) -> CourtVisionConfig:
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    document: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is not None and not isinstance(raw, Mapping):
            raise ConfigError(f"Config file {path} must contain a YAML mapping")
        document = dict(raw or {})
    elif path != DEFAULT_CONFIG_PATH:
        raise ConfigError(f"Config file not found: {path}")

    config = CourtVisionConfig.from_dict(document)
    if use_env:
        config.apply_env()
    if overrides:
        for section, values in overrides.items():
            if section not in CourtVisionConfig.SECTIONS:
                raise ConfigError(f"Unknown override section {section!r}")
            section_obj = getattr(config, section)
            for key, value in dict(values or {}).items():
                if key not in {f.name for f in fields(section_obj)}:
                    raise ConfigError(f"Unknown override {section}.{key}")
                if value is not None:
                    setattr(section_obj, key, value)
    return config


def to_dict(config: CourtVisionConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for section in CourtVisionConfig.SECTIONS:
        section_obj = getattr(config, section)
        out[section] = {
            spec.name: (
                str(getattr(section_obj, spec.name))
                if isinstance(getattr(section_obj, spec.name), Path)
                else getattr(section_obj, spec.name)
            )
            for spec in fields(section_obj)
        }
    return out
