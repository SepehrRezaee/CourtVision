"""Detector fine-tuning with full experiment provenance.

Every run writes ``experiment.json`` next to the weights containing the resolved
arguments, the dataset fingerprint and split id, and an environment snapshot, so a
reported metric can always be traced to the code, data and hardware behind it.
"""

from __future__ import annotations

import csv
import shutil
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import TrainConfig
from ..repro import environment_snapshot, resolve_device
from ..utils import write_json, write_text
from .ultralytics_detector import verify_model_identifier


@dataclass
class TrainResult:
    run_dir: Path
    weights: Path | None
    best_weights: Path | None
    last_weights: Path | None
    metrics: dict[str, float]
    config: dict
    environment: dict
    duration_seconds: float
    dataset_yaml: str
    experiment_file: Path | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "weights": str(self.weights) if self.weights else None,
            "best_weights": str(self.best_weights) if self.best_weights else None,
            "last_weights": str(self.last_weights) if self.last_weights else None,
            "metrics": {key: _round(value) for key, value in sorted(self.metrics.items())},
            "config": self.config,
            "environment": self.environment,
            "duration_seconds": round(self.duration_seconds, 3),
            "dataset_yaml": self.dataset_yaml,
            "experiment_file": str(self.experiment_file) if self.experiment_file else None,
            "notes": self.notes,
        }


def _round(value: Any, digits: int = 6) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _serialisable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def train_detector(
    config: TrainConfig | None = None,
    *,
    data: str | Path | None = None,
    output_dir: str | Path | None = None,
    split_id: str | None = None,
    dataset_fingerprint: str | None = None,
    extra: Mapping[str, Any] | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> TrainResult:
    """Fine-tune a YOLO detector. ``dry_run`` validates the configuration only."""
    config = config or TrainConfig()
    dataset_yaml = Path(data or config.data)
    run_root = Path(output_dir or "runs/detect")
    environment = environment_snapshot(seed=config.seed)
    notes: list[str] = []

    if not dataset_yaml.exists() and not dry_run:
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}. Run `courtvision data prepare` first.")

    arguments: dict[str, Any] = {
        "data": str(dataset_yaml),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device if dry_run else resolve_device(config.device),
        "workers": config.workers,
        "seed": config.seed,
        "deterministic": True,
        "patience": config.patience,
        "optimizer": config.optimizer,
        "cos_lr": config.cos_lr,
        "close_mosaic": config.close_mosaic,
        "project": str(run_root),
        "name": config.name,
        "exist_ok": False,
        "resume": resume,
        "plots": False,
    }
    if not config.augment:
        # There is no single "augment" switch in Ultralytics; record the flags used.
        arguments.update(
            {"mosaic": 0.0, "mixup": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "flipud": 0.0, "fliplr": 0.0}
        )
        notes.append("Augmentation disabled: mosaic/mixup/colour/flip turned off explicitly.")
    if extra:
        arguments.update(dict(extra))

    if dry_run:
        return TrainResult(
            run_dir=run_root / config.name,
            weights=None,
            best_weights=None,
            last_weights=None,
            metrics={},
            config={"arguments": arguments, "train_config": _as_dict(config)},
            environment=environment,
            duration_seconds=0.0,
            dataset_yaml=str(dataset_yaml),
            notes=[*notes, "Dry run: configuration validated, no training performed."],
        )

    verify_model_identifier(config.model)
    from ultralytics import YOLO

    model = YOLO(config.model)
    started = time.perf_counter()
    model.train(**arguments)
    duration = time.perf_counter() - started

    trainer = getattr(model, "trainer", None)
    run_dir = Path(str(getattr(trainer, "save_dir", run_root / config.name)))
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    metrics = _read_final_metrics(run_dir)
    if not metrics:
        notes.append("No results.csv found in the run directory; training metrics are NOT MEASURED.")

    experiment_file = write_json(
        run_dir / "experiment.json",
        {
            "kind": "detector-training",
            "dataset_yaml": str(dataset_yaml),
            "dataset_split_id": split_id,
            "dataset_fingerprint": dataset_fingerprint,
            "arguments": arguments,
            "train_config": _as_dict(config),
            "environment": environment,
            "duration_seconds": round(duration, 3),
            "run_dir": str(run_dir),
            "final_metrics": metrics,
            "notes": notes,
        },
    )
    # Keep the config beside the weights so a model is never orphaned from its settings.
    write_text(run_dir / "resolved_config.yaml", yaml.safe_dump({k: _serialisable(v) for k, v in arguments.items()}, sort_keys=True))
    with suppress(OSError):  # pragma: no cover - a copy failure must not fail a finished run
        shutil.copy2(dataset_yaml, run_dir / "dataset.yaml")

    return TrainResult(
        run_dir=run_dir,
        weights=best if best.exists() else (last if last.exists() else None),
        best_weights=best if best.exists() else None,
        last_weights=last if last.exists() else None,
        metrics=metrics,
        config={"arguments": arguments, "train_config": _as_dict(config)},
        environment=environment,
        duration_seconds=duration,
        dataset_yaml=str(dataset_yaml),
        experiment_file=experiment_file,
        notes=notes,
    )


def _read_final_metrics(run_dir: Path) -> dict[str, float]:
    """Read the last row of ``results.csv`` as plain floats."""
    results = run_dir / "results.csv"
    if not results.exists():
        return {}
    with results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    out: dict[str, float] = {}
    for key, value in rows[-1].items():
        try:
            out[str(key).strip()] = float(str(value).strip())
        except (TypeError, ValueError):
            continue
    return out


def _as_dict(config: TrainConfig) -> dict:
    return {key: _serialisable(value) for key, value in asdict(config).items()}


def model_size_report(model: str) -> dict:
    """Parameter count and file size for a checkpoint, without training."""
    verify_model_identifier(model)
    from ultralytics import YOLO

    loaded = YOLO(model)
    try:
        parameters = int(sum(parameter.numel() for parameter in loaded.model.parameters()))
    except (AttributeError, TypeError):  # pragma: no cover
        parameters = None
    path = Path(model)
    return {
        "model": model,
        "parameters": parameters,
        "file_bytes": path.stat().st_size if path.exists() else None,
        "task": getattr(loaded, "task", None),
    }
