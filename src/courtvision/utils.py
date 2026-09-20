"""Shared helpers: deterministic IO, structured logging, percentile maths."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file and rename, so a crash cannot truncate a report."""
    directory = path.parent if str(path.parent) else Path(".")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(directory), encoding="utf-8", suffix=".tmp") as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def ensure_jsonable(value: Any) -> Any:
    """Recursively convert to JSON/YAML-safe primitives (NaN becomes None)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_default(value)
    if isinstance(value, np.ndarray):
        return ensure_jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): ensure_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [ensure_jsonable(item) for item in value]
    return str(value)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    _atomic_write(path, json.dumps(ensure_jsonable(payload), indent=indent, default=json_default))
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    _atomic_write(path, text)
    return path


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    """Write rows to CSV without depending on pandas."""
    import csv

    path = Path(path)
    rows = [dict(row) for row in rows]
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    directory = path.parent if str(path.parent) else Path(".")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(directory), newline="", encoding="utf-8", suffix=".tmp") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(ensure_jsonable(row))
        temporary = handle.name
    os.replace(temporary, path)
    return path


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile, matching ``numpy.percentile``'s default."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - only when torch is installed
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:  # pragma: no cover
        pass


class _ExtraFormatter(logging.Formatter):
    """Formatter that appends ``extra=`` fields as ``key=value``."""

    RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self.RESERVED and not key.startswith("_")
        }
        if not extras:
            return base
        return base + " [" + " ".join(f"{k}={v}" for k, v in sorted(extras.items())) + "]"


def configure_logging(level: str | int = "INFO", *, stream: Any = None) -> logging.Logger:
    logger = logging.getLogger("courtvision")
    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(_ExtraFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"courtvision.{name}" if name else "courtvision")
