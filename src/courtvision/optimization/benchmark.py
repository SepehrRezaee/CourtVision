"""Layered runtime benchmarking.

Methodology, stated because a latency number is meaningless without it:

* **Warm-up is excluded.** The first frames pay for CUDA context creation, cuDNN
  autotuning and allocator growth; timing them inflates p95 and hides steady-state
  cost. If a clip is shorter than the warm-up, the run reports that **no frames were
  measured** rather than reporting a warm-up number.
* **Cold and steady state are not mixed.** Decode happens before the timed loop, so it
  is not attributed to inference; decode throughput is reported separately.
* **Stages are separated.** ``tracking_and_post_ms`` is derived as
  ``end_to_end - preprocess - inference`` because Ultralytics' ``speed`` dict has no
  tracker stage; the derivation is in the column name.
* **Raw data is kept.** Every measured frame is a row in ``raw.csv`` and summaries are
  computed from those rows.
* **Percentiles carry their sample count**, so a p95 from 20 frames cannot be quoted
  as if it were not.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..repro import environment_snapshot, resolve_device
from ..utils import ensure_dir, write_csv, write_json

DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 95.0, 99.0)


@dataclass
class BenchmarkPlan:
    """What to measure. Every field is recorded with the results."""

    warmup_frames: int = 10
    measured_frames: int = 200
    repeats: int = 1
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES
    preload_frames: bool = True
    max_decode_frames: int | None = None

    def __post_init__(self) -> None:
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be >= 0")
        if self.measured_frames <= 0:
            raise ValueError("measured_frames must be > 0")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")

    def to_dict(self) -> dict:
        return {
            "warmup_frames": self.warmup_frames,
            "measured_frames": self.measured_frames,
            "repeats": self.repeats,
            "percentiles": list(self.percentiles),
            "preload_frames": self.preload_frames,
            "max_decode_frames": self.max_decode_frames,
        }


@dataclass
class BenchmarkRun:
    """Raw rows, derived summary and the system context of one benchmark."""

    label: str
    plan: dict
    config: dict
    rows: list[dict] = field(default_factory=list)
    decode: dict = field(default_factory=dict)
    system: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> dict:
        return summarize_rows(self.rows, percentiles=tuple(self.plan.get("percentiles", DEFAULT_PERCENTILES)))

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "plan": self.plan,
            "config": self.config,
            "decode": self.decode,
            "system": self.system,
            "summary": self.summary,
            "warnings": self.warnings,
            "raw_rows": len(self.rows),
        }

    def save(self, output_dir: str | Path, *, prefix: str = "") -> dict[str, Path]:
        output_dir = ensure_dir(output_dir)
        stem = f"{prefix}{self.label}" if prefix else self.label
        paths = {
            "raw": write_csv(output_dir / f"{stem}_raw.csv", self.rows),
            "summary": write_json(output_dir / f"{stem}_summary.json", self.to_dict()),
        }
        rows = self.summary.get("rows", [])
        if rows:
            paths["summary_csv"] = write_csv(output_dir / f"{stem}_summary.csv", rows)
        return paths


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, percentiles: Sequence[float]) -> dict:
    """Aggregate per-frame latency rows. FPS comes from the mean end-to-end latency."""
    if not rows:
        return {"rows": [], "frames": 0}
    metrics = ("decode_ms", "preprocess_ms", "inference_ms", "tracking_and_post_ms", "end_to_end_ms", "detections", "tracks")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("repeat", 1)), []).append(dict(row))

    summary_rows: list[dict] = []
    for repeat, group in sorted(groups.items()):
        entry: dict[str, Any] = {"repeat": repeat, "frames": len(group)}
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            if not values:
                continue
            array = np.asarray(values, dtype=float)
            entry[f"{metric}_mean"] = round(float(array.mean()), 4)
            entry[f"{metric}_std"] = round(float(array.std(ddof=0)), 4)
            entry[f"{metric}_min"] = round(float(array.min()), 4)
            entry[f"{metric}_max"] = round(float(array.max()), 4)
            for percentile in percentiles:
                entry[f"{metric}_p{int(percentile)}"] = round(float(np.percentile(array, percentile)), 4)
        if entry.get("end_to_end_ms_mean"):
            entry["fps_from_mean"] = round(1000.0 / float(entry["end_to_end_ms_mean"]), 4)
        summary_rows.append(entry)

    pooled: dict[str, Any] = {}
    for metric in metrics:
        values = [entry[f"{metric}_mean"] for entry in summary_rows if f"{metric}_mean" in entry]
        if values:
            pooled[f"{metric}_mean_of_repeats"] = round(float(statistics.fmean(values)), 4)
    return {"frames": len(rows), "pooled": pooled, "rows": summary_rows}


def load_frames(source: str | Path, *, max_frames: int | None = None) -> tuple[list[np.ndarray], dict]:
    """Decode a video into memory, timing the decode separately."""
    import cv2

    source = Path(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video for benchmarking: {source}")
    declared_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    info = {
        "source": str(source),
        "frames": len(frames),
        "decode_seconds": round(elapsed, 6),
        "decode_ms_per_frame": round(elapsed / len(frames) * 1000.0, 4) if frames else None,
        "declared_fps": declared_fps or None,
        "frame_size": [int(frames[0].shape[1]), int(frames[0].shape[0])] if frames else None,
    }
    return frames, info


def benchmark_frames(
    frames: Sequence[np.ndarray],
    make_processor: Callable[[], Callable[[np.ndarray], Any]],
    *,
    label: str,
    plan: BenchmarkPlan,
    config: Mapping[str, Any] | None = None,
    decode: Mapping[str, Any] | None = None,
) -> BenchmarkRun:
    """Time per-frame processing over pre-decoded frames, excluding warm-up.

    ``make_processor`` is called once per repeat and must return a per-frame callable
    whose result has a ``detections`` attribute and an optional ``timings_ms`` mapping.
    A fresh processor per repeat is what keeps repeats independent.
    """
    run = BenchmarkRun(
        label=label,
        plan=plan.to_dict(),
        config=dict(config or {}),
        decode=dict(decode or {}),
        system=environment_snapshot(),
    )
    available = len(frames) - plan.warmup_frames
    if available <= 0:
        run.warnings.append(
            f"Only {len(frames)} frame(s) with a {plan.warmup_frames}-frame warm-up; no frames "
            "were measured. Use a longer clip or a smaller warm-up."
        )
    elif available < plan.measured_frames:
        run.warnings.append(
            f"Requested {plan.measured_frames} measured frames but only {available} are available "
            "after warm-up; percentiles rest on the smaller sample."
        )

    for repeat in range(1, plan.repeats + 1):
        process = make_processor()
        measured = 0
        for index, frame in enumerate(frames):
            started = time.perf_counter()
            result = process(frame)
            end_to_end = (time.perf_counter() - started) * 1000.0
            if index < plan.warmup_frames:
                continue  # warm-up: executed and discarded
            if measured >= plan.measured_frames:
                break
            measured += 1
            timings = dict(getattr(result, "timings_ms", {}) or {})
            preprocess = float(timings.get("preprocess", 0.0))
            inference = float(timings.get("inference", 0.0))
            detections = getattr(result, "detections", None) or []
            run.rows.append(
                {
                    "repeat": repeat,
                    "frame_index": index + 1,
                    "end_to_end_ms": round(end_to_end, 4),
                    "preprocess_ms": round(preprocess, 4),
                    "inference_ms": round(inference, 4),
                    "postprocess_ms": round(float(timings.get("postprocess", 0.0)), 4),
                    # Derived: `speed` has no tracker stage, so this is the remainder.
                    "tracking_and_post_ms": round(max(0.0, end_to_end - preprocess - inference), 4),
                    "detections": len(detections),
                    "tracks": len({d.track_id for d in detections if d.track_id is not None}),
                    "fps_instantaneous": round(1000.0 / end_to_end, 4) if end_to_end > 0 else None,
                }
            )
    if not run.rows and not run.warnings:
        run.warnings.append("No frames were measured.")
    return run


def benchmark_video(
    source: str | Path,
    make_processor: Callable[[], Callable[[np.ndarray], Any]],
    *,
    label: str,
    plan: BenchmarkPlan | None = None,
    config: Mapping[str, Any] | None = None,
) -> BenchmarkRun:
    """Decode a video (optionally fully preloaded) and benchmark per-frame processing."""
    plan = plan or BenchmarkPlan()
    budget = plan.max_decode_frames if plan.preload_frames else (plan.warmup_frames + plan.measured_frames) * plan.repeats
    frames, decode_info = load_frames(source, max_frames=budget)
    if not frames:
        raise ValueError(f"No frames decoded from {source}")
    decode_info["preloaded"] = plan.preload_frames
    return benchmark_frames(frames, make_processor, label=label, plan=plan, config=config, decode=decode_info)


def system_report(device: str | None = None) -> dict:
    """Hardware/software context written next to benchmark results."""
    snapshot = environment_snapshot()
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": snapshot["python_version"],
        "packages": snapshot["packages"],
        "hardware": snapshot["hardware"],
        "requested_device": device,
        "resolved_device": None,
        "device_note": None,
    }
    if device is not None:
        try:
            report["resolved_device"] = resolve_device(device)
        except RuntimeError as exc:
            report["device_note"] = str(exc)
    return report


def save_benchmark(run: BenchmarkRun, output_dir: str | Path, *, prefix: str = "") -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    paths = run.save(output_dir, prefix=prefix)
    paths["system"] = write_json(output_dir / f"{prefix}system.json" if prefix else output_dir / "system.json", run.system or system_report())
    return paths
