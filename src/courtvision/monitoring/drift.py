"""Input-distribution drift monitoring.

Two different things are monitored and never conflated:

* **Data drift** — the distribution of input signals (detections per frame,
  confidence, box area, track duration, latency, resolution) has moved relative to a
  reference batch. This is a **proxy signal**: it may indicate a new camera, resolution
  or lighting, and does **not** by itself demonstrate model degradation.
* **Performance degradation** — a *labelled* metric got worse. That lives in
  :func:`detect_performance_regression` and is direction-aware.

Batches are histogrammed onto the **reference's** bin edges; comparing two
independently binned histograms would not be a comparison. PSI cut-offs are rules of
thumb and are configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import ensure_dir, read_json, write_json

#: Monitored signals with a human description used in reports.
SIGNALS: dict[str, str] = {
    "detections_per_frame": "number of detections per processed frame",
    "confidence": "detection confidence",
    "box_area_px": "detection box area in pixels",
    "box_aspect_ratio": "detection box width/height",
    "track_duration_frames": "frames each track is observed for",
    "track_fragmentation": "number of frame gaps per track",
    "track_length": "number of tracks per analysed clip",
    "inference_latency_ms": "per-frame inference latency",
    "frame_width": "input frame width",
    "frame_height": "input frame height",
}

DEFAULT_QUANTILES = (10, 25, 50, 75, 90)
DEFAULT_BINS = 21


@dataclass
class SignalSummary:
    name: str
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    quantiles: dict[str, float]
    histogram: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.minimum, 6),
            "max": round(self.maximum, 6),
            "quantiles": {key: round(value, 6) for key, value in self.quantiles.items()},
            "histogram": self.histogram,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SignalSummary:
        return cls(
            name=str(data["name"]),
            count=int(data["count"]),
            mean=float(data["mean"]),
            std=float(data["std"]),
            minimum=float(data["min"]),
            maximum=float(data["max"]),
            quantiles={str(k): float(v) for k, v in dict(data["quantiles"]).items()},
            histogram=dict(data["histogram"]),
        )


def summarize_signal(
    name: str, values: Sequence[float] | np.ndarray, *, bins: int = DEFAULT_BINS, quantiles: Sequence[int] = DEFAULT_QUANTILES
) -> SignalSummary:
    """Summarise into moments, quantiles and a histogram **that retains its edges**."""
    array = np.asarray(list(values), dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError(f"Signal {name!r} has no finite values to summarise")
    counts, edges = np.histogram(array, bins=bins)
    return SignalSummary(
        name=name,
        count=int(array.size),
        mean=float(array.mean()),
        std=float(array.std(ddof=0)),
        minimum=float(array.min()),
        maximum=float(array.max()),
        quantiles={f"p{value}": float(np.percentile(array, value)) for value in quantiles},
        histogram={"counts": [int(count) for count in counts], "edges": [float(edge) for edge in edges]},
    )


def reference_from_signals(
    signals: Mapping[str, Sequence[float]], *, bins: int = DEFAULT_BINS, metadata: Mapping[str, Any] | None = None
) -> dict:
    """Build a reference document from a baseline batch of signals."""
    summaries = {
        name: summarize_signal(name, values, bins=bins).to_dict()
        for name, values in sorted(signals.items())
        if len(list(values)) > 0
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "reference-distribution",
        "bins": bins,
        "signals": summaries,
        "metadata": dict(metadata or {}),
    }


def population_stability_index(reference_counts, current_counts, *, epsilon: float = 1e-6) -> float:
    """PSI between two aligned histogram count vectors.

    Convention: ``<0.1`` stable, ``0.1-0.25`` moderate shift, ``>0.25`` large shift.
    Those cut-offs are rules of thumb, not theory.
    """
    reference = np.asarray(reference_counts, dtype=float)
    current = np.asarray(current_counts, dtype=float)
    if reference.shape != current.shape:
        raise ValueError("PSI requires identically binned histograms")
    reference_share = np.clip(reference / max(reference.sum(), epsilon), epsilon, None)
    current_share = np.clip(current / max(current.sum(), epsilon), epsilon, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def histogram_counts_with_edges(values: Sequence[float] | np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Histogram onto *given* edges, so two batches share bins."""
    counts, _ = np.histogram(np.asarray(list(values), dtype=float), bins=np.asarray(edges, dtype=float))
    return counts


def kolmogorov_smirnov_statistic(reference: Sequence[float], current: Sequence[float]) -> float:
    """Two-sample KS statistic (max CDF gap).

    Only the statistic, not a p-value: with large samples a trivial difference becomes
    "significant", so the effect size is the honest number to act on.
    """
    a = np.sort(np.asarray(list(reference), dtype=float))
    b = np.sort(np.asarray(list(current), dtype=float))
    if a.size == 0 or b.size == 0:
        raise ValueError("KS statistic needs two non-empty samples")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def jensen_shannon_divergence(reference_counts, current_counts, *, epsilon: float = 1e-12) -> float:
    """Jensen-Shannon divergence, base 2, so the result is in ``[0, 1]``."""
    reference = np.asarray(reference_counts, dtype=float)
    current = np.asarray(current_counts, dtype=float)
    if reference.shape != current.shape:
        raise ValueError("JS divergence requires identically binned histograms")
    p = reference / max(reference.sum(), epsilon)
    q = current / max(current.sum(), epsilon)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], epsilon, None))))

    return float(np.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


@dataclass
class DriftReport:
    signals: dict[str, dict] = field(default_factory=dict)
    alerts: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    interpretation: str = (
        "Drift signals describe changes in the input distribution. They are PROXY signals and do "
        "not by themselves demonstrate model degradation; a labelled metric is required for that."
    )

    @property
    def max_psi(self) -> float:
        values = [entry["psi"] for entry in self.signals.values() if entry.get("psi") is not None]
        return max(values) if values else 0.0

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "kind": "input-drift",
            "is_proxy_signal": True,
            "interpretation": self.interpretation,
            "signals": self.signals,
            "max_psi": round(self.max_psi, 6),
            "alerts": self.alerts,
            "warnings": self.warnings,
        }

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())

    def format_text(self) -> str:
        lines = ["Drift comparison (proxy signals; not a quality verdict)"]
        lines.append(f"  {'signal':26s} {'n_ref':>7s} {'n_cur':>7s} {'PSI':>8s} {'KS':>7s} {'JS':>7s} {'verdict':>10s}")
        for name, entry in self.signals.items():
            psi = entry["psi"] if entry["psi"] is not None else float("nan")
            ks = entry["ks"] if entry["ks"] is not None else float("nan")
            js = entry["js"] if entry["js"] is not None else float("nan")
            lines.append(
                f"  {name:26s} {entry['reference_count']:7d} {entry['current_count']:7d} "
                f"{psi:8.4f} {ks:7.4f} {js:7.4f} {entry['verdict']:>10s}"
            )
        return "\n".join(lines)


def compare_distributions(
    reference: Mapping[str, Any], current_signals: Mapping[str, Sequence[float]], *, psi_warn: float = 0.1, psi_alert: float = 0.25
) -> DriftReport:
    """Compare a current batch against a stored reference document."""
    report = DriftReport()
    reference_signals = dict(reference.get("signals", {}))
    for name, values in sorted(current_signals.items()):
        array = np.asarray(list(values), dtype=float).reshape(-1)
        array = array[np.isfinite(array)]
        if array.size == 0:
            report.warnings.append(f"Signal {name!r} has no finite values in the current batch")
            continue
        entry: dict[str, Any] = {
            "reference_count": 0,
            "current_count": int(array.size),
            "current_mean": round(float(array.mean()), 6),
            "current_std": round(float(array.std(ddof=0)), 6),
            "psi": None,
            "ks": None,
            "js": None,
            "verdict": "no-reference",
        }
        reference_entry = reference_signals.get(name)
        if reference_entry is None:
            report.warnings.append(
                f"Signal {name!r} is not present in the reference; it cannot be compared. Rebuild the "
                "reference with `courtvision drift reference` if this signal is new."
            )
            report.signals[name] = entry
            continue

        reference_counts = np.asarray(reference_entry["histogram"]["counts"], dtype=float)
        current_counts = histogram_counts_with_edges(array, reference_entry["histogram"]["edges"])
        psi = population_stability_index(reference_counts, current_counts)
        entry.update(
            {
                "reference_count": int(reference_entry["count"]),
                "psi": round(psi, 6),
                "js": round(jensen_shannon_divergence(reference_counts, current_counts), 6),
                "reference_mean": round(float(reference_entry["mean"]), 6),
                "mean_shift": round(float(array.mean()) - float(reference_entry["mean"]), 6),
            }
        )
        if psi >= psi_alert:
            entry["verdict"] = "alert"
            report.alerts.append(
                {"signal": name, "psi": round(psi, 6), "message": f"{name} shifted substantially (PSI {psi:.3f} >= {psi_alert})"}
            )
        elif psi >= psi_warn:
            entry["verdict"] = "warn"
        else:
            entry["verdict"] = "stable"
        report.signals[name] = entry
    return report


def load_reference(path: str | Path) -> dict:
    document = read_json(path)
    if not isinstance(document, dict) or "signals" not in document:
        raise ValueError(f"{path} is not a CourtVision reference distribution document")
    return document


def save_reference(path: str | Path, reference: Mapping[str, Any]) -> Path:
    ensure_dir(Path(path).parent)
    return write_json(path, dict(reference))


def detect_performance_regression(
    baseline_metrics: Mapping[str, float],
    current_metrics: Mapping[str, float],
    *,
    relative_tolerance: float = 0.02,
    higher_is_better: Sequence[str] = ("map50_95", "map50", "idf1", "hota", "mota", "precision", "recall"),
) -> dict:
    """Compare labelled metrics and report direction-aware changes.

    Only valid with labels: a drop in IDF1 is a regression, a drop in ID switches is an
    improvement.
    """
    regressions: list[dict] = []
    improvements: list[dict] = []
    unchanged: list[str] = []
    for metric, current in sorted(current_metrics.items()):
        if metric not in baseline_metrics:
            continue
        baseline = float(baseline_metrics[metric])
        current_value = float(current)
        relative = 0.0 if baseline == 0 else (current_value - baseline) / abs(baseline)
        good_direction = 1.0 if metric in higher_is_better else -1.0
        entry = {
            "metric": metric,
            "baseline": round(baseline, 6),
            "current": round(current_value, 6),
            "relative_change": round(relative, 6),
        }
        if relative * good_direction <= -relative_tolerance:
            regressions.append(entry)
        elif relative * good_direction >= relative_tolerance:
            improvements.append(entry)
        else:
            unchanged.append(metric)
    return {
        "kind": "performance-comparison",
        "requires_labels": True,
        "relative_tolerance": relative_tolerance,
        "higher_is_better": list(higher_is_better),
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "verdict": "regression" if regressions else "no-regression",
    }


def signals_from_tracking_result(result, *, latencies_ms: Sequence[float] | None = None) -> dict[str, list[float]]:
    """Derive monitorable signals from a tracking result.

    Centralised here so a reference file and every later batch are built from identical
    definitions; a baseline computed differently from the live batch is worse than none.
    """
    signals: dict[str, list[float]] = {
        "detections_per_frame": [float(count) for count in result.detection_counts()],
        "confidence": [float(value) for value in result.confidences()],
        "track_duration_frames": [float(value) for value in result.track_durations()],
        "track_length": [float(len(result.tracks()))],
    }
    gaps: list[float] = []
    for detections in result.tracks().values():
        frames = sorted(detection.frame for detection in detections)
        gaps.append(float(sum(1 for a, b in pairwise(frames) if b - a > 1)))
    signals["track_fragmentation"] = gaps or [0.0]

    areas: list[float] = []
    aspects: list[float] = []
    for detection in result.iter_detections():
        areas.append(float(detection.area))
        if detection.height > 0:
            aspects.append(float(detection.width / detection.height))
    signals["box_area_px"] = areas or [0.0]
    signals["box_aspect_ratio"] = aspects or [0.0]
    if latencies_ms:
        signals["inference_latency_ms"] = [float(value) for value in latencies_ms]
    if result.frame_size:
        signals["frame_width"] = [float(result.frame_size[0])]
        signals["frame_height"] = [float(result.frame_size[1])]
    return signals
