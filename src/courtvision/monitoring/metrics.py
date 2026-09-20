"""Runtime metrics for the serving path.

Counters, gauges and a latency histogram are recorded in-process and exposed in
Prometheus text exposition format at ``/metrics``.

The two families mean different things on a dashboard: ``courtvision_requests_*`` and
``courtvision_request_latency_ms`` are **operational**, while the last-observed
detections-per-frame gauge summarises **input signals** whose drift is analysed
separately in :mod:`courtvision.monitoring.drift` — a change there is a proxy signal,
not a quality verdict.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

LATENCY_BUCKETS_MS: tuple[float, ...] = (
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0, 60000.0
)


@dataclass
class Histogram:
    """Fixed-bucket histogram with a running sum and count."""

    name: str
    buckets: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    observations: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        self.total += float(value)
        self.observations += 1
        for index, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[index] += 1
                return
        self.counts[-1] += 1

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "buckets": list(self.buckets),
            "counts": list(self.counts),
            "count": self.observations,
            "sum": round(self.total, 6),
            "mean": round(self.total / self.observations, 6) if self.observations else None,
        }

    def render(self) -> list[str]:
        lines: list[str] = []
        cumulative = 0
        for index, bound in enumerate(self.buckets):
            cumulative += self.counts[index]
            lines.append(f'{self.name}_bucket{{le="{bound}"}} {cumulative}')
        cumulative += self.counts[-1]
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {cumulative}')
        lines.append(f"{self.name}_sum {self.total}")
        lines.append(f"{self.name}_count {self.observations}")
        return lines


class MetricsRegistry:
    """Thread-safe counters, gauges and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, Histogram] = {}

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + float(value)

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            histogram = self._histograms.setdefault(name, Histogram(name=name, buckets=LATENCY_BUCKETS_MS))
            histogram.observe(float(value))

    def record_analysis(
        self, *, frames: int, detections: int, unique_tracks: int, latency_ms: float, failed: bool = False
    ) -> None:
        """Record one completed analysis request."""
        self.increment("courtvision_requests_total")
        if failed:
            self.increment("courtvision_requests_failed_total")
        self.increment("courtvision_frames_processed_total", frames)
        self.increment("courtvision_detections_total", detections)
        self.increment("courtvision_tracks_total", unique_tracks)
        self.observe("courtvision_request_latency_ms", latency_ms)
        if frames:
            self.set_gauge("courtvision_last_detections_per_frame", detections / frames)
            self.set_gauge("courtvision_last_frames", float(frames))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": {key: round(value, 6) for key, value in sorted(self._counters.items())},
                "gauges": {key: round(value, 6) for key, value in sorted(self._gauges.items())},
                "histograms": {key: value.snapshot() for key, value in sorted(self._histograms.items())},
            }

    def render_prometheus(self) -> str:
        with self._lock:
            lines: list[str] = []
            for name, value in sorted(self._counters.items()):
                lines += [f"# TYPE {name} counter", f"{name} {value}"]
            for name, value in sorted(self._gauges.items()):
                lines += [f"# TYPE {name} gauge", f"{name} {value}"]
            for name, histogram in sorted(self._histograms.items()):
                lines.append(f"# TYPE {name} histogram")
                lines += histogram.render()
            return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


#: Process-wide registry. A module-level singleton is intentional: metrics must
#: aggregate across requests and the API is single-process by design.
REGISTRY = MetricsRegistry()


def percentile_summary(values: Iterable[float], percentiles: Iterable[float] = (50, 95, 99)) -> dict:
    """Percentile summary that always reports ``n``, so a p95 from 20 samples is visible as such."""
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size == 0:
        return {"n": 0}
    out: dict = {
        "n": int(array.size),
        "mean": round(float(array.mean()), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
    }
    for percentile in percentiles:
        out[f"p{int(percentile)}"] = round(float(np.percentile(array, percentile)), 4)
    return out
