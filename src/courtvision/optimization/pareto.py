"""Quality/throughput Pareto analysis.

A configuration *dominates* another when it is at least as good on both axes and
strictly better on one; the frontier is the non-dominated set. Configurations missing a
measurement are **excluded and listed**, because an unmeasured point is not a
dominating point.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils import ensure_dir, read_json, write_csv, write_json

#: Objective spellings where smaller is better.
DEFAULT_MINIMIZE = ("p95_latency_ms", "p50_latency_ms", "mean_latency_ms", "num_switches", "model_bytes")


@dataclass
class ConfigurationRecord:
    """One measured configuration: what it is, how good it is, how fast it is."""

    name: str
    detector: str | None = None
    tracker: str | None = None
    imgsz: int | None = None
    confidence: float | None = None
    device: str | None = None
    backend: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def value(self, key: str) -> float | None:
        if key in self.metrics and self.metrics[key] is not None:
            return float(self.metrics[key])
        if key in self.metadata and self.metadata[key] is not None:
            return float(self.metadata[key])
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "detector": self.detector,
            "tracker": self.tracker,
            "imgsz": self.imgsz,
            "confidence": self.confidence,
            "device": self.device,
            "backend": self.backend,
            "metrics": {key: _round(value) for key, value in sorted(self.metrics.items())},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfigurationRecord":
        return cls(
            name=str(data["name"]),
            detector=data.get("detector"),
            tracker=data.get("tracker"),
            imgsz=data.get("imgsz"),
            confidence=data.get("confidence"),
            device=data.get("device"),
            backend=data.get("backend"),
            metrics={str(k): float(v) for k, v in dict(data.get("metrics", {})).items() if v is not None},
            metadata=dict(data.get("metadata", {})),
        )


def _round(value: Any, digits: int = 6) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def dominates(a: ConfigurationRecord, b: ConfigurationRecord, *, quality_key: str, latency_key: str) -> bool:
    a_quality, b_quality = a.value(quality_key), b.value(quality_key)
    a_latency, b_latency = a.value(latency_key), b.value(latency_key)
    if None in (a_quality, b_quality, a_latency, b_latency):
        return False
    return (
        a_quality >= b_quality
        and a_latency <= b_latency
        and (a_quality > b_quality or a_latency < b_latency)
    )


def pareto_frontier(
    records: Sequence[ConfigurationRecord], *, quality_key: str, latency_key: str
) -> tuple[list[ConfigurationRecord], list[dict]]:
    usable: list[ConfigurationRecord] = []
    excluded: list[dict] = []
    for record in records:
        missing = [
            key
            for key, value in ((quality_key, record.value(quality_key)), (latency_key, record.value(latency_key)))
            if value is None
        ]
        if missing:
            excluded.append({"name": record.name, "reason": f"missing measurement(s): {', '.join(missing)}", "record": record.to_dict()})
            continue
        usable.append(record)

    frontier = [
        candidate
        for candidate in usable
        if not any(
            dominates(other, candidate, quality_key=quality_key, latency_key=latency_key)
            for other in usable
            if other is not candidate
        )
    ]
    frontier.sort(key=lambda record: record.value(latency_key) or 0.0)
    return frontier, excluded


def select_recommendations(
    frontier: Sequence[ConfigurationRecord], *, quality_key: str, latency_key: str
) -> dict[str, ConfigurationRecord | None]:
    """Best-quality, best-throughput and balanced picks from the frontier.

    "Balanced" maximises normalised quality plus normalised inverse latency. With a
    single frontier point all three roles are that point, and the report says so.
    """
    if not frontier:
        return {"best_quality": None, "best_throughput": None, "balanced": None}

    best_quality = max(frontier, key=lambda record: record.value(quality_key) or float("-inf"))
    best_throughput = min(frontier, key=lambda record: record.value(latency_key) or float("inf"))

    qualities = [record.value(quality_key) or 0.0 for record in frontier]
    latencies = [record.value(latency_key) or 0.0 for record in frontier]
    quality_span = max(qualities) - min(qualities)
    latency_span = max(latencies) - min(latencies)

    def score(record: ConfigurationRecord) -> float:
        normalised_quality = (
            ((record.value(quality_key) or 0.0) - min(qualities)) / quality_span if quality_span else 1.0
        )
        normalised_speed = (
            (max(latencies) - (record.value(latency_key) or 0.0)) / latency_span if latency_span else 1.0
        )
        return normalised_quality + normalised_speed

    return {"best_quality": best_quality, "best_throughput": best_throughput, "balanced": max(frontier, key=score)}


@dataclass
class ParetoReport:
    quality_key: str
    latency_key: str
    frontier: list[ConfigurationRecord]
    excluded: list[dict]
    recommendations: dict[str, ConfigurationRecord | None]
    all_records: list[ConfigurationRecord] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "quality_metric": self.quality_key,
            "latency_metric": self.latency_key,
            "definitions": {
                "dominates": "at least as good on both axes and strictly better on one",
                "frontier": "configurations not dominated by any other measured configuration",
            },
            "notes": self.notes,
            "recommendations": {role: (record.to_dict() if record else None) for role, record in self.recommendations.items()},
            "frontier": [record.to_dict() for record in self.frontier],
            "excluded": self.excluded,
            "all_configurations": [record.to_dict() for record in self.all_records],
        }

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())

    def markdown_table(self) -> str:
        columns = ["name", "detector", "tracker", "imgsz", "confidence", self.quality_key, self.latency_key]
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for record in self.frontier:
            lines.append(
                "| "
                + " | ".join(
                    [
                        record.name,
                        record.detector or "—",
                        record.tracker or "—",
                        str(record.imgsz) if record.imgsz else "—",
                        f"{record.confidence:.2f}" if record.confidence is not None else "—",
                        _fmt(record.value(self.quality_key)),
                        _fmt(record.value(self.latency_key)),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def build_pareto_report(
    records: Iterable[ConfigurationRecord], *, quality_key: str = "idf1", latency_key: str = "p95_latency_ms"
) -> ParetoReport:
    record_list = list(records)
    frontier, excluded = pareto_frontier(record_list, quality_key=quality_key, latency_key=latency_key)
    notes: list[str] = []
    if not frontier:
        notes.append(
            f"No configuration has both {quality_key} and {latency_key} measured; the frontier is "
            "empty and no recommendation can be made."
        )
    if len(frontier) == 1:
        notes.append("Only one configuration is non-dominated, so all three recommendations refer to it.")
    if excluded:
        notes.append(f"{len(excluded)} configuration(s) were excluded for missing measurements; see 'excluded'.")
    return ParetoReport(
        quality_key=quality_key,
        latency_key=latency_key,
        frontier=frontier,
        excluded=excluded,
        recommendations=select_recommendations(frontier, quality_key=quality_key, latency_key=latency_key),
        all_records=record_list,
        notes=notes,
    )


def configuration_records_from_reports(
    evaluations: Sequence[Mapping[str, Any]], benchmarks: Sequence[Mapping[str, Any]], *, key: str = "name"
) -> list[ConfigurationRecord]:
    """Join evaluation and benchmark reports into comparable records.

    Records present on only one side are still returned, so a missing measurement
    surfaces as an excluded configuration rather than disappearing.
    """
    merged: dict[str, ConfigurationRecord] = {}

    def record_for(name: str) -> ConfigurationRecord:
        return merged.setdefault(name, ConfigurationRecord(name=name))

    for evaluation in evaluations:
        record = record_for(str(evaluation.get(key, evaluation.get("label", "unnamed"))))
        for attribute in ("detector", "tracker", "imgsz", "confidence", "device", "backend"):
            if evaluation.get(attribute) is not None:
                setattr(record, attribute, evaluation[attribute])
        metrics = evaluation.get("metrics")
        if isinstance(metrics, Mapping):
            record.metrics.update(
                {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            )
        tracking = evaluation.get("tracking")
        if isinstance(tracking, Mapping):
            aggregate = tracking.get("aggregate")
            if isinstance(aggregate, Mapping):
                for metric, value in aggregate.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        record.metrics.setdefault(str(metric), float(value))
            hota = tracking.get("hota")
            if isinstance(hota, Mapping) and "HOTA" in hota:
                record.metrics.setdefault("hota", float(hota["HOTA"]))

    for benchmark in benchmarks:
        record = record_for(str(benchmark.get(key, benchmark.get("label", "unnamed"))))
        for attribute in ("detector", "tracker", "imgsz", "confidence", "device", "backend"):
            if benchmark.get(attribute) is not None:
                setattr(record, attribute, benchmark[attribute])
        summary = benchmark.get("summary")
        if isinstance(summary, Mapping):
            pooled = summary.get("pooled")
            if isinstance(pooled, Mapping):
                for metric, value in pooled.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        record.metrics[str(metric)] = float(value)
            rows = summary.get("rows")
            if isinstance(rows, list) and rows:
                for metric in ("end_to_end_ms_mean", "end_to_end_ms_p50", "end_to_end_ms_p95"):
                    if metric in rows[0]:
                        record.metrics[metric] = float(rows[0][metric])
        for field_name in ("p95_latency_ms", "p50_latency_ms"):
            if benchmark.get(field_name) is not None:
                record.metrics[field_name] = float(benchmark[field_name])

    # Normalise the common latency spellings so gates and the frontier can use one name.
    for record in merged.values():
        for canonical, aliases in {
            "p95_latency_ms": ("end_to_end_ms_p95",),
            "p50_latency_ms": ("end_to_end_ms_p50",),
            "mean_latency_ms": ("end_to_end_ms_mean",),
            "fps": ("fps_from_mean",),
        }.items():
            for alias in aliases:
                if canonical not in record.metrics and alias in record.metrics:
                    record.metrics[canonical] = record.metrics[alias]
    return sorted(merged.values(), key=lambda item: item.name)


def save_pareto_csv(report: ParetoReport, path: str | Path) -> Path:
    frontier_names = {record.name for record in report.frontier}
    rows = []
    for record in report.all_records:
        row = {
            "name": record.name,
            "detector": record.detector,
            "tracker": record.tracker,
            "imgsz": record.imgsz,
            "confidence": record.confidence,
            "device": record.device,
            "backend": record.backend,
            "on_frontier": record.name in frontier_names,
        }
        row.update({key: _round(value) for key, value in sorted(record.metrics.items())})
        rows.append(row)
    return write_csv(path, rows)


def plot_pareto(report: ParetoReport, path: str | Path) -> Path | None:
    """Render the frontier chart if matplotlib is available; JSON/CSV are authoritative."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    if not report.all_records:
        return None

    path = Path(path)
    ensure_dir(path.parent)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    frontier_names = {record.name for record in report.frontier}
    for record in report.all_records:
        latency, quality = record.value(report.latency_key), record.value(report.quality_key)
        if latency is None or quality is None:
            continue
        on_frontier = record.name in frontier_names
        axis.scatter(latency, quality, color="#d62728" if on_frontier else "#7f7f7f", zorder=3 if on_frontier else 2)
        axis.annotate(record.name, (latency, quality), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ordered = sorted(report.frontier, key=lambda record: record.value(report.latency_key) or 0.0)
    if len(ordered) > 1:
        axis.plot(
            [record.value(report.latency_key) for record in ordered],
            [record.value(report.quality_key) for record in ordered],
            linestyle="--",
            color="#d62728",
            linewidth=1,
            zorder=1,
        )
    axis.set_xlabel(f"{report.latency_key} (ms, lower is better)")
    axis.set_ylabel(f"{report.quality_key} (higher is better)")
    axis.set_title("Quality / throughput Pareto frontier (red = non-dominated)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def load_records(path: str | Path) -> list[ConfigurationRecord]:
    document = read_json(path)
    if isinstance(document, Mapping) and "all_configurations" in document:
        return [ConfigurationRecord.from_dict(item) for item in document["all_configurations"]]
    if isinstance(document, list):
        return [ConfigurationRecord.from_dict(item) for item in document]
    raise ValueError(f"{path} is not a pareto report or a list of configuration records")
