"""Report generation from measured artefacts.

Built only from files on disk that an actual run produced. Nothing is typed by hand,
and anything not measured appears as an explicit ``NOT MEASURED`` entry rather than
being omitted or estimated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils import ensure_dir, read_json, write_csv, write_json, write_text

ARTIFACT_PATHS: dict[str, str] = {
    "data_validation": "data_validation.json",
    "detection_metrics": "detection/metrics.json",
    "tracking_metrics": "tracking/metrics.json",
    "benchmarks": "benchmarks/summary.json",
    "pareto": "pareto.json",
    "gates": "quality_gates.json",
    "drift": "monitoring/drift.json",
    "export": "optimization/export.json",
}

EXPECTED_METRICS: tuple[tuple[str, str], ...] = (
    ("mAP50 / mAP50-95 (detection)", "detection_metrics"),
    ("precision / recall (detection)", "detection_metrics"),
    ("MOTA / MOTP (tracking)", "tracking_metrics"),
    ("IDF1 (tracking)", "tracking_metrics"),
    ("HOTA (tracking)", "tracking_metrics"),
    ("ID switches (tracking)", "tracking_metrics"),
    ("latency percentiles and FPS", "benchmarks"),
    ("Pareto trade-off analysis", "pareto"),
    ("quality gates", "gates"),
    ("input drift", "drift"),
    ("model export validation", "export"),
)


@dataclass
class ReportBundle:
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    reports_dir: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    files: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "reports_dir": self.reports_dir,
            "artifacts_present": sorted(self.artifacts),
            "missing_artifacts": self.missing,
            "not_measured": [
                {"metric": metric, "reason": f"no {artifact} artefact found"}
                for metric, artifact in EXPECTED_METRICS
                if artifact in self.missing
            ],
            "artifacts": self.artifacts,
            "notes": self.notes,
        }


def collect_artifacts(reports_dir: str | Path) -> ReportBundle:
    """Load every artefact that exists; record the ones that do not."""
    reports_dir = Path(reports_dir)
    bundle = ReportBundle(reports_dir=str(reports_dir))
    for name, relative in ARTIFACT_PATHS.items():
        path = reports_dir / relative
        if not path.exists():
            bundle.missing.append(name)
            continue
        try:
            bundle.artifacts[name] = read_json(path)
        except (ValueError, OSError) as exc:
            bundle.missing.append(name)
            bundle.notes.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
    return bundle


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def render_markdown(bundle: ReportBundle, gates: Mapping[str, Any] | None = None) -> str:
    lines = ["# CourtVision experiment report", ""]
    lines.append(f"Generated: `{bundle.generated_at}`")
    lines.append(f"Reports directory: `{bundle.reports_dir}`")
    lines.append("")
    lines.append(
        "Every number below was read from an artefact produced by an actual run. Anything not "
        "measured is listed under *Not measured*."
    )
    lines.append("")

    if gates:
        lines.append("## Quality gates")
        lines.append("")
        lines.append(f"**Status: {gates.get('status', 'unknown')}**")
        if not gates.get("is_quality_claim", False):
            lines.append("")
            lines.append("> Thresholds are placeholders (`kind: example`), not a measured quality claim.")
        for failure in gates.get("failures") or []:
            lines.append(f"- FAIL `{failure['metric']}`: {failure['message']}")
        lines.append("")

    detection = bundle.artifacts.get("detection_metrics")
    if isinstance(detection, Mapping):
        lines.append("## Detection")
        lines.append("")
        lines.append("| Group | Images | GT boxes | Predictions | mAP50 | mAP50-95 | Precision | Recall |")
        lines.append("| --- | --: | --: | --: | --: | --: | --: | --: |")
        for name, group in sorted((detection.get("groups") or {}).items()):
            lines.append(
                f"| {name} | {group.get('images', 0)} | {group.get('gt_boxes', 0)} | {group.get('predictions', 0)} | "
                f"{_fmt(group.get('map50'))} | {_fmt(group.get('map50_95'))} | {_fmt(group.get('precision'))} | "
                f"{_fmt(group.get('recall'))} |"
            )
        lines.append("")
        reference = detection.get("reference_ultralytics")
        if isinstance(reference, Mapping):
            metrics = reference.get("metrics", {})
            lines.append(
                f"Reference (Ultralytics `val()`): mAP50 {_fmt(metrics.get('map50'))}, "
                f"mAP50-95 {_fmt(metrics.get('map50_95'))}, precision {_fmt(metrics.get('precision'))}, "
                f"recall {_fmt(metrics.get('recall'))}"
            )
            lines.append("")
        cross = detection.get("cross_check")
        if isinstance(cross, Mapping):
            lines.append(f"Cross-check against the reference implementation: **{cross.get('status', 'unknown')}**")
            for comparison in cross.get("comparisons", []):
                lines.append(
                    f"- `{comparison['metric']}`: ours {_fmt(comparison.get('ours'))}, reference "
                    f"{_fmt(comparison.get('reference'))}, delta {_fmt(comparison.get('delta'))}"
                )
            lines.append("")

    tracking = bundle.artifacts.get("tracking_metrics")
    if isinstance(tracking, Mapping):
        aggregate = tracking.get("aggregate", {})
        derived = tracking.get("derived", {})
        lines.append("## Tracking")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --: |")
        for key in (
            "mota", "motp", "idf1", "idp", "idr", "precision", "recall", "num_switches",
            "num_fragmentations", "mostly_tracked", "mostly_lost",
        ):
            if key in aggregate:
                lines.append(f"| {key} | {_fmt(aggregate[key])} |")
        for key in ("mt_ratio", "ml_ratio", "id_switches_per_object"):
            if key in derived:
                lines.append(f"| {key} | {_fmt(derived[key])} |")
        hota = tracking.get("hota")
        if isinstance(hota, Mapping):
            for key in ("HOTA", "DetA", "AssA", "LocA"):
                if key in hota:
                    lines.append(f"| HOTA:{key} | {_fmt(hota[key])} |")
        elif tracking.get("hota_note"):
            lines.append(f"| HOTA | {tracking['hota_note']} |")
        lines.append("")
        lines.append("`motp` is a mean IoU *distance* (lower is better); HOTA values are means over TrackEval's 19 alpha thresholds.")
        lines.append("")

    benchmarks = bundle.artifacts.get("benchmarks")
    if isinstance(benchmarks, Mapping):
        lines.append("## Runtime")
        lines.append("")
        runs = benchmarks.get("runs")
        if isinstance(runs, list) and runs:
            lines.append("| Configuration | Frames | mean ms | p50 ms | p95 ms | FPS | Device |")
            lines.append("| --- | --: | --: | --: | --: | --: | --- |")
            for run in runs:
                summary = run.get("summary", {})
                row = (summary.get("rows") or [{}])[0]
                lines.append(
                    f"| {run.get('label', '—')} | {summary.get('frames', 0)} | {_fmt(row.get('end_to_end_ms_mean'))} | "
                    f"{_fmt(row.get('end_to_end_ms_p50'))} | {_fmt(row.get('end_to_end_ms_p95'))} | "
                    f"{_fmt(row.get('fps_from_mean'))} | {(run.get('config') or {}).get('device', '—')} |"
                )
            lines.append("")
        system = benchmarks.get("system")
        if isinstance(system, Mapping):
            hardware = system.get("hardware", {})
            lines.append(
                f"Hardware: {hardware.get('platform', 'unknown')}, CPU x{hardware.get('cpu_count')}, "
                f"GPU: {hardware.get('gpu') or 'none'}, CUDA: {hardware.get('cuda_version') or 'n/a'}"
            )
            lines.append("")

    pareto = bundle.artifacts.get("pareto")
    if isinstance(pareto, Mapping):
        lines.append("## Configuration trade-offs")
        lines.append("")
        for role, record in (pareto.get("recommendations") or {}).items():
            if record:
                quality = record.get("metrics", {}).get(pareto.get("quality_metric", ""))
                latency = record.get("metrics", {}).get(pareto.get("latency_metric", ""))
                lines.append(
                    f"- **{role.replace('_', ' ')}**: `{record.get('name')}` "
                    f"({pareto.get('quality_metric')} {_fmt(quality)}, {pareto.get('latency_metric')} {_fmt(latency)})"
                )
        lines.append("")
        for note in pareto.get("notes", []):
            lines.append(f"> {note}")
        lines.append("")

    drift = bundle.artifacts.get("drift")
    if isinstance(drift, Mapping):
        lines.append("## Input drift")
        lines.append("")
        lines.append("| Signal | Reference n | Current n | PSI | Verdict |")
        lines.append("| --- | --: | --: | --: | --- |")
        for name, entry in (drift.get("signals") or {}).items():
            lines.append(
                f"| {name} | {entry.get('reference_count', 0)} | {entry.get('current_count', 0)} | "
                f"{_fmt(entry.get('psi'))} | {entry.get('verdict', '—')} |"
            )
        lines.append("")
        lines.append("Drift is a **proxy signal** about input distributions; it does not by itself demonstrate model degradation.")
        lines.append("")

    export = bundle.artifacts.get("export")
    if isinstance(export, Mapping):
        lines.append("## Export & optimization")
        lines.append("")
        for entry in export.get("exports", []):
            lines.append(f"- `{entry.get('format')}`: {'ok' if entry.get('ok') else 'not produced'} {entry.get('error') or ''}")
        for entry in export.get("validations", []):
            lines.append(
                f"- Validation of `{entry.get('exported')}`: passed={entry.get('passed')}, "
                f"match fraction {_fmt(entry.get('match_fraction'))}, mean IoU {_fmt(entry.get('mean_iou_matched'))}"
            )
        for entry in export.get("benchmarks", []):
            lines.append(f"- Measured speedup: {_fmt(entry.get('speedup_exported_vs_reference'))} on {entry.get('device', 'unknown')}")
        lines.append("")

    lines.append("## Not measured")
    lines.append("")
    if bundle.missing:
        for metric, artifact in EXPECTED_METRICS:
            if artifact in bundle.missing:
                lines.append(f"- **{metric}** — no `{ARTIFACT_PATHS[artifact]}` artefact was produced")
    else:
        lines.append("All expected artefacts are present.")
    lines.append("")
    if bundle.notes:
        lines.append("## Notes")
        lines.append("")
        lines += [f"- {note}" for note in bundle.notes]
        lines.append("")
    return "\n".join(lines)


def build_report(
    reports_dir: str | Path, *, output_dir: str | Path | None = None, extra_notes: Sequence[str] | None = None
) -> ReportBundle:
    """Assemble ``REPORT.md``, ``report.json`` and ``results.csv``."""
    reports_dir = Path(reports_dir)
    output_dir = Path(output_dir) if output_dir is not None else reports_dir / "latest"
    ensure_dir(output_dir)

    bundle = collect_artifacts(reports_dir)
    if extra_notes:
        bundle.notes.extend(extra_notes)
    gates = bundle.artifacts.get("gates")
    bundle.files["markdown"] = write_text(
        output_dir / "REPORT.md", render_markdown(bundle, gates if isinstance(gates, Mapping) else None)
    )
    bundle.files["json"] = write_json(output_dir / "report.json", bundle.to_dict())

    rows = _results_rows(bundle)
    if rows:
        bundle.files["csv"] = write_csv(output_dir / "results.csv", rows)
    return bundle


def _results_rows(bundle: ReportBundle) -> list[dict]:
    rows: list[dict] = []
    detection = bundle.artifacts.get("detection_metrics")
    if isinstance(detection, Mapping):
        for name, group in sorted((detection.get("groups") or {}).items()):
            rows.append(
                {
                    "kind": "detection",
                    "group": name,
                    "images": group.get("images"),
                    "gt_boxes": group.get("gt_boxes"),
                    "predictions": group.get("predictions"),
                    "map50": group.get("map50"),
                    "map50_95": group.get("map50_95"),
                    "precision": group.get("precision"),
                    "recall": group.get("recall"),
                }
            )
    tracking = bundle.artifacts.get("tracking_metrics")
    if isinstance(tracking, Mapping):
        aggregate = tracking.get("aggregate", {})
        row: dict[str, Any] = {"kind": "tracking", "group": "aggregate"}
        row.update({key: aggregate.get(key) for key in ("mota", "motp", "idf1", "num_switches", "num_fragmentations")})
        hota = tracking.get("hota") or {}
        row["hota"] = hota.get("HOTA")
        row["deta"] = hota.get("DetA")
        row["assa"] = hota.get("AssA")
        rows.append(row)
    benchmarks = bundle.artifacts.get("benchmarks")
    if isinstance(benchmarks, Mapping):
        for run in benchmarks.get("runs", []):
            summary = run.get("summary", {})
            first = (summary.get("rows") or [{}])[0]
            rows.append(
                {
                    "kind": "runtime",
                    "group": run.get("label"),
                    "frames": summary.get("frames"),
                    "mean_ms": first.get("end_to_end_ms_mean"),
                    "p50_ms": first.get("end_to_end_ms_p50"),
                    "p95_ms": first.get("end_to_end_ms_p95"),
                    "fps": first.get("fps_from_mean"),
                    "device": (run.get("config") or {}).get("device"),
                }
            )
    return rows
