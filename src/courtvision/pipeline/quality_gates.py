"""Machine-readable release quality gates.

A gate is a dotted metric path with ``min`` and/or ``max``, evaluated against a
measurement file. The result is PASS/FAIL **plus the exact failing criteria**, so a
pipeline can fail a release with an actionable message.

Two kinds of threshold, never conflated:

* ``example`` thresholds are placeholders. ``is_quality_claim`` is false for them and
  the CLI warns whenever such a set is evaluated.
* ``baseline`` thresholds are derived from a measured run with
  :func:`baseline_from_measurements`, optionally with a relative tolerance so
  run-to-run noise does not fail a release while a real regression does.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..utils import ensure_jsonable, read_json, write_json

GATE_KINDS = ("baseline", "example")

#: Metrics where smaller is better; everything else gets a lower bound.
LOWER_IS_BETTER = ("num_switches", "switches", "latency_ms", "p95", "p99", "max_ms", "fragmentations")


class GateConfigError(ValueError):
    """Raised when a gate definition is malformed."""


@dataclass(frozen=True)
class Gate:
    """One criterion: ``metric`` must be ``>= min`` or ``<= max``."""

    metric: str
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise GateConfigError(f"Gate for {self.metric!r} needs a 'min' or a 'max'")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise GateConfigError(f"Gate for {self.metric!r} has min {self.minimum} > max {self.maximum}")

    def check(self, value: float | None) -> tuple[bool, str]:
        if value is None:
            if self.required:
                return False, f"{self.metric}: missing from measurements (required)"
            return True, f"{self.metric}: missing but not required"
        if self.minimum is not None and value < self.minimum:
            return False, f"{self.metric}={value:.6g} < min {self.minimum:.6g}"
        if self.maximum is not None and value > self.maximum:
            return False, f"{self.metric}={value:.6g} > max {self.maximum:.6g}"
        return True, f"{self.metric}={value:.6g} ok"

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "min": self.minimum,
            "max": self.maximum,
            "description": self.description,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str) -> Gate:
        if not isinstance(data, Mapping):
            raise GateConfigError(f"Gate {name!r} must be a mapping with 'min' and/or 'max'")
        minimum = data.get("min", data.get("minimum"))
        maximum = data.get("max", data.get("maximum"))
        return cls(
            metric=str(data.get("metric", name)),
            minimum=None if minimum is None else float(minimum),
            maximum=None if maximum is None else float(maximum),
            description=str(data.get("description", "")),
            required=bool(data.get("required", True)),
        )


@dataclass
class GateSet:
    name: str
    kind: str
    gates: list[Gate] = field(default_factory=list)
    source: str | None = None
    generated_at: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in GATE_KINDS:
            raise GateConfigError(f"kind must be one of {GATE_KINDS}, got {self.kind!r}")

    @property
    def is_quality_claim(self) -> bool:
        return self.kind == "baseline"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "generated_at": self.generated_at,
            "notes": self.notes,
            "is_quality_claim": self.is_quality_claim,
            "gates": [gate.to_dict() for gate in self.gates],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateSet:
        raw_gates = data.get("gates", data.get("quality_gates", {}))
        if isinstance(raw_gates, list):
            # A saved gate set serialises as a list of gate dicts. An empty list is a
            # valid (if useless) gate set and must reload rather than raise, otherwise a
            # file written by --write-baseline with nothing to gate cannot be read back.
            gates = [Gate.from_dict(item, name=str(item.get("metric", index))) for index, item in enumerate(raw_gates)]
        elif isinstance(raw_gates, Mapping):
            gates = [Gate.from_dict(spec, name=name) for name, spec in raw_gates.items()]
        else:
            raise GateConfigError("'gates' must be a mapping of name -> {min|max}, or a list of gate objects")
        return cls(
            name=str(data.get("name", "quality_gates")),
            kind=str(data.get("kind", "example")),
            gates=gates,
            source=data.get("source"),
            generated_at=data.get("generated_at"),
            notes=[str(note) for note in data.get("notes", [])],
        )


def load_gate_set(path: str | Path) -> GateSet:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise GateConfigError(f"{path} must contain a YAML mapping")
    gate_set = GateSet.from_dict(document)
    if gate_set.source is None:
        gate_set.source = str(path)
    return gate_set


def flatten_measurements(measurements: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten a nested report into ``dotted.key -> float`` lookups.

    Lists are flattened with numeric indices (``runs.0.summary...``), so a gate can
    address a specific benchmark run inside the aggregate report.
    """
    flat: dict[str, float] = {}
    for key, value in measurements.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_measurements(value, prefix=f"{name}."))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                flat.update(flatten_measurements({str(index): item}, prefix=f"{name}."))
        elif isinstance(value, bool) or value is None:
            continue
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


@dataclass
class GateOutcome:
    metric: str
    passed: bool
    message: str
    value: float | None
    gate: dict

    def to_dict(self) -> dict:
        return {"metric": self.metric, "passed": self.passed, "message": self.message, "value": self.value, "gate": self.gate}


@dataclass
class GateReport:
    gate_set: dict
    passed: bool
    outcomes: list[GateOutcome]
    measurements_file: str | None = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    warnings: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[GateOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.passed]

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "passed": self.passed,
            "generated_at": self.generated_at,
            "measurements_file": self.measurements_file,
            "gate_set": self.gate_set,
            "is_quality_claim": bool(self.gate_set.get("is_quality_claim")),
            "criteria_total": len(self.outcomes),
            "criteria_failed": len(self.failures),
            "failures": [outcome.to_dict() for outcome in self.failures],
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "warnings": self.warnings,
        }

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())

    def format_text(self) -> str:
        lines = [f"Quality gates: {self.status}"]
        if not self.gate_set.get("is_quality_claim"):
            lines.append("  NOTE: thresholds are placeholders ('example' kind), not a measured quality claim.")
        lines += [f"  [{'PASS' if outcome.passed else 'FAIL'}] {outcome.message}" for outcome in self.outcomes]
        return "\n".join(lines)


def evaluate_gates(
    gate_set: GateSet, measurements: Mapping[str, Any], *, measurements_file: str | Path | None = None
) -> GateReport:
    """Evaluate every gate against flattened measurements."""
    flat = flatten_measurements(measurements)
    outcomes: list[GateOutcome] = []
    warnings: list[str] = []
    for gate in gate_set.gates:
        value = flat.get(gate.metric)
        passed, message = gate.check(value)
        if value is None and gate.required:
            warnings.append(
                f"Metric {gate.metric!r} was not found in the measurements. "
                f"Available metrics include: {', '.join(sorted(flat)[:12])}"
            )
        outcomes.append(GateOutcome(metric=gate.metric, passed=passed, message=message, value=value, gate=gate.to_dict()))
    return GateReport(
        gate_set=gate_set.to_dict(),
        passed=bool(outcomes) and all(outcome.passed for outcome in outcomes),
        outcomes=outcomes,
        measurements_file=str(measurements_file) if measurements_file else None,
        warnings=warnings,
    )


def evaluate_gates_from_file(gate_path: str | Path, measurements_path: str | Path) -> GateReport:
    gate_set = load_gate_set(gate_path)
    measurements = read_json(measurements_path)
    if not isinstance(measurements, Mapping):
        raise GateConfigError(f"Measurements file {measurements_path} must be a JSON object")
    return evaluate_gates(gate_set, measurements, measurements_file=measurements_path)


def baseline_from_measurements(
    measurements: Mapping[str, Any],
    metrics: Iterable[str],
    *,
    tolerance: float = 0.0,
    name: str = "measured-baseline",
) -> GateSet:
    """Build a ``baseline`` gate set from a real measurement file.

    Cost-like metrics get upper bounds, quality metrics get lower bounds, and the
    tolerance loosens each bound by a relative fraction.
    """
    flat = flatten_measurements(measurements)
    gates: list[Gate] = []
    missing: list[str] = []
    for metric in metrics:
        value = flat.get(metric)
        if value is None:
            missing.append(metric)
            continue
        if any(token in metric.lower() for token in LOWER_IS_BETTER):
            gates.append(Gate(metric=metric, maximum=value * (1.0 + tolerance), description="baseline (lower is better)"))
        else:
            gates.append(Gate(metric=metric, minimum=value * (1.0 - tolerance), description="baseline (higher is better)"))
    gate_set = GateSet(
        name=name,
        kind="baseline",
        gates=gates,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=[
            "Thresholds derived from a measured run of this repository.",
            f"Relative tolerance applied: {tolerance:.4f}.",
        ],
    )
    if missing:
        gate_set.notes.append(f"Metrics requested but absent from the measurements (not gated): {sorted(missing)}")
    return gate_set


def save_gate_set(gate_set: GateSet, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(ensure_jsonable(gate_set.to_dict()), sort_keys=False), encoding="utf-8")
    return path


def default_gate_metrics() -> Sequence[str]:
    """Metric paths a complete detection+tracking report should gate on.

    Paths are relative to ``report.json``, where every artefact sits under ``artifacts``.
    """
    return (
        "artifacts.detection_metrics.groups.overall.map50_95",
        "artifacts.detection_metrics.groups.overall.map50",
        "artifacts.tracking_metrics.aggregate.idf1",
        "artifacts.tracking_metrics.aggregate.mota",
        "artifacts.tracking_metrics.aggregate.num_switches",
    )
