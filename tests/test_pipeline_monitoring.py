"""Tests for drift monitoring, runtime metrics, quality gates, Pareto and reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from courtvision.monitoring.drift import (
    compare_distributions,
    detect_performance_regression,
    histogram_counts_with_edges,
    jensen_shannon_divergence,
    kolmogorov_smirnov_statistic,
    population_stability_index,
    reference_from_signals,
    signals_from_tracking_result,
    summarize_signal,
)
from courtvision.monitoring.metrics import REGISTRY, MetricsRegistry, percentile_summary
from courtvision.optimization.benchmark import BenchmarkPlan, benchmark_frames, summarize_rows
from courtvision.optimization.pareto import (
    ConfigurationRecord,
    build_pareto_report,
    configuration_records_from_reports,
    dominates,
)
from courtvision.pipeline.quality_gates import (
    Gate,
    GateConfigError,
    GateSet,
    baseline_from_measurements,
    default_gate_metrics,
    evaluate_gates,
    flatten_measurements,
    save_gate_set,
)
from courtvision.reports.builder import build_report, collect_artifacts
from courtvision.tracking.base import Detection, FrameResult, TrackingResult


@pytest.fixture
def tracking_result() -> TrackingResult:
    frames = []
    for frame in range(1, 21):
        detections = [
            Detection(frame=frame, xyxy=(10.0 + frame * 4, 50.0, 40.0 + frame * 4, 110.0), confidence=0.9, track_id=1)
        ]
        if frame not in (7, 8):  # deliberate gap in track 2
            detections.append(
                Detection(frame=frame, xyxy=(200.0 - frame * 4, 60.0, 230.0 - frame * 4, 120.0), confidence=0.8, track_id=2)
            )
        frames.append(FrameResult(frame=frame, detections=detections))
    return TrackingResult(source="synthetic://monitoring", frames=frames, frame_size=(320, 180), source_fps=25.0)


# -- drift statistics --------------------------------------------------------


def test_psi_is_zero_for_identical_histograms_and_needs_matching_bins() -> None:
    counts = [10.0, 20.0, 30.0]
    assert population_stability_index(counts, counts) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="identically binned"):
        population_stability_index([1.0, 2.0], [1.0, 2.0, 3.0])


def test_psi_grows_with_shift_and_survives_empty_bins() -> None:
    reference = [100.0, 100.0, 100.0, 100.0]
    mild = population_stability_index(reference, [110.0, 100.0, 95.0, 95.0])
    severe = population_stability_index(reference, [400.0, 50.0, 30.0, 20.0])
    assert 0.0 <= mild < severe
    assert severe > 0.25
    assert np.isfinite(population_stability_index([0.0, 10.0, 0.0], [0.0, 5.0, 5.0]))


def test_ks_statistic_bounds() -> None:
    sample = np.linspace(0, 1, 50)
    assert kolmogorov_smirnov_statistic(sample, sample) == pytest.approx(0.0, abs=1e-9)
    assert kolmogorov_smirnov_statistic([0.0, 0.1], [10.0, 11.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        kolmogorov_smirnov_statistic([], [1.0])


def test_js_divergence_is_bounded_by_one() -> None:
    counts = [10.0, 20.0, 30.0]
    assert jensen_shannon_divergence(counts, counts) == pytest.approx(0.0, abs=1e-9)
    assert jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=1e-6)


def test_histogram_uses_reference_edges_and_keeps_outliers() -> None:
    counts = histogram_counts_with_edges([0.5, 0.5, 1.5], [0.0, 1.0, 2.0, 3.0])
    assert list(counts) == [2, 1, 0]
    assert int(histogram_counts_with_edges([-5.0, 99.0], [0.0, 1.0, 2.0, 3.0]).sum()) == 2


def test_summarize_signal_rejects_empty_and_ignores_non_finite() -> None:
    with pytest.raises(ValueError, match="no finite values"):
        summarize_signal("x", [])
    assert summarize_signal("x", [1.0, np.nan, 3.0]).count == 2


def test_drift_between_two_draws_of_one_distribution_is_stable() -> None:
    rng = np.random.default_rng(0)
    reference = reference_from_signals({"x": rng.normal(0, 1, 2000)}, bins=20)
    stable = compare_distributions(reference, {"x": rng.normal(0, 1, 2000).tolist()})
    assert stable.signals["x"]["verdict"] == "stable"
    assert not stable.alerts
    assert stable.max_psi < 0.1, "two draws of one distribution must not read as drift"

    shifted = compare_distributions(reference, {"x": rng.normal(6, 1, 2000).tolist()})
    assert shifted.signals["x"]["verdict"] == "alert"
    assert shifted.alerts and shifted.alerts[0]["signal"] == "x"


def test_out_of_range_mass_lands_in_the_edge_bins() -> None:
    """Mass outside the reference range must stay visible: dropping it would understate drift."""
    reference = reference_from_signals({"x": list(range(100))}, bins=10)
    far_away = compare_distributions(reference, {"x": [10_000.0] * 100})
    assert far_away.signals["x"]["verdict"] == "alert", "total distribution replacement must alert"


def test_drift_is_labelled_a_proxy_signal() -> None:
    reference = reference_from_signals({"x": list(range(100))}, bins=10)
    report = compare_distributions(reference, {"x": list(range(100))})
    payload = report.to_dict()
    assert payload["is_proxy_signal"] is True
    assert "do not by themselves demonstrate model degradation" in payload["interpretation"]


def test_unseen_signal_is_reported_not_silently_dropped() -> None:
    reference = reference_from_signals({"x": list(range(10))}, bins=5)
    report = compare_distributions(reference, {"x": list(range(10)), "new": [1.0, 2.0]})
    assert report.signals["new"]["verdict"] == "no-reference"
    assert any("new" in warning for warning in report.warnings)


def test_signals_from_tracking_result_cover_the_documented_set(tracking_result) -> None:
    signals = signals_from_tracking_result(tracking_result, latencies_ms=[10.0, 11.0])
    for name in (
        "detections_per_frame",
        "confidence",
        "track_duration_frames",
        "track_fragmentation",
        "track_length",
        "box_area_px",
        "box_aspect_ratio",
        "inference_latency_ms",
        "frame_width",
        "frame_height",
    ):
        assert name in signals
    assert max(signals["track_fragmentation"]) >= 1.0, "the fixture leaves a gap in track 2"


def test_performance_regression_is_direction_aware() -> None:
    payload = detect_performance_regression(
        {"idf1": 0.60, "num_switches": 100.0}, {"idf1": 0.55, "num_switches": 130.0}, relative_tolerance=0.02
    )
    assert {entry["metric"] for entry in payload["regressions"]} == {"idf1", "num_switches"}
    fewer = detect_performance_regression({"num_switches": 100.0}, {"num_switches": 80.0})
    assert not fewer["regressions"]
    assert fewer["requires_labels"] is True


# -- runtime metrics ---------------------------------------------------------


def test_registry_counters_gauges_and_prometheus_text() -> None:
    registry = MetricsRegistry()
    registry.record_analysis(frames=10, detections=25, unique_tracks=4, latency_ms=120.0)
    snapshot = registry.snapshot()
    assert snapshot["counters"]["courtvision_requests_total"] == 1
    assert snapshot["gauges"]["courtvision_last_detections_per_frame"] == pytest.approx(2.5)
    text = registry.render_prometheus()
    assert "courtvision_requests_total" in text and "# TYPE" in text
    assert "courtvision_request_latency_ms_bucket" in text


def test_percentile_summary_always_reports_n() -> None:
    summary = percentile_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["n"] == 4
    assert summary["p50"] == pytest.approx(2.5)
    assert percentile_summary([]) == {"n": 0}


def test_process_wide_registry_is_a_singleton() -> None:
    assert REGISTRY is not None
    REGISTRY.reset()
    REGISTRY.increment("courtvision_test_counter", 2)
    assert REGISTRY.snapshot()["counters"]["courtvision_test_counter"] == 2
    REGISTRY.reset()


# -- quality gates -----------------------------------------------------------


def measurements() -> dict:
    return {
        "artifacts": {
            "detection_metrics": {"groups": {"overall": {"map50": 0.72, "map50_95": 0.41}}},
            "tracking_metrics": {"aggregate": {"idf1": 0.55, "mota": 0.6, "num_switches": 120}},
        }
    }


def test_flatten_builds_dotted_paths_and_skips_non_numerics() -> None:
    flat = flatten_measurements(measurements())
    assert flat["artifacts.detection_metrics.groups.overall.map50"] == pytest.approx(0.72)
    assert flatten_measurements({"a": {"b": True, "c": None, "d": "x", "e": 1}}) == {"a.e": 1.0}


def test_gate_requires_a_bound_and_rejects_contradictions() -> None:
    with pytest.raises(GateConfigError, match="'min' or a 'max'"):
        Gate(metric="m")
    with pytest.raises(GateConfigError, match=r"has min 10\.0 > max 1\.0"):
        Gate(metric="m", minimum=10.0, maximum=1.0)
    with pytest.raises(GateConfigError, match="kind"):
        GateSet(name="x", kind="measured")


def test_gate_set_passes_and_fails_with_named_criteria() -> None:
    gate_set = GateSet(
        name="t",
        kind="baseline",
        gates=[
            Gate(metric="artifacts.detection_metrics.groups.overall.map50", minimum=0.5),
            Gate(metric="artifacts.tracking_metrics.aggregate.num_switches", maximum=200),
        ],
    )
    passing = evaluate_gates(gate_set, measurements())
    assert passing.passed and passing.status == "PASS"

    failing = GateSet(
        name="t",
        kind="baseline",
        gates=[Gate(metric="artifacts.detection_metrics.groups.overall.map50", minimum=0.9)],
    )
    report = evaluate_gates(failing, measurements())
    assert not report.passed
    assert report.failures[0].metric.endswith("map50")
    assert "map50=0.72" in report.format_text()


def test_missing_metric_fails_with_available_names_and_optional_passes() -> None:
    gate_set = GateSet(name="t", kind="baseline", gates=[Gate(metric="does.not.exist", minimum=0.1)])
    report = evaluate_gates(gate_set, measurements())
    assert not report.passed
    assert "Available metrics include" in report.warnings[0]
    optional = GateSet(
        name="t", kind="baseline", gates=[Gate(metric="does.not.exist", minimum=0.1, required=False)]
    )
    assert evaluate_gates(optional, measurements()).passed


def test_example_kind_is_not_a_quality_claim() -> None:
    gate_set = GateSet(name="x", kind="example", gates=[Gate(metric="a", minimum=0.0)])
    assert gate_set.to_dict()["is_quality_claim"] is False
    assert "placeholders" in evaluate_gates(gate_set, {"a": 1.0}).format_text()


def test_baseline_bounds_are_direction_aware() -> None:
    gate_set = baseline_from_measurements(
        measurements(),
        default_gate_metrics(),
        tolerance=0.1,
    )
    bounds = {gate.metric: gate for gate in gate_set.gates}
    assert bounds["artifacts.detection_metrics.groups.overall.map50"].minimum == pytest.approx(0.72 * 0.9)
    assert bounds["artifacts.tracking_metrics.aggregate.num_switches"].maximum == pytest.approx(120 * 1.1)
    assert bounds["artifacts.tracking_metrics.aggregate.num_switches"].minimum is None


def test_baseline_gates_detect_regression_beyond_tolerance() -> None:
    gate_set = baseline_from_measurements(
        measurements(), ["artifacts.detection_metrics.groups.overall.map50"], tolerance=0.02
    )
    assert evaluate_gates(gate_set, measurements()).passed
    degraded = {"artifacts": {"detection_metrics": {"groups": {"overall": {"map50": 0.60}}}}}
    assert not evaluate_gates(gate_set, degraded).passed
    noisy = {"artifacts": {"detection_metrics": {"groups": {"overall": {"map50": 0.70}}}}}
    tolerant = baseline_from_measurements(
        measurements(), ["artifacts.detection_metrics.groups.overall.map50"], tolerance=0.05
    )
    assert evaluate_gates(tolerant, noisy).passed


def test_gate_set_round_trips_through_yaml(tmp_path: Path) -> None:
    gate_set = GateSet(name="roundtrip", kind="baseline", gates=[Gate(metric="a", minimum=0.1)])
    from courtvision.pipeline.quality_gates import load_gate_set

    loaded = load_gate_set(save_gate_set(gate_set, tmp_path / "gates.yaml"))
    assert loaded.gates[0].metric == "a"
    assert loaded.gates[0].minimum == pytest.approx(0.1)


def test_empty_baseline_gate_set_round_trips(tmp_path: Path) -> None:
    """A --write-baseline run with nothing to gate must produce a readable file."""
    from courtvision.pipeline.quality_gates import load_gate_set

    empty = baseline_from_measurements(measurements(), ["absent.metric"])
    assert empty.gates == []
    path = save_gate_set(empty, tmp_path / "empty.yaml")
    assert load_gate_set(path).gates == []


# -- Pareto ------------------------------------------------------------------


def record(name: str, quality: float | None, latency: float | None) -> ConfigurationRecord:
    metrics = {}
    if quality is not None:
        metrics["idf1"] = quality
    if latency is not None:
        metrics["p95_latency_ms"] = latency
    return ConfigurationRecord(name=name, metrics=metrics)


def test_frontier_excludes_unmeasured_configurations() -> None:
    records = [record("a", 0.6, 50.0), record("b", 0.7, 50.0), record("unmeasured", None, 10.0)]
    report = build_pareto_report(records, quality_key="idf1", latency_key="p95_latency_ms")
    assert [item.name for item in report.frontier] == ["b"]
    assert report.excluded[0]["name"] == "unmeasured"
    assert report.recommendations["best_quality"].name == "b"
    assert report.recommendations["best_throughput"].name == "b"
    assert any("excluded" in note for note in report.notes)


def test_dominates_requires_better_on_one_axis() -> None:
    a, b = record("a", 0.6, 50.0), record("b", 0.6, 50.0)
    c = record("c", 0.7, 50.0)
    assert not dominates(a, b, quality_key="idf1", latency_key="p95_latency_ms")
    assert dominates(c, a, quality_key="idf1", latency_key="p95_latency_ms")
    assert not dominates(a, c, quality_key="idf1", latency_key="p95_latency_ms")


def test_single_point_frontier_says_all_recommendations_coincide() -> None:
    report = build_pareto_report([record("only", 0.5, 40.0)], quality_key="idf1", latency_key="p95_latency_ms")
    assert len(report.frontier) == 1
    assert any("refer to it" in note for note in report.notes)


def test_empty_frontier_is_stated_not_crashed() -> None:
    report = build_pareto_report([record("x", None, None)], quality_key="idf1", latency_key="p95_latency_ms")
    assert report.frontier == []
    assert report.recommendations["best_quality"] is None
    assert any("frontier is empty" in note for note in report.notes)


def test_records_normalise_latency_spellings() -> None:
    benchmarks = [
        {
            "label": "run_a",
            "tracker": "bytetrack.yaml",
            "detector": "yolo26n.pt",
            "imgsz": 640,
            "summary": {"pooled": {}, "rows": [{"end_to_end_ms_p95": 42.5, "end_to_end_ms_p50": 40.0}]},
        }
    ]
    records = configuration_records_from_reports([], benchmarks)
    assert records[0].metrics["p95_latency_ms"] == pytest.approx(42.5)
    assert records[0].metrics["p50_latency_ms"] == pytest.approx(40.0)


# -- benchmark summarising ---------------------------------------------------


class FakeFrame:
    def __init__(self, detections: int) -> None:
        self.timings_ms = {"preprocess": 1.0, "inference": 5.0, "postprocess": 0.5}
        self.detections = [
            Detection(frame=index + 1, xyxy=(0, 0, 1, 1), confidence=0.9, track_id=index % 3) for index in range(detections)
        ]


def test_benchmark_reports_short_clip_rather_than_warmup_numbers() -> None:
    frames = [object() for _ in range(6)]
    plan = BenchmarkPlan(warmup_frames=10, measured_frames=5)
    run = benchmark_frames(frames, lambda: (lambda frame: FakeFrame(2)), label="x", plan=plan)
    assert run.rows == []
    assert run.warnings and "no frames were measured" in run.warnings[0].lower()


def test_benchmark_excludes_warmup_and_derives_tracker_stage() -> None:
    frames = [object() for _ in range(15)]
    plan = BenchmarkPlan(warmup_frames=5, measured_frames=10)
    run = benchmark_frames(frames, lambda: (lambda frame: FakeFrame(3)), label="x", plan=plan)
    assert len(run.rows) == 10
    assert [row["frame_index"] for row in run.rows] == list(range(6, 16))
    row = run.rows[0]
    # Stage timings are backend-reported; with a fake processor the wall clock is near
    # zero, so the derived remainder clamps at zero instead of going negative.
    assert row["end_to_end_ms"] > 0
    assert row["tracking_and_post_ms"] == pytest.approx(
        max(0.0, row["end_to_end_ms"] - row["preprocess_ms"] - row["inference_ms"]), abs=0.01
    )
    assert row["detections"] == 3 and row["tracks"] == 3


def test_summarize_rows_derives_fps_from_mean_latency() -> None:
    rows = [
        {"repeat": 1, "end_to_end_ms": 40.0, "inference_ms": 20.0, "detections": 2, "tracks": 1},
        {"repeat": 1, "end_to_end_ms": 60.0, "inference_ms": 30.0, "detections": 2, "tracks": 1},
    ]
    summary = summarize_rows(rows, percentiles=(50.0,))
    entry = summary["rows"][0]
    assert entry["end_to_end_ms_mean"] == pytest.approx(50.0)
    assert entry["fps_from_mean"] == pytest.approx(20.0)
    assert summary["pooled"]["end_to_end_ms_mean_of_repeats"] == pytest.approx(50.0)


# -- reports -----------------------------------------------------------------


def test_report_lists_missing_artefacts_explicitly(tmp_path: Path) -> None:
    (tmp_path / "detection").mkdir(parents=True)
    (tmp_path / "detection" / "metrics.json").write_text(
        '{"groups": {"overall": {"images": 1, "gt_boxes": 2, "predictions": 2, "map50": 0.5, '
        '"map50_95": 0.3, "precision": 0.9, "recall": 0.8}}}',
        encoding="utf-8",
    )
    bundle = collect_artifacts(tmp_path)
    assert "detection_metrics" in bundle.artifacts
    assert "tracking_metrics" in bundle.missing

    result = build_report(tmp_path, output_dir=tmp_path / "latest")
    markdown = result.files["markdown"].read_text(encoding="utf-8")
    assert "Not measured" in markdown
    assert "HOTA" in markdown, "missing tracking artefacts must be called out by metric"
    payload = result.files["json"].read_text(encoding="utf-8")
    assert "detection_metrics" in payload and "tracking_metrics" in payload
    rows = result.files["csv"].read_text(encoding="utf-8")
    assert "detection" in rows
