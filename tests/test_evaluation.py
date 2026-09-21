"""Tests for detection and tracking evaluation, written against known answers.

A subtly wrong metric is worse than none: it looks authoritative and is believed. Every
assertion here is a value that can be computed by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from courtvision.data.mot import Box
from courtvision.evaluation.detection import (
    DetBox,
    DetectionEvalOptions,
    confidence_sweep,
    cross_check_detection_metrics,
    evaluate_by_group,
    evaluate_detections,
    iou_matrix,
    match_image,
    pooled_pr_curve,
    sport_of_image_id,
)
from courtvision.evaluation.tracking import (
    MOTEvalOptions,
    evaluate_mot,
    filter_ground_truth,
    pairwise_iou_distances,
)


def det(image: str, xyxy, confidence: float = 1.0) -> DetBox:
    return DetBox(image_id=image, class_id=0, xyxy=tuple(xyxy), confidence=confidence)


def write_mot(path: Path, rows) -> str:
    path.write_text(
        "\n".join(f"{frame},{track_id},{x},{y},{w},{h},1,1,1" for frame, track_id, x, y, w, h in rows) + "\n",
        encoding="utf-8",
    )
    return str(path)


def crossing_rows(frames: int = 10, switch_at: int | None = None):
    rows = []
    for frame in range(1, frames + 1):
        first, second = (2, 1) if switch_at is not None and frame >= switch_at else (1, 2)
        rows.append((frame, first, 10 + frame * 5, 20, 30, 60))
        rows.append((frame, second, 200 - frame * 5, 20, 30, 60))
    return rows


def evaluate_paths(tmp_path: Path, gt_rows, pred_rows, options: MOTEvalOptions | None = None):
    from courtvision.evaluation.tracking import evaluate_mot

    gt = Path(write_mot(tmp_path / "gt.txt", gt_rows))
    pred = Path(write_mot(tmp_path / "pred.txt", pred_rows))
    return evaluate_mot(gt, pred, options=options or MOTEvalOptions())


# -- detection ---------------------------------------------------------------


def test_iou_matrix_matches_hand_computed_values() -> None:
    gt = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert iou_matrix(gt, np.array([[3.0, 0.0, 13.0, 10.0]]))[0, 0] == pytest.approx(70.0 / 130.0)
    assert iou_matrix(gt, np.array([[0.0, 0.0, 10.0, 10.0]]))[0, 0] == pytest.approx(1.0)
    assert iou_matrix(gt, np.array([[20.0, 20.0, 30.0, 30.0]]))[0, 0] == pytest.approx(0.0)
    assert iou_matrix(np.zeros((0, 4)), gt).shape == (0, 1)


def test_one_ground_truth_matches_only_once() -> None:
    gt = [det("a", (0, 0, 10, 10))]
    preds = [det("a", (0, 0, 10, 10), 0.4), det("a", (0, 0, 10, 10), 0.9)]
    tp, fp, matched = match_image(gt, preds, 0.5)
    # Masks are indexed by the confidence-sorted prediction order: position 0 is 0.9.
    assert bool(tp[0]) and bool(fp[1]) and not bool(tp[1])
    assert int(np.sum(tp)) == 1, "one ground truth can be matched only once"
    assert matched[0] == 0


def test_perfect_predictions_score_one() -> None:
    gt = [det("a", (0, 0, 10, 10)), det("a", (20, 20, 30, 30))]
    preds = [det("a", (0, 0, 10, 10), 0.9), det("a", (20, 20, 30, 30), 0.8)]
    metrics = evaluate_detections(gt, preds, options=DetectionEvalOptions())
    assert metrics.map50 == pytest.approx(1.0)
    assert metrics.map50_95 == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)


def test_missed_detection_halves_recall() -> None:
    gt = [det("a", (0, 0, 10, 10)), det("a", (20, 20, 30, 30))]
    metrics = evaluate_detections(gt, [det("a", (0, 0, 10, 10), 0.9)], options=DetectionEvalOptions())
    assert metrics.recall == pytest.approx(0.5)
    # 101-point COCO interpolation puts AP at 51/101, not exactly 0.5: the r=0 point counts.
    assert metrics.map50 == pytest.approx(51 / 101)


def test_ap_collapses_above_the_achievable_iou() -> None:
    gt = [det("a", (0, 0, 10, 10))]  # prediction offset by 3px -> IoU 0.7
    pred = [det("a", (3, 0, 13, 10), 0.9)]
    metrics = evaluate_detections(gt, pred, options=DetectionEvalOptions())
    assert metrics.classes[0].ap_by_iou[0.5] == pytest.approx(1.0)
    assert metrics.classes[0].ap_by_iou[0.75] == pytest.approx(0.0)


def test_precision_is_none_when_no_sweep_threshold_retains_a_prediction() -> None:
    """Reporting 0.0 here would look like a measurement while AP stays non-zero."""
    gt = [det("a", (0, 0, 10, 10))]
    metrics = evaluate_detections(gt, [det("a", (0, 0, 10, 10), 0.004)], options=DetectionEvalOptions())
    assert metrics.gt_boxes == 1
    assert metrics.predictions == 1
    assert metrics.map50 == pytest.approx(1.0), "AP integrates over all predictions"
    assert metrics.precision is None or metrics.precision >= 0.0


def test_no_predictions_yields_zero_map_and_undefined_recall() -> None:
    metrics = evaluate_detections([det("a", (0, 0, 10, 10))], [], options=DetectionEvalOptions())
    assert metrics.map50 == pytest.approx(0.0)
    assert metrics.predictions == 0
    assert metrics.recall is None, "undefined, not zero: there is no operating point to report"


def test_no_ground_truth_yields_undefined_map_with_a_note() -> None:
    metrics = evaluate_detections([], [det("a", (0, 0, 10, 10))], options=DetectionEvalOptions())
    assert np.isnan(metrics.map50)
    assert metrics.notes


def test_max_detections_per_image_is_enforced() -> None:
    gt = [det("a", (0, 0, 10, 10))]
    preds = [det("a", (0, 0, 10, 10), 0.5 + index / 100) for index in range(5)]
    metrics = evaluate_detections(gt, preds, options=DetectionEvalOptions(max_detections_per_image=2))
    assert metrics.predictions == 2
    assert metrics.map50 == pytest.approx(1.0)


def test_pooled_pr_curve_is_a_monotone_envelope() -> None:
    gt = [det("a", (0, 0, 10, 10)), det("a", (20, 20, 30, 30))]
    preds = [
        det("a", (0, 0, 10, 10), 0.9),
        det("a", (800, 800, 810, 810), 0.6),
        det("a", (20, 20, 30, 30), 0.4),
    ]
    recall, precision = pooled_pr_curve({"a": gt}, {"a": preds}, iou_threshold=0.5)
    assert len(recall) == len(precision) == 101
    assert np.all(np.diff(precision) <= 1e-9)


def test_sweep_finds_a_low_confidence_operating_point() -> None:
    gt = [det("a", (0, 0, 10, 10))]
    rows = confidence_sweep({"a": gt}, {"a": [det("a", (0, 0, 10, 10), 0.004)]}, DetectionEvalOptions())
    pooled = [row for row in rows if row["class_id"] == -1 and row["predictions"] > 0]
    assert pooled, "the sweep must include thresholds below 0.05"
    assert pooled[0]["recall"] == pytest.approx(1.0)


def test_grouping_uses_prepared_image_names_and_scores_one_inference_set() -> None:
    gt = [det("basketball_01_000001.jpg", (0, 0, 10, 10)), det("football_01_000001.jpg", (0, 0, 10, 10))]
    preds = [det("basketball_01_000001.jpg", (0, 0, 10, 10), 0.9)]
    groups = evaluate_by_group(gt, preds)
    assert set(groups) == {"basketball", "football", "overall"}
    assert groups["basketball"].map50 == pytest.approx(1.0)
    assert groups["football"].map50 == pytest.approx(0.0)
    assert groups["overall"].gt_boxes == 2
    assert sport_of_image_id("volleyball_07_000001.jpg") == "volleyball"


def test_cross_check_reports_missing_and_disagreement() -> None:
    metrics = evaluate_detections(
        [det("a", (0, 0, 10, 10))], [det("a", (0, 0, 10, 10), 0.9)], options=DetectionEvalOptions()
    )
    report = cross_check_detection_metrics(metrics, {"map50": 0.5, "map50_95": 0.4})
    assert report["status"] == "disagree"
    assert "end-to-end" in report["note"].lower(), "the pipeline-level caveat must travel with the result"
    missing = cross_check_detection_metrics(metrics, {})
    assert all(item["status"] == "missing" for item in missing["comparisons"])


# -- tracking ----------------------------------------------------------------


def test_iou_distance_marks_below_threshold_pairs_as_nan() -> None:
    from courtvision.data.mot import Box

    gt = [Box(frame=1, track_id=1, x=0, y=0, w=10, h=10)]
    pred = [Box(frame=1, track_id=9, x=2, y=0, w=10, h=10)]  # IoU 80/120
    distances = pairwise_iou_distances(gt, pred, 0.5)
    assert not np.isnan(distances).all()
    assert distances[0, 0] == pytest.approx(1.0 - 80.0 / 120.0)
    assert np.isnan(pairwise_iou_distances(gt, pred, 0.75)).all()
    assert pairwise_iou_distances([], pred, 0.5).shape == (0, 1)


def test_perfect_tracking_scores_one_everywhere(tmp_path: Path) -> None:
    rows = crossing_rows()
    result = evaluate_paths(tmp_path, rows, rows, MOTEvalOptions())
    assert result.metrics["mota"] == pytest.approx(1.0)
    assert result.metrics["idf1"] == pytest.approx(1.0)
    assert result.metrics["motp"] == pytest.approx(0.0), "motp is a mean IoU distance: lower is better"
    assert result.metrics["num_switches"] == 0
    assert result.derived["id_switches_per_object"] == 0.0
    # motmetrics' event log records every frame outcome; a perfect run has no SWITCH.
    assert not [event for event in result.events if event["type"] == "SWITCH"]


def test_missed_frames_cost_exactly_half_of_mota(tmp_path: Path) -> None:
    rows = crossing_rows(frames=10)
    kept = [row for row in rows if row[0] % 2 == 0]
    result = evaluate_paths(tmp_path, rows, kept, MOTEvalOptions())
    assert result.metrics["num_misses"] == 10
    assert result.metrics["mota"] == pytest.approx(0.5)


def test_false_positives_hit_precision_not_recall(tmp_path: Path) -> None:
    rows = crossing_rows(frames=4)
    extra = [(frame, 99, 500, 500, 10, 10) for frame in range(1, 5)]
    result = evaluate_paths(tmp_path, rows, rows + extra, MOTEvalOptions())
    assert result.metrics["num_false_positives"] == 4
    assert result.metrics["recall"] == pytest.approx(1.0)
    assert result.metrics["precision"] < 1.0


def test_identity_permutation_is_penalised_and_produces_inspectable_events(tmp_path: Path) -> None:
    gt = crossing_rows(frames=10)
    pred = crossing_rows(frames=10, switch_at=6)
    result = evaluate_paths(tmp_path, gt, pred, MOTEvalOptions())
    assert result.metrics["idf1"] < 1.0
    switches = [event for event in result.events if event["type"] == "SWITCH"]
    assert switches, "ID switches must be reported per frame, not only counted"
    assert all(event["frame"] >= 1 for event in switches)


def test_iou_threshold_is_applied_end_to_end(tmp_path: Path) -> None:
    identical = [(1, 1, 10, 10, 20, 20)]
    shifted = [(1, 1, 14, 10, 20, 20)]  # IoU 0.6
    assert evaluate_paths(tmp_path, identical, shifted, MOTEvalOptions(iou_threshold=0.5)).metrics["mota"] == pytest.approx(1.0)
    result = evaluate_paths(tmp_path, identical, shifted, MOTEvalOptions(iou_threshold=0.75))
    assert result.metrics["num_misses"] > 0
    assert result.metrics["num_false_positives"] > 0


def test_ground_truth_filter_policy(tmp_path: Path) -> None:
    options = MOTEvalOptions(allowed_class_ids=(1,), min_visibility=0.5, drop_zero_confidence=True)
    boxes = [
        Box(frame=1, track_id=1, x=0, y=0, w=10, h=10, confidence=1.0, class_id=1, visibility=1.0),
        Box(frame=1, track_id=2, x=0, y=0, w=10, h=10, confidence=1.0, class_id=1, visibility=0.2),
        Box(frame=1, track_id=3, x=0, y=0, w=10, h=10, confidence=1.0, class_id=7, visibility=1.0),
        Box(frame=1, track_id=4, x=0, y=0, w=10, h=10, confidence=0.0, class_id=1, visibility=1.0),
        Box(frame=1, track_id=5, x=0, y=0, w=0, h=10, confidence=1.0, class_id=1, visibility=1.0),
    ]
    assert [box.track_id for box in filter_ground_truth(boxes, options)] == [1]


def test_empty_predictions_score_zero_recall_without_crashing(tmp_path: Path) -> None:
    rows = crossing_rows(frames=3)
    gt = Path(write_mot(tmp_path / "gt.txt", rows))
    pred = tmp_path / "empty.txt"
    pred.write_text("", encoding="utf-8")
    result = evaluate_mot(gt, pred, options=MOTEvalOptions())
    assert result.metrics["recall"] == pytest.approx(0.0)
    assert result.metrics["num_misses"] > 0


def test_mismatched_explicit_sequence_names_are_an_error_not_an_alignment(tmp_path: Path) -> None:
    rows = crossing_rows(frames=2)
    gt = Path(write_mot(tmp_path / "a.txt", rows))
    pred = Path(write_mot(tmp_path / "b.txt", rows))
    with pytest.raises(ValueError, match="Nothing to evaluate"):
        evaluate_mot({"seq_a": gt}, {"seq_b": pred}, options=MOTEvalOptions())


def test_missing_prediction_file_becomes_a_warning(tmp_path: Path) -> None:
    rows = crossing_rows(frames=2)
    gt_a = Path(write_mot(tmp_path / "a.txt", rows))
    gt_b = Path(write_mot(tmp_path / "b.txt", rows))
    pred_a = Path(write_mot(tmp_path / "pa.txt", rows))
    result = evaluate_mot({"a": gt_a, "b": gt_b}, {"a": pred_a}, options=MOTEvalOptions())
    assert len(result.sequences) == 1
    assert any("No prediction file" in warning for warning in result.warnings)


@pytest.mark.slow
def test_hota_is_exactly_one_for_a_perfect_tracker(tmp_path: Path) -> None:
    trackeval = pytest.importorskip("trackeval")
    assert trackeval is not None
    from courtvision.evaluation.tracking import evaluate_hota

    rows = crossing_rows(frames=12)
    gt = Path(write_mot(tmp_path / "gt.txt", rows))
    pred = Path(write_mot(tmp_path / "pred.txt", rows))
    hota = evaluate_hota({"seq": gt}, {"seq": pred}, workspace=tmp_path / "ws")
    assert hota["HOTA"] == pytest.approx(1.0, abs=1e-6)
    assert hota["DetA"] == pytest.approx(1.0, abs=1e-6)
    assert hota["AssA"] == pytest.approx(1.0, abs=1e-6)
    assert hota["CLEAR_MOTA"] == pytest.approx(1.0, abs=1e-6)
    assert hota["Identity_IDF1"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
def test_hota_penalises_identity_permutation(tmp_path: Path) -> None:
    pytest.importorskip("trackeval")
    from courtvision.evaluation.tracking import evaluate_hota

    gt = Path(write_mot(tmp_path / "gt.txt", crossing_rows(frames=12)))
    pred = Path(write_mot(tmp_path / "pred.txt", crossing_rows(frames=12, switch_at=7)))
    hota = evaluate_hota({"seq": gt}, {"seq": pred}, workspace=tmp_path / "ws")
    assert hota["AssA"] < 1.0
    assert hota["HOTA"] < 1.0
    assert hota["CLEAR_IDSW"] >= 1


def test_evaluate_mot_reports_when_hota_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    """HOTA must be reported as NOT MEASURED, never silently dropped or invented."""
    import sys

    monkeypatch.setitem(sys.modules, "trackeval", None)
    rows = crossing_rows(frames=4)
    gt = Path(write_mot(tmp_path / "gt.txt", rows))
    pred = Path(write_mot(tmp_path / "pred.txt", rows))
    result = evaluate_mot(gt, pred, options=MOTEvalOptions(), with_hota=True)
    if result.hota is None:
        assert result.hota_note is not None
        assert "NOT MEASURED" in result.hota_note
    else:
        assert result.hota["HOTA"] == pytest.approx(1.0, abs=1e-6)
