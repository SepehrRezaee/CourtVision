import numpy as np
import pytest

from courtvision.metrics import greedy_detection_counts, iou_xyxy


def test_iou_identity() -> None:
    box = np.array([0, 0, 10, 10], dtype=float)
    assert iou_xyxy(box, box) == pytest.approx(1.0)


def test_iou_disjoint() -> None:
    a = np.array([0, 0, 10, 10], dtype=float)
    b = np.array([20, 20, 30, 30], dtype=float)
    assert iou_xyxy(a, b) == 0.0


def test_greedy_counts() -> None:
    gt = [
        np.array([0, 0, 10, 10], dtype=float),
        np.array([20, 20, 30, 30], dtype=float),
    ]
    pred = [
        np.array([0, 0, 10, 10], dtype=float),
        np.array([50, 50, 60, 60], dtype=float),
    ]
    counts = greedy_detection_counts(gt, pred, iou_threshold=0.5)
    assert counts.true_positive == 1
    assert counts.false_positive == 1
    assert counts.false_negative == 1
    assert counts.precision == pytest.approx(0.5)
    assert counts.recall == pytest.approx(0.5)
