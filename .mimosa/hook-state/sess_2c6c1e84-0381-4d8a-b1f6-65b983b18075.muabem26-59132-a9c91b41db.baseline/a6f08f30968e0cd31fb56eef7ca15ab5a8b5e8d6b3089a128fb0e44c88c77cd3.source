from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DetectionCounts:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_detection_counts(ground_truth: list[np.ndarray], predictions: list[np.ndarray], iou_threshold: float = 0.5) -> DetectionCounts:
    matched_gt: set[int] = set()
    tp = 0
    for pred in predictions:
        candidates = [
            (iou_xyxy(pred, gt), idx)
            for idx, gt in enumerate(ground_truth)
            if idx not in matched_gt
        ]
        best_iou, best_idx = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1
    return DetectionCounts(tp, len(predictions) - tp, len(ground_truth) - tp)
