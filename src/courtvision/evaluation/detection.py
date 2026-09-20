"""Detection evaluation: COCO-style mAP, confidence sweeps and grouped breakdowns.

Implemented in-repo rather than only calling ``model.val()`` so that per-group
breakdowns, confidence sweeps and PR curves can all be computed from **one** set of
predictions. Re-running inference per group would confound the comparison with
inference variance. Ultralytics' ``val()`` is kept as a reference implementation and
cross-checked against this one.

Matching rule (the VOC/COCO convention, as used by Ultralytics and pycocotools):
within each IoU threshold, sort predictions by descending confidence and greedily
match each to the highest-IoU unmatched ground-truth box above the threshold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..utils import ensure_dir, write_csv, write_json

DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * index, 2) for index in range(10))


@dataclass(frozen=True)
class DetBox:
    """A detection or ground-truth box in pixel ``xyxy``."""

    image_id: str
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0]) * max(0.0, self.xyxy[3] - self.xyxy[1])

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "class_id": self.class_id,
            "confidence": round(float(self.confidence), 6),
            "xyxy": [round(float(value), 3) for value in self.xyxy],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "DetBox":
        return cls(
            image_id=str(data["image_id"]),
            class_id=int(data["class_id"]),
            xyxy=tuple(float(value) for value in data["xyxy"]),  # type: ignore[arg-type]
            confidence=float(data.get("confidence", 1.0)),
        )


def iou_matrix(gt_xyxy: np.ndarray, pred_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N,4)`` and ``(M,4)`` xyxy arrays -> ``(N,M)``."""
    gt = np.asarray(gt_xyxy, dtype=float).reshape(-1, 4)
    pred = np.asarray(pred_xyxy, dtype=float).reshape(-1, 4)
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)), dtype=float)
    x1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    gt_area = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)
    pred_area = np.clip(pred[:, 2] - pred[:, 0], 0, None) * np.clip(pred[:, 3] - pred[:, 1], 0, None)
    union = gt_area[:, None] + pred_area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def group_by_image(boxes: Iterable[DetBox]) -> dict[str, list[DetBox]]:
    grouped: dict[str, list[DetBox]] = defaultdict(list)
    for box in boxes:
        grouped[box.image_id].append(box)
    return dict(grouped)


def match_image(gt: Sequence[DetBox], pred: Sequence[DetBox], iou_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy confidence-ordered matching for one image.

    Returns ``(tp_mask, fp_mask, matched_gt_index)`` indexed by the confidence-sorted
    prediction order; ``matched_gt_index`` is ``-1`` where unmatched.
    """
    if not pred:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), np.zeros(0, dtype=int)
    order = sorted(range(len(pred)), key=lambda index: -pred[index].confidence)
    ious = iou_matrix(
        np.array([box.xyxy for box in gt], dtype=float).reshape(-1, 4),
        np.array([pred[index].xyxy for index in order], dtype=float).reshape(-1, 4),
    )
    tp = np.zeros(len(pred), dtype=bool)
    fp = np.zeros(len(pred), dtype=bool)
    matched = np.full(len(pred), -1, dtype=int)
    taken = np.zeros(len(gt), dtype=bool)
    for position in range(len(pred)):
        if len(gt) == 0:
            fp[position] = True
            continue
        candidates = ious[:, position].copy()
        candidates[taken] = -1.0
        best = int(np.argmax(candidates))
        if candidates[best] >= iou_threshold:
            tp[position] = True
            taken[best] = True
            matched[position] = best
        else:
            fp[position] = True
    return tp, fp, matched


def _average_precision(tp: np.ndarray, fp: np.ndarray, n_gt: int, *, points: int = 101) -> tuple[float, np.ndarray, np.ndarray]:
    """COCO-style 101-point interpolated AP on the monotone precision envelope."""
    if n_gt == 0:
        return float("nan"), np.zeros(points), np.zeros(points)
    sample_points = np.linspace(0, 1, points)
    if len(tp) == 0:
        return 0.0, sample_points, np.zeros(points)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(float).eps)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, sample_points, side="left")
    valid = indices < len(recall)
    curve = np.zeros(points)
    curve[valid] = envelope[indices[valid]]
    return float(curve.mean()), sample_points, curve


@dataclass
class ClassMetrics:
    class_id: int
    class_name: str
    gt_boxes: int
    predictions: int
    ap_by_iou: dict[float, float] = field(default_factory=dict)
    #: ``None`` when no sweep threshold retained a prediction, i.e. undefined rather than 0.
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    best_f1_confidence: float | None = None
    mean_iou_matched: float = 0.0

    @property
    def map50(self) -> float:
        return self.ap_by_iou.get(0.5, float("nan"))

    @property
    def map50_95(self) -> float:
        values = [value for value in self.ap_by_iou.values() if not np.isnan(value)]
        return float(np.mean(values)) if values else float("nan")

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "gt_boxes": self.gt_boxes,
            "predictions": self.predictions,
            "map50": _round(self.map50),
            "map50_95": _round(self.map50_95),
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "f1": _round(self.f1),
            "best_f1_confidence": _round(self.best_f1_confidence),
            "mean_iou_matched": round(self.mean_iou_matched, 6),
            "ap_by_iou": {str(key): _round(value) for key, value in sorted(self.ap_by_iou.items())},
        }


@dataclass
class DetectionMetrics:
    group: str
    images: int
    gt_boxes: int
    predictions: int
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS
    classes: dict[int, ClassMetrics] = field(default_factory=dict)
    confidence_sweep: list[dict] = field(default_factory=list)
    pr_curve_50: dict[str, list[float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def _gated_classes(self) -> list[ClassMetrics]:
        """Classes with ground truth. A class with no GT contributes no AP and no P/R.

        Without this gate a single-class dataset's aggregate precision/recall would be
        dragged to zero by the detector's other-class false positives — classes that the
        dataset does not contain and that mAP already ignores.
        """
        return [item for item in self.classes.values() if item.gt_boxes > 0]

    @property
    def map50(self) -> float:
        values = [item.map50 for item in self._gated_classes() if not np.isnan(item.map50)]
        return float(np.mean(values)) if values else float("nan")

    @property
    def map50_95(self) -> float:
        values = [item.map50_95 for item in self._gated_classes() if not np.isnan(item.map50_95)]
        return float(np.mean(values)) if values else float("nan")

    @property
    def precision(self) -> float | None:
        values = [item.precision for item in self._gated_classes() if item.precision is not None]
        return float(np.mean(values)) if values else None

    @property
    def recall(self) -> float | None:
        values = [item.recall for item in self._gated_classes() if item.recall is not None]
        return float(np.mean(values)) if values else None

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "images": self.images,
            "gt_boxes": self.gt_boxes,
            "predictions": self.predictions,
            "map50": _round(self.map50),
            "map50_95": _round(self.map50_95),
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "iou_thresholds": list(self.iou_thresholds),
            "per_class": {str(key): value.to_dict() for key, value in sorted(self.classes.items())},
            "confidence_sweep": self.confidence_sweep,
            "pr_curve_50": self.pr_curve_50,
            "notes": self.notes,
        }


def _round(value, digits: int = 6):
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    return None if np.isnan(as_float) else round(as_float, digits)


@dataclass
class DetectionEvalOptions:
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS
    class_names: Mapping[int, str] = field(default_factory=lambda: {0: "player"})
    #: Includes sub-0.05 thresholds so a low-confidence operating point is still found:
    #: on some data every prediction scores below 0.05, and a sweep that starts at 0.05
    #: would report P/R as undefined even though predictions exist.
    sweep_confidences: tuple[float, ...] = (
        0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    )
    min_confidence: float = 0.001
    max_detections_per_image: int | None = 300


def evaluate_detections(
    ground_truth: Sequence[DetBox],
    predictions: Sequence[DetBox],
    *,
    group: str = "all",
    options: DetectionEvalOptions | None = None,
    image_filter: Iterable[str] | None = None,
) -> DetectionMetrics:
    """Score predictions for one group of images (``image_filter`` selects a subset)."""
    options = options or DetectionEvalOptions()
    allowed = set(image_filter) if image_filter is not None else None

    gt_by_image = {
        image: list(boxes)
        for image, boxes in group_by_image(ground_truth).items()
        if allowed is None or image in allowed
    }
    pred_by_image: dict[str, list[DetBox]] = {}
    for image, boxes in group_by_image(predictions).items():
        if allowed is not None and image not in allowed:
            continue
        kept = [box for box in boxes if box.confidence >= options.min_confidence]
        if options.max_detections_per_image and len(kept) > options.max_detections_per_image:
            # Mirrors Ultralytics' per-image cap so the two implementations compare.
            kept = sorted(kept, key=lambda box: -box.confidence)[: options.max_detections_per_image]
        pred_by_image[image] = kept

    images = sorted(set(gt_by_image) | set(pred_by_image))
    metrics = DetectionMetrics(
        group=group,
        images=len(images),
        gt_boxes=sum(len(boxes) for boxes in gt_by_image.values()),
        predictions=sum(len(boxes) for boxes in pred_by_image.values()),
        iou_thresholds=options.iou_thresholds,
    )
    swept = confidence_sweep(gt_by_image, pred_by_image, options)
    metrics.confidence_sweep = swept

    class_ids = sorted(
        {box.class_id for boxes in gt_by_image.values() for box in boxes}
        | {box.class_id for boxes in pred_by_image.values() for box in boxes}
    )
    for class_id in class_ids:
        metrics.classes[class_id] = _evaluate_class(
            class_id, gt_by_image, pred_by_image, options, swept
        )

    recall_points, precision_curve = pooled_pr_curve(gt_by_image, pred_by_image, iou_threshold=0.5)
    metrics.pr_curve_50 = {
        "recall": [round(float(value), 6) for value in recall_points],
        "precision": [round(float(value), 6) for value in precision_curve],
    }
    if metrics.gt_boxes == 0:
        metrics.notes.append("No ground-truth boxes in this group; mAP is undefined (reported as null).")
    return metrics


def _evaluate_class(
    class_id: int,
    gt_by_image: Mapping[str, Sequence[DetBox]],
    pred_by_image: Mapping[str, Sequence[DetBox]],
    options: DetectionEvalOptions,
    sweep: Sequence[Mapping],
) -> ClassMetrics:
    n_gt = sum(1 for boxes in gt_by_image.values() for box in boxes if box.class_id == class_id)
    per_iou_tp: dict[float, list[bool]] = {}
    ious_matched: list[float] = []

    for threshold in options.iou_thresholds:
        tp_all: list[bool] = []
        score_all: list[float] = []
        for image in sorted(set(gt_by_image) | set(pred_by_image)):
            gt = [box for box in gt_by_image.get(image, []) if box.class_id == class_id]
            pred = [box for box in pred_by_image.get(image, []) if box.class_id == class_id]
            if not pred:
                continue
            tp, _fp, matched = match_image(gt, pred, threshold)
            order = sorted(range(len(pred)), key=lambda index: -pred[index].confidence)
            score_all.extend(pred[index].confidence for index in order)
            tp_all.extend(bool(value) for value in tp)
            if abs(threshold - 0.5) < 1e-9:
                ious = iou_matrix(
                    np.array([box.xyxy for box in gt], dtype=float).reshape(-1, 4),
                    np.array([pred[index].xyxy for index in order], dtype=float).reshape(-1, 4),
                )
                ious_matched.extend(float(ious[gt_index, position]) for position, gt_index in enumerate(matched) if gt_index >= 0)
        order = np.argsort(-np.asarray(score_all, dtype=float)) if score_all else np.zeros(0, dtype=int)
        per_iou_tp[threshold] = [tp_all[index] for index in order]

    ap_by_iou: dict[float, float] = {}
    for threshold in options.iou_thresholds:
        flags = np.asarray(per_iou_tp[threshold], dtype=bool)
        ap, _recall, _precision = _average_precision(flags, ~flags, n_gt)
        ap_by_iou[threshold] = ap

    rows = [row for row in sweep if row["class_id"] == class_id and row["predictions"] > 0]
    best = max(rows, key=lambda row: row["f1"], default=None)
    # Precision/recall are reported at the best-F1 operating point, and are UNDEFINED
    # (None) rather than 0.0 when either there is no ground truth for this class or no
    # sweep threshold retained a prediction. Reporting 0.0 would look like a
    # measurement, while AP — which integrates over all predictions, including
    # very-low-confidence ones — can still be non-zero.
    defined = best is not None and n_gt > 0
    return ClassMetrics(
        class_id=class_id,
        class_name=str(options.class_names.get(class_id, f"class_{class_id}")),
        gt_boxes=n_gt,
        predictions=sum(1 for boxes in pred_by_image.values() for box in boxes if box.class_id == class_id),
        ap_by_iou=ap_by_iou,
        precision=float(best["precision"]) if defined else None,
        recall=float(best["recall"]) if defined else None,
        f1=float(best["f1"]) if defined else None,
        best_f1_confidence=float(best["confidence"]) if defined else None,
        mean_iou_matched=float(np.mean(ious_matched)) if ious_matched else 0.0,
    )


def confidence_sweep(
    gt_by_image: Mapping[str, Sequence[DetBox]],
    pred_by_image: Mapping[str, Sequence[DetBox]],
    options: DetectionEvalOptions | None = None,
    *,
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Precision/recall/F1 across confidence thresholds, per class and pooled (-1)."""
    options = options or DetectionEvalOptions()
    images = sorted(set(gt_by_image) | set(pred_by_image))
    class_ids = sorted(
        {box.class_id for boxes in gt_by_image.values() for box in boxes}
        | {box.class_id for boxes in pred_by_image.values() for box in boxes}
    )
    rows: list[dict] = []
    for class_id in [-1, *class_ids]:
        matches: list[tuple[float, bool]] = []
        n_gt = 0
        for image in images:
            gt = [box for box in gt_by_image.get(image, []) if class_id in (-1, box.class_id)]
            pred = [box for box in pred_by_image.get(image, []) if class_id in (-1, box.class_id)]
            n_gt += len(gt)
            if not pred:
                continue
            tp, _fp, _matched = match_image(gt, pred, iou_threshold)
            order = sorted(range(len(pred)), key=lambda index: -pred[index].confidence)
            matches.extend((pred[index].confidence, bool(tp[position])) for position, index in enumerate(order))
        matches.sort(key=lambda pair: -pair[0])
        for confidence in options.sweep_confidences:
            kept = [flag for score, flag in matches if score >= confidence]
            tp_count = sum(1 for flag in kept if flag)
            fp_count = len(kept) - tp_count
            fn_count = max(0, n_gt - tp_count)
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) else 0.0
            recall = tp_count / n_gt if n_gt else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            rows.append(
                {
                    "class_id": class_id,
                    "confidence": confidence,
                    "true_positives": tp_count,
                    "false_positives": fp_count,
                    "false_negatives": fn_count,
                    "predictions": len(kept),
                    "precision": round(precision, 6),
                    "recall": round(recall, 6),
                    "f1": round(f1, 6),
                    "iou_threshold": iou_threshold,
                }
            )
    return rows


def pooled_pr_curve(
    gt_by_image: Mapping[str, Sequence[DetBox]],
    pred_by_image: Mapping[str, Sequence[DetBox]],
    *,
    iou_threshold: float = 0.5,
    points: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    matches: list[tuple[float, bool]] = []
    n_gt = 0
    for image in sorted(set(gt_by_image) | set(pred_by_image)):
        gt = list(gt_by_image.get(image, []))
        pred = list(pred_by_image.get(image, []))
        n_gt += len(gt)
        if not pred:
            continue
        tp, _fp, _matched = match_image(gt, pred, iou_threshold)
        order = sorted(range(len(pred)), key=lambda index: -pred[index].confidence)
        matches.extend((pred[index].confidence, bool(tp[position])) for position, index in enumerate(order))
    matches.sort(key=lambda pair: -pair[0])
    flags = np.asarray([flag for _score, flag in matches], dtype=bool)
    _ap, recall_points, precision_curve = _average_precision(flags, ~flags, n_gt, points=points)
    return recall_points, precision_curve


def sport_of_image_id(image_id: str) -> str:
    """Recover the stratum from ``<sequence>_<frame>.<ext>`` prepared image names."""
    from ..data.splitting import infer_sport

    stem = Path(image_id).stem
    if "_" in stem:
        stem = stem.rsplit("_", 1)[0]
    return infer_sport(stem)


def evaluate_by_group(
    ground_truth: Sequence[DetBox],
    predictions: Sequence[DetBox],
    *,
    group_key: str = "sport",
    options: DetectionEvalOptions | None = None,
) -> dict[str, DetectionMetrics]:
    """Score each subgroup separately from a single set of predictions."""
    options = options or DetectionEvalOptions()
    images = sorted({box.image_id for box in ground_truth} | {box.image_id for box in predictions})
    if group_key == "sport":
        grouper = sport_of_image_id
    elif group_key == "none":
        grouper = lambda _image: "all"  # noqa: E731 - intentional trivial grouper
    else:
        raise ValueError(f"Unsupported group_key {group_key!r}; use 'sport' or 'none'")

    buckets: dict[str, list[str]] = defaultdict(list)
    for image in images:
        buckets[grouper(image)].append(image)

    out = {
        name: evaluate_detections(ground_truth, predictions, group=name, options=options, image_filter=members)
        for name, members in sorted(buckets.items())
    }
    out["overall"] = evaluate_detections(ground_truth, predictions, group="overall", options=options)
    return out


def load_yolo_ground_truth(labels_dir: str | Path, images_dir: str | Path, *, class_id: int = 0) -> list[DetBox]:
    """Read YOLO label files into pixel-space boxes, using real image dimensions."""
    from ..data.validation import IMAGE_SUFFIXES, image_dimensions

    labels_dir, images_dir = Path(labels_dir), Path(images_dir)
    boxes: list[DetBox] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        width, height = image_dimensions(image_path)
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed YOLO label {stripped!r} in {label_path}")
            cls = int(float(parts[0]))
            if cls != class_id:
                continue
            cx, cy, bw, bh = (float(value) for value in parts[1:5])
            boxes.append(
                DetBox(
                    image_id=image_path.name,
                    class_id=cls,
                    xyxy=(
                        (cx - bw / 2) * width,
                        (cy - bh / 2) * height,
                        (cx + bw / 2) * width,
                        (cy + bh / 2) * height,
                    ),
                    confidence=1.0,
                )
            )
    return boxes


def save_detection_reports(
    metrics: Mapping[str, DetectionMetrics], output_dir: str | Path, *, extra: Mapping | None = None
) -> dict[str, Path]:
    """Write ``metrics.json``, ``metrics.csv`` and ``per_sport.csv``."""
    output_dir = ensure_dir(output_dir)
    payload: dict = {"groups": {name: metric.to_dict() for name, metric in sorted(metrics.items())}}
    if extra:
        payload.update(extra)
    paths = {"metrics_json": write_json(output_dir / "metrics.json", payload)}

    rows = []
    for name, metric in sorted(metrics.items()):
        for class_metrics in sorted(metric.classes.values(), key=lambda item: item.class_id):
            row = {"group": name, "images": metric.images, "gt_boxes": metric.gt_boxes}
            row.update(class_metrics.to_dict())
            row.pop("ap_by_iou", None)
            rows.append(row)
    if rows:
        paths["metrics_csv"] = write_csv(output_dir / "metrics.csv", rows)

    paths["per_group_csv"] = write_csv(
        output_dir / "per_sport.csv",
        [
            {
                "group": name,
                "images": metric.images,
                "gt_boxes": metric.gt_boxes,
                "predictions": metric.predictions,
                "map50": _round(metric.map50),
                "map50_95": _round(metric.map50_95),
                "precision": _round(metric.precision),
                "recall": _round(metric.recall),
            }
            for name, metric in sorted(metrics.items())
        ],
    )
    return paths


def cross_check_detection_metrics(
    ours: DetectionMetrics, reference: Mapping[str, float], *, tolerance: float = 0.01
) -> dict:
    """Compare our mAP against a reference implementation on identical inputs.

    Precision/recall are deliberately not compared: ours are reported at the best-F1
    confidence while Ultralytics reports its own operating point, so comparing them
    would manufacture a disagreement.
    """
    comparisons = []
    agree = True
    for name, our_value in (("map50", ours.map50), ("map50_95", ours.map50_95)):
        reference_value = reference.get(name)
        if our_value is None or reference_value is None or np.isnan(our_value):
            comparisons.append({"metric": name, "ours": our_value, "reference": reference_value, "status": "missing"})
            continue
        delta = float(our_value) - float(reference_value)
        within = abs(delta) <= tolerance
        agree = agree and within
        comparisons.append(
            {
                "metric": name,
                "ours": round(float(our_value), 6),
                "reference": round(float(reference_value), 6),
                "delta": round(delta, 6),
                "within_tolerance": within,
                "tolerance": tolerance,
            }
        )
    return {
        "status": "agree" if agree else "disagree",
        "comparisons": comparisons,
        "note": (
            "This compares two END-TO-END pipelines, not two evaluators on identical input: ours "
            "scores the predictions from our own predict() call while the reference re-runs "
            "inference inside val() with its own batching, rect settings and max_det. A small delta "
            "is therefore expected and is not by itself evidence of an evaluator bug. To compare "
            "evaluators cleanly, score one shared prediction set with both (e.g. val(save_json=True) "
            "COCO output). Precision/recall is likewise not compared, since the two use different "
            "operating points."
        ),
    }


def format_metrics_table(metrics: Mapping[str, DetectionMetrics]) -> str:
    lines = [
        "| Group | Images | GT boxes | Predictions | mAP50 | mAP50-95 | Precision | Recall |",
        "| --- | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for name, metric in sorted(metrics.items()):
        lines.append(
            f"| {name} | {metric.images} | {metric.gt_boxes} | {metric.predictions} | "
            f"{_fmt(metric.map50)} | {_fmt(metric.map50_95)} | {_fmt(metric.precision)} | {_fmt(metric.recall)} |"
        )
    return "\n".join(lines)


def _fmt(value) -> str:
    return "—" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.4f}"
