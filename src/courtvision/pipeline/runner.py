"""Pipeline orchestration: one place that runs an experiment end to end.

Each function writes its artefacts to a directory and returns them, so a report is
assembled from files that exist rather than from values held in memory.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..evaluation.detection import (
    DEFAULT_IOU_THRESHOLDS,
    DetBox,
    DetectionEvalOptions,
    DetectionMetrics,
    cross_check_detection_metrics,
    evaluate_by_group,
    evaluate_detections,
    load_yolo_ground_truth,
    save_detection_reports,
)
from ..evaluation.tracking import MOTEvalOptions, MOTResult, evaluate_mot
from ..repro import environment_snapshot
from ..utils import ensure_dir, get_logger, write_csv, write_json

LOGGER = get_logger("pipeline.runner")


@dataclass
class DetectionEvaluation:
    """Our metrics, the reference metrics and the agreement between them."""

    metrics: dict[str, DetectionMetrics]
    predictions: list[DetBox]
    files: dict[str, Path] = field(default_factory=dict)
    reference: dict | None = None
    cross_check: dict | None = None
    ground_truth_boxes: int = 0
    prediction_meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "groups": {name: metric.to_dict() for name, metric in sorted(self.metrics.items())},
            "reference_ultralytics": self.reference,
            "cross_check": self.cross_check,
            "ground_truth_boxes": self.ground_truth_boxes,
            "predictor": self.prediction_meta,
        }


def predict_dataset(
    weights: str,
    images_dir: str | Path,
    *,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "cpu",
    max_images: int | None = None,
    batch: int = 8,
) -> tuple[list[DetBox], dict]:
    """Run a detector over an image directory and return per-image predictions.

    Predictions are materialised once so every downstream analysis (overall, per-group,
    sweeps, failure cases) scores identical inferences.
    """
    from ultralytics import YOLO

    from ..detection.ultralytics_detector import ultralytics_version
    from ..repro import resolve_device

    images_dir = Path(images_dir)
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    device_resolved = resolve_device(device)
    model = YOLO(weights)
    boxes: list[DetBox] = []
    timings: list[float] = []
    started = time.perf_counter()
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device_resolved,
        batch=batch,
        stream=True,
        verbose=False,
    )
    for image_path, result in zip(image_paths, results, strict=True):
        speed = getattr(result, "speed", None) or {}
        if speed.get("inference"):
            timings.append(float(speed["inference"]))
        detection_boxes = getattr(result, "boxes", None)
        if detection_boxes is None or len(detection_boxes) == 0:
            continue
        xyxy = detection_boxes.xyxy.cpu().numpy()
        confidences = detection_boxes.conf.cpu().numpy()
        classes = detection_boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), score, class_id in zip(xyxy, confidences, classes, strict=True):
            boxes.append(
                DetBox(
                    image_id=image_path.name,
                    class_id=int(class_id),
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(score),
                )
            )
    elapsed = time.perf_counter() - started
    meta = {
        "weights": weights,
        "images": len(image_paths),
        "images_with_detections": len({box.image_id for box in boxes}),
        "detections": len(boxes),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "device": device_resolved,
        "ultralytics_version": ultralytics_version(),
        "wall_seconds": round(elapsed, 4),
        "inference_ms_mean": round(sum(timings) / len(timings), 4) if timings else None,
        "throughput_images_per_second": round(len(image_paths) / elapsed, 4) if elapsed > 0 else None,
        "predictions_shared_across_analyses": True,
    }
    return boxes, meta


def evaluate_detector_on_dataset(
    weights: str,
    prepared_root: str | Path,
    *,
    split: str = "val",
    output_dir: str | Path | None = None,
    conf: float = 0.001,
    imgsz: int = 640,
    device: str = "cpu",
    max_images: int | None = None,
    cross_check_with_ultralytics: bool = True,
    class_names: Mapping[int, str] | None = None,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> DetectionEvaluation:
    """Evaluate a detector on a prepared YOLO split, with a reference cross-check.

    Two independent evaluations are produced deliberately: ours (which supports
    per-group breakdowns and sweeps) and Ultralytics' ``val()``. Agreement is reported;
    a disagreement means one of the two is wrong and must be investigated.
    """
    prepared_root = Path(prepared_root)
    images_dir = prepared_root / "images" / split
    labels_dir = prepared_root / "labels" / split
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Prepared split not found: {images_dir}. Run `courtvision data prepare` first.")

    ground_truth = load_yolo_ground_truth(labels_dir, images_dir)
    predictions, prediction_meta = predict_dataset(
        weights, images_dir, conf=conf, imgsz=imgsz, device=device, max_images=max_images
    )
    options = DetectionEvalOptions(
        iou_thresholds=tuple(iou_thresholds), class_names=dict(class_names or {0: "player"}), min_confidence=0.0
    )
    by_group = evaluate_by_group(ground_truth, predictions, group_key="sport", options=options)
    by_group["overall"] = evaluate_detections(ground_truth, predictions, group="overall", options=options)

    evaluation = DetectionEvaluation(
        metrics=by_group,
        predictions=predictions,
        ground_truth_boxes=len(ground_truth),
        prediction_meta=prediction_meta,
    )

    if cross_check_with_ultralytics:
        try:
            from ..detection.ultralytics_detector import UltralyticsDetector

            validation = UltralyticsDetector(weights=weights, device=device, imgsz=imgsz).validate(
                prepared_root / "dataset.yaml", split=split
            )
            evaluation.reference = validation.to_dict()
            evaluation.cross_check = cross_check_detection_metrics(by_group["overall"], validation.metrics)
        except Exception as exc:
            evaluation.cross_check = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}

    if output_dir is not None:
        directory = ensure_dir(output_dir)
        evaluation.files = save_detection_reports(
            by_group,
            directory,
            extra={
                "detector": prediction_meta,
                "ground_truth_boxes": len(ground_truth),
                "reference_ultralytics": evaluation.reference,
                "cross_check": evaluation.cross_check,
                "evaluation_notes": [
                    "Predictions were computed once and re-scored per group; all groups share "
                    "identical inferences.",
                    "mAP uses 101-point COCO interpolation and confidence-ordered greedy matching, "
                    "matching the convention used by Ultralytics and pycocotools.",
                ],
            },
        )
        evaluation.files["environment_json"] = write_json(directory / "environment.json", environment_snapshot())
    return evaluation


@dataclass
class TrackingEvaluation:
    result: MOTResult
    files: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.result.to_dict()


def evaluate_tracking(
    ground_truth: Mapping[str, Path] | Path,
    predictions: Mapping[str, Path] | Path,
    *,
    output_dir: str | Path | None = None,
    iou_threshold: float = 0.5,
    min_confidence: float = 0.0,
    allowed_class_ids: Sequence[int] | None = (1,),
    min_visibility: float = 0.0,
    with_hota: bool = True,
    dataset_name: str = "courtvision",
) -> TrackingEvaluation:
    """Evaluate MOT predictions, writing ``metrics.json`` and ``metrics.csv``."""
    options = MOTEvalOptions(
        iou_threshold=iou_threshold,
        min_confidence=min_confidence,
        allowed_class_ids=tuple(allowed_class_ids) if allowed_class_ids else None,
        min_visibility=min_visibility,
    )
    result = evaluate_mot(ground_truth, predictions, options=options, dataset_name=dataset_name, with_hota=with_hota)
    evaluation = TrackingEvaluation(result=result)
    if output_dir is not None:
        directory = ensure_dir(output_dir)
        evaluation.files["metrics_json"] = result.save(directory / "metrics.json")
        evaluation.files["metrics_csv"] = write_csv(directory / "metrics.csv", result.to_rows())
        evaluation.files["environment_json"] = write_json(directory / "environment.json", environment_snapshot())
    return evaluation


def ground_truth_paths_from_prepared(raw_root: str | Path, *, split: str | None = None) -> dict[str, Path]:
    """Map sequence name -> ``gt.txt`` for a raw MOT tree (optionally one split)."""
    root = Path(raw_root)
    search_roots = [root / split] if split else [root / name for name in ("train", "val", "test")]
    search_roots = [candidate for candidate in search_roots if candidate.is_dir()] or [root]
    found: dict[str, Path] = {}
    for search_root in search_roots:
        for sequence in sorted(search_root.iterdir()):
            gt = sequence / "gt" / "gt.txt"
            if sequence.is_dir() and gt.exists():
                found[sequence.name] = gt
    return found


def write_experiment_metadata(
    output_dir: str | Path,
    *,
    name: str,
    parameters: Mapping[str, Any],
    dataset: Mapping[str, Any] | None = None,
    seed: int | None = None,
    command: str | None = None,
) -> Path:
    """Write the provenance block every experiment directory should carry."""
    return write_json(
        ensure_dir(output_dir) / "experiment.json",
        {
            "experiment": name,
            "parameters": dict(parameters),
            "dataset": dict(dataset or {}),
            "environment": environment_snapshot(seed=seed, command=command),
        },
    )


def log_environment_summary() -> dict:
    """Log a one-line environment summary, used at the start of long runs."""
    snapshot = environment_snapshot()
    hardware = snapshot["hardware"]
    LOGGER.info(
        "environment",
        extra={
            "python": snapshot["python_version"],
            "torch": snapshot["packages"].get("torch"),
            "ultralytics": snapshot["packages"].get("ultralytics"),
            "cuda": hardware.get("cuda_available"),
            "gpu": hardware.get("gpu"),
            "cpu_count": hardware.get("cpu_count"),
        },
    )
    return snapshot
