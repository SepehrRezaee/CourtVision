"""Multi-object tracking evaluation.

``motmetrics`` provides CLEAR-MOT and identity metrics; TrackEval (the official HOTA
implementation) provides HOTA when installed.

Two conventions are pinned by tests because both are easy to get wrong:

* The distance matrix is ``1 - IoU`` with ``NaN`` marking pairs below the threshold.
  It is built here (:func:`pairwise_iou_distances`) rather than by
  ``motmetrics.distances.iou_matrix``, which calls ``np.asfarray`` — removed in NumPy
  2.0 — so the library helper raises on any modern NumPy.
* ``motp`` is the mean IoU **distance** of matched pairs: lower is better, and it is
  not a percentage. TrackEval's ``MOTP`` is the opposite convention (a similarity
  where 100 is perfect), so TrackEval's MOTP is deliberately not reported.
"""

from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stdout, suppress
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from ..data.mot import Box, load_mot_file
from ..utils import ensure_dir, get_logger, write_csv, write_json

DESIRED_MOT_METRICS: tuple[str, ...] = (
    "num_frames",
    "mota",
    "motp",
    "idf1",
    "idp",
    "idr",
    "precision",
    "recall",
    "num_unique_objects",
    "num_objects",
    "num_predictions",
    "num_matches",
    "num_detections",
    "num_misses",
    "num_false_positives",
    "num_switches",
    "num_fragmentations",
    "num_transfer",
    "num_ascend",
    "num_migrate",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
)

REQUIRED_MOT_METRICS: tuple[str, ...] = ("mota", "motp", "idf1", "num_switches", "num_frames")


@dataclass(frozen=True)
class MOTEvalOptions:
    """Filtering and matching policy. Recorded in every report, since it changes results."""

    iou_threshold: float = 0.5
    min_confidence: float = 0.0
    allowed_class_ids: tuple[int, ...] | None = (1,)
    min_visibility: float = 0.0
    drop_zero_confidence: bool = True
    frame_range: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "iou_threshold": self.iou_threshold,
            "min_confidence": self.min_confidence,
            "allowed_class_ids": list(self.allowed_class_ids) if self.allowed_class_ids else None,
            "min_visibility": self.min_visibility,
            "drop_zero_confidence": self.drop_zero_confidence,
            "frame_range": list(self.frame_range) if self.frame_range else None,
        }


def available_mot_metrics() -> tuple[list[str], list[str]]:
    """Return ``(available, missing)`` for :data:`DESIRED_MOT_METRICS`."""
    import motmetrics as mm

    known = set(mm.metrics.create().metrics.keys())
    return (
        [name for name in DESIRED_MOT_METRICS if name in known],
        [name for name in DESIRED_MOT_METRICS if name not in known],
    )


def pairwise_iou_distances(gt: Sequence[Box], pred: Sequence[Box], iou_threshold: float) -> np.ndarray:
    """IoU distance matrix: ``1 - IoU``, with ``NaN`` where IoU < threshold.

    ``NaN`` is motmetrics' marker for "do not pair".
    """
    from .detection import iou_matrix

    if not gt or not pred:
        return np.empty((len(gt), len(pred)), dtype=float)
    ious = iou_matrix(
        np.array([box.xyxy for box in gt], dtype=float).reshape(-1, 4),
        np.array([box.xyxy for box in pred], dtype=float).reshape(-1, 4),
    )
    return np.where(ious < iou_threshold, np.nan, 1.0 - ious)


def filter_ground_truth(boxes: Sequence[Box], options: MOTEvalOptions) -> list[Box]:
    out: list[Box] = []
    for box in boxes:
        if options.drop_zero_confidence and box.confidence <= 0.0:
            continue
        if not 0.0 < box.confidence <= 1.0:
            continue
        if options.allowed_class_ids is not None and box.class_id not in options.allowed_class_ids:
            continue
        if box.visibility < options.min_visibility:
            continue
        if box.is_degenerate:
            continue
        if options.frame_range and not (options.frame_range[0] <= box.frame <= options.frame_range[1]):
            continue
        out.append(box)
    return out


def filter_predictions(boxes: Sequence[Box], options: MOTEvalOptions) -> list[Box]:
    out: list[Box] = []
    for box in boxes:
        if box.confidence < options.min_confidence or box.is_degenerate:
            continue
        if options.frame_range and not (options.frame_range[0] <= box.frame <= options.frame_range[1]):
            continue
        out.append(box)
    return out


def boxes_by_frame(boxes: Sequence[Box]) -> dict[int, list[Box]]:
    grouped: dict[int, list[Box]] = defaultdict(list)
    for box in boxes:
        grouped[box.frame].append(box)
    return dict(grouped)


@dataclass
class SequenceMOTResult:
    sequence: str
    frames: int
    ground_truth_objects: int
    ground_truth_tracks: int
    predictions: int
    predicted_tracks: int
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "sequence": self.sequence,
            "frames": self.frames,
            "ground_truth_objects": self.ground_truth_objects,
            "ground_truth_tracks": self.ground_truth_tracks,
            "predictions": self.predictions,
            "predicted_tracks": self.predicted_tracks,
        }
        payload.update({key: _round(value) for key, value in sorted(self.metrics.items())})
        return payload


@dataclass
class MOTResult:
    dataset: str
    options: dict
    sequences: dict[str, SequenceMOTResult] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    derived: dict[str, float] = field(default_factory=dict)
    hota: dict[str, float] | None = None
    hota_note: str | None = None
    unavailable_metrics: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    @property
    def mota(self) -> float | None:
        return self.metrics.get("mota")

    @property
    def idf1(self) -> float | None:
        return self.metrics.get("idf1")

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "options": self.options,
            "aggregate": {key: _round(value) for key, value in sorted(self.metrics.items())},
            "derived": {key: _round(value) for key, value in sorted(self.derived.items())},
            "hota": self.hota,
            "hota_note": self.hota_note,
            "unavailable_metrics": self.unavailable_metrics,
            "warnings": self.warnings,
            "events_total": len(self.events),
            "events_by_type": dict(Counter(event["type"] for event in self.events)),
            "sequences": {name: item.to_dict() for name, item in sorted(self.sequences.items())},
            "totals": {
                "sequences": len(self.sequences),
                "frames": sum(item.frames for item in self.sequences.values()),
                "ground_truth_objects": sum(item.ground_truth_objects for item in self.sequences.values()),
                "predictions": sum(item.predictions for item in self.sequences.values()),
            },
        }

    def save(self, path: str | Path) -> Path:
        payload = self.to_dict()
        payload["events"] = self.events[:5000]
        return write_json(path, payload)

    def to_rows(self) -> list[dict]:
        rows = []
        for name, sequence in sorted(self.sequences.items()):
            row = {
                "sequence": name,
                "frames": sequence.frames,
                "ground_truth_objects": sequence.ground_truth_objects,
                "predictions": sequence.predictions,
            }
            row.update({key: _round(value) for key, value in sorted(sequence.metrics.items())})
            rows.append(row)
        if rows:
            aggregate = {"sequence": "AGGREGATE"}
            aggregate.update({key: _round(value) for key, value in sorted(self.metrics.items())})
            rows.append(aggregate)
        return rows


def _round(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    return None if np.isnan(as_float) else round(as_float, digits)


def _accumulate(
    gt_by_frame: Mapping[int, Sequence[Box]],
    pred_by_frame: Mapping[int, Sequence[Box]],
    options: MOTEvalOptions,
    accumulator,
) -> None:
    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt = gt_by_frame.get(frame, [])
        pred = pred_by_frame.get(frame, [])
        accumulator.update(
            [box.track_id for box in gt],
            [box.track_id for box in pred],
            pairwise_iou_distances(gt, pred, options.iou_threshold),
        )


def _metrics_from_accumulator(accumulator, name: str) -> tuple[dict[str, float], list[str]]:
    import motmetrics as mm

    available, missing = available_mot_metrics()
    summary = mm.metrics.create().compute(accumulator, metrics=available, name=name)
    return {metric: float(summary.loc[name, metric]) for metric in available}, missing


def _derive(metrics: Mapping[str, float]) -> dict[str, float]:
    """Direction-aware normalisations that make runs of different length comparable."""
    objects = metrics.get("num_objects", 0.0) or 0.0
    frames = metrics.get("num_frames", 0.0) or 0.0
    unique = metrics.get("num_unique_objects", 0.0) or 0.0
    derived: dict[str, float] = {}
    if objects > 0:
        derived["id_switches_per_object"] = metrics.get("num_switches", 0.0) / objects
        derived["false_positives_per_object"] = metrics.get("num_false_positives", 0.0) / objects
        derived["misses_per_object"] = metrics.get("num_misses", 0.0) / objects
        derived["fragmentations_per_object"] = metrics.get("num_fragmentations", 0.0) / objects
    if unique > 0:
        derived["mt_ratio"] = metrics.get("mostly_tracked", 0.0) / unique
        derived["ml_ratio"] = metrics.get("mostly_lost", 0.0) / unique
        derived["pt_ratio"] = metrics.get("partially_tracked", 0.0) / unique
    if frames > 0:
        derived["objects_per_frame"] = objects / frames
        derived["predictions_per_frame"] = metrics.get("num_predictions", 0.0) / frames
        derived["id_switches_per_frame"] = metrics.get("num_switches", 0.0) / frames
    return derived


def _extract_events(accumulator) -> list[dict]:
    """Per-frame identity events, so "num_switches: 412" becomes inspectable frames."""
    frame_events = getattr(accumulator, "events", None)
    if frame_events is None:
        return []
    try:
        records = frame_events.reset_index().to_dict("records")
    except (AttributeError, TypeError):  # pragma: no cover - library variation
        return []
    out: list[dict] = []
    for record in records:
        event_type = str(record.get("Type", "")).upper()
        if event_type in {"", "RAW"}:
            continue
        out.append(
            {
                "type": event_type,
                "frame": int(record.get("FrameId", 0)),
                "ground_truth_id": _maybe_int(record.get("OId")),
                "predicted_id": _maybe_int(record.get("HId")),
            }
        )
    return out


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(as_float) else int(as_float)


def _as_sequence_map(source: Mapping[str, Path] | Path, label: str) -> dict[str, Path]:
    if isinstance(source, Mapping):
        return {str(name): Path(path) for name, path in source.items()}
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"MOT {label} file not found: {path}")
    return {path.parent.parent.name or path.stem: path}


def evaluate_mot(
    ground_truth: Mapping[str, Path] | Path,
    predictions: Mapping[str, Path] | Path,
    *,
    options: MOTEvalOptions | None = None,
    dataset_name: str = "courtvision",
    with_hota: bool = False,
    hota_tracker_name: str = "courtvision",
) -> MOTResult:
    """Evaluate MOT predictions against ground truth, optionally including HOTA."""
    options = options or MOTEvalOptions()
    gt_map = _as_sequence_map(ground_truth, "ground truth")
    pred_map = _as_sequence_map(predictions, "predictions")

    # Two bare file paths are treated as the same sequence; explicit mappings with
    # mismatched names are a caller error and must not be silently aligned.
    if (
        isinstance(ground_truth, Path)
        and isinstance(predictions, Path)
        and len(gt_map) == 1
        and len(pred_map) == 1
        and set(gt_map) != set(pred_map)
    ):
        pred_map = {next(iter(gt_map)): next(iter(pred_map.values()))}

    result = MOTResult(dataset=dataset_name, options=options.to_dict())
    missing = sorted(set(gt_map) - set(pred_map))
    if missing:
        result.warnings.append(
            f"No prediction file for sequence(s) {missing}; excluded from the aggregate."
        )

    import motmetrics as mm

    aggregate = mm.MOTAccumulator(auto_id=True)
    for name in sorted(set(gt_map) & set(pred_map)):
        gt_boxes = filter_ground_truth(load_mot_file(gt_map[name]), options)
        pred_boxes = filter_predictions(load_mot_file(pred_map[name]), options)
        gt_frames, pred_frames = boxes_by_frame(gt_boxes), boxes_by_frame(pred_boxes)

        sequence_acc = mm.MOTAccumulator(auto_id=True)
        _accumulate(gt_frames, pred_frames, options, sequence_acc)
        _accumulate(gt_frames, pred_frames, options, aggregate)

        metrics, missing_metrics = _metrics_from_accumulator(sequence_acc, name)
        result.sequences[name] = SequenceMOTResult(
            sequence=name,
            frames=len(set(gt_frames) | set(pred_frames)),
            ground_truth_objects=len(gt_boxes),
            ground_truth_tracks=len({box.track_id for box in gt_boxes}),
            predictions=len(pred_boxes),
            predicted_tracks=len({box.track_id for box in pred_boxes}),
            metrics=metrics,
        )
        if missing_metrics and not result.unavailable_metrics:
            result.unavailable_metrics = missing_metrics
        if not gt_boxes:
            result.warnings.append(f"Sequence {name} has no ground-truth boxes after filtering.")

    if not result.sequences:
        raise ValueError(
            "Nothing to evaluate: no sequence has both ground truth and predictions "
            f"(gt={sorted(gt_map)}, pred={sorted(pred_map)})"
        )

    result.metrics, result.unavailable_metrics = _metrics_from_accumulator(aggregate, "AGGREGATE")
    for required in REQUIRED_MOT_METRICS:
        if required not in result.metrics:
            raise RuntimeError(
                f"motmetrics did not expose {required!r}; available: {sorted(result.metrics)}"
            )
    result.derived = _derive(result.metrics)
    result.events = _extract_events(aggregate)

    if with_hota:
        hota, note = _try_hota(gt_map, pred_map, options, hota_tracker_name)
        result.hota, result.hota_note = hota, note
    return result


@contextmanager
def numpy_legacy_aliases() -> Iterator[list[str]]:
    """Restore the NumPy-1 scalar aliases TrackEval still references.

    TrackEval uses ``np.float``/``np.int``/``np.bool``, removed in NumPy 2. Rather than
    fork the reference implementation or pin NumPy below 2 — which would break
    Ultralytics 8.4 and OpenCV 5 — the aliases are set for the duration of the call.
    ``np.float`` *is* ``float``, so this is a compatibility shim, not a change to the
    metric. It mutates module state, so HOTA evaluation should not run concurrently
    with other threads.
    """
    import numpy as np

    aliases = {"float": float, "int": int, "bool": bool}
    installed = [name for name in aliases if not hasattr(np, name)]
    try:
        for name in installed:
            setattr(np, name, aliases[name])
        yield installed
    finally:
        for name in installed:
            with suppress(AttributeError):  # pragma: no cover
                delattr(np, name)


def _try_hota(
    gt_map: Mapping[str, Path], pred_map: Mapping[str, Path], options: MOTEvalOptions, tracker_name: str
) -> tuple[dict[str, float] | None, str | None]:
    """Run TrackEval's HOTA, or explain precisely why it could not run."""
    try:
        return evaluate_hota(gt_map, pred_map, options=options, tracker_name=tracker_name), None
    except ModuleNotFoundError:
        return None, (
            "HOTA NOT MEASURED: the optional 'trackeval' package is not installed. "
            'Install with: pip install -e ".[eval]"'
        )
    except Exception as exc:
        return None, f"HOTA NOT MEASURED: TrackEval failed with {type(exc).__name__}: {exc}"


def evaluate_hota(
    gt_map: Mapping[str, Path],
    pred_map: Mapping[str, Path],
    *,
    options: MOTEvalOptions | None = None,
    tracker_name: str = "courtvision",
    workspace: str | Path | None = None,
) -> dict[str, float]:
    """Compute HOTA/DetA/AssA with TrackEval's reference implementation.

    TrackEval expects a MOTChallenge layout, so a throwaway workspace is materialised
    from the caller's ground truth and predictions.
    """
    from trackeval import Evaluator
    from trackeval.datasets import MotChallenge2DBox
    from trackeval.metrics import CLEAR, HOTA, Identity

    options = options or MOTEvalOptions()
    sequences = sorted(set(gt_map) & set(pred_map))
    if not sequences:
        raise ValueError("HOTA needs at least one sequence in both ground truth and predictions")

    temporary = None
    if workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="courtvision-hota-")
        workspace = temporary.name
    workspace = ensure_dir(workspace)
    gt_root = ensure_dir(Path(workspace) / "gt")
    trackers_root = ensure_dir(Path(workspace) / "trackers")

    seq_info: dict[str, int] = {}
    try:
        for sequence in sequences:
            boxes = filter_ground_truth(load_mot_file(gt_map[sequence]), options)
            _write_mot(boxes, ensure_dir(gt_root / sequence / "gt") / "gt.txt")
            length = max((box.frame for box in boxes), default=1)
            seq_info[sequence] = length
            _write_seqinfo(gt_root / sequence / "seqinfo.ini", sequence, length)
            _write_mot(
                filter_predictions(load_mot_file(pred_map[sequence]), options),
                ensure_dir(trackers_root / tracker_name / "data") / f"{sequence}.txt",
            )

        with numpy_legacy_aliases() as shimmed:
            dataset = MotChallenge2DBox(
                {
                    "GT_FOLDER": str(gt_root),
                    "TRACKERS_FOLDER": str(trackers_root),
                    "TRACKERS_TO_EVAL": [tracker_name],
                    "CLASSES_TO_EVAL": ["pedestrian"],
                    "BENCHMARK": "MOT20",
                    "SPLIT_TO_EVAL": "train",
                    # The workspace already has <gt>/<seq> and <trackers>/<name>/data.
                    "SKIP_SPLIT_FOL": True,
                    "SEQ_INFO": seq_info,
                    "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
                    "OUTPUT_FOLDER": None,
                    "PRINT_CONFIG": False,
                    "DO_PREPROC": False,
                }
            )
            evaluator = Evaluator(
                {
                    "PRINT_CONFIG": False,
                    "USE_PARALLEL": False,
                    "NUM_PARALLEL_CORES": 1,
                    "TIME_PROGRESS": False,
                    "OUTPUT_SUMMARY": False,
                    "OUTPUT_DETAILED": False,
                    "PLOT_CURVES": False,
                }
            )
            # TrackEval prints tables to stdout; capture them so CLI output stays parseable.
            with redirect_stdout(StringIO()):
                raw, _messages = evaluator.evaluate(
                    [dataset],
                    [CLEAR({"PRINT_CONFIG": False}), HOTA({"PRINT_CONFIG": False}), Identity({"PRINT_CONFIG": False})],
                    show_progressbar=False,
                )
    finally:
        if temporary is not None:
            temporary.cleanup()

    if shimmed:
        import numpy as np

        get_logger("evaluation.tracking").info(
            "TrackEval needed the NumPy-1 alias shim (%s) on NumPy %s",
            ", ".join(f"np.{name}" for name in shimmed),
            np.__version__,
        )
    return _extract_hota(raw, tracker_name)


def _extract_hota(raw: Mapping, tracker_name: str) -> dict[str, float]:
    """Pull combined-sequence HOTA/CLEAR/Identity values out of TrackEval output.

    TrackEval nests results as ``raw[dataset][tracker][sequence][class][metric][field]``
    with the aggregate under ``"COMBINED_SEQ"``, and reports HOTA-family fields as
    arrays over the 19 alpha thresholds — the headline figure is their mean.
    """
    dataset_name = next(iter(raw))
    per_sequence = raw[dataset_name][tracker_name]
    combined = per_sequence.get("COMBINED_SEQ") or next(iter(per_sequence.values()))
    class_block = combined.get("pedestrian") or next(iter(combined.values()))

    out: dict[str, float] = {}
    for metric_name, wanted in {
        "HOTA": ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA"),
        "CLEAR": ("MOTA", "CLR_Re", "CLR_Pr", "IDSW", "FP", "FN", "Frag", "MT", "PT", "ML"),
        "Identity": ("IDF1", "IDR", "IDP"),
    }.items():
        block = class_block.get(metric_name)
        if not isinstance(block, Mapping):
            continue
        for field_name in wanted:
            if field_name not in block:
                continue
            value = _aggregate_field(block[field_name])
            if value is None:
                continue
            out[field_name if metric_name == "HOTA" else f"{metric_name}_{field_name}"] = round(value, 6)

    hota_block = class_block.get("HOTA")
    if isinstance(hota_block, Mapping):
        for scalar_name in ("HOTA(0)", "LocA(0)", "HOTALocA(0)"):
            if scalar_name in hota_block:
                value = _aggregate_field(hota_block[scalar_name])
                if value is not None:
                    out[scalar_name.replace("(", "_at_").replace(")", "")] = round(value, 6)
    if not out:
        raise KeyError("TrackEval returned no HOTA/CLEAR/Identity fields; its output layout changed")
    return out


def _aggregate_field(value: Any) -> float | None:
    """Reduce a TrackEval field to one number, averaging arrays over thresholds."""
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    if array.ndim > 0 and array.size > 1:
        if not np.issubdtype(array.dtype, np.number):
            return None
        return float(array.astype(float).mean())
    try:
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def _write_mot(boxes: Sequence[Box], path: Path) -> None:
    ensure_dir(path.parent)
    lines = [
        f"{box.frame},{box.track_id},{box.x:.3f},{box.y:.3f},{box.w:.3f},{box.h:.3f},"
        f"{box.confidence if box.confidence > 0 else 1.0:.5f},-1,-1,-1"
        for box in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_seqinfo(path: Path, name: str, length: int, width: int = 1920, height: int = 1080) -> None:
    ensure_dir(path.parent)
    path.write_text(
        "[Sequence]\n"
        f"name={name}\n"
        "imDir=img1\n"
        "frameRate=25\n"
        f"seqLength={length}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n",
        encoding="utf-8",
    )


def compare_trackers(rows: Sequence[Mapping[str, Any]]) -> str:
    """Markdown comparison table for the tracker benchmark report."""
    columns = (
        "tracker",
        "detector",
        "imgsz",
        "confidence",
        "MOTA",
        "HOTA",
        "IDF1",
        "ID_switches",
        "FPS",
        "p95_latency_ms",
    )
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append("—" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_tracker_comparison(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    return write_csv(path, [dict(row) for row in rows])
