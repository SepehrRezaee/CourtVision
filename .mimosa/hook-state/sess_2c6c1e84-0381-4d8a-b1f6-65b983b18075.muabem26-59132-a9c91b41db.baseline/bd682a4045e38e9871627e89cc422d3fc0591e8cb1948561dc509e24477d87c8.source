from __future__ import annotations

from pathlib import Path

import motmetrics as mm
import pandas as pd
from ultralytics import YOLO

MOT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]


def evaluate_detector(weights: str, data: str, imgsz: int = 1280, device: str = "cpu") -> dict:
    result = YOLO(weights).val(data=data, imgsz=imgsz, device=device, verbose=False)
    return {
        "map50_95": float(result.box.map),
        "map50": float(result.box.map50),
        "map75": float(result.box.map75),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
    }


def export_mot_predictions(weights: str, video: str, output: str | Path, tracker: str = "bytetrack.yaml", conf: float = 0.25, imgsz: int = 1280, device: str = "cpu") -> Path:
    model = YOLO(weights)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    stream = model.track(source=video, tracker=tracker, conf=conf, imgsz=imgsz, device=device, stream=True, persist=True, verbose=False)
    for frame_idx, result in enumerate(stream, start=1):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        for box, track_id, score in zip(boxes.xywh.cpu().numpy(), boxes.id.int().cpu().numpy(), boxes.conf.cpu().numpy()):
            cx, cy, w, h = [float(v) for v in box]
            lines.append(f"{frame_idx},{int(track_id)},{cx-w/2:.3f},{cy-h/2:.3f},{w:.3f},{h:.3f},{float(score):.5f},1,1")
    output.write_text("\n".join(lines) + ("\n" if lines else ""))
    return output


def _load_mot(path: str | Path) -> pd.DataFrame:
    rows = []
    for raw in Path(path).read_text().splitlines():
        if raw.strip():
            values = [float(x.strip()) for x in raw.split(",")]
            rows.append((values + [1.0] * 9)[:9])
    return pd.DataFrame(rows, columns=MOT_COLUMNS)


def evaluate_mot(ground_truth: str | Path, prediction: str | Path, iou_threshold: float = 0.5) -> dict:
    gt = _load_mot(ground_truth)
    pred = _load_mot(prediction)
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt["frame"].astype(int)) | set(pred["frame"].astype(int)))
    for frame_id in frames:
        g = gt[gt["frame"].astype(int) == frame_id]
        p = pred[pred["frame"].astype(int) == frame_id]
        distances = mm.distances.iou_matrix(
            g[["x", "y", "w", "h"]].to_numpy(dtype=float),
            p[["x", "y", "w", "h"]].to_numpy(dtype=float),
            max_iou=1 - iou_threshold,
        )
        acc.update(g["id"].astype(int).tolist(), p["id"].astype(int).tolist(), distances)
    metrics = ["mota", "motp", "idf1", "precision", "recall", "num_switches", "mostly_tracked", "mostly_lost"]
    summary = mm.metrics.create().compute(acc, metrics=metrics, name="courtvision")
    return {name: float(summary.loc["courtvision", name]) for name in metrics}
