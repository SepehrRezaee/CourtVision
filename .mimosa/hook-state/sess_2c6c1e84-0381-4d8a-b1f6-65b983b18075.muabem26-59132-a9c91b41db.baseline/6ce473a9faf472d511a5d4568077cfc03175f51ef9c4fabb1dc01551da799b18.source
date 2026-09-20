from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO


@dataclass
class AnalysisSummary:
    frames: int
    detections: int
    unique_tracks: int
    mean_confidence: float
    elapsed_seconds: float
    fps: float
    per_frame: list[dict]


class VideoAnalyzer:
    def __init__(self, weights: str = "yolo26s.pt", device: str = "cpu") -> None:
        self.weights = weights
        self.device = device
        self.model = YOLO(weights)

    def analyze(self, video: str | Path, tracker: str = "tracktrack.yaml", conf: float = 0.25, imgsz: int = 960) -> AnalysisSummary:
        started = time.perf_counter()
        stream = self.model.track(
            source=str(video),
            tracker=tracker,
            conf=conf,
            imgsz=imgsz,
            device=self.device,
            stream=True,
            persist=True,
            verbose=False,
        )
        frames = detections = 0
        confidences: list[float] = []
        track_ids: set[int] = set()
        per_frame: list[dict] = []

        for frame_idx, result in enumerate(stream, start=1):
            frames += 1
            frame_tracks = []
            boxes = result.boxes
            if boxes is not None and len(boxes):
                detections += len(boxes)
                xyxy = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                for coords, score, track_id in zip(xyxy, confs, ids):
                    confidences.append(float(score))
                    if track_id is not None:
                        track_ids.add(int(track_id))
                    frame_tracks.append(
                        {
                            "track_id": int(track_id) if track_id is not None else None,
                            "confidence": round(float(score), 5),
                            "xyxy": [round(float(v), 2) for v in coords],
                        }
                    )
            per_frame.append({"frame": frame_idx, "tracks": frame_tracks})

        elapsed = time.perf_counter() - started
        return AnalysisSummary(
            frames=frames,
            detections=detections,
            unique_tracks=len(track_ids),
            mean_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
            elapsed_seconds=elapsed,
            fps=(frames / elapsed) if elapsed else 0.0,
            per_frame=per_frame,
        )
