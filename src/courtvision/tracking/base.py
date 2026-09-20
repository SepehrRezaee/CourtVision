"""Framework-agnostic tracking primitives.

Keeping the data model free of Ultralytics types means evaluation, analytics and
serving all consume the same structures, and a different backend can be slotted in
without touching them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Detection:
    """One bounding box observed in one frame, in pixel ``xyxy``."""

    frame: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    track_id: int | None = None

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xyxy[0] + self.xyxy[2]) / 2.0, (self.xyxy[1] + self.xyxy[3]) / 2.0)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def to_mot_line(self) -> str:
        x1, y1, _x2, _y2 = self.xyxy
        return (
            f"{self.frame},{self.track_id},{x1:.3f},{y1:.3f},{self.width:.3f},{self.height:.3f},"
            f"{self.confidence:.5f},1,1"
        )

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "track_id": self.track_id,
            "confidence": round(float(self.confidence), 5),
            "xyxy": [round(float(value), 2) for value in self.xyxy],
            "class_id": self.class_id,
        }


@dataclass
class FrameResult:
    """All detections in one frame plus backend-reported stage timings in ms."""

    frame: int
    detections: list[Detection] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def track_count(self) -> int:
        return sum(1 for detection in self.detections if detection.track_id is not None)

    def to_dict(self, *, include_detections: bool = True) -> dict:
        payload: dict = {"frame": self.frame, "count": len(self.detections)}
        if include_detections:
            payload["detections"] = [detection.to_dict() for detection in self.detections]
        return payload


@dataclass
class TrackingResult:
    """The output of tracking (or detecting over) one video."""

    source: str
    frames: list[FrameResult] = field(default_factory=list)
    frame_size: tuple[int, int] | None = None
    source_fps: float | None = None
    tracker: str | None = None
    weights: str | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def detections(self) -> int:
        return sum(len(frame.detections) for frame in self.frames)

    @property
    def unique_track_ids(self) -> set[int]:
        return {d.track_id for frame in self.frames for d in frame.detections if d.track_id is not None}

    def iter_detections(self) -> Iterator[Detection]:
        for frame in self.frames:
            yield from frame.detections

    def tracks(self) -> dict[int, list[Detection]]:
        grouped: dict[int, list[Detection]] = {}
        for detection in self.iter_detections():
            if detection.track_id is not None:
                grouped.setdefault(detection.track_id, []).append(detection)
        for detections in grouped.values():
            detections.sort(key=lambda item: item.frame)
        return grouped

    def detections_by_frame(self) -> dict[int, list[Detection]]:
        return {frame.frame: list(frame.detections) for frame in self.frames}

    def mot_lines(self, *, include_untracked: bool = False) -> list[str]:
        return [
            detection.to_mot_line()
            for detection in self.iter_detections()
            if detection.track_id is not None or include_untracked
        ]

    def write_mot(self, path: str | Path, *, include_untracked: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = self.mot_lines(include_untracked=include_untracked)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def detection_counts(self) -> list[int]:
        return [len(frame.detections) for frame in self.frames]

    def confidences(self) -> list[float]:
        return [detection.confidence for detection in self.iter_detections()]

    def track_durations(self) -> list[int]:
        return [len(detections) for detections in self.tracks().values()]

    def mean_confidence(self) -> float:
        values = self.confidences()
        return float(sum(values) / len(values)) if values else 0.0

    def frame_range(self) -> tuple[int, int]:
        if not self.frames:
            return (0, 0)
        numbers = [frame.frame for frame in self.frames]
        return (min(numbers), max(numbers))

    def summary(self) -> dict:
        counts = self.detection_counts()
        return {
            "source": self.source,
            "frames": len(self.frames),
            "detections": self.detections,
            "unique_tracks": len(self.unique_track_ids),
            "mean_confidence": round(self.mean_confidence(), 6),
            "detections_per_frame_mean": round(sum(counts) / len(counts), 4) if counts else 0.0,
            "detections_per_frame_max": max(counts) if counts else 0,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "source_fps": self.source_fps,
            "tracker": self.tracker,
            "weights": self.weights,
        }

    def to_dict(self, *, include_frames: bool = False) -> dict:
        payload = self.summary()
        payload["meta"] = self.meta
        if include_frames:
            payload["frames_detail"] = [frame.to_dict() for frame in self.frames]
        return payload


@runtime_checkable
class Tracker(Protocol):
    """Backend interface: turn a video into a :class:`TrackingResult`."""

    name: str

    def track(self, source: str | Path, **kwargs) -> TrackingResult:  # pragma: no cover
        ...


@runtime_checkable
class Detector(Protocol):
    """Backend interface for pure detection (no identity assignment)."""

    name: str

    def detect(self, source: str | Path, **kwargs) -> TrackingResult:  # pragma: no cover
        ...


def filter_detections(
    detections: Iterable[Detection],
    *,
    min_confidence: float | None = None,
    min_area: float | None = None,
    frame_range: tuple[int, int] | None = None,
    track_ids: Sequence[int] | None = None,
) -> list[Detection]:
    allowed = set(track_ids) if track_ids is not None else None
    out: list[Detection] = []
    for detection in detections:
        if min_confidence is not None and detection.confidence < min_confidence:
            continue
        if min_area is not None and detection.area < min_area:
            continue
        if frame_range is not None and not (frame_range[0] <= detection.frame <= frame_range[1]):
            continue
        if allowed is not None and detection.track_id not in allowed:
            continue
        out.append(detection)
    return out


def tracking_result_from_mot(path: str | Path) -> TrackingResult:
    """Load a MOTChallenge prediction file back into a :class:`TrackingResult`."""
    from ..data.mot import load_mot_file

    by_frame: dict[int, list[Detection]] = {}
    for box in load_mot_file(path):
        by_frame.setdefault(box.frame, []).append(
            Detection(
                frame=box.frame,
                xyxy=(box.x, box.y, box.x + box.w, box.y + box.h),
                confidence=box.confidence,
                track_id=box.track_id,
            )
        )
    return TrackingResult(
        source=str(path),
        frames=[
            FrameResult(frame=number, detections=sorted(items, key=lambda item: (item.track_id or -1, item.xyxy)))
            for number, items in sorted(by_frame.items())
        ],
    )


def tracking_result_to_records(result: TrackingResult) -> list[Mapping]:
    """Flatten to CSV rows."""
    return [
        {
            "frame": detection.frame,
            "track_id": detection.track_id,
            "x1": round(detection.xyxy[0], 3),
            "y1": round(detection.xyxy[1], 3),
            "x2": round(detection.xyxy[2], 3),
            "y2": round(detection.xyxy[3], 3),
            "confidence": round(detection.confidence, 5),
            "class_id": detection.class_id,
        }
        for detection in result.iter_detections()
    ]
