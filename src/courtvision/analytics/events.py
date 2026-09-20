"""Deterministic, threshold-based event detection over trajectories.

This is the **heuristic** event layer. Every event carries the measured value, the
threshold that fired and the unit, so an event can be audited rather than trusted.

**Event accuracy is NOT MEASURED.** No labelled temporal sports-event dataset exists in
this repository, so the detector's correctness is established by unit tests and its
determinism, never by an accuracy figure. The interface
(:class:`EventDetector` protocol, and ``temporal/features.py``) is where a learned
sequence classifier would plug in; that path is interface-only here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import EventConfig
from .trajectories import TrajectoryFeatures, angular_change_deg

EVENT_TYPES = (
    "TRACK_APPEARANCE",
    "TRACK_DISAPPEARANCE",
    "ZONE_ENTER",
    "ZONE_EXIT",
    "DIRECTION_CHANGE",
    "ACCELERATION",
    "DECELERATION",
    "SPRINT",
    "STATIONARY",
    "CLOSE_PROXIMITY",
    "CONVERGENCE",
    "DIVERGENCE",
)


@dataclass(frozen=True)
class Event:
    """One detected event with the evidence that produced it."""

    event_type: str
    track_id: int
    start_frame: int
    end_frame: int
    value: float | None = None
    threshold: float | None = None
    unit: str = ""
    related_track_id: int | None = None
    zone_id: int | None = None
    details: dict = field(default_factory=dict)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "related_track_id": self.related_track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "value": None if self.value is None else round(float(self.value), 6),
            "threshold": None if self.threshold is None else round(float(self.threshold), 6),
            "unit": self.unit,
            "zone_id": self.zone_id,
            "details": self.details,
        }


@runtime_checkable
class EventDetector(Protocol):
    """Anything that turns trajectories into events."""

    name: str

    def detect(self, features: Sequence[TrajectoryFeatures]) -> list[Event]:  # pragma: no cover
        ...


def _runs(mask: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
    """Index runs ``(start, end_inclusive)`` where ``mask`` is True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0] - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True) if (end - start + 1) >= min_length]


class HeuristicEventDetector:
    """Threshold rules over :class:`TrajectoryFeatures`, in pixel space."""

    name = "heuristic-thresholds"

    def __init__(self, config: EventConfig | None = None) -> None:
        self.config = config or EventConfig()

    def _presence_events(self, feature: TrajectoryFeatures) -> list[Event]:
        if len(feature.frames) == 0:
            return []
        return [
            Event(
                event_type="TRACK_APPEARANCE",
                track_id=feature.track_id,
                start_frame=int(feature.frames[0]),
                end_frame=int(feature.frames[0]),
                details={"confidence": round(float(feature.confidences[0]), 5)},
            ),
            Event(
                event_type="TRACK_DISAPPEARANCE",
                track_id=feature.track_id,
                start_frame=int(feature.frames[-1]),
                end_frame=int(feature.frames[-1]),
                details={"confidence": round(float(feature.confidences[-1]), 5)},
            ),
        ]

    def _zone_events(self, feature: TrajectoryFeatures) -> list[Event]:
        events: list[Event] = []
        zones = feature.zone_ids
        if len(zones) < 2 or np.all(zones < 0):
            return events
        for index in range(1, len(zones)):
            previous, current = int(zones[index - 1]), int(zones[index])
            if previous == current:
                continue
            frame = int(feature.frames[index])
            if previous >= 0:
                events.append(Event("ZONE_EXIT", feature.track_id, frame, frame, zone_id=previous))
            if current >= 0:
                events.append(Event("ZONE_ENTER", feature.track_id, frame, frame, zone_id=current))
        return events

    def _direction_events(self, feature: TrajectoryFeatures) -> list[Event]:
        """Heading changes from *consecutive* positions, not centrally differenced velocity.

        Central differences average across a corner, so a 90-degree turn would appear as
        two 45-degree changes and never reach a 90-degree threshold.
        """
        centers = feature.centers_px
        if len(centers) < 3:
            return []
        steps = np.diff(centers, axis=0)
        magnitudes = np.linalg.norm(steps, axis=1)
        headings = np.full(len(steps), np.nan)
        moving = magnitudes > 1e-9
        headings[moving] = np.degrees(np.arctan2(steps[moving, 1], steps[moving, 0]))
        changes = np.zeros(len(centers))
        changes[1:] = angular_change_deg(np.concatenate([[headings[0]], headings]))[1:]
        speeds = np.concatenate([[magnitudes[0]], magnitudes])

        mask = (np.abs(changes) >= self.config.direction_change_deg) & (
            speeds >= self.config.direction_change_min_speed
        )
        mask[:2] = False  # the first two samples have no usable heading pair
        return [
            Event(
                event_type="DIRECTION_CHANGE",
                track_id=feature.track_id,
                start_frame=int(feature.frames[index]),
                end_frame=int(feature.frames[index]),
                value=float(changes[index]),
                threshold=float(self.config.direction_change_deg),
                unit="degrees",
                details={"speed_px_per_frame": round(float(speeds[index]), 4)},
            )
            for index in np.nonzero(mask)[0]
        ]

    def _acceleration_events(self, feature: TrajectoryFeatures) -> list[Event]:
        speeds = feature.speeds_px_per_frame
        if len(speeds) < 2:
            return []
        deltas = np.concatenate([[0.0], np.diff(speeds)])
        events: list[Event] = []
        for index, delta in enumerate(deltas):
            if index == 0:
                continue
            if delta >= self.config.accel_threshold:
                event_type, threshold = "ACCELERATION", self.config.accel_threshold
            elif delta <= self.config.decel_threshold:
                event_type, threshold = "DECELERATION", self.config.decel_threshold
            else:
                continue
            events.append(
                Event(
                    event_type=event_type,
                    track_id=feature.track_id,
                    start_frame=int(feature.frames[index]),
                    end_frame=int(feature.frames[index]),
                    value=float(delta),
                    threshold=float(threshold),
                    unit="px_per_frame2",
                    details={
                        "speed_before_px_per_frame": round(float(speeds[index - 1]), 4),
                        "speed_after_px_per_frame": round(float(speeds[index]), 4),
                    },
                )
            )
        return events

    def _speed_regime_events(self, feature: TrajectoryFeatures) -> list[Event]:
        speeds, frames = feature.speeds_px_per_frame, feature.frames
        events: list[Event] = []
        if len(speeds) == 0:
            return events
        for start, end in _runs(speeds >= self.config.sprint_speed, self.config.sprint_min_frames):
            events.append(
                Event(
                    event_type="SPRINT",
                    track_id=feature.track_id,
                    start_frame=int(frames[start]),
                    end_frame=int(frames[end]),
                    value=float(speeds[start : end + 1].max()),
                    threshold=float(self.config.sprint_speed),
                    unit="px_per_frame",
                    details={"mean_speed_px_per_frame": round(float(speeds[start : end + 1].mean()), 4)},
                )
            )
        for start, end in _runs(speeds <= self.config.stationary_speed, self.config.stationary_min_frames):
            events.append(
                Event(
                    event_type="STATIONARY",
                    track_id=feature.track_id,
                    start_frame=int(frames[start]),
                    end_frame=int(frames[end]),
                    value=float(speeds[start : end + 1].mean()),
                    threshold=float(self.config.stationary_speed),
                    unit="px_per_frame",
                    details={"max_speed_px_per_frame": round(float(speeds[start : end + 1].max()), 4)},
                )
            )
        return events

    def _pairwise_events(self, first: TrajectoryFeatures, second: TrajectoryFeatures) -> list[Event]:
        common = np.intersect1d(first.frames, second.frames)
        if len(common) < max(3, self.config.proximity_min_frames):
            return []
        first_index = {int(frame): index for index, frame in enumerate(first.frames)}
        second_index = {int(frame): index for index, frame in enumerate(second.frames)}
        distances = np.array(
            [
                float(np.linalg.norm(first.centers_px[first_index[int(frame)]] - second.centers_px[second_index[int(frame)]]))
                for frame in common
            ]
        )
        events: list[Event] = []
        for start, end in _runs(distances <= self.config.close_proximity_distance, self.config.proximity_min_frames):
            events.append(
                Event(
                    event_type="CLOSE_PROXIMITY",
                    track_id=first.track_id,
                    related_track_id=second.track_id,
                    start_frame=int(common[start]),
                    end_frame=int(common[end]),
                    value=float(distances[start : end + 1].min()),
                    threshold=float(self.config.close_proximity_distance),
                    unit="px",
                    details={"mean_distance_px": round(float(distances[start : end + 1].mean()), 4)},
                )
            )
        window = self.config.convergence_window
        if len(common) >= window + 1:
            delta = distances[window:] - distances[:-window]
            span = self.config.convergence_delta
            for index, change in enumerate(delta):
                if change <= -span:
                    event_type = "CONVERGENCE"
                elif change >= span:
                    event_type = "DIVERGENCE"
                else:
                    continue
                events.append(
                    Event(
                        event_type=event_type,
                        track_id=first.track_id,
                        related_track_id=second.track_id,
                        start_frame=int(common[index]),
                        end_frame=int(common[index + window]),
                        value=float(change),
                        threshold=float(span),
                        unit="px",
                        details={
                            "window_frames": int(window),
                            "distance_start_px": round(float(distances[index]), 4),
                            "distance_end_px": round(float(distances[index + window]), 4),
                        },
                    )
                )
        return events

    def detect(self, features: Sequence[TrajectoryFeatures]) -> list[Event]:
        """Detect all events, in a deterministic and stable order."""
        usable = [feature for feature in features if len(feature.frames) >= 1]
        events: list[Event] = []
        for feature in sorted(usable, key=lambda item: item.track_id):
            events += self._presence_events(feature)
            events += self._zone_events(feature)
            events += self._direction_events(feature)
            events += self._acceleration_events(feature)
            events += self._speed_regime_events(feature)
        ordered_pairs = sorted(
            ((a, b) for index, a in enumerate(usable) for b in usable[index + 1 :]),
            key=lambda pair: (pair[0].track_id, pair[1].track_id),
        )
        for first, second in ordered_pairs:
            events += self._pairwise_events(first, second)
        events.sort(key=lambda event: (event.start_frame, event.event_type, event.track_id, event.related_track_id or -1))
        return events


def detect_events(features: Sequence[TrajectoryFeatures], config: EventConfig | None = None) -> list[Event]:
    return HeuristicEventDetector(config).detect(features)


def summarize_events(events: Iterable[Event]) -> dict:
    events = list(events)
    counts = Counter(event.event_type for event in events)
    durations: dict[str, list[int]] = {}
    for event in events:
        durations.setdefault(event.event_type, []).append(event.duration_frames)
    return {
        "total": len(events),
        "by_type": {event_type: int(counts.get(event_type, 0)) for event_type in EVENT_TYPES},
        "mean_duration_frames": {
            event_type: round(float(np.mean(values)), 3) for event_type, values in sorted(durations.items())
        },
        "tracks_with_events": len({event.track_id for event in events}),
    }


def events_to_records(events: Iterable[Event]) -> list[dict]:
    return [
        {
            "event_type": event.event_type,
            "track_id": event.track_id,
            "related_track_id": event.related_track_id,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "duration_frames": event.duration_frames,
            "value": event.value,
            "threshold": event.threshold,
            "unit": event.unit,
            "zone_id": event.zone_id,
        }
        for event in events
    ]


def event_rate(events: Iterable[Event], frames: int, fps: float | None) -> dict:
    """Events per minute of footage when FPS is known, else per frame.

    An unknown frame rate is never invented; the normalisation used is recorded.
    """
    counts = Counter(event.event_type for event in events)
    if fps and fps > 0 and frames > 0:
        minutes = frames / fps / 60.0
        return {"normalised_by": "fps", "fps": float(fps), **{key: round(value / minutes, 6) for key, value in counts.items()}}
    if not frames:
        return {"normalised_by": "frames", "fps": None}
    return {"normalised_by": "frames", "fps": None, **{key: round(value / frames, 6) for key, value in counts.items()}}
