"""Trajectory extraction, pixel-space kinematics and spatial zones.

Everything here is **pixel space**: distances in pixels, rates per frame, units encoded
in the field names. No metre-based or m/s quantity is produced anywhere, because no
court calibration or homography exists for these datasets — a physical speed would be
a fabricated number.

Two things that silently corrupt trajectory maths are handled explicitly:

* **frame gaps** — a track missing for 40 frames must not look like one fast step.
  Gaps are recorded and interpolation is opt-in and marked.
* **detector jitter** — positions are smoothed before differencing, because
  differentiating raw box centres turns a couple of pixels of box noise into fake
  velocity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from ..tracking.base import Detection, TrackingResult

PX_UNITS_NOTE = (
    "Pixel space: positions and distances in image pixels, rates per frame. "
    "No physical (metre/second) interpretation without court calibration."
)


def as_positions(points) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected an (N, 2) array of pixel points, got {array.shape}")
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError("Positions contain non-finite values")
    return array


def smooth_positions(points, window: int = 5) -> np.ndarray:
    """Centred moving average with shrinking windows at the edges.

    Centred rather than causal so a smoothed position does not lag the true position.
    """
    points = as_positions(points)
    if window <= 1 or len(points) < 3:
        return points.copy()
    window = min(window, len(points))
    if window % 2 == 0:
        window += 1
    half = window // 2
    return np.array([points[max(0, index - half) : min(len(points), index + half + 1)].mean(axis=0) for index in range(len(points))])


def finite_difference(values: np.ndarray, dt: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, dt, axis=0, edge_order=1)


def velocities(points, dt: float = 1.0) -> np.ndarray:
    """Velocity in pixels per frame."""
    return finite_difference(as_positions(points), dt)


def speeds(points, dt: float = 1.0) -> np.ndarray:
    """Speed magnitude in pixels per frame."""
    return np.linalg.norm(velocities(points, dt), axis=1)


def accelerations(points, dt: float = 1.0) -> np.ndarray:
    return finite_difference(velocities(points, dt), dt)


def directions_deg(points, dt: float = 1.0) -> np.ndarray:
    """Heading in degrees from +x; ``nan`` where the object is effectively stationary."""
    velocity = velocities(points, dt)
    magnitude = np.linalg.norm(velocity, axis=1)
    angles = np.full(len(velocity), np.nan)
    moving = magnitude > 1e-9
    angles[moving] = np.degrees(np.arctan2(velocity[moving, 1], velocity[moving, 0]))
    return angles


def angular_change_deg(angles: np.ndarray) -> np.ndarray:
    """Signed shortest angular difference between consecutive headings, in ``(-180, 180]``."""
    angles = np.asarray(angles, dtype=float)
    if len(angles) < 2:
        return np.zeros(len(angles))
    wrapped = (np.diff(angles) + 180.0) % 360.0 - 180.0
    return np.concatenate([[0.0], wrapped])


def path_length(points) -> float:
    points = as_positions(points)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def displacement(points) -> float:
    points = as_positions(points)
    return float(np.linalg.norm(points[-1] - points[0])) if len(points) > 1 else 0.0


def straightness(points) -> float:
    """``displacement / path_length`` in ``[0, 1]``; 1 is a perfectly straight path."""
    travelled = path_length(points)
    return float(min(1.0, displacement(points) / travelled)) if travelled > 1e-9 else 0.0


@dataclass(frozen=True)
class MotionSummary:
    """Per-track kinematic summary. Lengths are pixels, rates are per frame."""

    frames_observed: int
    path_length_px: float
    displacement_px: float
    straightness: float
    speed_px_per_frame_mean: float
    speed_px_per_frame_max: float
    speed_px_per_frame_std: float
    acceleration_px_per_frame2_mean: float
    acceleration_px_per_frame2_max_abs: float
    direction_change_deg_mean_abs: float
    direction_change_deg_max_abs: float
    stationary_fraction: float

    def to_dict(self) -> dict:
        return {
            "frames_observed": self.frames_observed,
            "path_length_px": round(self.path_length_px, 4),
            "displacement_px": round(self.displacement_px, 4),
            "straightness": round(self.straightness, 6),
            "speed_px_per_frame_mean": round(self.speed_px_per_frame_mean, 6),
            "speed_px_per_frame_max": round(self.speed_px_per_frame_max, 6),
            "speed_px_per_frame_std": round(self.speed_px_per_frame_std, 6),
            "acceleration_px_per_frame2_mean": round(self.acceleration_px_per_frame2_mean, 6),
            "acceleration_px_per_frame2_max_abs": round(self.acceleration_px_per_frame2_max_abs, 6),
            "direction_change_deg_mean_abs": round(self.direction_change_deg_mean_abs, 4),
            "direction_change_deg_max_abs": round(self.direction_change_deg_max_abs, 4),
            "stationary_fraction": round(self.stationary_fraction, 6),
            "units": PX_UNITS_NOTE,
        }


def summarize_motion(points, *, stationary_speed_threshold: float = 1.0, smoothing_window: int = 0) -> MotionSummary:
    points = as_positions(points)
    if smoothing_window and smoothing_window > 1:
        points = smooth_positions(points, smoothing_window)
    if len(points) == 0:
        return MotionSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    speed = speeds(points)
    accel = np.linalg.norm(accelerations(points), axis=1)
    changes = np.abs(angular_change_deg(directions_deg(points)))
    finite = changes[np.isfinite(changes)]
    return MotionSummary(
        frames_observed=len(points),
        path_length_px=path_length(points),
        displacement_px=displacement(points),
        straightness=straightness(points),
        speed_px_per_frame_mean=float(speed.mean()),
        speed_px_per_frame_max=float(speed.max()),
        speed_px_per_frame_std=float(speed.std()),
        acceleration_px_per_frame2_mean=float(accel.mean()),
        acceleration_px_per_frame2_max_abs=float(accel.max()),
        direction_change_deg_mean_abs=float(finite.mean()) if finite.size else 0.0,
        direction_change_deg_max_abs=float(finite.max()) if finite.size else 0.0,
        stationary_fraction=float(np.mean(speed <= stationary_speed_threshold)),
    )


@dataclass(frozen=True)
class Zone:
    """An axis-aligned pixel rectangle with a stable integer id."""

    zone_id: int
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "x1": round(self.x1, 3),
            "y1": round(self.y1, 3),
            "x2": round(self.x2, 3),
            "y2": round(self.y2, 3),
        }


class ZoneGrid:
    """A ``rows`` x ``cols`` uniform grid over the frame.

    Zones are defined relative to the *frame*, not the court: "the middle-left ninth of
    the image" is defensible without a homography, "the defensive third" is not.
    """

    def __init__(self, rows: int, cols: int, frame_size: tuple[int, int]) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame_size must be positive")
        self.rows, self.cols, self.frame_size = rows, cols, (width, height)
        self.zones = [
            Zone(
                zone_id=row * cols + col,
                name=f"r{row}c{col}",
                x1=width * col / cols,
                y1=height * row / rows,
                x2=width * (col + 1) / cols,
                y2=height * (row + 1) / rows,
            )
            for row in range(rows)
            for col in range(cols)
        ]

    def __len__(self) -> int:
        return len(self.zones)

    def assign(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        width, height = self.frame_size
        cols = np.clip((points[:, 0] / width * self.cols).astype(int), 0, self.cols - 1)
        rows = np.clip((points[:, 1] / height * self.rows).astype(int), 0, self.rows - 1)
        return rows * self.cols + cols


@dataclass
class TrackTrajectory:
    """One tracked object's observation history."""

    track_id: int
    frames: np.ndarray
    xyxy: np.ndarray
    confidences: np.ndarray
    class_id: int = 0

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int)
        self.xyxy = np.asarray(self.xyxy, dtype=float).reshape(-1, 4)
        self.confidences = np.asarray(self.confidences, dtype=float).reshape(-1)
        if not (len(self.frames) == len(self.xyxy) == len(self.confidences)):
            raise ValueError("frames, xyxy and confidences must have equal length")

    @classmethod
    def from_detections(cls, track_id: int, detections: Sequence[Detection]) -> TrackTrajectory:
        ordered = sorted(detections, key=lambda item: item.frame)
        return cls(
            track_id=track_id,
            frames=np.array([item.frame for item in ordered], dtype=int),
            xyxy=np.array([item.xyxy for item in ordered], dtype=float),
            confidences=np.array([item.confidence for item in ordered], dtype=float),
            class_id=ordered[0].class_id if ordered else 0,
        )

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def centers(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0], axis=1)

    @property
    def areas(self) -> np.ndarray:
        width = self.xyxy[:, 2] - self.xyxy[:, 0]
        height = self.xyxy[:, 3] - self.xyxy[:, 1]
        return np.maximum(0.0, width) * np.maximum(0.0, height)

    @property
    def start_frame(self) -> int:
        return int(self.frames[0]) if len(self.frames) else 0

    @property
    def end_frame(self) -> int:
        return int(self.frames[-1]) if len(self.frames) else 0

    @property
    def duration_frames(self) -> int:
        return (self.end_frame - self.start_frame + 1) if len(self.frames) else 0

    @property
    def gaps(self) -> list[tuple[int, int]]:
        return [(int(self.frames[i]), int(self.frames[i + 1])) for i in np.nonzero(np.diff(self.frames) > 1)[0]]

    @property
    def is_contiguous(self) -> bool:
        return not self.gaps

    @property
    def mean_confidence(self) -> float:
        return float(self.confidences.mean()) if len(self.confidences) else 0.0

    def fill_gaps(self, *, max_gap: int = 0) -> tuple[TrackTrajectory, np.ndarray]:
        """Linearly interpolate gaps of at most ``max_gap`` frames.

        Returns ``(trajectory, interpolated_mask)``. Interpolated samples carry
        ``confidence 0.0`` and ``mask True`` so callers can exclude them from statistics.
        A gap longer than ``max_gap`` is never bridged — linear interpolation over a long
        absence invents a path the object never travelled — so the returned trajectory
        simply **ends before** such a gap, and unobserved leading frames are skipped
        rather than fabricated.
        """
        if max_gap <= 0 or self.is_contiguous:
            return self, np.zeros(len(self), dtype=bool)

        targets = [int(frame) for frame in range(self.start_frame, self.end_frame + 1)]
        known = {int(frame): index for index, frame in enumerate(self.frames)}
        centers, areas = self.centers, self.areas
        sides = np.sqrt(np.maximum(areas, 0.0))

        out_frames: list[int] = []
        out_centers: list[list[float]] = []
        out_sides: list[float] = []
        out_conf: list[float] = []
        out_mask: list[bool] = []
        pending: list[int] = []
        last_known: int | None = None

        for position, frame in enumerate(targets):
            if frame in known:
                index = known[frame]
                if pending:
                    if last_known is None:
                        # Frames before the track's first observation: the track simply
                        # starts here, there is nothing to interpolate toward.
                        pending = []
                    else:
                        before_index = known[last_known]
                        span = frame - last_known
                        for gap_position in pending:
                            weight = (targets[gap_position] - last_known) / span
                            blended = centers[before_index] * (1 - weight) + centers[index] * weight
                            out_centers.append(blended.tolist())
                            out_sides.append(float(sides[before_index] * (1 - weight) + sides[index] * weight))
                            out_conf.append(0.0)
                            out_mask.append(True)
                            out_frames.append(targets[gap_position])
                    pending = []
                out_frames.append(frame)
                out_centers.append(centers[index].tolist())
                out_sides.append(float(sides[index]))
                out_conf.append(float(self.confidences[index]))
                out_mask.append(False)
                last_known = frame
            elif last_known is not None and len(pending) < max_gap:
                pending.append(position)
            elif last_known is None:
                continue
            else:
                # A gap longer than max_gap: stop here. Continuing would either invent
                # motion across it or emit uninitialised array contents.
                break

        if not out_frames:  # pragma: no cover - a trajectory always has known frames
            return self, np.zeros(len(self), dtype=bool)

        frames = np.array(out_frames, dtype=int)
        filled = np.asarray(out_centers, dtype=float).reshape(-1, 2)
        sizes = np.asarray(out_sides, dtype=float)
        xyxy = np.stack(
            [
                filled[:, 0] - sizes / 2,
                filled[:, 1] - sizes / 2,
                filled[:, 0] + sizes / 2,
                filled[:, 1] + sizes / 2,
            ],
            axis=1,
        )
        return (
            TrackTrajectory(
                track_id=self.track_id,
                frames=frames,
                xyxy=xyxy,
                confidences=np.asarray(out_conf, dtype=float),
                class_id=self.class_id,
            ),
            np.asarray(out_mask, dtype=bool),
        )

    def to_dict(self, *, include_points: bool = False) -> dict:
        payload = {
            "track_id": self.track_id,
            "frames_observed": len(self),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "contiguous": self.is_contiguous,
            "gaps": [list(gap) for gap in self.gaps],
            "mean_confidence": round(self.mean_confidence, 6),
            "median_box_area_px": round(float(np.median(self.areas)), 3) if len(self.areas) else 0.0,
        }
        if include_points:
            payload["points"] = [
                {
                    "frame": int(frame),
                    "center_px": [round(float(cx), 2), round(float(cy), 2)],
                    "xyxy": [round(float(value), 2) for value in box],
                    "confidence": round(float(confidence), 5),
                }
                for frame, box, confidence, (cx, cy) in zip(
                    self.frames, self.xyxy, self.confidences, self.centers, strict=True
                )
            ]
        return payload


@dataclass
class TrajectoryFeatures:
    """Per-frame kinematic and geometric features for one track."""

    track_id: int
    frames: np.ndarray
    centers_raw_px: np.ndarray
    centers_px: np.ndarray
    speeds_px_per_frame: np.ndarray
    accelerations_px_per_frame2: np.ndarray
    directions_deg: np.ndarray
    direction_change_deg: np.ndarray
    confidences: np.ndarray
    box_areas_px: np.ndarray
    zone_ids: np.ndarray
    interpolated: np.ndarray
    summary: MotionSummary
    smoothing_window: int = 0
    units_note: str = PX_UNITS_NOTE

    def __len__(self) -> int:
        return len(self.frames)

    def to_dict(self, *, include_series: bool = False) -> dict:
        payload = {
            "track_id": self.track_id,
            "frames": len(self.frames),
            "start_frame": int(self.frames[0]) if len(self.frames) else 0,
            "end_frame": int(self.frames[-1]) if len(self.frames) else 0,
            "interpolated_frames": int(self.interpolated.sum()),
            "zones_visited": sorted({int(zone) for zone in self.zone_ids if zone >= 0}),
            "smoothing_window": self.smoothing_window,
            "summary": self.summary.to_dict(),
            "units": self.units_note,
        }
        if include_series:
            payload["series"] = [
                {
                    "frame": int(frame),
                    "center_px": [round(float(cx), 2), round(float(cy), 2)],
                    "speed_px_per_frame": round(float(speed), 5),
                    "acceleration_px_per_frame2": round(float(accel), 5),
                    "direction_deg": None if math.isnan(direction) else round(float(direction), 3),
                    "direction_change_deg": round(float(change), 3),
                    "zone_id": int(zone),
                    "confidence": round(float(confidence), 5),
                    "interpolated": bool(interpolated),
                }
                for frame, (cx, cy), speed, accel, direction, change, zone, confidence, interpolated in zip(
                    self.frames,
                    self.centers_px,
                    self.speeds_px_per_frame,
                    self.accelerations_px_per_frame2,
                    self.directions_deg,
                    self.direction_change_deg,
                    self.zone_ids,
                    self.confidences,
                    self.interpolated,
                    strict=True,
                )
            ]
        return payload


def build_trajectories(result: TrackingResult, *, min_length: int = 1) -> list[TrackTrajectory]:
    return [
        TrackTrajectory.from_detections(track_id, detections)
        for track_id, detections in sorted(result.tracks().items())
        if len(detections) >= max(1, min_length)
    ]


def extract_features(
    trajectory: TrackTrajectory,
    *,
    smoothing_window: int = 5,
    zone_grid: tuple[int, int] | None = None,
    frame_size: tuple[int, int] | None = None,
    stationary_speed_threshold: float = 1.0,
    fill_gap: int = 0,
) -> TrajectoryFeatures:
    interpolated = np.zeros(len(trajectory), dtype=bool)
    working = trajectory
    if fill_gap > 0 and not trajectory.is_contiguous:
        working, interpolated = trajectory.fill_gaps(max_gap=fill_gap)

    raw_centers = working.centers
    centers = smooth_positions(raw_centers, smoothing_window) if smoothing_window > 1 else raw_centers
    if zone_grid is not None:
        if frame_size is None:
            raise ValueError("frame_size is required to assign zones")
        zone_ids = ZoneGrid(rows=zone_grid[0], cols=zone_grid[1], frame_size=frame_size).assign(centers)
    else:
        zone_ids = np.full(len(centers), -1, dtype=int)

    return TrajectoryFeatures(
        track_id=trajectory.track_id,
        frames=working.frames,
        centers_raw_px=raw_centers,
        centers_px=centers,
        speeds_px_per_frame=speeds(centers),
        accelerations_px_per_frame2=np.linalg.norm(accelerations(centers), axis=1),
        directions_deg=directions_deg(centers),
        direction_change_deg=angular_change_deg(directions_deg(centers)),
        confidences=working.confidences,
        box_areas_px=working.areas,
        zone_ids=zone_ids,
        interpolated=interpolated,
        summary=summarize_motion(centers, stationary_speed_threshold=stationary_speed_threshold),
        smoothing_window=smoothing_window,
    )


def analyze_trajectories(
    result: TrackingResult,
    *,
    min_track_length: int = 1,
    smoothing_window: int = 5,
    zone_grid: tuple[int, int] | None = None,
    stationary_speed_threshold: float = 1.0,
    fill_gap: int = 0,
) -> tuple[list[TrackTrajectory], list[TrajectoryFeatures]]:
    trajectories = build_trajectories(result, min_length=min_track_length)
    features = [
        extract_features(
            trajectory,
            smoothing_window=smoothing_window,
            zone_grid=zone_grid,
            frame_size=result.frame_size,
            stationary_speed_threshold=stationary_speed_threshold,
            fill_gap=fill_gap,
        )
        for trajectory in trajectories
    ]
    return trajectories, features


def trajectory_records(features: Sequence[TrajectoryFeatures]) -> list[dict]:
    """Flat per-frame rows for CSV export."""
    rows: list[dict] = []
    for feature in features:
        for index, frame in enumerate(feature.frames):
            rows.append(
                {
                    "track_id": feature.track_id,
                    "frame": int(frame),
                    "cx_px": round(float(feature.centers_px[index, 0]), 3),
                    "cy_px": round(float(feature.centers_px[index, 1]), 3),
                    "speed_px_per_frame": round(float(feature.speeds_px_per_frame[index]), 6),
                    "acceleration_px_per_frame2": round(float(feature.accelerations_px_per_frame2[index]), 6),
                    "direction_deg": (
                        None if math.isnan(feature.directions_deg[index]) else round(float(feature.directions_deg[index]), 3)
                    ),
                    "direction_change_deg": round(float(feature.direction_change_deg[index]), 3),
                    "zone_id": int(feature.zone_ids[index]),
                    "confidence": round(float(feature.confidences[index]), 5),
                    "box_area_px": round(float(feature.box_areas_px[index]), 2),
                    "interpolated": bool(feature.interpolated[index]),
                }
            )
    return rows


def occupancy_map(features: Sequence[TrajectoryFeatures], frame_size: tuple[int, int], grid: tuple[int, int] = (8, 12)) -> dict:
    """Trajectory samples per grid cell — a resolution-independent heatmap."""
    rows, cols = grid
    width, height = frame_size
    if rows <= 0 or cols <= 0 or width <= 0 or height <= 0:
        raise ValueError("grid and frame_size must be positive")
    counts = np.zeros((rows, cols), dtype=int)
    for feature in features:
        for cx, cy in feature.centers_px:
            counts[min(rows - 1, max(0, int(cy / height * rows))), min(cols - 1, max(0, int(cx / width * cols)))] += 1
    total = int(counts.sum())
    return {
        "grid": [rows, cols],
        "frame_size": [width, height],
        "counts": counts.tolist(),
        "density": (counts / total).tolist() if total else counts.tolist(),
        "total_samples": total,
        "cell_size_px": [round(width / cols, 3), round(height / rows, 3)],
    }


def track_statistics(features: Sequence[TrajectoryFeatures]) -> dict:
    """Dataset-level aggregation over tracks."""
    if not features:
        return {"tracks": 0, "units": PX_UNITS_NOTE}
    durations = np.array([len(feature) for feature in features], dtype=float)
    displacements = np.array([feature.summary.displacement_px for feature in features], dtype=float)
    confidences = np.concatenate([feature.confidences for feature in features])
    areas = np.concatenate([feature.box_areas_px for feature in features])
    speeds = np.concatenate([feature.speeds_px_per_frame for feature in features])
    return {
        "tracks": len(features),
        "duration_frames_mean": round(float(durations.mean()), 4),
        "duration_frames_max": int(durations.max()),
        "displacement_px_mean": round(float(displacements.mean()), 4),
        "mean_confidence": round(float(confidences.mean()), 6) if confidences.size else 0.0,
        "median_box_area_px": round(float(np.median(areas)), 3) if areas.size else 0.0,
        "speed_px_per_frame_mean": round(float(speeds.mean()), 6) if speeds.size else 0.0,
        "units": PX_UNITS_NOTE,
    }


def gapped_track_count(result: TrackingResult) -> int:
    return sum(1 for detections in result.tracks().values() if any(b - a > 1 for a, b in pairwise(sorted(d.frame for d in detections))))
