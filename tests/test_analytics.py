"""Tests for trajectory kinematics, zones and event detection.

Assertions are analytically known values (a straight path has straightness 1, a 90-degree
turn is detected as 90 degrees) because a kinematics bug is invisible in aggregate output
but silently wrong in every derived event.
"""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.analytics.events import (
    EVENT_TYPES,
    HeuristicEventDetector,
    detect_events,
    event_rate,
    events_to_records,
    summarize_events,
)
from courtvision.analytics.trajectories import (
    TrackTrajectory,
    ZoneGrid,
    analyze_trajectories,
    angular_change_deg,
    as_positions,
    directions_deg,
    gapped_track_count,
    occupancy_map,
    smooth_positions,
    speeds,
    straightness,
    summarize_motion,
    track_statistics,
)
from courtvision.config import EventConfig
from courtvision.tracking.base import Detection, FrameResult, TrackingResult


def build_result(track_points: dict[int, list[tuple[float, float]]], frame_size=(400, 400), fps=25.0) -> TrackingResult:
    """Turn per-track pixel centres into a tracking result with a constant box size."""
    frames: dict[int, list[Detection]] = {}
    for track_id, points in track_points.items():
        for index, (cx, cy) in enumerate(points, start=1):
            frames.setdefault(index, []).append(
                Detection(frame=index, xyxy=(cx - 10, cy - 20, cx + 10, cy + 20), confidence=0.9, track_id=track_id)
            )
    return TrackingResult(
        source="synthetic://analytics",
        frames=[FrameResult(frame=number, detections=items) for number, items in sorted(frames.items())],
        frame_size=frame_size,
        source_fps=fps,
    )


def features_for(track_points, config: EventConfig | None = None, frame_size=(400, 400)):
    config = config or EventConfig()
    _trajectories, features = analyze_trajectories(
        build_result(track_points, frame_size),
        min_track_length=1,
        smoothing_window=0,
        zone_grid=config.zone_grid,
        stationary_speed_threshold=config.stationary_speed,
    )
    return features


# -- kinematics --------------------------------------------------------------


def test_straight_line_kinematics() -> None:
    points = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]])
    assert speeds(points) == pytest.approx([3.0, 3.0, 3.0, 3.0])
    assert directions_deg(points) == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert straightness(points) == pytest.approx(1.0)


def test_downward_motion_is_90_degrees_because_y_grows_downwards() -> None:
    points = np.array([[0.0, 0.0], [0.0, 5.0], [0.0, 10.0]])
    assert directions_deg(points) == pytest.approx([90.0, 90.0, 90.0])


def test_stationary_points_have_undefined_direction_not_zero() -> None:
    angles = directions_deg(np.zeros((5, 2)))
    assert np.all(np.isnan(angles)), "a stationary object has no heading; NaN, not 0"


def test_angular_change_wraps_shortest_way() -> None:
    assert angular_change_deg(np.array([170.0, -170.0]))[1] == pytest.approx(20.0)
    assert abs(angular_change_deg(np.array([0.0, 180.0]))[1]) == pytest.approx(180.0)


def test_smoothing_reduces_jitter_and_keeps_the_trend() -> None:
    rng = np.random.default_rng(0)
    base = np.stack([np.arange(50.0), np.zeros(50)], axis=1)
    noisy = base + rng.normal(0, 3.0, base.shape)
    smoothed = smooth_positions(noisy, window=7)
    assert np.std(np.diff(smoothed[:, 0])) <= np.std(np.diff(noisy[:, 0]))
    assert smoothed[-1, 0] == pytest.approx(49.0, abs=3.0)


def test_invalid_positions_are_rejected() -> None:
    with pytest.raises(ValueError, match="N, 2"):
        as_positions(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        as_positions(np.array([[0.0, 0.0], [np.nan, 1.0]]))


def test_summarize_motion_uses_explicit_pixel_units() -> None:
    points = np.stack([np.arange(0.0, 30.0), np.zeros(30)], axis=1)
    summary = summarize_motion(points, smoothing_window=0, stationary_speed_threshold=0.5)
    payload = summary.to_dict()
    assert summary.speed_px_per_frame_mean == pytest.approx(1.0, abs=1e-6)
    assert summary.path_length_px == pytest.approx(29.0)
    assert "path_length_px" in payload and "speed_px_per_frame_mean" in payload
    assert "pixel" in payload["units"]


def test_stationary_threshold_is_inclusive() -> None:
    points = np.stack([np.arange(0.0, 10.0), np.zeros(10)], axis=1)
    assert summarize_motion(points, stationary_speed_threshold=1.0).stationary_fraction == pytest.approx(1.0)
    assert summarize_motion(points, stationary_speed_threshold=0.99).stationary_fraction == pytest.approx(0.0)


def test_zigzag_has_low_straightness() -> None:
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    assert straightness(points) < 0.4
    assert summarize_motion(points).path_length_px == pytest.approx(30.0)


# -- zones and occupancy -----------------------------------------------------


def test_zone_grid_assigns_clamped_ids() -> None:
    grid = ZoneGrid(rows=2, cols=2, frame_size=(400, 400))
    # (-5, 450) clamps to column 0, row 1 -> id = row * cols + col = 2
    assert grid.assign(np.array([[10.0, 10.0], [390.0, 390.0], [-5.0, 450.0]])).tolist() == [0, 3, 2]
    assert len(grid) == 4


def test_occupancy_map_counts_land_in_the_right_cells() -> None:
    features = features_for({1: [(10.0, 10.0), (30.0, 30.0)]})
    counts = occupancy_map(features, frame_size=(400, 400), grid=(2, 2))
    assert counts["total_samples"] == 2
    assert counts["counts"][0][0] == 2


# -- trajectories ------------------------------------------------------------


def trajectory_from_points(points: list[float], track_id: int = 1) -> TrackTrajectory:
    frames = list(range(1, len(points) + 1))
    return TrackTrajectory(
        track_id=track_id,
        frames=np.array(frames, dtype=int),
        xyxy=np.array([[x - 10, 80.0, x + 10, 120.0] for x in points], dtype=float),
        confidences=np.full(len(points), 0.9),
    )


def test_gaps_are_detected() -> None:
    trajectory = trajectory_from_points([10.0, 20.0, 30.0, 40.0])
    holed = TrackTrajectory(
        track_id=1,
        frames=np.array([1, 2, 5, 6], dtype=int),
        xyxy=trajectory.xyxy[[0, 1, 4 % 4, 5 % 4]][[0, 1, 0, 1]],
        confidences=np.full(4, 0.9),
    )
    assert holed.gaps == [(2, 5)]
    assert not holed.is_contiguous
    assert gapped_track_count(build_result({1: [(10 + i, 100) for i in range(5)]})) == 0


def test_fill_gaps_interpolates_and_marks_synthetic_samples() -> None:
    trajectory = TrackTrajectory(
        track_id=1,
        frames=np.array([1, 2, 5], dtype=int),
        xyxy=np.array([[0, 0, 10, 10], [10, 0, 20, 10], [40, 0, 50, 10]], dtype=float),
        confidences=np.array([0.9, 0.9, 0.9]),
    )
    filled, mask = trajectory.fill_gaps(max_gap=2)
    assert filled.frames.tolist() == [1, 2, 3, 4, 5]
    assert mask.tolist() == [False, False, True, True, False]
    assert filled.confidences[[2, 3]].tolist() == [0.0, 0.0], "interpolated samples carry no fake confidence"
    # Linear interpolation between centre x=15 (frame 2) and x=45 (frame 5).
    assert filled.centers[2, 0] == pytest.approx(25.0)
    assert filled.centers[3, 0] == pytest.approx(35.0)
    assert not filled.gaps


def test_fill_gaps_ends_before_a_gap_it_cannot_bridge() -> None:
    """Regression: the previous implementation returned uninitialised array contents."""
    trajectory = TrackTrajectory(
        track_id=1,
        frames=np.array([1, 2, 9, 10], dtype=int),
        xyxy=np.array(
            [[0, 0, 10, 10], [10, 0, 20, 10], [80, 0, 90, 10], [90, 0, 100, 10]], dtype=float
        ),
        confidences=np.full(4, 0.9),
    )
    filled, mask = trajectory.fill_gaps(max_gap=2)
    assert filled.frames.tolist() == [1, 2], "the trajectory must end before the unbridged gap"
    assert mask.sum() == 0
    assert np.all(np.isfinite(filled.xyxy)), "no uninitialised coordinates may be emitted"


def test_fill_gaps_skips_unobserved_leading_frames() -> None:
    trajectory = TrackTrajectory(
        track_id=1,
        frames=np.array([4, 5], dtype=int),
        xyxy=np.array([[0, 0, 10, 10], [10, 0, 20, 10]], dtype=float),
        confidences=np.array([0.9, 0.9]),
    )
    filled, _mask = trajectory.fill_gaps(max_gap=2)
    assert filled.frames.tolist() == [4, 5], "leading absence is not a gap and must not be fabricated"


def test_features_expose_pixel_units_and_zone_ids() -> None:
    features = features_for({1: [(10.0 + i * 5, 100.0) for i in range(20)]}, EventConfig(zone_grid=(3, 3)))
    assert features[0].zone_ids.min() >= 0
    assert "pixel" in features[0].summary.to_dict()["units"].lower()
    stats = track_statistics(features)
    assert stats["tracks"] == 1
    assert "pixel" in stats["units"].lower()


# -- events ------------------------------------------------------------------


def test_event_types_are_stable() -> None:
    assert len(EVENT_TYPES) == len(set(EVENT_TYPES))
    assert "SPRINT" in EVENT_TYPES and "CLOSE_PROXIMITY" in EVENT_TYPES


def test_appearance_and_disappearance_fire_once_per_track() -> None:
    events = detect_events(features_for({1: [(10 + i * 5, 200) for i in range(10)]}), EventConfig())
    assert [e.start_frame for e in events if e.event_type == "TRACK_APPEARANCE"] == [1]
    assert [e.start_frame for e in events if e.event_type == "TRACK_DISAPPEARANCE"] == [10]


def test_stationary_event_carries_value_threshold_and_unit() -> None:
    config = EventConfig(stationary_min_frames=5)
    events = [e for e in detect_events(features_for({1: [(200.0, 200.0)] * 12}, config), config) if e.event_type == "STATIONARY"]
    assert events
    assert events[0].value == pytest.approx(0.0, abs=1e-6)
    assert events[0].threshold == pytest.approx(1.0)
    assert events[0].unit == "px_per_frame"
    assert events[0].duration_frames >= 5


def test_sprint_requires_minimum_duration() -> None:
    config = EventConfig(sprint_speed=5.0, sprint_min_frames=6)
    points = [(10.0, 100.0), (30.0, 100.0)] + [(30.0, 100.0)] * 4 + [(30.0 + i * 20.0, 100.0) for i in range(1, 9)]
    sprints = [e for e in detect_events(features_for({1: points}, config), config) if e.event_type == "SPRINT"]
    assert len(sprints) == 1
    assert sprints[0].duration_frames >= 6
    assert sprints[0].value >= 5.0


def test_right_angle_turn_is_detected_at_full_angle_and_slow_turn_is_not() -> None:
    """Heading changes come from consecutive positions: central differences would halve a corner."""
    config = EventConfig(direction_change_deg=90.0, direction_change_min_speed=2.0)
    turning = [(50.0 + i * 10, 100.0) for i in range(6)] + [(100.0, 100.0 + i * 10) for i in range(1, 6)]
    turns = [e for e in detect_events(features_for({1: turning}, config), config) if e.event_type == "DIRECTION_CHANGE"]
    assert turns, "a 90-degree turn at speed must be detected"
    assert abs(turns[0].value) >= 90.0
    assert turns[0].unit == "degrees"

    slow = [(50.0 + i * 0.5, 100.0) for i in range(6)] + [(52.5, 100.0 + i * 0.5) for i in range(1, 6)]
    assert not [
        e for e in detect_events(features_for({1: slow}, config), config) if e.event_type == "DIRECTION_CHANGE"
    ], "the same corner taken slowly must not fire"


def test_acceleration_and_deceleration_are_reported() -> None:
    config = EventConfig(accel_threshold=1.0, decel_threshold=-1.0)
    points = [(10.0 + i, 50.0) for i in range(5)] + [(14.0 + i * 10, 50.0) for i in range(1, 6)] + [(54.0, 50.0)] * 4
    events = detect_events(features_for({1: points}, config), config)
    assert [e for e in events if e.event_type == "ACCELERATION"]
    assert [e for e in events if e.event_type == "DECELERATION"]


def test_zone_transitions_fire_on_crossing() -> None:
    config = EventConfig(zone_grid=(2, 2))
    features = features_for({1: [(20.0 + i * 40, 200.0) for i in range(10)]}, config)
    events = detect_events(features, config)
    assert [e for e in events if e.event_type == "ZONE_ENTER"]
    assert [e for e in events if e.event_type == "ZONE_EXIT"]


def test_close_proximity_names_both_tracks() -> None:
    config = EventConfig(close_proximity_distance=30.0, proximity_min_frames=2)
    first = [(50.0 + i * 15.0, 200.0) for i in range(12)]
    second = [(350.0 - i * 15.0, 200.0) for i in range(12)]
    events = [e for e in detect_events(features_for({1: first, 2: second}, config), config) if e.event_type == "CLOSE_PROXIMITY"]
    assert events and events[0].related_track_id == 2
    assert events[0].unit == "px"


def test_events_are_deterministic_and_sorted() -> None:
    points = {1: [(10.0 + i * 9, 100.0) for i in range(30)], 2: [(300.0 - i * 9, 120.0) for i in range(30)]}
    features = features_for(points)
    first = detect_events(features, EventConfig())
    second = detect_events(features, EventConfig())
    assert [event.to_dict() for event in first] == [event.to_dict() for event in second]
    keys = [(event.start_frame, event.event_type, event.track_id) for event in first]
    assert keys == sorted(keys)


def test_thresholds_come_from_config_not_code() -> None:
    features = features_for({1: [(200.0, 200.0)] * 15}, EventConfig(stationary_min_frames=5))
    loose = HeuristicEventDetector(EventConfig(stationary_min_frames=5))
    strict = HeuristicEventDetector(EventConfig(stationary_min_frames=50))
    assert [e for e in loose.detect(features) if e.event_type == "STATIONARY"]
    assert not [e for e in strict.detect(features) if e.event_type == "STATIONARY"]


def test_summary_records_and_rate_normalisation() -> None:
    events = detect_events(features_for({1: [(200.0, 200.0)] * 60}), EventConfig(stationary_min_frames=5))
    summary = summarize_events(events)
    assert summary["total"] == len(events)
    assert set(summary["by_type"]) == set(EVENT_TYPES)
    assert events_to_records(events)[0]["event_type"]
    with_fps = event_rate(events, frames=60, fps=25.0)
    without_fps = event_rate(events, frames=60, fps=None)
    assert with_fps["normalised_by"] == "fps"
    assert without_fps["normalised_by"] == "frames"
    assert without_fps["fps"] is None, "an unknown frame rate must never be invented"
