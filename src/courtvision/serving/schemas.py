"""Request/response schemas for the HTTP API.

The large per-frame arrays are explicitly optional: ``/v1/analyze`` returns a summary
and frame-level detail must be requested. Returning tens of thousands of boxes by
default is how a service falls over under its own success.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

OutputDetail = Literal["summary", "tracks", "full"]
JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class VideoInfo(BaseModel):
    filename: str
    bytes: int = Field(ge=0)
    frames: int = Field(ge=0)
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None


class RuntimeInfo(BaseModel):
    wall_seconds: float
    frames_per_second: float
    frames_processed: int
    device: str


class DetectionSummary(BaseModel):
    total: int
    per_frame_mean: float
    per_frame_max: int
    mean_confidence: float
    confidence_min: float | None = None
    confidence_max: float | None = None


class TrackSummary(BaseModel):
    unique_tracks: int
    mean_duration_frames: float
    max_duration_frames: int
    tracks_with_gaps: int


class ModelInfo(BaseModel):
    weights: str
    tracker: str
    imgsz: int
    confidence: float
    backend: str | None = None
    parameters: int | None = None


class QualityNotes(BaseModel):
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class SystemInfo(BaseModel):
    python_version: str | None = None
    packages: dict[str, str] = Field(default_factory=dict)
    cuda_available: bool = False
    gpu: str | None = None


class AnalysisResponse(BaseModel):
    request_id: str
    video: VideoInfo
    runtime: RuntimeInfo
    detections: DetectionSummary
    tracks: TrackSummary
    model: ModelInfo
    quality: QualityNotes = Field(default_factory=QualityNotes)
    system: SystemInfo | None = None
    events: dict[str, Any] | None = None
    event_list: list[dict] | None = None
    trajectories: list[dict] | None = None
    trajectory_statistics: dict[str, Any] | None = None
    tracks_detail: list[dict] | None = None


class AnalyzeForm(BaseModel):
    """Validated form fields for ``POST /v1/analyze``."""

    tracker: str | None = None
    conf: float = Field(default=0.25, gt=0.0, le=1.0)
    imgsz: int = Field(default=640, ge=32, le=4096)
    max_frames: int | None = Field(default=None, ge=1)
    detail: OutputDetail = "summary"
    include_events: bool = True


class JobAccepted(BaseModel):
    job_id: str
    state: JobState
    submitted_at: str
    poll_url: str
    result_url: str


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    detail: OutputDetail = "summary"


class JobResult(BaseModel):
    job_id: str
    state: JobState
    result: AnalysisResponse | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    ready: bool
    model_loaded: bool
    weights: str
    device: str
    tracker: str
    tracker_supported: bool
    reasons: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
