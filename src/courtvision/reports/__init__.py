"""Aggregation of experiment artefacts into JSON/CSV/Markdown reports."""

from .builder import (
    ARTIFACT_PATHS,
    EXPECTED_METRICS,
    ReportBundle,
    build_report,
    collect_artifacts,
    render_markdown,
)

__all__ = [
    "ARTIFACT_PATHS",
    "EXPECTED_METRICS",
    "ReportBundle",
    "build_report",
    "collect_artifacts",
    "render_markdown",
]
