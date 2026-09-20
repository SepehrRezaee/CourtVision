"""CourtVision sports-video analytics package.

The version is read from the installed distribution metadata so it cannot drift from
``pyproject.toml`` the way a hardcoded literal does. The fallback covers running from a
source tree that was never installed.
"""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("courtvision")
except metadata.PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
