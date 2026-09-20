"""Environment and hardware provenance attached to every result.

A reported metric is only meaningful alongside the code revision, dependency
versions and hardware that produced it, including whether the working tree was dirty.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "ultralytics",
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "motmetrics",
    "fastapi",
    "uvicorn",
    "pydantic",
    "PyYAML",
    "trackeval",
)

_GIT_TIMEOUT = 10


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=_GIT_TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def git_info(cwd: str | Path | None = None) -> dict[str, Any]:
    cwd = Path(cwd) if cwd is not None else Path.cwd()
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None}
    status = _git(["status", "--porcelain"], cwd)
    return {
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": _cpu_count(),
        "gpu": None,
        "gpu_count": 0,
        "gpus": [],
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "torch_version": None,
    }
    try:
        import torch
    except ModuleNotFoundError:
        return info
    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_properties(index).name,
                "total_memory_gb": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2),
            }
            for index in range(torch.cuda.device_count())
        ]
        info["gpu"] = info["gpus"][0]["name"]
    return info


def _cpu_count() -> int | None:
    import os

    try:
        return os.cpu_count()
    except (NotImplementedError, OSError):  # pragma: no cover
        return None


def environment_snapshot(
    *, seed: int | None = None, command: str | None = None, cwd: str | Path | None = None
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "git": git_info(cwd),
        "packages": package_versions(),
        "hardware": hardware_info(),
        "seed": seed,
        "command": command,
    }


def resolve_device(requested: str) -> str:
    """Validate a device string against the runtime.

    Raises rather than falling back to CPU: a GPU-configured run that silently
    degrades would report CPU latency as if it were GPU latency.
    """
    requested = str(requested).strip().lower()
    if requested in {"cpu", ""}:
        return "cpu"
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed; no CUDA device is available") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} requested but torch.cuda.is_available() is False"
        )
    if requested.startswith("cuda"):
        return requested
    if requested.isdigit():
        index = int(requested)
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} requested but only {torch.cuda.device_count()} visible"
            )
        return f"cuda:{index}"
    if requested in {"mps", "xpu"}:
        return requested
    raise RuntimeError(f"Unrecognised device specification: {requested!r}")
