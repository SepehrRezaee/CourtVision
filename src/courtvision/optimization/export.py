"""Model export and *behavioural* validation of exported artefacts.

Exporting is only useful if the exported model computes the same thing. This module
therefore does not stop at "the file exists": it runs the artefact on real frames,
matches its detections against the reference model's, and reports agreement against
explicit tolerances. No speedup is reported that was not measured in the same run.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.detection import iou_matrix
from ..repro import hardware_info, resolve_device
from ..utils import ensure_dir, write_json

EXPORT_FORMATS = ("onnx", "openvino", "torchscript", "engine", "tflite")

#: Ultralytics' TensorRT path is POSIX-only, so 'engine' is never attempted on Windows.
LINUX_ONLY_FORMATS = ("engine",)


@dataclass
class ExportCapability:
    format: str
    supported_here: bool
    reason: str = ""
    required_package: str | None = None

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "supported_here": self.supported_here,
            "reason": self.reason,
            "required_package": self.required_package,
        }


def export_capabilities() -> list[ExportCapability]:
    """Which export formats can run here, and precisely why not when they cannot."""
    is_windows = platform.system() == "Windows"
    has_cuda = bool(hardware_info().get("cuda_available"))
    capabilities: list[ExportCapability] = []
    for export_format in EXPORT_FORMATS:
        if export_format == "torchscript":
            capabilities.append(ExportCapability(export_format, True, "always available with torch"))
            continue
        if export_format == "engine":
            if is_windows:
                reason = "TensorRT export is not supported by Ultralytics on Windows"
            elif not has_cuda:
                reason = "no CUDA device is visible to PyTorch"
            else:
                reason = "TensorRT is not installed in this environment"
            capabilities.append(ExportCapability(export_format, False, reason, "tensorrt"))
            continue
        package = {"onnx": "onnx", "openvino": "openvino", "tflite": "tensorflow"}[export_format]
        available = importlib.util.find_spec(package) is not None
        capabilities.append(
            ExportCapability(
                export_format,
                available,
                "" if available else f"the {package!r} package is not installed",
                None if available else package,
            )
        )
    return capabilities


@dataclass
class ExportResult:
    format: str
    ok: bool
    path: str | None = None
    seconds: float = 0.0
    imgsz: int | None = None
    half: bool = False
    device: str | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "ok": self.ok,
            "path": self.path,
            "seconds": round(self.seconds, 3),
            "imgsz": self.imgsz,
            "half": self.half,
            "device": self.device,
            "notes": self.notes,
            "error": self.error,
        }


def export_model(
    weights: str | Path,
    *,
    format: str = "onnx",  # noqa: A002 - matches the library/CLI parameter name
    imgsz: int = 640,
    device: str = "cpu",
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
    opset: int | None = None,
    output_dir: str | Path | None = None,
) -> ExportResult:
    """Export a checkpoint and return the produced artefact."""
    from ultralytics import YOLO

    if format in LINUX_ONLY_FORMATS and platform.system() == "Windows":
        return ExportResult(
            format=format,
            ok=False,
            notes=["NOT ATTEMPTED"],
            error=(
                "TensorRT export is not supported by Ultralytics on Windows; run on a Linux host "
                "with an NVIDIA GPU and the tensorrt package installed."
            ),
        )

    kwargs: dict[str, Any] = {"format": format, "imgsz": imgsz, "device": resolve_device(device), "half": half}
    if format == "onnx":
        kwargs.update({"dynamic": dynamic, "simplify": simplify})
        if opset is not None:
            kwargs["opset"] = opset

    started = time.perf_counter()
    try:
        produced = Path(str(YOLO(str(weights)).export(**kwargs)))
    except Exception as exc:  # noqa: BLE001 - any backend failure is reportable, not fatal
        return ExportResult(
            format=format,
            ok=False,
            imgsz=imgsz,
            half=half,
            device=device,
            seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    elapsed = time.perf_counter() - started

    target = produced
    if output_dir is not None:
        destination = ensure_dir(output_dir) / produced.name
        if produced.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(produced, destination)
        else:
            shutil.copy2(produced, destination)
        target = destination

    return ExportResult(
        format=format,
        ok=True,
        path=str(target),
        seconds=elapsed,
        imgsz=imgsz,
        half=half,
        device=device,
        notes=["Exported; run validate_export to compare behaviour against the reference."],
    )


@dataclass
class ExportValidation:
    """Numeric comparison between a reference model and its exported artefact."""

    reference: str
    exported: str
    frames: int
    reference_detections: int
    exported_detections: int
    matched: int
    match_fraction: float
    mean_iou_matched: float
    max_coordinate_deviation_px: float
    tolerances: dict
    passed: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "exported": self.exported,
            "frames": self.frames,
            "reference_detections": self.reference_detections,
            "exported_detections": self.exported_detections,
            "matched": self.matched,
            "match_fraction": round(self.match_fraction, 6),
            "mean_iou_matched": round(self.mean_iou_matched, 6),
            "max_coordinate_deviation_px": round(self.max_coordinate_deviation_px, 4),
            "tolerances": self.tolerances,
            "passed": self.passed,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())


def validate_export(
    reference_weights: str | Path,
    exported_weights: str | Path,
    frames: Sequence[np.ndarray],
    *,
    imgsz: int = 640,
    device: str = "cpu",
    conf: float = 0.25,
    min_match_iou: float = 0.9,
    min_match_fraction: float = 0.95,
    max_coordinate_deviation_px: float = 2.0,
) -> ExportValidation:
    """Run both models on the same frames and compare detections numerically."""
    from ultralytics import YOLO

    if not frames:
        raise ValueError("Export validation needs at least one frame")

    def predict(weights: str | Path) -> list[np.ndarray]:
        model = YOLO(str(weights))
        outputs: list[np.ndarray] = []
        for frame in frames:
            result = model.predict(source=frame, imgsz=imgsz, conf=conf, device=resolve_device(device), verbose=False)[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                outputs.append(np.zeros((0, 5), dtype=float))
                continue
            outputs.append(
                np.hstack([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy().reshape(-1, 1)])
            )
        return outputs

    reference_outputs = predict(reference_weights)
    exported_outputs = predict(exported_weights)

    matched = reference_total = exported_total = 0
    ious: list[float] = []
    deviations: list[float] = []
    for reference_boxes, exported_boxes in zip(reference_outputs, exported_outputs, strict=True):
        reference_total += len(reference_boxes)
        exported_total += len(exported_boxes)
        if len(reference_boxes) == 0 or len(exported_boxes) == 0:
            continue
        overlap = iou_matrix(reference_boxes[:, :4], exported_boxes[:, :4])
        for index in range(len(reference_boxes)):
            best = int(np.argmax(overlap[index]))
            if overlap[index, best] >= min_match_iou:
                matched += 1
                ious.append(float(overlap[index, best]))
                deviations.append(float(np.abs(reference_boxes[index, :4] - exported_boxes[best, :4]).max()))

    match_fraction = matched / reference_total if reference_total else 1.0
    mean_iou = float(np.mean(ious)) if ious else 1.0
    max_deviation = float(np.max(deviations)) if deviations else 0.0
    notes: list[str] = []
    if reference_total == 0:
        notes.append(
            "The reference model produced no detections on these frames, so agreement is "
            "untestable; use frames containing the objects of interest."
        )
    return ExportValidation(
        reference=str(reference_weights),
        exported=str(exported_weights),
        frames=len(frames),
        reference_detections=reference_total,
        exported_detections=exported_total,
        matched=matched,
        match_fraction=match_fraction,
        mean_iou_matched=mean_iou,
        max_coordinate_deviation_px=max_deviation,
        tolerances={
            "min_match_iou": min_match_iou,
            "min_match_fraction": min_match_fraction,
            "max_coordinate_deviation_px": max_coordinate_deviation_px,
            "conf": conf,
            "imgsz": imgsz,
        },
        passed=(
            reference_total > 0
            and match_fraction >= min_match_fraction
            and mean_iou >= min_match_iou
            and max_deviation <= max_coordinate_deviation_px
        ),
        notes=notes,
    )


def benchmark_export_pair(
    reference_weights: str | Path,
    exported_weights: str | Path,
    frames: Sequence[np.ndarray],
    *,
    imgsz: int = 640,
    device: str = "cpu",
    warmup: int = 5,
    repeats: int = 3,
) -> dict:
    """Measure both models on identical frames with warm-up excluded."""
    from ultralytics import YOLO

    timings: dict[str, list[float]] = {}
    for label, weights in (("reference", reference_weights), ("exported", exported_weights)):
        model = YOLO(str(weights))
        for _ in range(warmup):
            model.predict(source=frames[0], imgsz=imgsz, device=resolve_device(device), verbose=False)
        per_repeat: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            for frame in frames:
                model.predict(source=frame, imgsz=imgsz, device=resolve_device(device), verbose=False)
            per_repeat.append((time.perf_counter() - started) / len(frames) * 1000.0)
        timings[label] = per_repeat

    summary = {
        label: {
            "per_frame_ms_repeats": [round(value, 4) for value in values],
            "mean_ms": round(float(np.mean(values)), 4),
            "std_ms": round(float(np.std(values)), 4),
            "fps": round(1000.0 / float(np.mean(values)), 4),
        }
        for label, values in timings.items()
    }
    reference_ms = summary["reference"]["mean_ms"]
    exported_ms = summary["exported"]["mean_ms"]
    summary["speedup_exported_vs_reference"] = round(reference_ms / exported_ms, 4) if exported_ms > 0 else None
    summary["device"] = device
    summary["note"] = (
        "Speedup is reference_ms / exported_ms on identical frames, image size and device, with "
        "warm-up excluded. It is a measured ratio for this hardware, not a portable claim."
    )
    return summary
