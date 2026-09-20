from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO


def train_detector(
    data: str | Path,
    model: str = "yolo26x.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int | float = -1,
    device: str | int = 0,
    workers: int = 8,
    seed: int = 42,
    name: str = "courtvision",
    **kwargs: Any,
):
    detector = YOLO(model)
    return detector.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        seed=seed,
        deterministic=True,
        project="runs/detect",
        name=name,
        patience=20,
        cos_lr=True,
        close_mosaic=10,
        **kwargs,
    )
