from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .inference import VideoAnalyzer

MODEL = os.getenv("COURTVISION_MODEL", "yolo26s.pt")
DEVICE = os.getenv("COURTVISION_DEVICE", "cpu")
app = FastAPI(title="CourtVision API", version="0.1.0")
_analyzer: VideoAnalyzer | None = None


def get_analyzer() -> VideoAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = VideoAnalyzer(weights=MODEL, device=DEVICE)
    return _analyzer


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL, "device": DEVICE}


@app.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    tracker: str = Form("tracktrack.yaml"),
    conf: float = Form(0.25),
    imgsz: int = Form(960),
) -> dict:
    if not 0.0 < conf < 1.0:
        raise HTTPException(status_code=422, detail="conf must be between 0 and 1")
    if imgsz < 320 or imgsz > 2048:
        raise HTTPException(status_code=422, detail="imgsz must be in [320, 2048]")
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.TemporaryDirectory(prefix="courtvision-") as tmp:
        path = Path(tmp) / f"input{suffix}"
        content = await video.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty upload")
        path.write_bytes(content)
        summary = get_analyzer().analyze(path, tracker=tracker, conf=conf, imgsz=imgsz)
        payload = asdict(summary)
        payload["mean_confidence"] = round(payload["mean_confidence"], 6)
        payload["elapsed_seconds"] = round(payload["elapsed_seconds"], 4)
        payload["fps"] = round(payload["fps"], 3)
        return payload
