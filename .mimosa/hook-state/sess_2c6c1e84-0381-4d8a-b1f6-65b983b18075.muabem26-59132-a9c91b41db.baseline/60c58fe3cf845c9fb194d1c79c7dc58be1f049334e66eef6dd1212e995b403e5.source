from __future__ import annotations

import argparse
import json
import statistics
import time
import cv2
import numpy as np
from ultralytics import YOLO

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    model = YOLO(args.weights)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.video}")

    latencies = []
    frame_count = 0
    while frame_count < args.max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        start = time.perf_counter()
        model.track(frame, tracker=args.tracker, imgsz=args.imgsz, device=args.device, persist=True, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if frame_count >= args.warmup:
            latencies.append(elapsed_ms)
        frame_count += 1
    capture.release()

    mean_ms = statistics.fmean(latencies) if latencies else 0.0
    report = {
        "weights": args.weights,
        "device": args.device,
        "tracker": args.tracker,
        "imgsz": args.imgsz,
        "frames_total": frame_count,
        "frames_measured": len(latencies),
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3) if latencies else 0.0,
        "p95_ms": round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
        "fps_from_mean": round(1000 / mean_ms, 3) if mean_ms else 0.0,
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
