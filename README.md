# CourtVision — Production Sports Video Analytics

A portfolio-grade applied computer-vision project demonstrating the complete ML lifecycle:

**data preparation → detector fine-tuning → player tracking → quantitative evaluation → latency benchmarking → FastAPI/Docker deployment**

CourtVision uses SportsMOT for basketball, volleyball, and football footage, a YOLO26 detector, and configurable multi-object trackers.

## What this proves

- sequence-level dataset splitting to prevent temporal leakage;
- MOT annotations → YOLO detection labels;
- YOLO26 fine-tuning with explicit accuracy/latency profiles;
- multi-object tracking with configurable ByteTrack / BoT-SORT / OC-SORT / TrackTrack-family trackers;
- independent detection and MOT evaluation;
- p50/p95 latency and FPS benchmarking;
- FastAPI inference service, Docker packaging, and CI tests.

> Metrics are never fabricated. The repository generates mAP, IDF1/MOTA-style tracking metrics, and hardware-specific latency from the dataset and machine actually used.

## Recommended profiles

| Profile | Detector | Tracker | Goal |
|---|---|---|---|
| Accuracy | `yolo26x.pt` or fine-tuned `best.pt` | `tracktrack.yaml` | maximize quality / ID stability |
| Throughput | `yolo26s.pt` or exported ONNX/TensorRT | `bytetrack.yaml` | lower latency and simpler deployment |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/prepare_sportsmot.py \
  --source /data/SportsMOT \
  --output data/sportsmot_yolo \
  --val-ratio 0.2

python scripts/train.py \
  --data data/sportsmot_yolo/dataset.yaml \
  --model yolo26x.pt \
  --epochs 100 \
  --imgsz 1280

python scripts/evaluate.py detect \
  --weights runs/detect/courtvision/weights/best.pt \
  --data data/sportsmot_yolo/dataset.yaml

python scripts/evaluate.py track \
  --weights runs/detect/courtvision/weights/best.pt \
  --video path/to/sequence.mp4 \
  --tracker tracktrack.yaml \
  --output runs/tracking/pred.txt

python scripts/benchmark.py \
  --weights runs/detect/courtvision/weights/best.pt \
  --video path/to/video.mp4 \
  --device 0

uvicorn courtvision.api:app --host 0.0.0.0 --port 8000
```

## Architecture

```text
SportsMOT
   │
   ├── sequence-level split (prevents adjacent-frame leakage)
   ├── MOT annotations → YOLO labels
   └── original MOT ground truth retained
          │
          ▼
YOLO26 detector ── train/val ──> best.pt
          │
          ├── detection: precision / recall / mAP
          ├── tracking: MOTA / MOTP / IDF1 / ID switches
          └── serving: p50 / p95 latency + FPS
          │
          ▼
FastAPI service ──> Docker ──> CPU / CUDA / TensorRT
```

## API

```bash
curl -X POST http://localhost:8000/analyze \
  -F "video=@match.mp4" \
  -F "tracker=tracktrack.yaml" \
  -F "conf=0.25"
```

The response includes processed frames, total detections, unique track IDs, mean confidence, elapsed time, effective FPS, and per-frame tracks.

## Tracking evaluation

First export predictions in MOTChallenge format:

```bash
python scripts/evaluate.py track \
  --weights best.pt \
  --video sequence.mp4 \
  --output runs/tracking/pred.txt
```

Then compare them with the sequence ground truth:

```bash
python scripts/evaluate.py mot \
  --ground-truth /data/SportsMOT/train/<sequence>/gt/gt.txt \
  --prediction runs/tracking/pred.txt
```

## Resume-ready description

**CourtVision — Sports Video Detection & Multi-Object Tracking**  
Built an end-to-end sports video analytics pipeline on SportsMOT, covering sequence-level data splitting, YOLO26 fine-tuning, player detection, multi-object tracking, MOT evaluation, latency benchmarking, FastAPI serving, Docker packaging, and CI tests. Designed explicit accuracy/throughput profiles and reproducible reporting for CPU/GPU deployment.

After running the project, replace generic wording with your *measured* mAP, IDF1/MOTA, p95 latency, FPS, hardware, and dataset split.

## Licensing

This is a portfolio/research implementation. Ultralytics packages and model weights have their own licensing terms; review those terms before commercial redistribution or SaaS deployment.
