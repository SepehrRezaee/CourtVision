# CourtVision

Sports-video analytics: detection, multi-object tracking, trajectory and event analysis,
quantitative evaluation, latency benchmarking and a hardened HTTP service.

```bash
pip install -e ".[dev]"          # runtime plus tests/lint
pip install -e ".[eval]"         # adds TrackEval, for HOTA

courtvision doctor                                    # what this environment can actually do
courtvision data validate --source /data/SportsMOT    # integrity report on the raw tree
courtvision data prepare --source /data/SportsMOT --output data/sportsmot_yolo
courtvision eval detector --weights runs/detect/courtvision/weights/best.pt \
    --data data/sportsmot_yolo --output reports/detection
courtvision benchmark --video match.mp4 \
    --weights yolo26n.pt,yolo26s.pt --trackers bytetrack.yaml,botsort.yaml --imgsz 480,640
courtvision report                                    # aggregate every artefact on disk
courtvision gate --write-baseline                     # write gates from *measured* values
courtvision serve                                     # FastAPI on :8000
```

Every subcommand shares one configuration loader and records the same provenance, so a
reported number can be traced to the commit, dependencies and hardware behind it.

## What is implemented

| Area | Status |
| --- | --- |
| Dataset validation (integrity report, per-code severities, aggregate counts) | implemented, tested |
| Leak-safe sequence-level split: seeded, stratified, content-addressed | implemented, tested |
| MOT → YOLO conversion with an explicit, recorded filter policy | implemented, tested |
| Detection evaluation: COCO mAP, confidence sweeps, PR curves, per-group | implemented, tested |
| MOT evaluation: motmetrics CLEAR/ID plus TrackEval HOTA | implemented, tested |
| Tracker backends (all 6 trackers shipped by Ultralytics) | implemented, run |
| Trajectory and kinematic analytics, frame-relative zones | implemented, tested |
| Deterministic event detection, 12 event types | implemented, tested |
| Layered latency/throughput benchmarking | implemented, measured |
| FastAPI service: upload validation, bounded frames, in-process jobs | implemented, tested |
| Runtime metrics plus input-drift comparison (PSI/KS/JS) | implemented, tested |
| Machine-readable quality gates (example vs measured baseline) | implemented, tested |
| Model export and export *validation* (ONNX/OpenVINO) | implemented; ONNX measured, fails the strict equivalence gate |
| Pose estimation | not implemented |
| Airflow DAG, MLflow experiment tracking | not implemented |

## Capability claims are queried, not hardcoded

`courtvision doctor` reads these from the installed packages, so this file cannot drift from
the runtime. Verified against `ultralytics==8.4.156`:

* trackers shipped: `botsort`, `bytetrack`, `deepocsort`, `fasttrack`, `ocsort`, `tracktrack`;
* YOLO26 assets available: 60, including `yolo26n/s/m/l/x.pt` and `-pose`, `-seg`, `-cls`, `-obb`, `-depth` variants.

The original README's tracker and model claims were checked and are **accurate** — they were
not stale. The defects in the original code were elsewhere (see "The two original defects").

## Repository layout

```text
src/courtvision/
  config.py            central config: defaults < YAML < COURTVISION_* env < CLI overrides
  repro.py             git/package/hardware provenance attached to every result
  utils.py             atomic JSON/CSV writes, structured logging, seeding
  cli.py               the `courtvision` entry point
  data/                mot.py validation.py splitting.py conversion.py sportsmot.py synthetic.py
  detection/           base.py ultralytics_detector.py trainer.py
  tracking/            base.py ultralytics_tracker.py
  analytics/           trajectories.py (kinematics, zones) events.py (event detector)
  evaluation/          detection.py (mAP) tracking.py (motmetrics + TrackEval HOTA)
  optimization/        benchmark.py export.py pareto.py
  monitoring/          metrics.py (Prometheus exposition) drift.py (PSI/KS/JS)
  serving/             schemas.py service.py api.py (FastAPI, in-process jobs)
  pipeline/            runner.py quality_gates.py
  reports/             builder.py (assembles REPORT.md from artefacts on disk)
configs/               base.yaml, accuracy/throughput/cpu profiles, quality_gates.yaml
reports/               measurement evidence from real runs
docs/                  ARCHITECTURE.md EVALUATION.md PERFORMANCE.md LIMITATIONS.md INTERVIEW_GUIDE.md
tests/                 pytest suite (data, evaluation, analytics, monitoring, serving, pipeline)
```

## Measured results

Everything below was produced by a command in this repository on this machine:
**AMD Ryzen 9 5900HS, 16 threads, no CUDA device visible to PyTorch (`torch 2.14.0+cpu`)**,
`ultralytics 8.4.156`, `numpy 2.4.6`.

### Runtime, CPU — 60 measured frames after 5 warm-up frames, warm-up excluded

| Detector | Tracker | imgsz | mean ms | p50 ms | p95 ms | FPS |
| --- | --- | --: | --: | --: | --: | --: |
| yolo26n | botsort | 480 | **31.64** | 31.95 | **38.35** | **31.60** |
| yolo26n | bytetrack | 480 | 34.92 | 34.36 | 40.59 | 28.64 |
| yolo26n | botsort | 640 | 42.63 | 42.12 | 54.01 | 23.46 |
| yolo26n | bytetrack | 640 | 43.66 | 43.28 | 50.91 | 22.90 |
| yolo26s | botsort | 480 | 55.65 | 55.89 | 66.43 | 17.97 |
| yolo26s | bytetrack | 480 | 55.99 | 56.41 | 66.07 | 17.86 |
| yolo26s | botsort | 640 | 82.45 | 83.59 | 95.74 | 12.13 |
| yolo26s | bytetrack | 640 | 89.60 | 89.47 | **107.15** | 11.16 |

### Runtime, CPU — 120 measured frames after 5 warm-up frames, warm-up excluded

Measured over a real street scene (`yolo26n.pt`, 25 fps source, per-frame detections from
the model itself), 2 repeats with fresh tracker state per repeat:

| Detector | Tracker | imgsz | mean ms | p50 ms | p95 ms | FPS |
| --- | --- | --: | --: | --: | --: | --: |
| yolo26n | bytetrack | 480 | **52.77** | 54.02 | **67.77** | **18.95** |
| yolo26n | bytetrack | 640 | 57.03 | 55.00 | 79.69 | 17.54 |
| yolo26n | botsort | 480 | 63.37 | 61.42 | 81.90 | 15.78 |
| yolo26n | botsort | 640 | 65.11 | 63.63 | 83.98 | 15.36 |
| yolo26s | bytetrack | 480 | 80.27 | 77.12 | 106.97 | 12.46 |
| yolo26s | botsort | 480 | 87.82 | 86.71 | 103.10 | 11.39 |
| yolo26s | bytetrack | 640 | 120.44 | 116.72 | 157.84 | 8.30 |
| yolo26s | botsort | 640 | 126.92 | 127.15 | 149.90 | 7.88 |

Artefacts: `reports/benchmarks/*_raw.csv` (one row per measured frame), `_summary.json`,
`system.json`. Reading: on this CPU **yolo26n + bytetrack @480** is the throughput
configuration (67.8 ms p95, 19.0 FPS); moving to yolo26s @640 costs ~2.3× the p95. That is
a measured trade-off. No GPU row exists because no CUDA device was available.

### Model export — ONNX vs the reference, on the same frames

`yolo26n.onnx` ran against the PyTorch reference over the same 12 frames of the street
clip (61 detections each): match fraction **0.885** against the 0.95 gate, mean IoU of
matches 0.976, one box off by 27 px — so the strict behavioural gate **fails**, and the
artefact is not treated as drop-in equivalent. CPU timing at imgsz 640 showed no speedup
(PyTorch 58.6 ms vs ONNX 62.1 ms per frame). Full numbers:
`reports/optimization/export.json`.

### Configuration trade-off (Pareto)

With latency on both axes there is no accuracy dimension to trade against; the frontier in
`reports/pareto.json` ranks mean throughput against tail latency across the eight
configurations above. `yolo26n_bytetrack_480` dominates — it is simultaneously the fastest
and the most consistent — so the report states that the quality, throughput and balanced
recommendations all coincide.

### The two original defects, both fixed and reproduced

1. **Tracking evaluation crashed on NumPy 2.** `evaluate_mot` called
   `motmetrics.distances.iou_matrix`, which uses `np.asfarray`, removed in NumPy 2.0, while
   `pyproject.toml` permits `numpy>=1.26,<3`. The distance matrix is now built in-repo
   (`NaN` marks pairs below the IoU threshold).
2. **Labels were corrupted at frame edges.** `mot_to_yolo` clipped the four normalised
   outputs independently, so a box at `x=180..220` in a 200 px frame became `cx=1.0, w=0.2`
   — a label 20 px outside the image, centred on the edge instead of at the true clipped
   centre 190.0. Clipping now happens in pixel space and yields `cx=0.95, w=0.10`.

## Design rules this repository follows

* **Sequence-level splitting, never frame-level.** Adjacent frames are near-duplicates;
  splitting by frame turns validation into memorisation. The split is seeded, stratified,
  content-addressed and guarded by a leakage check.
* **Pixel space is labelled as pixel space.** No m/s, metres or biomechanics, because no
  homography exists. Units live in field names and events carry their threshold.
* **Warm-up is excluded from latency**, percentiles carry their sample count, and a clip
  shorter than the warm-up reports *no frames measured* rather than a warm-up number.
* **Not measured is stated, not estimated.** `reports/latest/REPORT.md` has an explicit
  *Not measured* section listing every missing artefact.
* **Drift is a proxy signal.** Input drift is never presented as model degradation;
  labelled regression detection is a separate, direction-aware function.
