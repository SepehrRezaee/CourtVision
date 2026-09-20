# CourtVision

Sports-video analytics: detection, multi-object tracking, trajectory and event analysis,
quantitative evaluation, latency benchmarking and a hardened HTTP service.

```bash
pip install -e ".[dev]"          # runtime plus tests/lint
pip install -e ".[eval]"         # adds TrackEval, for HOTA

courtvision doctor                                    # what this environment can actually do
courtvision data synth --output data/synth            # synthetic dataset (no licensed download)
courtvision data prepare --source data/synth --output data/prepared
courtvision eval detector --weights yolo26n.pt --data data/prepared --output reports/detection
courtvision data video --output data/clips            # synthetic clip plus matching MOT ground truth
courtvision benchmark --video data/clips/synthetic_track.mp4 \
    --weights yolo26n.pt,yolo26s.pt --trackers bytetrack.yaml,botsort.yaml --imgsz 480,640
courtvision report                                    # aggregate every artefact on disk
courtvision gate --write-baseline                     # write gates from *measured* values
courtvision serve                                     # FastAPI on :8000
```

## What is implemented

| Area | Status |
| --- | --- |
| Dataset validation (integrity report, per-code severities, aggregate counts) | implemented, verified |
| Leak-safe sequence-level split: seeded, stratified, content-addressed | implemented, verified |
| MOT → YOLO conversion with an explicit, recorded filter policy | implemented, verified |
| Detection evaluation: COCO mAP, confidence sweeps, PR curves, per-group | implemented, cross-checked |
| MOT evaluation: motmetrics CLEAR/ID plus TrackEval HOTA | implemented, validated on known answers |
| Tracker backends (all 6 trackers shipped by Ultralytics) | implemented, run |
| Trajectory and kinematic analytics, frame-relative zones | implemented |
| Deterministic event detection, 12 event types | implemented |
| Layered latency/throughput benchmarking | implemented, measured |
| FastAPI service: upload validation, bounded frames, in-process jobs | implemented |
| Runtime metrics plus input-drift comparison (PSI/KS/JS) | implemented |
| Machine-readable quality gates (example vs measured baseline) | implemented, exercised |
| Model export and export *validation* (ONNX/OpenVINO) | **not executed here** — see Limitations |
| Pose estimation | **not implemented** |
| Airflow DAG, MLflow experiment tracking | **not implemented** |
| Test suite for the new modules | **not written** — see Limitations 3 |

## Capability claims are queried, not hardcoded

`courtvision doctor` reads these from the installed packages, so this file cannot drift from
the runtime. Verified against `ultralytics==8.4.156`:

* trackers shipped: `botsort`, `bytetrack`, `deepocsort`, `fasttrack`, `ocsort`, `tracktrack`;
* YOLO26 assets available: 60, including `yolo26n/s/m/l/x.pt` and `-pose`, `-seg`, `-cls`, `-obb`, `-depth` variants.

The original README's tracker and model claims were checked and are **accurate** — they were
not stale. The defects in the original code were elsewhere (see the audit notes below).

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

Artefacts: `reports/benchmarks/*_raw.csv` (one row per measured frame), `_summary.json`,
`system.json`. Reading: on this CPU **yolo26n + botsort @480** is the throughput
configuration (38.4 ms p95, 31.6 FPS); moving to yolo26s @640 costs about 2.8× the p95.
That is a measured trade-off. No GPU row exists because no CUDA device was available.

### Detection — synthetic validation split, explicitly NOT model quality

72 images, 284 ground-truth boxes, 323 predictions, `yolo26n.pt` @640 on CPU:

| Group | Images | mAP50 | mAP50-95 | Precision | Recall |
| --- | --: | --: | --: | --: | --: |
| basketball | 24 | 0.0876 | 0.0223 | 0.5882 | 0.1075 |
| football | 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| volleyball | 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| overall | 72 | 0.0282 | 0.0075 | 0.3571 | 0.0352 |

**These are not model-quality numbers.** The labels come from the synthetic generator and the
imagery is coloured rectangles, so a COCO detector has almost nothing to find. They show the
evaluation path runs end to end and serialises metrics.

The informative part of that run is the cross-check: our mAP50 **0.0282** against
Ultralytics' `val()` **0.0177** — reported as `disagree` at a 0.01 tolerance. That comparison
runs two *end-to-end* pipelines (ours scores our own `predict()` output; the reference
re-runs inference inside `val()` with its own batching and rect settings), so a small delta
is expected and is **not** evidence that either evaluator is wrong. A clean evaluator
comparison requires one shared prediction set scored by both, which is not implemented.

### MOT metric correctness on analytically known cases

| Case | MOTA | IDF1 | HOTA | AssA |
| --- | --: | --: | --: | --: |
| ground truth scored against itself | 1.000 | 1.000 | 1.000 | 1.000 |
| identity permutation of the predictions | -0.300 | 0.350 | 0.194 | 0.186 |

Threshold semantics are pinned as well: a box at IoU 0.7 scores AP 1.0 at threshold 0.5 and
0.0 at 0.75, and MOTA is 1.0 at threshold 0.5 and -1.0 at 0.9. HOTA comes from TrackEval's
own implementation.

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

## Limitations

1. **No real dataset was used.** SportsMOT was unavailable, so no real mAP, IDF1, HOTA or
   MOTA exists in this repository — every quality number above is synthetic. With the
   dataset staged, run `courtvision data prepare --source /data/SportsMOT --output data/sportsmot_yolo`,
   then `courtvision eval detector --weights <best.pt> --data data/sportsmot_yolo` and
   `courtvision eval trackers --weights <best.pt> --video <clip> --ground-truth <raw root>`.
2. **No GPU measurement.** CUDA was verified working earlier in the session (RTX 3050 Ti,
   4 GB, sm_86, CUDA 12.6) with the CUDA build of PyTorch, but this environment deleted that
   install, so all numbers above are CPU. GPU and TensorRT results are `NOT MEASURED`.
3. **No test suite for the new modules.** Only the repository's original tests
   (`tests/test_data.py`, `tests/test_metrics.py`) exist, and they pass. Correctness evidence
   for the new code is the analytical MOT/detection results above plus per-command
   verification, not pytest coverage. This is the largest remaining gap.
4. **Event-detection accuracy is NOT MEASURED.** No labelled temporal sports-event dataset is
   available here. The detector is deterministic; no accuracy is claimed.
5. **Export paths unverified.** ONNX/OpenVINO export and numerical validation are implemented
   but were not executed in the final environment. `courtvision doctor` reports TensorRT
   unavailable on Windows.
6. **The job store is in-process.** Jobs are lost on restart and are not shared between
   replicas — documented rather than hidden behind a fake distributed layer.
7. **The detection cross-check compares two pipelines**, not two evaluators on shared
   predictions (see above).
8. **`docs/rebuild-notes/` is design rationale, not status.** Those documents describe
   methodology written for an implementation that was destroyed mid-session and later
   rebuilt; this README is the accurate statement of what exists now.

## Third-party licensing

Ultralytics packages, Ultralytics model weights and the SportsMOT dataset carry their own
licence terms; review them before commercial or SaaS use. TrackEval is pinned to a git commit
because it is not distributed on PyPI.
