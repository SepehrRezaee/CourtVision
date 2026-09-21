# Architecture

## The data path

```text
raw sports video / SportsMOT tree
    │  courtvision.data.validation     integrity report, no silent repair
    ▼
sequence-level split (splitting.py)   seeded, sport-stratified, content-addressed id
    │  conversion.py                  MOT → YOLO labels, recorded filter policy
    ▼
prepared YOLO dataset (images/ labels/ dataset.yaml splits.json prepare_report.json)
    │
    ├── detection/   UltralyticsDetector ── trainer.py ──► best.pt + experiment.json
    │        └────── evaluation/detection.py   mAP, sweeps, PR curves, per-sport
    ▼
tracking/  UltralyticsTracker         ByteTrack / BoT-SORT / OC-SORT / DeepOCSORT /
    │                                 FastTrack / TrackTrack; MOT-format export
    ├── evaluation/tracking.py        motmetrics CLEAR+ID, TrackEval HOTA
    ▼
analytics/  trajectories.py (kinematics, zones) events.py (12 event primitives)
    ▼
optimization/ benchmark.py export.py pareto.py
    ▼
serving/    AnalysisService → FastAPI (/v1/analyze, /v1/jobs) → Docker
    │
    ├── monitoring/   runtime metrics (Prometheus text) + input drift (PSI/KS/JS)
    └── pipeline/     quality gates + reports/builder.py → REPORT.md
```

## Why each piece exists

**`data/`** — every downstream number depends on label correctness, and label bugs are
silent. Parsing is faithful (structure validated, geometry reported, never repaired), the
split is by **sequence** rather than frame — frame-level splitting of video is the classic
leakage bug that turns validation into memorisation — and `splits.json` carries a
content-addressed `split_id` so an experiment can name its exact partition.

**`detection/` separate from `tracking/`** — detection and tracking are evaluated
independently so the question "is the tracker bad, or is the detector feeding it bad boxes?"
is answerable. Capability checks query the *installed* Ultralytics rather than hardcoding
names, so documentation cannot drift from the runtime (`courtvision doctor`).

**`evaluation/`** — the in-repo detection evaluator supports per-sport breakdowns and
confidence sweeps from one prediction set, and Ultralytics' `val()` is kept as a reference
to cross-check against. For MOT, `motmetrics` provides CLEAR/identity metrics and TrackEval
provides HOTA; both are third-party implementations rather than in-house rewrites. The IoU
distance matrix is built in-repo because `motmetrics.distances.iou_matrix` uses
`np.asfarray`, removed in NumPy 2.0.

**`analytics/`** — tracking output is a stream of boxes; analytics needs objects with
history. Everything stays in **pixel space** with units in the field names: there is no
court calibration or homography, so a metre-based speed would be fabricated.

**`optimization/`** — accuracy/latency/cost comparisons are only meaningful with a fixed
methodology: warm-up excluded, stages separated, raw per-frame rows kept, percentiles
reported with their sample count, exports validated behaviourally instead of by file
existence.

**`serving/`** — model loaded once at startup (a broken checkpoint fails the deploy, not a
request), tracker state reset per call, analysis handlers synchronous so inference runs in
Starlette's threadpool rather than on the event loop, uploads validated and size-capped
while streaming, per-request temp dirs always removed, request ids in structured logs.

**`pipeline/` + `reports/`** — gates return PASS/FAIL with the exact failing criteria and
separate `example` (placeholder) from `baseline` (measured) thresholds; the report is built
only from artefacts on disk and lists everything absent under *Not measured*.

## Deployment

### CPU container

The Dockerfile installs runtime dependencies only, runs as uid 10001, sets
`YOLO_CONFIG_DIR`/`XDG_CACHE_HOME` to writable locations, and declares a HEALTHCHECK on
`/health`. Weights are not baked in — the first start downloads the configured checkpoint
unless you mount a weights directory.

```bash
docker build -t courtvision:local .
docker run --rm -p 8000:8000 courtvision:local
curl localhost:8000/health   # liveness
curl localhost:8000/ready    # readiness (includes the reason when not ready)
```

### GPU

Use a CUDA base image, install the CUDA build of torch
(`--index-url https://download.pytorch.org/whl/cu126`), run with the NVIDIA container
runtime, and set `COURTVISION_SERVING_DEVICE=0`. Device strings are validated —
`resolve_device` raises rather than falling back to CPU, so GPU-configured runs cannot
silently publish CPU numbers. `half: true` is forced off on CPU.

### Concurrency model

Single process, multi-threaded: synchronous routes run in the threadpool, one model per
process, tracker access serialised. The job store is in-process and bounded (max jobs, TTL,
worker count); jobs are lost on restart and are not shared between replicas. Scale-out means
multiple processes behind a load balancer, each with its own model.

## Configuration

```
dataclass defaults < configs/base.yaml < COURTVISION_<SECTION>_<KEY> env < CLI overrides
```

Unknown sections and keys are hard errors, so a typo cannot silently leave a setting at its
default. Profiles (`accuracy.yaml`, `throughput.yaml`, `cpu.yaml`) are overlays loaded with
`--config`.

## Provenance

`courtvision.repro.environment_snapshot()` captures timestamp, git commit **and dirty
flag**, Python, package versions, OS/CPU/GPU/CUDA and the seed. It is embedded in training
runs (`experiment.json`), benchmarks (`system.json`), evaluations (`environment.json`) and
the aggregate report.
