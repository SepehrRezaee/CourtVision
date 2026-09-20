# CourtVision architecture

## The data path

```text
raw sports video / SportsMOT tree
        │
        │  discover + validate (integrity report, no silent repair)
        ▼
  data/  ── mot.py ── validation.py ── splitting.py ── conversion.py ── sportsmot.py
        │        sequence-level leak-safe split, MOT→YOLO labels, dataset fingerprint
        ▼
   prepared YOLO dataset (images/ + labels/ + dataset.yaml + splits.json)
        │
        ├──────────────► detection/  UltralyticsDetector ──► train ──► best.pt
        │                       │
        │                       └──► evaluation/detection.py  (mAP, PR, conf sweep, per-sport)
        ▼
   tracking/  UltralyticsTracker  (ByteTrack / BoT-SORT / OC-SORT / DeepOCSORT / FastTrack / TrackTrack)
        │            per-video state isolation; MOT-format export
        │
        ├──► evaluation/tracking.py  (motmetrics CLEAR+ID, TrackEval HOTA)
        │
        ▼
   analytics/  trajectories → motion → zones → events
        │            pixel-space kinematics, deterministic event primitives
        ▼
   temporal/  windows → window features → EventDetector interface
        │            (heuristic detector shipped; learned classifier is a seam)
        ▼
   optimization/  benchmark → export → pareto
        │            layered latency, ONNX/OpenVINO, measured trade-offs
        ▼
   serving/  AnalysisService → FastAPI (/v1/analyze, /v1/jobs) → Docker
        │
        └──► monitoring/  runtime metrics + input-drift comparison
        │
        └──► pipeline/  quality gates + reports/ aggregation
```

## Why each component exists

**`data/`** — Every downstream number depends on label correctness, and label bugs
are silent. Parsing is kept faithful (structure validated, geometry reported not
repaired), and the split is by **sequence** rather than by frame. Frame-level
splitting of video data is the classic temporal-leakage bug: adjacent frames are
near-duplicates, so validation accuracy becomes memorisation. `build_split` is
seeded, stratified by sport, persisted with a content-addressed `split_id`, and
guarded by a leakage check that is itself tested.

**`detection/`** — Detection and tracking are evaluated separately on purpose. When
tracking metrics are poor, the first question is whether the detector is feeding the
tracker bad boxes; without an independent mAP number that question is unanswerable.
Capability checks (`available_trackers`, `verify_model_identifier`) query the
*installed* Ultralytics rather than hardcoding names, so documentation cannot drift
from the runtime.

**`tracking/`** — Ultralytics keeps tracker objects on the model, so with
`persist=True` track ids continue across videos. A service that analyses video A and
then video B would report B's players with ids continuing A's sequence. The backend
therefore resets tracker state after every call (see the module docstring and
`tests/test_tracker_backend.py`).

**`analytics/`** — Tracking output is a stream of boxes; almost every sports question
needs objects with history. Trajectories add kinematics, and everything stays in
**pixel space** with units in the field names, because without court calibration a
"speed in m/s" would be a fabricated number.

**`temporal/`** — The window abstraction is the interface a learned sequence model
would consume: a fixed, named, order-stable feature vector. The shipped detector is
deterministic and threshold-based; the learned path is *interface only* because no
labelled temporal sports-event dataset is available here. `docs/LIMITATIONS.md` says
so explicitly.

**`evaluation/`** — Two independent implementations of detection metrics are
deliberately kept: ours (which supports per-sport breakdowns and confidence sweeps
from one set of predictions) and Ultralytics' `val()` (the reference). They are
cross-checked, and a disagreement is surfaced rather than averaged away. For MOT,
`motmetrics` provides CLEAR/ID metrics and TrackEval provides HOTA.

**`optimization/`** — Accuracy, latency and cost are only comparable if measurement
methodology is fixed: warm-up excluded, stages separated, raw per-frame data kept,
percentiles reported with sample counts. Export validation compares *outputs*, not
the existence of a file.

**`serving/`** — Model loaded once, tracker state reset per request, blocking work
off the event loop, uploads validated while streaming, temporary files always
removed, per-request ids in structured logs.

**`monitoring/`** — Distinguishes **input drift** (a proxy signal) from **measured
performance degradation** (needs labels). Conflating them is the standard ML
observability mistake.

**`pipeline/`** — Machine-readable gates returning PASS/FAIL *and the failing
criteria*, with `example` thresholds clearly separated from measured `baseline`
thresholds.

## Deployment shapes

### CPU container (built and tested)

The `Dockerfile` installs only the runtime dependencies, runs as a non-root user
(uid 10001), declares a `HEALTHCHECK` on `/health`, and configures Ultralytics'
config and cache directories to a writable home. Weights are not baked in: the first
start downloads them unless a weights directory is mounted.

```bash
docker build -t courtvision:local .
docker run --rm -p 8000:8000 courtvision:local
curl localhost:8000/health
curl localhost:8000/ready
curl localhost:8000/v1/capabilities
```

`/health` is a liveness probe and deliberately says nothing about the model.
`/ready` is the readiness probe and reports *why* it is not ready (model not loaded,
unsupported tracker). An orchestrator should route traffic on `/ready`.

### GPU deployment (documented, not built here)

A CUDA image needs three things this CPU image does not have:

1. a CUDA base image, e.g. `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`;
2. the CUDA build of PyTorch, installed from the PyTorch index rather than PyPI:
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`;
3. the NVIDIA container runtime on the host, and `--gpus all` (or the equivalent
   `deploy.resources.reservations.devices` block).

At runtime set `COURTVISION_SERVING_DEVICE=0`. The device string is validated:
`courtvision.repro.resolve_device` raises instead of silently falling back to CPU, so
a GPU-configured deployment that has no visible GPU fails loudly rather than
reporting CPU latency as if it were GPU latency. Note also that `half: true` is
ignored on CPU by design, and the code forces it off rather than measuring a
different arithmetic precision than the report claims.

### Air-gapped variant

Set `YOLO_CONFIG_DIR` and `XDG_CACHE_HOME` to a mounted volume and pre-seed the
checkpoint there, or mount a directory containing the `.pt` file and point
`COURTVISION_SERVING_MODEL` at the mounted path. Nothing in the request path
downloads anything.

## Concurrency model

* The API is **single-process, multi-threaded**: `uvicorn` with one worker, and
  FastAPI/Starlette runs the synchronous, CPU-bound handlers in its threadpool.
  Analysis therefore does not block the event loop.
* The job store is an in-process thread pool with a bounded queue, a TTL and
  capacity limit. It is explicitly **not distributed**: jobs are lost on restart and
  are not shared between replicas. Scaling out requires a broker, for which the
  Airflow DAG (`dags/courtvision_pipeline.py`) is the documented path.
* PyTorch inference through one model instance is not thread-safe, and the service
  serialises access to the tracker. For higher throughput, run multiple worker
  processes behind a load balancer, each with its own model copy.

## Configuration and precedence

Configuration is resolved with one loader (`courtvision.config.load_config`):

```
dataclass defaults  <  YAML file  <  COURTVISION_* environment  <  explicit overrides
```

Explicit overrides win so that a CLI flag beats an inherited environment variable.
Unknown sections and unknown keys are hard errors, so a typo cannot silently leave a
setting at its default. `configs/base.yaml` documents every field; `configs/*.yaml`
provide profiles (accuracy / throughput / CPU) that can be layered with
`--config`.

## Provenance

`courtvision.repro.environment_snapshot()` captures timestamp, git commit *and dirty
flag*, Python version, package versions, OS, CPU, GPU, CUDA/cuDNN and seed. It is
embedded in training runs (`experiment.json`), benchmark output (`system.json`),
evaluation output (`environment.json`) and the generated report, so any reported
number can be traced to the code and hardware that produced it.

## Repository layout

```text
src/courtvision/
  config.py repro.py cli.py
  utils/            io.py (atomic writes, logging, seeding)
  data/             mot.py validation.py splitting.py conversion.py sportsmot.py synthetic.py
  detection/        base.py ultralytics_detector.py trainer.py
  tracking/         base.py ultralytics_tracker.py
  analytics/        motion.py trajectories.py zones.py events.py team_assignment.py
  temporal/         windows.py features.py event_detector.py
  evaluation/       detection.py tracking.py error_analysis.py
  optimization/     benchmark.py export.py pareto.py
  monitoring/       metrics.py drift.py
  serving/          schemas.py service.py jobs.py api.py
  pipeline/         runner.py quality_gates.py
  reports/          builder.py
  pose/             estimator.py features.py
  integrations/     mlflow_tracking.py
dags/               courtvision_pipeline.py       (optional Airflow)
configs/            base.yaml accuracy.yaml throughput.yaml cpu.yaml quality_gates*.yaml
tests/              unit + integration + model-marked end-to-end
docs/               ARCHITECTURE.md EVALUATION.md PERFORMANCE.md LIMITATIONS.md INTERVIEW_GUIDE.md
reports/            generated artefacts (never hand-edited)
```
