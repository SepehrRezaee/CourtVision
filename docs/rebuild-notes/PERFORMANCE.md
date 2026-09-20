# Benchmark methodology

Latency and throughput numbers are only comparable when the measurement rules are
fixed and stated. These are the rules CourtVision uses; the runner implements them
(`courtvision.optimization.benchmark`) and every benchmark artefact records them
under `plan`.

## Rules

**1. Warm-up is excluded.** The first frames of a run pay for CUDA context creation,
cuDNN autotuning, lazy module initialisation and allocator growth. Timing them
inflates p95 and hides steady-state cost. `warmup_frames` frames are executed and
discarded, and the plan records how many. If a clip is shorter than the warm-up, the
run reports that **no frames were measured** rather than reporting a warm-up number.

**2. Cold start and steady state are never mixed.** Decode happens before the timed
loop (`preload_frames`), so decode is not silently attributed to inference. Decode
throughput is reported separately as `decode_ms_per_frame`.

**3. Stages are separated.** Per frame:

| Column | Meaning |
| --- | --- |
| `preprocess_ms` | backend-reported preprocessing (letterbox, normalise, to-device) |
| `inference_ms` | backend-reported forward pass |
| `postprocess_ms` | backend-reported NMS (detect-only path) |
| `tracking_and_post_ms` | **derived**: `end_to_end_ms - preprocess_ms - inference_ms` |
| `end_to_end_ms` | wall clock around the whole per-frame call |
| `detections`, `tracks` | outputs in that frame, so a latency figure can be read next to its workload |

`tracking_and_post_ms` is derived because Ultralytics' `speed` dictionary has no
tracker stage. The derivation is stated in the column name and in the module
docstring rather than presented as a directly measured stage.

**4. Raw data is kept.** Every measured frame is one row in `raw.csv`, including the
`frame_index` and `repeat`. Summaries are computed from those rows. Nothing in a
summary is typed by hand, and a reader can recompute every statistic.

**5. Percentiles come with their sample count.** Every summary reports `n`. A p95 from
20 frames is an anecdote; the number is published next to `n` so it cannot be quoted
as if it were not. `p99` is included only when there are enough samples for it to
mean anything.

**6. Repeats are independent.** Each repeat gets a fresh processor (and therefore
fresh tracker state), so repeats are separate measurements rather than slices of one
long run. Cross-repeat aggregates are the mean of per-repeat means, so a longer
repeat cannot dominate a shorter one.

**7. FPS is derived from the mean end-to-end latency**, not from summing stage means,
so the throughput figure is internally consistent with the latency figure.

**8. The environment is recorded.** `system.json` captures OS, CPU count, GPU, CUDA
version, torch and Ultralytics versions, plus the requested and *resolved* device.
`resolve_device` raises rather than falling back to CPU, so a GPU-configured
benchmark on a GPU-less machine fails instead of publishing CPU numbers as GPU ones.

## What is measured

* **Detector/tracker combinations** — model size × tracker × image size, as a sweep
  (`--weights a,b --trackers x,y --imgsz 480,640`).
* **Export backends** — reference PyTorch vs ONNX/OpenVINO on identical frames, with
  the same warm-up rule. `benchmark_export_pair` returns per-repeat means and the
  measured ratio; no speedup is asserted anywhere that was not measured in that run.
* **Layered stages** as above, so a bottleneck can be attributed rather than guessed.

The export comparison for TensorRT is **not** produced here: Ultralytics does not
support TensorRT export on Windows, and this environment is Windows. That path is
reported as `supported path — not benchmarked on current hardware`
(`courtvision doctor` prints the reason from a runtime capability query).

## Artefacts

```text
reports/benchmarks/
    <label>_raw.csv        one row per measured frame
    <label>_summary.json   plan, config, system, summary
    <label>_summary.csv    per-repeat summary rows
    system.json            hardware/software context
reports/pareto.json        non-dominated configurations and recommendations
reports/pareto.csv         every configuration with an on_frontier flag
reports/pareto.png         chart (written only if matplotlib is installed)
```

## Pareto analysis

A configuration *dominates* another when it is at least as good on **both** axes and
strictly better on one. The frontier is the set of non-dominated configurations.
Quality is normally `idf1` (or `map50_95`) and cost is normally `p95_latency_ms`; if
only throughput is available, the report says so in a note instead of inventing a
quality axis.

Configurations missing either measurement are **excluded and listed**, never silently
dropped: an unmeasured point is not a dominating point. The report names a
best-quality, a best-throughput and a balanced configuration; when the frontier has
a single point, it says all three are that point rather than pretending a choice was
made.

## Reproducing the numbers in this repository

```bash
# 1. Environment and capability inventory (queries installed packages)
courtvision doctor

# 2. Latency/throughput sweep on a clip
courtvision benchmark --video clip.mp4 --weights yolo26n.pt,yolo26s.pt \
    --trackers bytetrack.yaml,botsort.yaml --imgsz 480,640 --device 0

# 3. Export and verify behaviour against the reference model
courtvision export onnx --weights yolo26n.pt --imgsz 640
courtvision export-validate --reference yolo26n.pt --exported yolo26n.onnx \
    --video clip.mp4 --benchmark

# 4. Aggregate everything that was produced
courtvision report
```

Every artefact carries the commit, the dirty flag and the package versions, so a
result measured on another machine can be compared only when the hardware line in
`system.json` is compared too.
