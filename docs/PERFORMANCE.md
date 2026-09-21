# Benchmark methodology

Latency numbers are meaningless without fixed measurement rules. These are the rules, and
`optimization/benchmark.py` implements all of them; every artefact records the plan under
`plan`.

1. **Warm-up is excluded.** The first frames pay for CUDA context creation, cuDNN
   autotuning and allocator growth; timing them inflates p95 and hides steady-state cost.
   A clip shorter than the warm-up reports *no frames were measured* rather than a warm-up
   number.
2. **Cold and steady state are never mixed.** Decode happens before the timed loop
   (`preload_frames`) and is reported separately as `decode_ms_per_frame`.
3. **Stages are separated.** `preprocess_ms` / `inference_ms` are backend-reported;
   `tracking_and_post_ms` is **derived** as `end_to_end − preprocess − inference` because
   Ultralytics' `speed` dict has no tracker stage — the derivation is in the name.
4. **Raw data is kept.** Every measured frame is one row of `raw.csv` with repeat index and
   frame number; summaries are computed from those rows, never typed by hand.
5. **Percentiles carry their sample count (`n`, `frames`).** A p95 from 20 frames is an
   anecdote and is published next to `n` so it cannot be quoted as if it were not.
6. **Repeats are independent.** Each repeat constructs a fresh processor (and therefore
   fresh tracker state), so repeats are separate measurements; cross-repeat aggregates are
   means of per-repeat means so a long repeat cannot dominate a short one.
7. **FPS is 1000 / mean end-to-end latency**, so throughput and latency are internally
   consistent.
8. **The environment is recorded** (`system.json`): OS, CPU, GPU, CUDA, torch/Ultralytics
   versions, requested *and* resolved device. `resolve_device` raises rather than falling
   back, so a GPU-configured benchmark on a GPU-less machine fails instead of publishing
   CPU numbers as GPU ones.

## What was measured here

Eight configurations — `yolo26n`/`yolo26s` × `bytetrack`/`botsort` × imgsz 480/640 — on CPU
(AMD Ryzen 9 5900HS, 16 threads, `torch 2.14.0+cpu`), 5 warm-up frames, 120 measured frames
per configuration over a real street scene, 2 repeats with fresh tracker state. Raw rows,
per-config summaries and `system.json` are under `reports/benchmarks/`. Headline:
`yolo26n + bytetrack @480` at 19.0 FPS / 67.8 ms p95; `yolo26s @640` costs ~2.3× the p95.
**No GPU rows exist** — no CUDA device was available.

## Export

`courtvision export` produces the artefact; `courtvision export-validate` runs both the
reference and the exported model on the same frames, matches detections, and reports match
fraction, mean IoU of matches and maximum coordinate deviation against explicit tolerances.
The ONNX result measured here **failed** the strict default gate (match fraction 0.885 <
0.95, one box off by 27 px on 61 detections): ONNX Runtime is not drop-in equivalent to
PyTorch for this model, with differences concentrated at threshold-boundary boxes. Its CPU
timing also showed no speedup (≈0.94× at imgsz 640). Both facts are in
`reports/optimization/export.json`. TensorRT export is not available on Windows
(`courtvision doctor` prints the runtime-checked reason).

## Pareto

A configuration dominates another when it is at least as good on both axes and strictly
better on one; the frontier is the non-dominated set. Configurations missing a measurement
are excluded and listed. `reports/pareto.json` is a **throughput-consistency** frontier
(mean FPS vs p95 latency) across the eight configurations above. Its frontier is a single
point — `yolo26n_bytetrack_480` wins on both axes — and the report states that all three
recommendations therefore coincide.

## Reproducing

```bash
courtvision doctor
courtvision benchmark --video match.mp4 \
    --weights yolo26n.pt,yolo26s.pt --trackers bytetrack.yaml,botsort.yaml --imgsz 480,640
courtvision export onnx --weights yolo26n.pt
courtvision export-validate --reference yolo26n.pt --exported yolo26n.onnx \
    --video match.mp4 --benchmark
courtvision report
```

Every artefact carries the commit, dirty flag and package versions; compare runs only
against the hardware line in `system.json`.
