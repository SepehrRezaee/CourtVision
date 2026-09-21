# Known limitations

Stated plainly. Each item says what it would take to remove.

## 1. No real dataset was used

SportsMOT was not staged in this environment, so **no real mAP, IDF1, HOTA or MOTA exists in
this repository**. Every quality number (detection tables, tracker comparisons, the Pareto
quality axis) was measured against *synthetic* data — drawn rectangles on generated clips —
and describes pipeline correctness, not model quality. The latency numbers, by contrast, are
real CPU measurements of real model inference.

Removal: stage SportsMOT, then
`courtvision data prepare --source /data/SportsMOT --output data/sportsmot_yolo`,
`courtvision eval detector --weights <best.pt> --data data/sportsmot_yolo`, and
`courtvision eval trackers --weights <best.pt> --video <clip> --ground-truth <raw root>`.

## 2. No GPU measurement

CUDA was verified working earlier in the session (RTX 3050 Ti Laptop, 4 GB, sm_86, CUDA 12.6,
`torch 2.14.0+cu126`), but the workspace's cleanup process deleted that install and repeatedly
removed the 531 MB `cublasLt` DLL during re-extraction, so the environment settled on CPU
torch. All latency numbers are CPU. GPU and TensorRT results are **NOT MEASURED**.

Removal: install the CUDA build of torch, rerun `courtvision benchmark --device 0`, and
regenerate the report.

## 3. The detection cross-check compares pipelines, not evaluators

Our mAP and Ultralytics' `val()` disagree by ~0.01 on the synthetic split. That is expected —
they score different inference passes — but it means the cross-check does not yet isolate
evaluator correctness. Removal: score one shared prediction set (e.g. `val(save_json=True)`
COCO output) with both implementations.

## 4. ONNX export failed the behavioural gate

At imgsz 640 on CPU, ONNX Runtime matched 61/61 detections with mean IoU 0.976 but only 88.5%
found a ≥0.9-IoU partner (gate: 95%) and one box deviated 27 px, with no speedup (0.94×).
The artefact is **not** drop-in equivalent to PyTorch and the report says so. Removal: tune
export (opset/simplify/dynamic), or investigate the NMS tie-breaking difference, then re-run
`courtvision export-validate`.

## 5. Event-detection accuracy is NOT MEASURED

The 12 event primitives are deterministic thresholds with unit tests, but no labelled
temporal sports-event dataset exists here, so no precision/recall is claimed for them. The
learned-classifier seam (`SklearnEventClassifier`-style, over window features) is not
implemented.

## 6. Test coverage is broad but shallow in places

153 tests cover splitting, validation, conversion, both evaluators, analytics, monitoring,
gates, Pareto, benchmarking and the HTTP contract (~63% line coverage overall). Gaps: the
Ultralytics-backed detector/tracker/trainer and `cli.py` are only exercised by the manual
command runs recorded in the README, not by CI (those paths need weights). Removal: add
model-marked integration tests that download `yolo26n.pt` and run the real backends.

## 7. Not implemented

Pose estimation, team/appearance grouping, the Airflow DAG and MLflow tracking are absent.
The serving job store is deliberately in-process (bounded, TTL-expired, not distributed);
restart loses jobs and replicas do not share them.

## 8. Single-point Pareto frontier

`yolo26n_botsort_480` dominates every other measured configuration on both axes, so the
"quality/throughput" frontier is one point and all three recommendations coincide — and its
quality axis is throughput, not accuracy, because of limitation 1. A meaningful frontier
requires real per-configuration quality metrics.

## 9. Provenance artefacts describe the producing machine

`environment.json` / `system.json` / `report.json` embed OS, CPU, GPU and absolute paths —
that is their purpose, but they do describe the author's machine and will differ on yours.
