# Known limitations

Stated plainly. Each item says what it would take to remove.

## 1. No GPU measurement

CUDA was verified working earlier in the session (RTX 3050 Ti Laptop, 4 GB, sm_86, CUDA 12.6,
`torch 2.14.0+cu126`), but the workspace's cleanup process deleted that install and repeatedly
removed the 531 MB `cublasLt` DLL during re-extraction, so the environment settled on CPU
torch. All published latency numbers are CPU (AMD Ryzen 9 5900HS, 16 threads). GPU and
TensorRT results are not published.

Removal: install the CUDA build of torch, rerun
`courtvision benchmark --device 0 --video match.mp4`, and regenerate the report.

## 2. ONNX export fails the strict behavioural gate

At imgsz 640 on CPU, ONNX Runtime matched PyTorch's 61 detections with mean IoU 0.976, but
only 88.5% found a ≥0.9-IoU partner (gate: 95%) and one box deviated by 27 px, with no CPU
speedup (0.94×). The artefact is not drop-in equivalent to PyTorch and `reports/optimization/
export.json` records the failure. Removal: tune export options (opset/simplify/dynamic) or
investigate the NMS tie-breaking difference, then rerun `courtvision export-validate`.

## 3. Pareto frontier is throughput-only

`reports/pareto.json` ranks mean throughput against tail latency across the benchmarked
configurations. Accuracy is not an axis: per-configuration mAP/IDF1 requires evaluation on a
labelled dataset, which is a separate run (`courtvision eval detector` /
`courtvision eval trackers` on the staged dataset). With latency on both axes the frontier is
a single point — `yolo26n_bytetrack_480` wins on both — and the report says the three
recommendations coincide.

## 4. Detector/tracker accuracy is not published

No mAP, IDF1, HOTA or MOTA numbers are published because accuracy evaluation requires the
staged SportsMOT dataset and a fine-tuned checkpoint. The evaluation implementations are
tested (see `tests/test_evaluation.py`: hand-computed mAP cases, motmetrics and TrackEval
agreement on constructed sequences); publishing numbers is a matter of running
`courtvision eval detector` and `courtvision eval trackers` on the staged dataset.

## 5. Test coverage is broad but shallow in places

155 tests cover splitting, validation, conversion, both evaluators, analytics, monitoring,
gates, Pareto, benchmarking and the HTTP contract (~63% line coverage overall). Gaps: the
Ultralytics-backed detector/tracker/trainer and `cli.py` are exercised by the manual command
runs recorded in the README rather than by CI, because those paths need weights. Removal:
add weight-marked integration tests that download `yolo26n.pt` and run the real backends.

## 6. Not implemented

Pose estimation, team/appearance grouping, the Airflow DAG and MLflow tracking are absent.
The serving job store is deliberately in-process (bounded, TTL-expired, not distributed);
restart loses jobs and replicas do not share them.

## 7. Provenance artefacts describe the producing machine

`environment.json` / `system.json` / `report.json` embed OS, CPU, GPU and absolute paths —
that is their purpose, but they describe the machine that produced a run and will differ on
yours.
