# CourtVision experiment report

Generated: `2026-09-20T21:37:12+00:00`
Reports directory: `reports`

Every number below was read from an artefact produced by an actual run. Anything not measured is listed under *Not measured*.

## Quality gates

**Status: PASS**

## Detection

| Group | Images | GT boxes | Predictions | mAP50 | mAP50-95 | Precision | Recall |
| --- | --: | --: | --: | --: | --: | --: | --: |
| basketball | 24 | 93 | 116 | 0.0876 | 0.0223 | 0.5882 | 0.1075 |
| football | 24 | 91 | 129 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| overall | 72 | 284 | 323 | 0.0282 | 0.0075 | 0.3571 | 0.0352 |
| volleyball | 24 | 100 | 78 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Reference (Ultralytics `val()`): mAP50 0.0177, mAP50-95 0.0039, precision 0.0833, recall 0.0458

Cross-check against the reference implementation: **disagree**
- `map50`: ours 0.0282, reference 0.0177, delta 0.0105
- `map50_95`: ours 0.0075, reference 0.0039, delta 0.0035

## Runtime

| Configuration | Frames | mean ms | p50 ms | p95 ms | FPS | Device |
| --- | --: | --: | --: | --: | --: | --- |
| yolo26n_bytetrack_480 | 60 | 34.9180 | 34.3596 | 40.5880 | 28.6385 | cpu |
| yolo26n_bytetrack_640 | 60 | 43.6640 | 43.2775 | 50.9068 | 22.9022 | cpu |
| yolo26n_botsort_480 | 60 | 31.6437 | 31.9492 | 38.3522 | 31.6019 | cpu |
| yolo26n_botsort_640 | 60 | 42.6325 | 42.1213 | 54.0078 | 23.4563 | cpu |
| yolo26s_bytetrack_480 | 60 | 55.9864 | 56.4059 | 66.0686 | 17.8615 | cpu |
| yolo26s_bytetrack_640 | 60 | 89.5990 | 89.4703 | 107.1485 | 11.1608 | cpu |
| yolo26s_botsort_480 | 60 | 55.6516 | 55.8916 | 66.4294 | 17.9689 | cpu |
| yolo26s_botsort_640 | 60 | 82.4495 | 83.5904 | 95.7389 | 12.1286 | cpu |

Hardware: Windows-10-10.0.26200-SP0, CPU x16, GPU: none, CUDA: n/a

## Not measured

- **MOTA / MOTP (tracking)** — no `tracking/metrics.json` artefact was produced
- **IDF1 (tracking)** — no `tracking/metrics.json` artefact was produced
- **HOTA (tracking)** — no `tracking/metrics.json` artefact was produced
- **ID switches (tracking)** — no `tracking/metrics.json` artefact was produced
- **Pareto trade-off analysis** — no `pareto.json` artefact was produced
- **input drift** — no `monitoring/drift.json` artefact was produced
- **model export validation** — no `optimization/export.json` artefact was produced
