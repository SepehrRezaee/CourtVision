# CourtVision experiment report

Generated: `2026-09-21T11:44:58+00:00`
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

## Tracking

| Metric | Value |
| --- | --: |
| mota | -0.3089 |
| motp | — |
| idf1 | 0.0000 |
| idp | 0.0000 |
| idr | 0.0000 |
| precision | 0.0000 |
| recall | 0.0000 |
| num_switches | 0.0000 |
| num_fragmentations | 0.0000 |
| mostly_tracked | 0.0000 |
| mostly_lost | 5.0000 |
| mt_ratio | 0.0000 |
| ml_ratio | 1.0000 |
| id_switches_per_object | 0.0000 |
| HOTA:HOTA | 0.0000 |
| HOTA:DetA | 0.0000 |
| HOTA:AssA | 0.0000 |
| HOTA:LocA | 1.0000 |

`motp` is a mean IoU *distance* (lower is better); HOTA values are means over TrackEval's 19 alpha thresholds.

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

## Configuration trade-offs

- **best quality**: `yolo26n_botsort_480` (fps_from_mean 31.6019, end_to_end_ms_p95 38.3522)
- **best throughput**: `yolo26n_botsort_480` (fps_from_mean 31.6019, end_to_end_ms_p95 38.3522)
- **balanced**: `yolo26n_botsort_480` (fps_from_mean 31.6019, end_to_end_ms_p95 38.3522)

> NO LABELLED QUALITY AXIS IS AVAILABLE: SportsMOT was not staged in this environment, so idf1/mAP cannot be measured per configuration. This frontier therefore trades mean throughput (fps_from_mean = 1000 / mean end-to-end latency) against tail latency (p95) - a consistency axis, not an accuracy axis. A real quality/latency frontier requires `courtvision eval` output per configuration on the licensed dataset.
> Only one configuration is non-dominated, so all three recommendations refer to it.

## Input drift

| Signal | Reference n | Current n | PSI | Verdict |
| --- | --: | --: | --: | --- |
| box_area_px | 86 | 122 | 14.2856 | alert |
| box_aspect_ratio | 86 | 122 | 11.9118 | alert |
| confidence | 86 | 122 | 5.9244 | alert |
| detections_per_frame | 60 | 24 | 16.2854 | alert |
| frame_height | 1 | 1 | 27.6310 | alert |
| frame_width | 1 | 1 | 27.6310 | alert |
| inference_latency_ms | 60 | 24 | 1.4804 | alert |
| track_duration_frames | 5 | 6 | 16.2758 | alert |
| track_fragmentation | 5 | 6 | 8.5108 | alert |
| track_length | 1 | 1 | 27.6310 | alert |

Drift is a **proxy signal** about input distributions; it does not by itself demonstrate model degradation.

## Export & optimization


## Not measured

All expected artefacts are present.
