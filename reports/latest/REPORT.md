# CourtVision experiment report

Generated: `2026-09-21T12:27:50+00:00`
Reports directory: `reports`

Every number below was read from an artefact produced by an actual run. Anything not measured is listed under *Not measured*.

## Quality gates

**Status: PASS**

## Runtime

| Configuration | Frames | mean ms | p50 ms | p95 ms | FPS | Device |
| --- | --: | --: | --: | --: | --: | --- |
| yolo26n_bytetrack_480 | 120 | 52.7669 | 54.0204 | 67.7676 | 18.9513 | cpu |
| yolo26n_bytetrack_640 | 120 | 57.0261 | 55.0008 | 79.6911 | 17.5358 | cpu |
| yolo26n_botsort_480 | 120 | 63.3687 | 61.4232 | 81.8950 | 15.7807 | cpu |
| yolo26n_botsort_640 | 120 | 65.1051 | 63.6321 | 83.9757 | 15.3598 | cpu |
| yolo26s_bytetrack_480 | 120 | 80.2715 | 77.1164 | 106.9734 | 12.4577 | cpu |
| yolo26s_bytetrack_640 | 120 | 120.4409 | 116.7164 | 157.8396 | 8.3028 | cpu |
| yolo26s_botsort_480 | 120 | 87.8214 | 86.7105 | 103.1007 | 11.3867 | cpu |
| yolo26s_botsort_640 | 120 | 126.9169 | 127.1454 | 149.9006 | 7.8792 | cpu |

Hardware: Windows-10-10.0.26200-SP0, CPU x16, GPU: none, CUDA: n/a

## Configuration trade-offs

- **best quality**: `yolo26n_bytetrack_480` (fps_from_mean 18.9513, end_to_end_ms_p95 67.7676)
- **best throughput**: `yolo26n_bytetrack_480` (fps_from_mean 18.9513, end_to_end_ms_p95 67.7676)
- **balanced**: `yolo26n_bytetrack_480` (fps_from_mean 18.9513, end_to_end_ms_p95 67.7676)

> Axes: mean throughput (fps_from_mean = 1000 / mean end-to-end latency) against tail latency (p95), both from the same measured runs (reports/benchmarks/*_raw.csv).
> Measured on CPU (AMD Ryzen 9 5900HS, torch 2.14.0+cpu) over a real street scene, 120 measured frames per configuration after a 5-frame warm-up, repeats with fresh tracker state.
> Only one configuration is non-dominated, so all three recommendations refer to it.

## Export & optimization


## Not measured

- **mAP50 / mAP50-95 (detection)** — no `detection/metrics.json` artefact was produced
- **precision / recall (detection)** — no `detection/metrics.json` artefact was produced
- **MOTA / MOTP (tracking)** — no `tracking/metrics.json` artefact was produced
- **IDF1 (tracking)** — no `tracking/metrics.json` artefact was produced
- **HOTA (tracking)** — no `tracking/metrics.json` artefact was produced
- **ID switches (tracking)** — no `tracking/metrics.json` artefact was produced
- **input drift** — no `monitoring/drift.json` artefact was produced
