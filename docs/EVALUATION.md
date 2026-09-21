# Evaluation methodology

Every metric CourtVision reports is defined here. If a number is not defined here, it is
not reported.

## Detection

Two evaluators run against the same prepared split:

| Evaluator | Purpose |
| --- | --- |
| `courtvision.evaluation.detection` | per-sport breakdowns, confidence sweeps, PR curves — from **one** prediction set |
| Ultralytics `model.val()` | reference implementation, cross-checked |

Predictions are computed once and re-scored per group, so breakdowns never differ because
of inference variance.

**Matching.** Per IoU threshold, predictions are sorted by descending confidence and each is
greedily matched to the highest-IoU unmatched ground-truth box at or above the threshold
(the VOC/COCO convention used by Ultralytics and pycocotools).

**mAP.** 101-point COCO interpolation on the monotone precision envelope. With few ground
truths this quantises: two boxes with one detection give AP = 51/101 ≈ 0.505, not exactly
0.5, because the r = 0 point counts.

| Metric | Definition |
| --- | --- |
| `map50` / `map50_95` | AP at IoU 0.50; mean over IoU 0.50…0.95 in steps of 0.05 |
| `precision`, `recall` | at the **best-F1 confidence**, which is reported alongside |
| `precision: null` | no sweep threshold retained a prediction — **undefined, not zero** |
| `ap_by_iou` | per-threshold AP breakdown |
| `mean_iou_matched` | mean IoU of matched pairs at IoU 0.5 — how tight the boxes are |

Group aggregates average only classes that have ground truth: a class with no GT (the
detector's other COCO classes) contributes no AP and no P/R, and letting it drag P/R to zero
made a single-class dataset look worse than it was.

**Cross-check.** `cross_check_detection_metrics` compares our mAP50/mAP50-95 against
Ultralytics' on the same split at a 0.01 absolute tolerance. This compares two *end-to-end
pipelines* (ours scores our `predict()` output; `val()` re-runs inference with its own
batching and rect settings), so a small delta is expected and is not by itself evidence of
an evaluator bug. A clean evaluator comparison needs one shared prediction set scored by
both (e.g. `val(save_json=True)` COCO output); that is not implemented.

## Multi-object tracking

`motmetrics` provides CLEAR-MOT and identity metrics; TrackEval provides HOTA.

**Matching.** Per frame, a distance matrix `d = 1 - IoU` is built with `NaN` marking pairs
below the threshold (`NaN` = motmetrics' "do not pair"), and motmetrics solves the
assignment. The matrix is built in `pairwise_iou_distances` rather than by
`motmetrics.distances.iou_matrix`, whose `np.asfarray` call raises on NumPy 2.

| Metric | Meaning | Direction |
| --- | --- | --- |
| `mota` | 1 − (FN + FP + IDSW) / GT | higher |
| `motp` | mean IoU **distance** of matches | **lower** |
| `idf1` / `idp` / `idr` | identity F1 / precision / recall | higher |
| `num_switches`, `num_fragmentations` | identity switches, track breaks | lower |
| `mostly_tracked` / `partially_tracked` / `mostly_lost` | ≥80% / 20–80% / ≤20% covered | counts |
| `mt_ratio` / `ml_ratio` / `pt_ratio` | the above as fractions of unique GT objects | mixed |
| `HOTA`, `DetA`, `AssA`, `LocA` | TrackEval, averaged over its 19 alpha thresholds | higher |

TrackEval's own `MOTP` is a similarity (100 = perfect) — the opposite convention from
motmetrics' distance — so it is deliberately **not** reported.

**Inspectable failures.** motmetrics' per-frame events (`SWITCH`, `TRANSFER`, …) are
extracted with frame and both identities, so `num_switches: 412` becomes a list of frames a
human can open.

**HOTA on NumPy 2.** TrackEval still references `np.float`/`np.int`/`np.bool`. Rather than
fork the reference implementation or pin NumPy below 2 (breaking Ultralytics 8.4 and OpenCV
5), `numpy_legacy_aliases` restores those aliases for the duration of the call and logs when
it does. `np.float` *is* `float`: a compatibility shim, not a metric change.

## Trajectory and event analytics

Pixel space only: distances in pixels, rates per frame, units in the field names. Positions
are smoothed with a centred moving average before differencing (differentiating raw box
centres turns box jitter into fake velocity). Frame gaps are recorded; interpolation is
opt-in (`fill_gap`), marks interpolated samples with confidence 0, ends the trajectory
before an unbridgeable gap, and never fabricates leading frames. Event headings use
**consecutive** positions, not central differences, which would average a 90° corner into
two 45° changes.

Events are deterministic thresholds: appearance, disappearance, zone enter/exit, direction
change, acceleration, deceleration, sprint, stationary, close proximity, convergence,
divergence. Every event carries its measured value, threshold and unit. The rules are covered
by unit tests on constructed trajectories; accuracy against human-labelled events is a
separate activity requiring a labelled dataset.

## Drift

`monitoring/drift` compares a reference batch against a later one with PSI, the two-sample
KS statistic and Jensen–Shannon divergence. Batches are histogrammed onto the **reference's**
bin edges; out-of-range values are clipped into the edge bins (dropping them would make
total replacement read as a mild shift). The reference retains a bounded raw sample per
signal so KS is computable; without one, KS is `null`, never approximated from bins.

Data drift is a **proxy signal** (`is_proxy_signal: true` in every report): it may indicate a
new camera or resolution and does **not** by itself demonstrate model degradation. Labelled
regression detection lives in `detect_performance_regression` and is direction-aware — fewer
ID switches is an improvement, lower IDF1 is a regression.

## Quality gates

A gate is a dotted metric path into `reports/latest/report.json` with `min` and/or `max`.
Two kinds, never conflated: `kind: example` (placeholders, `is_quality_claim: false`, the CLI
warns when evaluated) and `kind: baseline` (written by `courtvision gate --write-baseline`
from a real run, with a relative tolerance so noise does not fail a release while a
regression does). Cost-like metrics (`num_switches`, latencies) get upper bounds; quality
metrics get lower bounds.
