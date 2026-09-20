# Evaluation methodology

This document defines every metric CourtVision reports, how matching is performed,
and where each number comes from. If a number is not defined here, it is not
reported.

## Detection

**Implementation.** Two independent evaluators are run on the same predictions:

| Evaluator | Purpose |
| --- | --- |
| `courtvision.evaluation.detection` (ours) | per-sport breakdowns, confidence sweeps, PR curves, failure cases |
| Ultralytics `model.val()` | reference implementation, used as a cross-check |

Predictions are computed **once** into `DetBox` records and then re-scored per group,
so every breakdown describes identical inferences. Re-running inference per sport
would confound the comparison with inference variance.

**Matching.** Within each IoU threshold, predictions are sorted by descending
confidence and each is greedily matched to the highest-IoU *unmatched* ground-truth
box whose IoU is at least the threshold. This is the VOC/COCO convention and is what
Ultralytics and pycocotools do.

**mAP.** Average precision uses 101-point COCO interpolation on the monotone
precision envelope, i.e. `AP = mean over r in {0, 0.01, ..., 1} of max{precision(r') : r' >= r}`.

| Metric | Definition |
| --- | --- |
| `map50` | AP at IoU 0.50, averaged over classes |
| `map50_95` | mean of AP over IoU 0.50, 0.55, ..., 0.95 |
| `precision`, `recall` | **at the best-F1 confidence**, which is reported alongside (`best_f1_confidence`) |
| `ap_by_iou` | the full per-threshold AP breakdown |
| `mean_iou_matched` | mean IoU of matched pairs at IoU 0.5 — how tight the boxes are |

Precision/recall are reported at the best-F1 operating point rather than at an
arbitrary confidence, and the *confidence that was used* is part of the record.
Ultralytics reports them at its own operating point, so the cross-check compares mAP
only; comparing precision across two different operating points would manufacture a
disagreement.

**Grouping.** The stratum is recovered from the prepared image name
(`<sequence>_<frame>.jpg` → leading token of the sequence). Groups that cannot be
inferred are reported as their own token rather than being forced into a known sport.

**Cross-check.** `cross_check_detection_metrics` compares our mAP50 and mAP50-95
against Ultralytics' on identical inputs with an absolute tolerance (default 1e-2)
and reports `agree` / `disagree`. A disagreement is a bug signal in one of the two
implementations and is surfaced, never averaged away.

## Multi-object tracking

**Implementation.** `motmetrics` computes CLEAR-MOT and identity metrics;
TrackEval (the official HOTA implementation) computes HOTA when installed.

**Matching.** For each frame, a distance matrix is built as `d = 1 - IoU` with
`NaN` marking pairs whose IoU is below the threshold (`NaN` = "do not pair"), and
`motmetrics` solves the association.

Two conventions are pinned by tests because both are easy to get wrong:

* `motmetrics.distances.iou_matrix` names its parameter `max_iou` but treats it as a
  maximum **distance**, so keeping pairs with `IoU >= t` requires `max_iou = 1 - t`.
  CourtVision does not call that helper at all: it uses `np.asfarray`, removed in
  NumPy 2.0, so the helper raises on any modern NumPy. The distance matrix is built
  in `pairwise_iou_distances` instead, and `tests/test_tracking_eval.py` asserts the
  threshold semantics directly.
* `motp` is the mean IoU **distance** of matched pairs, so **lower is better** and it
  is not a percentage. TrackEval's `MOTP` is a *similarity* where 100 is perfect; the
  two are never mixed, and TrackEval's MOTP is deliberately not reported.

| Metric | Meaning | Direction |
| --- | --- | --- |
| `mota` | 1 - (FN + FP + IDSW) / GT | higher |
| `motp` | mean IoU distance of matches | **lower** |
| `idf1` / `idp` / `idr` | identity F1 / precision / recall | higher |
| `num_switches` | identity switches | lower |
| `num_fragmentations` | track interruptions | lower |
| `mostly_tracked` / `partially_tracked` / `mostly_lost` | tracks covered ≥80% / 20-80% / ≤20% | counts |
| `mt_ratio` / `ml_ratio` / `pt_ratio` | the above as fractions of unique GT objects (computed here, since `motmetrics` exposes only counts) | higher / lower / — |
| `id_switches_per_object`, `misses_per_object`, … | per-object normalisations that make runs of different length comparable | lower |
| HOTA / DetA / AssA / LocA / OWTA | TrackEval, averaged over the 19 alpha thresholds 0.05…0.95 | higher |
| `HOTA_at_0`, `LocA_at_0` | TrackEval scalars at the lowest threshold | higher |

**Inspectable failures.** `motmetrics`' per-frame events (SWITCH, TRANSFER, ASCEND,
MIGRATE) are extracted with their frame and both identities, so "num_switches: 412"
becomes a list of frames a human can open with
`courtvision … error analysis` / `reports/detection/failure_cases/`.

**Filtering policy** (recorded in every report): ground truth drops
`confidence <= 0` rows (MOTChallenge marks unreviewed targets that way), boxes whose
class is not in `allowed_class_ids` (SportsMOT labels players as class 1), boxes
below `min_visibility`, and degenerate boxes. Predictions are filtered by
`min_confidence`. Changing the policy changes the numbers, which is why the policy
is written into the output file.

### HOTA on NumPy 2

TrackEval's own sources reference `np.float`, `np.int` and `np.bool`, all removed in
NumPy 2. Rather than fork the reference implementation or pin NumPy below 2 (which
would break Ultralytics 8.4 and OpenCV 5), the aliases are restored for the duration
of the call by `courtvision.evaluation.tracking.numpy_legacy_aliases`. `np.float`
**is** `float`, so this is a compatibility shim and not a change to the metric; the
shim is logged when it activates. If TrackEval is fixed upstream the shim becomes a
no-op.

## Trajectory and event analytics

All kinematics are **pixel space**: distances in pixels, rates per frame. Field names
carry the units. No metre-based or speed-in-m/s quantity is produced anywhere,
because no court calibration or homography is available for these datasets; a
physical speed would be fabricated.

* Positions are smoothed with a centred moving average before differencing, since
  differentiating raw box centres turns a couple of pixels of detector jitter into
  fake velocity.
* Frame gaps are recorded. Linear interpolation is opt-in (`fill_gap`) and marks the
  interpolated samples so downstream statistics can exclude them; a long gap is never
  interpolated, because that invents a path the object did not travel.
* Direction changes are computed from **consecutive** centred positions, not from
  centrally differenced velocity: central differences average across a corner, so a
  90° turn would appear as two 45° changes and never reach a 90° threshold.

Events are deterministic thresholds over trajectories: `TRACK_APPEARANCE`,
`TRACK_DISAPPEARANCE`, `ZONE_ENTER`, `ZONE_EXIT`, `DIRECTION_CHANGE`,
`ACCELERATION`, `DECELERATION`, `SPRINT`, `STATIONARY`, `CLOSE_PROXIMITY`,
`CONVERGENCE`, `DIVERGENCE`. Every event carries the measured `value`, the
`threshold` that fired and the `unit`, so it can be audited.

**Event accuracy is NOT MEASURED.** There is no labelled temporal sports-event dataset
in this repository. What is measured is that the detector is deterministic, covered
by tests, and reproducible. `docs/LIMITATIONS.md` states this again.

## Pose (optional)

Pose output is an **inference capability only**. No pose ground truth is bundled, so
no PCK/OKS/keypoint accuracy is claimed, and the API attaches
`quantitatively_validated: false` to every pose payload. Posture features (torso
axis angle, shoulder/hip widths, stance ratio) are 2D image-space descriptors, not
calibrated biomechanics, and they are `null` — never `0` — when the required
keypoints are missing or below the confidence threshold.

## Drift

`courtvision.monitoring.drift` compares a reference distribution against a later
batch using PSI, the two-sample KS statistic and Jensen-Shannon divergence. Two
distinctions are enforced:

* **Data drift is a proxy signal.** A shift in detections-per-frame, confidence or
  box size may indicate a new camera, resolution or lighting. It does **not** by
  itself demonstrate model degradation, and the report says so in an
  `interpretation` field and in `is_proxy_signal: true`.
* **Performance degradation requires labels.** That comparison lives in
  `detect_performance_regression`, which is direction-aware: fewer ID switches is an
  improvement, a lower IDF1 is a regression.

Batches are always histogrammed onto the **reference's** bin edges; comparing two
independently binned histograms would not be a comparison. PSI cut-offs (0.1 warn,
0.25 alert) are rules of thumb and are configuration, not theory.

## Quality gates

A gate is a dotted metric path with `min` and/or `max`. Metric names index into
`reports/latest/report.json` (for example
`detection_metrics.groups.overall.map50_95`). Evaluation returns PASS/FAIL plus the
exact failing criteria and the available metric names when a path is missing.

Two kinds of threshold, never conflated:

* `kind: example` — placeholders. `is_quality_claim: false`, and the CLI prints a
  warning whenever such a set is evaluated.
* `kind: baseline` — produced by `courtvision gate --write-baseline` from a real
  measurement, optionally with a relative tolerance so run-to-run noise does not fail
  a release while a genuine regression does. Cost-like metrics (`num_switches`,
  `*_latency_ms`) get upper bounds; quality metrics get lower bounds.

## Reproducibility

Every artefact embeds `environment.json`/`system.json` from
`courtvision.repro.environment_snapshot()`: timestamp, git commit **and dirty flag**,
Python, package versions, OS/CPU/GPU/CUDA, and the seed. The dirty flag matters — a
result produced from a modified working tree is not reproducible from the recorded
commit, and the report says so.
