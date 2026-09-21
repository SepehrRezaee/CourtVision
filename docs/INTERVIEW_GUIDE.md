# Interview guide

Questions an interviewer is likely to ask about this repository, with answers grounded in
what is actually implemented and measured. Nothing here should be said in an interview that
is not true of the code.

## Data

**Why sequence-level splitting?**
Adjacent video frames are near-duplicates. A frame-level split puts frame *t* in training
and frame *t+1* in validation, and the model scores well by recognising the same scene
rather than by generalising. CourtVision splits by sequence, seeds and stratifies the
assignment by sport, persists it with a content-addressed `split_id` (hash of the
assignments, ratios and seed), and a leakage check verifies no sequence appears in two
splits — a check that itself has a test proving it can fail.

**What does dataset validation actually check?**
seqinfo presence and parseability, image directory population, frame-number contiguity,
declared-vs-actual sequence length, header-readable image dimensions, sampled decode checks,
degenerate boxes, out-of-bounds boxes, duplicate (frame, track) annotations, annotations on
frames with no image, confidence/visibility range violations, and class allow-listing. Each
anomaly is a stable issue code with a severity; nothing is silently repaired. MOT's
confidence-0 "unreviewed target" rows are reported as *information* by validation and dropped
by the conversion policy — two different layers, deliberately.

## Detection

**Why implement mAP yourself when Ultralytics has `val()`?**
Three reasons. Per-sport breakdowns need one prediction set re-scored per group, otherwise
group differences are confounded with inference variance. Confidence sweeps and PR curves
need per-detection scores that `val()` does not expose. And a second implementation lets the
two be cross-checked — the run in this repository disagrees by ~0.01 mAP50, which is
documented as an expected pipeline-level difference (the two score different inference
passes), not hidden.

**Precision/recall: why `null` instead of 0?**
If every prediction scores below the lowest sweep threshold, no operating point exists to
report precision or recall *at*. mAP is still defined (it integrates over all predictions),
so the two would contradict. Reporting `null` says "undefined"; reporting `0.0` would look
like a measurement. The aggregate also averages only classes with ground truth — otherwise
the detector's other-class false positives would drag a single-class dataset's P/R to zero.

## Tracking

**MOTA vs IDF1 vs HOTA — what does each tell you?**
MOTA is detection-heavy: 1 − (FN + FP + switches)/GT, and it can go negative — a tracker
that outputs nothing scores better than one that hallucinates. IDF1 is identity-heavy: how
much of the time are the *right* objects tracked under the *right* ids, regardless of frame
level detection noise. HOTA decomposes into DetA × AssA explicitly, so it answers *whether*
the failure is detection or association. In this repository a perfect run scores 1.0 on all
three, while an identity permutation of the predictions leaves MOTA at −0.30 but IDF1 0.35
and AssA 0.19 — the decomposition is what tells you the failure mode.

**A subtlety about motmetrics you found?**
`motmetrics.distances.iou_matrix` names its parameter `max_iou` but treats it as a maximum
*distance*: to keep pairs with IoU ≥ t you pass `1 − t`. More importantly, that helper calls
`np.asfarray`, removed in NumPy 2.0, so the original code crashed on any modern
environment — the fix was to build the `1 − IoU` matrix in-repo with `NaN` marking
sub-threshold pairs, and to test the threshold semantics directly.

**How did you make TrackEval work on NumPy 2?**
Its sources reference `np.float`/`np.int`/`np.bool`. Rather than fork the reference HOTA
implementation or pin NumPy below 2 (breaking Ultralytics 8.4 and OpenCV 5), a context
manager restores those aliases for the duration of the call and logs it. `np.float` *is*
`float`, so it is a compatibility shim, not a metric change.

**What do you do with ID switches beyond counting them?**
motmetrics records per-frame events (`SWITCH`, `TRANSFER`, …) with the frame number and both
identities. CourtVision extracts them, so "412 switches" becomes a list of frames an
engineer can open next to the video.

## Analytics

**Why is everything in pixels?**
There is no homography or court calibration for these datasets, so converting pixel
displacement to metres-per-second would fabricate a physical quantity that changes with
camera distance and zoom. Field names carry the units (`speed_px_per_frame`) and every event
carries its measured value, threshold and unit.

**How do you compute a direction change?**
From *consecutive* position differences, not centrally differenced velocity. Central
differences average across a corner: a 90° turn measured that way appears as two 45° changes
and never trips a 90° threshold. The tests assert the full angle fires.

**How do you handle frame gaps?**
They are recorded per track. Interpolation is opt-in, bounded by `max_gap`, marks
interpolated samples with confidence 0 and an explicit mask, and — after a regression test
caught it — *ends* the trajectory at an unbridgeable gap instead of returning uninitialised
array contents.

## Performance

**How do you benchmark without lying to yourself?**
Warm-up frames are executed and discarded (CUDA context, cuDNN autotuning, allocator growth),
decode happens outside the timed loop, per-frame rows are kept raw, percentiles are published
with their sample count, and repeats get fresh tracker state. If a clip is shorter than the
warm-up, the benchmark reports *no frames measured* rather than a warm-up number.

**What did the Pareto analysis actually show?**
On CPU, `yolo26n + botsort @480` dominated all eight configurations on *both* axes
(31.6 FPS, 38.4 ms p95), so the frontier is a single point and the report says the three
recommendations coincide — rather than inventing a trade-off. The honest caveat is written
into the artefact: with no labelled dataset, the quality axis is throughput, not accuracy.

**What happened when you validated the ONNX export?**
It matched PyTorch's 61 detections with mean IoU 0.976, but only 88.5% found a ≥0.9 partner
(gate: 95%) with one box off by 27 px — so the gate **failed**, correctly refusing to call
the artefact drop-in equivalent, and the CPU timing showed no speedup (0.94×). Reporting the
failure is the point of validating exports.

## Production

**Walk me through the API's failure modes.**
Blocking inference runs in the threadpool (sync handlers), so one video cannot stall the
event loop. Uploads are rejected by extension and content type (415), size while streaming
(413), emptiness (400), and bad parameters (422); the client filename never touches the
filesystem — uploads land under a generated name in a per-request temp dir that is removed
in a `finally`. `/health` is liveness; `/ready` reports *why* the service is not ready.
Jobs complete only after their temp files are deleted, so a client that sees "succeeded"
never races a file that is about to vanish.

**How would you know the model degraded in production?**
Two separate signals. Input drift (PSI/KS/JS against a reference batch, computed per frame
resolution, detections-per-frame, confidence, latency) says the *inputs* changed — it is
explicitly a proxy and cannot show degradation by itself. Actual degradation needs labels:
`detect_performance_regression` compares labelled metrics direction-aware, so fewer ID
switches counts as an improvement, not a regression.

**What would you build next?**
In order: real-dataset evaluation (everything else is instrumentation), model-marked
integration tests for the Ultralytics-backed classes, the pose and team-grouping modules
against the existing interfaces, and a distributed job queue if the API needs to scale
beyond one process.
