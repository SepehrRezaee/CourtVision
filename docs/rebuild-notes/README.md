# Rebuild notes — audit findings and a work loss

This directory is a **salvage record**, not part of the shipped package. It contains
a forensic audit of this repository's original implementation, two demonstrated
defects with reproduction evidence, and the design documents for a larger refactor
whose source files were destroyed mid-session by the environment (see below).

Nothing here describes code that is currently present in `src/`. The files are kept
because the audit findings are valid against the code that *is* present, and because
re-deriving them would waste the effort.

## What happened to the workspace

During the session, an external process deleted most of the working tree, twice:

| Time | What was removed |
| --- | --- |
| ~23:58 | `.venv` (≈5 GB, a working CUDA PyTorch install), `.wheels/` (2.6 GB of downloaded wheels), `.git/HEAD`, `.git/refs/`, `.git/index` |
| ~00:30 | `src/` (≈57 modules), `configs/`, `Dockerfile`, `pyproject.toml`, `README.md`, `.github/`, `dags/`, most of `tests/` |

A control probe confirmed the pattern: files smaller than roughly 500 MB survived,
while large artifacts were removed repeatedly. The clearest signal was installing
CUDA PyTorch: `uv` downloaded and began extracting `torch-2.14.0+cu126`, and the
531 MB `cublasLt64_12.dll` was deleted mid-extraction three separate times, each
attempt restarting in a fresh temp directory. Installing CPU-only PyTorch (largest
file ≈200 MB) completed in 12.7 seconds.

Consequence: **CUDA could not be kept installed**, so no GPU benchmarks could be
produced in this environment, even though CUDA was verified working beforehand
(`RTX 3050 Ti Laptop, 4 GB, sm_86, CUDA 12.6`, `torch.cuda.is_available() == True`).

The repository itself was recovered from the surviving git pack object
(`.git/objects/pack/pack-9bf157*.pack`, full history, tip `c469990`): a valid
`.git` skeleton was reconstructed, `refs/heads/main` recreated, and the working tree
restored with `git checkout -f`. `git log` shows all 15 original commits and
`pytest` passes.

## Verified findings against the original code

Two defects were reproduced empirically. Two further observations are code-review
findings and are labelled as such.

### 1. Tracking evaluation crashes on NumPy 2 — CONFIRMED

`courtvision/evaluate.py::evaluate_mot` calls `motmetrics.distances.iou_matrix`,
which uses `np.asfarray` at `motmetrics/distances.py:117`. NumPy 2.0 removed that
alias. `pyproject.toml` permits `numpy>=1.26,<3`, so a clean install resolves NumPy 2
and the evaluation path raises immediately:

```
$ python -c "from courtvision.evaluate import evaluate_mot; evaluate_mot(gt, gt)"
AttributeError: `np.asfarray` was removed in the NumPy 2.0 release.
  at: distances = mm.distances.iou_matrix(...)
  File ".../motmetrics/distances.py", line 117, in iou_matrix
```

The breakage is confined to `motmetrics/distances.py`; the accumulator and metric
arithmetic work correctly on NumPy 2 (verified: a perfect match scores `mota=1.0`,
`idf1=1.0`). The fix is to build the distance matrix in this repository rather than
call the broken helper:

```python
# IoU-thresholded distance matrix: NaN marks "do not pair", i.e. IoU < threshold.
ious = iou_matrix(gt_xyxy, pred_xyxy)            # vectorised, xyxy
distances = np.where(ious < iou_threshold, np.nan, 1.0 - ious)
acc.update(gt_ids, pred_ids, distances)
```

Note on a related convention: `iou_matrix`'s parameter is named `max_iou` but behaves
as a maximum *distance*, so keeping pairs with `IoU >= t` requires `max_iou = 1 - t`.
The existing call site already passes `1 - iou_threshold` and is therefore correct —
it was verified against the library source rather than assumed, and it is **not** a
bug.

### 2. `mot_to_yolo` corrupts labels for boxes crossing an image edge — CONFIRMED

`courtvision/data.py::mot_to_yolo` clips the four normalised outputs independently:

```python
return tuple(max(0.0, min(1.0, v)) for v in (cx, cy, w, h))
```

For a box at `x=180..220` in a 200×100 frame this yields `cx=1.0, w=0.2`, which
round-trips to pixels `180..220` — a label extending 20 px outside the image whose
centre sits at 200.0. The true clipped rectangle is `180..200` with centre 190.0
(`cx=0.95, w=0.10`). Clipping in pixel space before normalising, as below, keeps the
centre and size describing the same rectangle:

```python
x1 = min(max(box.x, 0.0), width);  x2 = min(max(box.x + box.w, 0.0), width)
y1 = min(max(box.y, 0.0), height); y2 = min(max(box.y + box.h, 0.0), height)
if x2 - x1 <= 0 or y2 - y1 <= 0:
    raise ValueError("box lies entirely outside the frame")
return ((x1 + (x2 - x1) / 2) / width, (y1 + (y2 - y1) / 2) / height,
        (x2 - x1) / width, (y2 - y1) / height)
```

Impact: supervision for every target at the frame boundary is wrong, which is exactly
where players enter and leave in broadcast footage.

### 3. Retracted finding — tracker state does **not** leak between videos

An initial code review of `courtvision/inference.py` suggested that holding one
`VideoAnalyzer` for the process lifetime with `persist=True` would leak track ids
between requests. This was tested and is **not true** in Ultralytics 8.4.156:
analysing two different clips back-to-back with one model gave ids `[1,2,3,4,5]` and
then `[1,2,3,4,6,7]` (both starting near 1), and three repeated analyses of the *same*
clip in one process gave byte-identical id sets `[1,2,3,4,5]` every time, with and
without an explicit `tracker.reset()`. The hypothesis was wrong and is recorded here
so it is not "fixed" again later.

### 4. Code-review findings (not empirically reproduced)

* `courtvision/api.py` defines `async def analyze_video` and calls blocking,
  CPU-bound inference inside it. Under Starlette this runs on the event loop and
  stalls every other request for the duration of the video. Fix: make the handler
  synchronous (`def`) so it runs in the threadpool, or await `run_in_threadpool`.
* The same endpoint has no upload size limit, no extension or content-type check,
  and reads the whole upload into memory before validating it.
* The response returns every frame's tracks by default; on a long clip that is a
  multi-megabyte payload.
* `courtvision/data.py::prepare_sportsmot` produces only train/val — there is no test
  split, no persisted split identifier beyond `splits.json`, and no leakage guard.
* `Dockerfile` runs as root and declares no `HEALTHCHECK`.
* `pyproject.toml` allows NumPy 2 while pinning a `motmetrics` that is incompatible
  with it (finding 1).

## Claims that were checked and found accurate

Not every README statement was wrong — these were verified against the installed
packages and hold:

* `tracktrack.yaml` is real: Ultralytics 8.4.156 ships `botsort.yaml`,
  `bytetrack.yaml`, `deepocsort.yaml`, `fasttrack.yaml`, `ocsort.yaml` and
  `tracktrack.yaml`. The README's tracker list is correct.
* `yolo26x.pt` is a real, downloadable asset, as are `yolo26n/s/m/l.pt` and the
  `-pose`, `-seg`, `-cls`, `-obb`, `-depth` variants (60 YOLO26 assets in total).
* The pinned `ultralytics==8.4.156` exists on PyPI and supports YOLO26.

So the repository's central capability claims were supported by the runtime. The
gap was in correctness, evaluation rigour and production hardening — not in the
tracker or model names.

## Status of the rebuild

The implementation described below was subsequently rebuilt. The design documents now live
in the parent directory — `../ARCHITECTURE.md`, `../EVALUATION.md` and
`../PERFORMANCE.md` — rewritten against the code that is actually present, alongside
`../LIMITATIONS.md` and `../INTERVIEW_GUIDE.md`. This file remains as the record of the
audit findings, the two confirmed defects and their reproductions, and the retracted
hypotheses.

## How to resume

1. `pip install -e ".[dev]"` (add `eval` for TrackEval/HOTA, `export` for
   ONNX/OpenVINO/openvino).
2. Fix findings 1 and 2 first — both are small, and both silently corrupt or block
   results.
3. Re-apply the module layout from `ARCHITECTURE.md`, keeping the invariants that
   `EVALUATION.md` and `PERFORMANCE.md` state, since those are the parts that make
   reported numbers trustworthy.
4. Run the test tiers: `pytest -m "not model"` offline, then `pytest -m model` with
   weights (the model tier downloads checkpoints and must be excluded from CI).
