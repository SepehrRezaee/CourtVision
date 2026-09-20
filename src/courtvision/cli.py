"""CourtVision command-line interface.

One entry point replaces the disconnected scripts, and every subcommand shares the same
configuration loader and records the same provenance::

    courtvision doctor
    courtvision data validate | prepare | synth
    courtvision train detector
    courtvision eval detector | tracking | trackers
    courtvision analyze VIDEO
    courtvision benchmark --video CLIP
    courtvision export onnx | export-validate
    courtvision drift reference | compare
    courtvision gate [--write-baseline]
    courtvision report
    courtvision serve

``argparse`` is used rather than adding a CLI framework dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import load_config, to_dict
from .utils import configure_logging, ensure_dir, write_json


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str) if isinstance(payload, (dict, list)) else payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="courtvision", description="Sports video analytics.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")

    doctor = sub.add_parser("doctor", help="Report runtime capabilities from installed packages")
    doctor.add_argument("--json", action="store_true")

    data = sub.add_parser("data", help="Dataset operations")
    data_sub = data.add_subparsers(dest="data_command")
    validate = data_sub.add_parser("validate", help="Validate a raw MOT/SportsMOT tree")
    validate.add_argument("--source", type=Path, required=True)
    validate.add_argument("--output", type=Path, default=None)
    validate.add_argument("--decode-sample", type=int, default=5)
    validate.add_argument("--max-sequences", type=int, default=None)
    validate.add_argument("--allowed-class-ids", default="1", help="Comma-separated ids, or 'any'")

    prepare = data_sub.add_parser("prepare", help="Validate + split + convert to a YOLO dataset")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=None)
    prepare.add_argument("--val-ratio", type=float, default=None)
    prepare.add_argument("--test-ratio", type=float, default=None)
    prepare.add_argument("--seed", type=int, default=None)
    prepare.add_argument("--link-mode", default="hardlink", choices=["hardlink", "copy", "symlink"])
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--max-sequences", type=int, default=None)

    synth = data_sub.add_parser("synth", help="Generate a synthetic SportsMOT-shaped dataset")
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("--sequences-per-sport", type=int, default=3)
    synth.add_argument("--frames", type=int, default=40)
    synth.add_argument("--tracks", type=int, default=6)
    synth.add_argument("--width", type=int, default=640)
    synth.add_argument("--height", type=int, default=360)
    synth.add_argument("--seed", type=int, default=1234)

    video = data_sub.add_parser("video", help="Generate a synthetic clip plus matching ground truth")
    video.add_argument("--output", type=Path, required=True)
    video.add_argument("--frames", type=int, default=60)
    video.add_argument("--tracks", type=int, default=5)
    video.add_argument("--seed", type=int, default=11)

    train = sub.add_parser("train", help="Train models")
    train_sub = train.add_subparsers(dest="train_command")
    train_detector = train_sub.add_parser("detector", help="Fine-tune a YOLO detector")
    train_detector.add_argument("--data", type=Path, default=None)
    train_detector.add_argument("--model", default=None)
    train_detector.add_argument("--epochs", type=int, default=None)
    train_detector.add_argument("--imgsz", type=int, default=None)
    train_detector.add_argument("--batch", type=int, default=None)
    train_detector.add_argument("--device", default=None)
    train_detector.add_argument("--seed", type=int, default=None)
    train_detector.add_argument("--name", default=None)
    train_detector.add_argument("--output", type=Path, default=None)
    train_detector.add_argument("--split-id", default=None)
    train_detector.add_argument("--dry-run", action="store_true")

    evaluate = sub.add_parser("eval", help="Evaluate detection and tracking")
    eval_sub = evaluate.add_subparsers(dest="eval_command")
    eval_detector = eval_sub.add_parser("detector", help="Detection mAP on a prepared split")
    eval_detector.add_argument("--weights", required=True)
    eval_detector.add_argument("--data", type=Path, required=True, help="Prepared dataset root")
    eval_detector.add_argument("--split", default="val", choices=["train", "val", "test"])
    eval_detector.add_argument("--imgsz", type=int, default=640)
    eval_detector.add_argument("--conf", type=float, default=0.001)
    eval_detector.add_argument("--device", default="cpu")
    eval_detector.add_argument("--max-images", type=int, default=None)
    eval_detector.add_argument("--output", type=Path, default=None)
    eval_detector.add_argument("--no-cross-check", action="store_true")

    eval_tracking = eval_sub.add_parser("tracking", help="MOT evaluation against ground truth")
    eval_tracking.add_argument("--ground-truth", type=Path, required=True, help="gt.txt or a directory of sequences")
    eval_tracking.add_argument("--predictions", type=Path, required=True, help="MOT file or a directory of them")
    eval_tracking.add_argument("--iou-threshold", type=float, default=0.5)
    eval_tracking.add_argument("--min-confidence", type=float, default=0.0)
    eval_tracking.add_argument("--allowed-class-ids", default="1")
    eval_tracking.add_argument("--no-hota", action="store_true")
    eval_tracking.add_argument("--output", type=Path, default=None)

    eval_trackers = eval_sub.add_parser("trackers", help="Track one clip with several trackers and compare")
    eval_trackers.add_argument("--weights", required=True)
    eval_trackers.add_argument("--video", type=Path, required=True)
    eval_trackers.add_argument("--trackers", default=None, help="Comma-separated; default: all shipped")
    eval_trackers.add_argument("--ground-truth", type=Path, default=None)
    eval_trackers.add_argument("--imgsz", type=int, default=640)
    eval_trackers.add_argument("--conf", type=float, default=0.25)
    eval_trackers.add_argument("--device", default="cpu")
    eval_trackers.add_argument("--max-frames", type=int, default=None)
    eval_trackers.add_argument("--output", type=Path, default=None)

    analyze = sub.add_parser("analyze", help="Analyse a video end to end")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--weights", default=None)
    analyze.add_argument("--tracker", default=None)
    analyze.add_argument("--device", default=None)
    analyze.add_argument("--imgsz", type=int, default=None)
    analyze.add_argument("--conf", type=float, default=None)
    analyze.add_argument("--max-frames", type=int, default=None)
    analyze.add_argument("--output", type=Path, default=None)

    benchmark = sub.add_parser("benchmark", help="Measure latency and throughput")
    benchmark.add_argument("--video", type=Path, required=True)
    benchmark.add_argument("--weights", default=None, help="Comma-separated for a sweep")
    benchmark.add_argument("--trackers", default=None, help="Comma-separated for a sweep")
    benchmark.add_argument("--imgsz", default=None, help="Comma-separated for a sweep, e.g. 480,640")
    benchmark.add_argument("--conf", type=float, default=None)
    benchmark.add_argument("--device", default=None)
    benchmark.add_argument("--warmup", type=int, default=None)
    benchmark.add_argument("--frames", type=int, default=None)
    benchmark.add_argument("--repeats", type=int, default=None)
    benchmark.add_argument("--output", type=Path, default=None)
    benchmark.add_argument("--pareto", action="store_true")

    export = sub.add_parser("export", help="Export a model to another runtime")
    export.add_argument("format", choices=["onnx", "openvino", "torchscript", "engine", "tflite"])
    export.add_argument("--weights", required=True)
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--device", default="cpu")
    export.add_argument("--half", action="store_true")
    export.add_argument("--opset", type=int, default=None)
    export.add_argument("--output", type=Path, default=None)

    export_validate = sub.add_parser("export-validate", help="Compare an exported model against the reference")
    export_validate.add_argument("--reference", required=True)
    export_validate.add_argument("--exported", required=True)
    export_validate.add_argument("--video", type=Path, required=True)
    export_validate.add_argument("--imgsz", type=int, default=640)
    export_validate.add_argument("--device", default="cpu")
    export_validate.add_argument("--frames", type=int, default=8)
    export_validate.add_argument("--benchmark", action="store_true")
    export_validate.add_argument("--output", type=Path, default=None)

    drift = sub.add_parser("drift", help="Input-distribution drift monitoring")
    drift_sub = drift.add_subparsers(dest="drift_command")
    drift_ref = drift_sub.add_parser("reference", help="Build a reference distribution")
    drift_cmp = drift_sub.add_parser("compare", help="Compare a clip against the reference")
    for sub_parser in (drift_ref, drift_cmp):
        sub_parser.add_argument("--video", type=Path, required=True)
        sub_parser.add_argument("--weights", default=None)
        sub_parser.add_argument("--tracker", default=None)
        sub_parser.add_argument("--device", default=None)
        sub_parser.add_argument("--max-frames", type=int, default=None)
        sub_parser.add_argument("--output", type=Path, default=None)
    drift_cmp.add_argument("--reference", type=Path, default=None)

    gate = sub.add_parser("gate", help="Evaluate release quality gates")
    gate.add_argument("--gates", type=Path, default=Path("configs/quality_gates.yaml"))
    gate.add_argument("--measurements", type=Path, default=Path("reports/latest/report.json"))
    gate.add_argument("--output", type=Path, default=Path("reports/quality_gates.json"))
    gate.add_argument("--write-baseline", action="store_true")
    gate.add_argument("--tolerance", type=float, default=0.0)

    report = sub.add_parser("report", help="Aggregate experiment artefacts into a report")
    report.add_argument("--reports-dir", type=Path, default=None)
    report.add_argument("--output", type=Path, default=None)

    serve = sub.add_parser("serve", help="Run the FastAPI service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__

        print(__version__)
        return 0
    if not args.command:
        parser.print_help()
        return 1

    configure_logging(args.log_level)
    config = load_config(args.config)
    handlers = {
        "doctor": _doctor,
        "data": _data,
        "train": _train,
        "eval": _eval,
        "analyze": _analyze,
        "benchmark": _benchmark,
        "export": _export,
        "export-validate": _export_validate,
        "drift": _drift,
        "gate": _gate,
        "report": _report,
        "serve": _serve,
    }
    handler = handlers.get(args.command)
    if handler is None:  # pragma: no cover - argparse prevents this
        parser.error(f"Unknown command {args.command!r}")
        return 2
    return handler(args, config)


def _doctor(args, config) -> int:
    from .detection.ultralytics_detector import describe_capabilities
    from .optimization.export import export_capabilities
    from .repro import environment_snapshot

    capabilities = describe_capabilities()
    if args.json:
        _print(
            {
                "capabilities": capabilities,
                "export_formats": [item.to_dict() for item in export_capabilities()],
                "environment": environment_snapshot(),
                "config": to_dict(config),
            }
        )
        return 0

    print("CourtVision environment")
    print(f"  ultralytics : {capabilities.get('ultralytics_version')}")
    print(f"  torch       : {capabilities.get('torch_version')}")
    print(f"  CUDA        : {capabilities.get('cuda_available')} ({capabilities.get('cuda_version')})")
    names = capabilities.get("device_names") or []
    print(f"  GPU         : {', '.join(names) if names else 'none visible'}")
    print(f"  trackers    : {', '.join(capabilities.get('trackers', []))}")
    print(f"  YOLO26 assets visible: {len(capabilities.get('yolo26_assets', []))}")
    print("\nExport formats:")
    for item in export_capabilities():
        print(f"  {item.format:12s} {'available' if item.supported_here else 'unavailable - ' + item.reason}")
    return 0


def _data(args, config) -> int:
    if args.data_command == "validate":
        from .data.validation import ValidationOptions, summarize_issues, validate_dataset

        allowed = None if args.allowed_class_ids == "any" else tuple(
            int(value) for value in args.allowed_class_ids.split(",") if value.strip()
        )
        report = validate_dataset(
            args.source,
            options=ValidationOptions(decode_sample=args.decode_sample, allowed_class_ids=allowed),
            max_sequences=args.max_sequences,
        )
        path = args.output or config.paths.reports_dir / "data_validation.json"
        report.save(path)
        print(f"sequences: {len(report.sequences)}  images: {sum(s.images for s in report.sequences)}")
        print(f"annotations: {sum(s.annotations for s in report.sequences)}")
        print(f"errors: {len(report.errors)}  warnings: {len(report.warnings)}")
        print(f"report: {path}")
        if report.errors:
            print(summarize_issues(report.errors, limit=10))
        return 0 if report.passed else 1

    if args.data_command == "prepare":
        from .data.conversion import FilterPolicy
        from .data.sportsmot import prepare_dataset

        if args.val_ratio is not None:
            config.split.val_ratio = args.val_ratio
        if args.test_ratio is not None:
            config.split.test_ratio = args.test_ratio
        if args.seed is not None:
            config.split.seed = args.seed
        report = prepare_dataset(
            args.source,
            args.output or config.paths.data_root,
            split_config=config.split,
            policy=FilterPolicy(),
            link_mode=args.link_mode,
            force=args.force,
            reports_dir=config.paths.reports_dir,
            max_sequences=args.max_sequences,
            progress=lambda message: print(f"  {message}"),
        )
        _print(report.conversion["totals"])
        print(f"strategy: {report.strategy}  split id: {report.split_manifest.split_id}")
        print(f"dataset yaml: {report.dataset_yaml}")
        for warning in report.warnings:
            print(f"warning: {warning}")
        return 0

    if args.data_command == "synth":
        from .data.synthetic import SyntheticSpec, generate_dataset

        spec = SyntheticSpec(
            sequences_per_sport=args.sequences_per_sport,
            frames=args.frames,
            tracks_per_sequence=args.tracks,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )
        sequences = generate_dataset(args.output, spec)
        print(f"generated {len(sequences)} sequence(s) under {args.output}")
        print("NOTE: synthetic data exercises the pipeline; metrics on it are not model quality.")
        return 0

    if args.data_command == "video":
        from .data.synthetic import SyntheticSpec, write_ground_truth_video_pair

        spec = SyntheticSpec(frames=args.frames, tracks_per_sequence=args.tracks, seed=args.seed)
        video, gt = write_ground_truth_video_pair(args.output, spec)
        print(f"video: {video}")
        print(f"ground truth: {gt}")
        print("NOTE: synthetic data exercises the pipeline; metrics on it are not model quality.")
        return 0

    print("usage: courtvision data {validate|prepare|synth|video} ...")
    return 1


def _train(args, config) -> int:
    if args.train_command != "detector":
        print("usage: courtvision train detector ...")
        return 1
    from .detection.trainer import train_detector

    for key, value in {
        "data": args.data,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
        "name": args.name,
    }.items():
        if value is not None:
            setattr(config.train, key, value)

    result = train_detector(
        config.train, output_dir=args.output or config.paths.runs_dir / "detect", split_id=args.split_id, dry_run=args.dry_run
    )
    if args.dry_run:
        _print({"dry_run": True, "arguments": result.config["arguments"]})
        return 0
    print(f"run dir: {result.run_dir}\nbest weights: {result.best_weights}")
    if result.metrics:
        _print(dict(sorted(result.metrics.items())[:12]))
    for note in result.notes:
        print(f"note: {note}")
    return 0


def _eval(args, config) -> int:
    if args.eval_command == "detector":
        from .evaluation.detection import format_metrics_table
        from .pipeline.runner import evaluate_detector_on_dataset

        evaluation = evaluate_detector_on_dataset(
            args.weights,
            args.data,
            split=args.split,
            output_dir=args.output or config.paths.reports_dir / "detection",
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            max_images=args.max_images,
            cross_check_with_ultralytics=not args.no_cross_check,
        )
        print(format_metrics_table(evaluation.metrics))
        cross = evaluation.cross_check or {}
        print(f"\ncross-check vs Ultralytics: {cross.get('status', cross.get('reason', 'n/a'))}")
        for path in evaluation.files.values():
            print(f"artefact: {path}")
        return 0 if cross.get("status") != "disagree" else 1

    if args.eval_command == "tracking":
        from .pipeline.runner import evaluate_tracking, ground_truth_paths_from_prepared

        ground_truth: Any = args.ground_truth
        if args.ground_truth.is_dir():
            ground_truth = ground_truth_paths_from_prepared(args.ground_truth)
            if not ground_truth:
                raise SystemExit(f"No gt.txt files found under {args.ground_truth}")
        predictions: Any = args.predictions
        if args.predictions.is_dir():
            predictions = {path.stem: path for path in sorted(args.predictions.glob("*.txt"))}
            if not predictions:
                raise SystemExit(f"No .txt prediction files found in {args.predictions}")
        allowed = None if args.allowed_class_ids == "any" else tuple(
            int(value) for value in args.allowed_class_ids.split(",") if value.strip()
        )
        evaluation = evaluate_tracking(
            ground_truth,
            predictions,
            output_dir=args.output or config.paths.reports_dir / "tracking",
            iou_threshold=args.iou_threshold,
            min_confidence=args.min_confidence,
            allowed_class_ids=allowed,
            with_hota=not args.no_hota,
        )
        result = evaluation.result
        print(f"sequences: {len(result.sequences)}  frames: {result.to_dict()['totals']['frames']}")
        for key in ("mota", "motp", "idf1", "num_switches", "num_fragmentations"):
            if key in result.metrics:
                print(f"  {key:20s} {result.metrics[key]:.4f}")
        if result.hota:
            print(f"  {'HOTA':20s} {result.hota.get('HOTA'):.4f}  (TrackEval)")
        elif result.hota_note:
            print(f"  HOTA: {result.hota_note}")
        for warning in result.warnings:
            print(f"warning: {warning}")
        for path in evaluation.files.values():
            print(f"artefact: {path}")
        return 0

    if args.eval_command == "trackers":
        return _trackers(args, config)

    print("usage: courtvision eval {detector|tracking|trackers} ...")
    return 1


def _trackers(args, config) -> int:
    from .detection.ultralytics_detector import available_trackers
    from .evaluation.tracking import compare_trackers
    from .pipeline.runner import evaluate_tracking
    from .tracking.ultralytics_tracker import UltralyticsTracker, export_tracking_artifacts
    from .utils import write_csv

    trackers = [name.strip() for name in args.trackers.split(",")] if args.trackers else available_trackers()
    output = ensure_dir(args.output or config.paths.reports_dir / "tracker_comparison")
    rows: list[dict] = []
    evaluations: list[dict] = []

    for tracker in trackers:
        print(f"tracking with {tracker} ...")
        with UltralyticsTracker(
            weights=args.weights, tracker=tracker, device=args.device, imgsz=args.imgsz, conf=args.conf
        ) as backend:
            result = backend.track(args.video, max_frames=args.max_frames)
        tracker_dir = ensure_dir(output / Path(tracker).stem)
        written = export_tracking_artifacts(result, tracker_dir)
        row = {
            "tracker": tracker,
            "detector": args.weights,
            "imgsz": args.imgsz,
            "confidence": args.conf,
            "device": args.device,
            "frames": len(result.frames),
            "detections": result.detections,
            "unique_tracks": len(result.unique_track_ids),
            "FPS": round(result.meta.get("fps_wall") or 0.0, 4),
            "wall_seconds": result.meta.get("wall_seconds"),
            "MOTA": None,
            "HOTA": None,
            "IDF1": None,
            "ID_switches": None,
            "p95_latency_ms": None,
        }
        if args.ground_truth is not None:
            ground_truth = (
                {args.ground_truth.parent.parent.name: args.ground_truth}
                if args.ground_truth.is_file()
                else {path.parent.parent.name: path for path in args.ground_truth.rglob("gt.txt")}
            )
            try:
                evaluation = evaluate_tracking(
                    ground_truth,
                    {name: written["mot"] for name in ground_truth},
                    with_hota=True,
                )
            except ValueError as exc:
                print(f"  MOT evaluation skipped: {exc}")
            else:
                row["MOTA"] = evaluation.result.metrics.get("mota")
                row["IDF1"] = evaluation.result.metrics.get("idf1")
                row["ID_switches"] = evaluation.result.metrics.get("num_switches")
                if evaluation.result.hota:
                    row["HOTA"] = evaluation.result.hota.get("HOTA")
                evaluations.append(
                    {
                        "name": Path(tracker).stem,
                        "tracker": tracker,
                        "detector": args.weights,
                        "imgsz": args.imgsz,
                        "confidence": args.conf,
                        "device": args.device,
                        "metrics": dict(evaluation.result.metrics),
                        "tracking": evaluation.result.to_dict(),
                    }
                )
        rows.append(row)

    write_csv(output / "tracker_comparison.csv", rows, columns=list(rows[0].keys()) if rows else None)
    write_json(output / "tracker_comparison.json", {"rows": rows, "evaluations": evaluations})
    print()
    print(compare_trackers(rows))
    if args.ground_truth is None:
        print("\nNOTE: no --ground-truth was given, so MOTA/HOTA/IDF1 are NOT MEASURED; only runtime and track counts are real.")
    return 0


def _analyze(args, config) -> int:
    from .serving.service import AnalysisRequest, AnalysisService

    service = AnalysisService(config.serving, config.api, config.events)
    service.load()
    try:
        payload = service.analyze(
            AnalysisRequest(
                video_path=args.video,
                tracker=args.tracker or config.serving.tracker,
                conf=args.conf if args.conf is not None else config.serving.conf,
                imgsz=args.imgsz or config.serving.imgsz,
                max_frames=args.max_frames or config.api.max_frames,
                include_tracks=False,
                include_trajectories=True,
                include_events=True,
                source_name=args.video.name,
            )
        )
    finally:
        service.unload()
    output = ensure_dir(args.output or config.paths.reports_dir / "analysis")
    write_json(output / "analysis.json", payload)
    _print({key: value for key, value in payload.items() if key not in {"event_list", "trajectories"}})
    print(f"artefact: {output / 'analysis.json'}")
    return 0


def _benchmark(args, config) -> int:
    from .optimization.benchmark import BenchmarkPlan, benchmark_video, save_benchmark, system_report
    from .optimization.pareto import build_pareto_report, configuration_records_from_reports, plot_pareto, save_pareto_csv
    from .tracking.ultralytics_tracker import UltralyticsTracker

    weights_list = (args.weights or config.detector.model).split(",")
    tracker_list = (args.trackers or config.tracker.name).split(",")
    imgsz_list = [int(value) for value in (str(args.imgsz) if args.imgsz else config.detector.imgsz).split(",")]
    plan = BenchmarkPlan(
        warmup_frames=args.warmup if args.warmup is not None else config.benchmark.warmup_frames,
        measured_frames=args.frames if args.frames is not None else config.benchmark.max_frames,
        repeats=args.repeats if args.repeats is not None else config.benchmark.repeats,
        percentiles=config.benchmark.percentiles,
    )
    output = ensure_dir(args.output or config.paths.reports_dir / "benchmarks")
    device = args.device or config.serving.device
    confidence = args.conf if args.conf is not None else config.detector.conf
    print(f"plan: warmup={plan.warmup_frames} measured={plan.measured_frames} repeats={plan.repeats} (warm-up excluded)")

    runs: list[dict] = []
    for weights in weights_list:
        for tracker in tracker_list:
            for imgsz in imgsz_list:
                label = f"{Path(weights).stem}_{Path(tracker).stem}_{imgsz}"
                print(f"benchmarking {label} ...")

                def make_processor(weights=weights, tracker=tracker, imgsz=imgsz):
                    backend = UltralyticsTracker(
                        weights=weights, tracker=tracker, device=device, imgsz=imgsz, conf=confidence
                    )
                    return lambda frame: backend.track_frames([frame]).frames[0]

                run = benchmark_video(
                    args.video,
                    make_processor,
                    label=label,
                    plan=plan,
                    config={"weights": weights, "tracker": tracker, "imgsz": imgsz, "device": device, "conf": confidence},
                )
                save_benchmark(run, output)
                summary = run.summary
                first = (summary.get("rows") or [{}])[0]
                print(
                    f"  frames={summary.get('frames')} mean={first.get('end_to_end_ms_mean')}ms "
                    f"p50={first.get('end_to_end_ms_p50')}ms p95={first.get('end_to_end_ms_p95')}ms "
                    f"fps={first.get('fps_from_mean')}"
                )
                for warning in run.warnings:
                    print(f"  warning: {warning}")
                runs.append(
                    {
                        "label": label,
                        "config": run.config,
                        "summary": summary,
                        "warnings": run.warnings,
                        "imgsz": imgsz,
                        "confidence": confidence,
                        "device": device,
                        "detector": weights,
                        "tracker": tracker,
                    }
                )

    write_json(output / "summary.json", {"runs": runs, "system": system_report(device)})
    print(f"\nartefacts written to {output}")

    if args.pareto:
        records = configuration_records_from_reports([], runs)
        report = build_pareto_report(records, quality_key="fps", latency_key="p95_latency_ms")
        report.notes.append(
            "Quality axis is throughput because no labelled quality metrics were supplied on this command "
            "line. Run `courtvision eval` first and combine the reports for a quality/latency frontier."
        )
        report.save(output.parent / "pareto.json")
        save_pareto_csv(report, output.parent / "pareto.csv")
        plotted = plot_pareto(report, output.parent / "pareto.png")
        print(report.markdown_table())
        if plotted:
            print(f"chart: {plotted}")
    return 0


def _export(args, config) -> int:
    from .optimization.export import export_model

    result = export_model(
        args.weights, format=args.format, imgsz=args.imgsz, device=args.device, half=args.half, opset=args.opset, output_dir=args.output
    )
    _print(result.to_dict())
    if result.ok:
        print("\nNOTE: a produced file is not evidence of correctness; run `courtvision export-validate` to compare behaviour.")
    return 0 if result.ok else 1


def _export_validate(args, config) -> int:
    from .optimization.benchmark import load_frames
    from .optimization.export import benchmark_export_pair, validate_export

    frames, info = load_frames(args.video, max_frames=args.frames)
    if not frames:
        raise SystemExit(f"No frames decoded from {args.video}")
    validation = validate_export(args.reference, args.exported, frames, imgsz=args.imgsz, device=args.device)
    payload: dict[str, Any] = {"validation": validation.to_dict(), "frames_info": info}
    if args.benchmark:
        payload["benchmark"] = benchmark_export_pair(args.reference, args.exported, frames, imgsz=args.imgsz, device=args.device)
    if args.output is not None:
        write_json(args.output, payload)
        print(f"artefact: {args.output}")
    _print(payload)
    return 0 if validation.passed else 1


def _drift(args, config) -> int:
    from .monitoring.drift import (
        compare_distributions,
        load_reference,
        reference_from_signals,
        save_reference,
        signals_from_tracking_result,
    )
    from .tracking.ultralytics_tracker import UltralyticsTracker

    if args.drift_command not in {"reference", "compare"}:
        print("usage: courtvision drift {reference|compare} --video ...")
        return 1

    with UltralyticsTracker(
        weights=args.weights or config.serving.model,
        tracker=args.tracker or config.serving.tracker,
        device=args.device or config.serving.device,
        imgsz=config.serving.imgsz,
        conf=config.serving.conf,
    ) as backend:
        result = backend.track(args.video, max_frames=args.max_frames)
    latencies = [frame.timings_ms.get("inference") for frame in result.frames if frame.timings_ms.get("inference")]
    signals = signals_from_tracking_result(result, latencies_ms=latencies)

    if args.drift_command == "reference":
        reference = reference_from_signals(
            signals, bins=config.monitoring.bins, metadata={"video": str(args.video), "weights": args.weights or config.serving.model}
        )
        path = args.output or config.monitoring.reference_path
        save_reference(path, reference)
        print(f"reference written to {path}")
        print(f"signals: {', '.join(sorted(reference['signals']))}")
        return 0

    report = compare_distributions(
        load_reference(args.reference or config.monitoring.reference_path),
        signals,
        psi_warn=config.monitoring.psi_warn,
        psi_alert=config.monitoring.psi_alert,
    )
    path = args.output or config.paths.reports_dir / "monitoring" / "drift.json"
    report.save(path)
    print(report.format_text())
    print(f"\nartefact: {path}")
    print("Reminder: drift is a proxy signal and does not by itself mean the model got worse.")
    return 0


def _gate(args, config) -> int:
    from .pipeline.quality_gates import (
        baseline_from_measurements,
        default_gate_metrics,
        evaluate_gates_from_file,
        save_gate_set,
    )
    from .utils import read_json

    if not args.measurements.exists():
        print(f"measurements file not found: {args.measurements}")
        print("Run `courtvision report` first, or point --measurements at a report.json.")
        return 2

    if args.write_baseline:
        gate_set = baseline_from_measurements(
            read_json(args.measurements), default_gate_metrics(), tolerance=args.tolerance
        )
        save_gate_set(gate_set, args.gates)
        print(f"wrote {len(gate_set.gates)} baseline gate(s) to {args.gates}")
        for gate in gate_set.gates:
            bound = f"min {gate.minimum}" if gate.minimum is not None else f"max {gate.maximum}"
            print(f"  {gate.metric}: {bound}")
        for note in gate_set.notes:
            print(f"note: {note}")
        return 0

    if not args.gates.exists():
        print(f"gate file not found: {args.gates}")
        print("Write a measured baseline with --write-baseline, or start from configs/quality_gates.example.yaml.")
        return 2
    report = evaluate_gates_from_file(args.gates, args.measurements)
    report.save(args.output)
    print(report.format_text())
    print(f"\nartefact: {args.output}")
    return 0 if report.passed else 1


def _report(args, config) -> int:
    from .reports.builder import build_report

    bundle = build_report(args.reports_dir or config.paths.reports_dir, output_dir=args.output)
    print(f"report written to {bundle.files['markdown']}")
    for name, path in bundle.files.items():
        print(f"  {name}: {path}")
    if bundle.missing:
        print("\nNOT MEASURED (no artefact produced):")
        for name in bundle.missing:
            print(f"  - {name}")
    return 0


def _serve(args, config) -> int:
    import uvicorn

    uvicorn.run("courtvision.serving.api:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
