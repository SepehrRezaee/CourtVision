from __future__ import annotations

import argparse
import json
from courtvision.evaluate import evaluate_detector, evaluate_mot, export_mot_predictions

def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect")
    detect.add_argument("--weights", required=True)
    detect.add_argument("--data", required=True)
    detect.add_argument("--imgsz", type=int, default=1280)
    detect.add_argument("--device", default="cpu")

    track = sub.add_parser("track")
    track.add_argument("--weights", required=True)
    track.add_argument("--video", required=True)
    track.add_argument("--output", required=True)
    track.add_argument("--tracker", default="bytetrack.yaml")
    track.add_argument("--conf", type=float, default=0.25)
    track.add_argument("--imgsz", type=int, default=1280)
    track.add_argument("--device", default="cpu")

    mot = sub.add_parser("mot")
    mot.add_argument("--ground-truth", required=True)
    mot.add_argument("--prediction", required=True)
    mot.add_argument("--iou-threshold", type=float, default=0.5)

    args = parser.parse_args()
    if args.command == "detect":
        print(json.dumps(evaluate_detector(args.weights, args.data, args.imgsz, args.device), indent=2))
    elif args.command == "track":
        print(export_mot_predictions(args.weights, args.video, args.output, args.tracker, args.conf, args.imgsz, args.device))
    else:
        print(json.dumps(evaluate_mot(args.ground_truth, args.prediction, args.iou_threshold), indent=2))

if __name__ == "__main__":
    main()
