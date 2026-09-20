from __future__ import annotations

import configparser
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Box:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0
    class_id: int = 0
    visibility: float = 1.0


def parse_mot_line(line: str) -> Box:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        raise ValueError(f"Expected at least 6 MOT fields, got {len(parts)}: {line!r}")
    values = [float(p) for p in parts]
    return Box(
        frame=int(values[0]),
        track_id=int(values[1]),
        x=values[2],
        y=values[3],
        w=values[4],
        h=values[5],
        confidence=values[6] if len(values) > 6 else 1.0,
        class_id=int(values[7]) if len(values) > 7 else 0,
        visibility=values[8] if len(values) > 8 else 1.0,
    )


def mot_to_yolo(box: Box, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    cx = (box.x + box.w / 2.0) / image_width
    cy = (box.y + box.h / 2.0) / image_height
    w = box.w / image_width
    h = box.h / image_height
    return tuple(max(0.0, min(1.0, v)) for v in (cx, cy, w, h))


def read_seqinfo(sequence_dir: Path) -> tuple[int, int]:
    config_path = sequence_dir / "seqinfo.ini"
    parser = configparser.ConfigParser()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    parser.read(config_path)
    return parser.getint("Sequence", "imWidth"), parser.getint("Sequence", "imHeight")


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_sportsmot(source: Path, output: Path, val_ratio: float = 0.2, seed: int = 42) -> dict[str, list[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    train_root = source / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Expected SportsMOT train directory at {train_root}")
    sequences = sorted(p for p in train_root.iterdir() if p.is_dir())
    if len(sequences) < 2:
        raise ValueError("Need at least two sequences for a train/validation split")

    rng = random.Random(seed)
    shuffled = sequences[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    val_names = {p.name for p in shuffled[:n_val]}
    manifest: dict[str, list[str]] = {"train": [], "val": []}

    for sequence in sequences:
        split = "val" if sequence.name in val_names else "train"
        manifest[split].append(sequence.name)
        width, height = read_seqinfo(sequence)
        gt_path = sequence / "gt" / "gt.txt"
        image_dir = sequence / "img1"
        if not gt_path.exists() or not image_dir.exists():
            raise FileNotFoundError(f"Sequence {sequence.name} is missing img1 or gt/gt.txt")

        by_frame: dict[int, list[Box]] = {}
        for raw in gt_path.read_text().splitlines():
            if not raw.strip():
                continue
            box = parse_mot_line(raw)
            if box.confidence <= 0:
                continue
            by_frame.setdefault(box.frame, []).append(box)

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                frame = int(image_path.stem)
            except ValueError:
                continue
            image_out = output / "images" / split / f"{sequence.name}_{image_path.name}"
            label_out = output / "labels" / split / f"{sequence.name}_{image_path.stem}.txt"
            _copy_or_link(image_path, image_out)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for box in by_frame.get(frame, []):
                cx, cy, bw, bh = mot_to_yolo(box, width, height)
                lines.append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
            label_out.write_text("\n".join(lines) + ("\n" if lines else ""))

    output.mkdir(parents=True, exist_ok=True)
    (output / "splits.json").write_text(json.dumps(manifest, indent=2))
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": {0: "player"}},
            sort_keys=False,
        )
    )
    return manifest
