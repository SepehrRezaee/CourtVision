from __future__ import annotations

import argparse
from pathlib import Path
from courtvision.data import prepare_sportsmot

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare_sportsmot(args.source, args.output, args.val_ratio, args.seed)
    print(f"train sequences: {len(manifest['train'])}")
    print(f"val sequences: {len(manifest['val'])}")

if __name__ == "__main__":
    main()
