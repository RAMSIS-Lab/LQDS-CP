#!/usr/bin/env python3
"""Run all requested dataset/seed combinations with the paper protocol."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("meps19", "meps20", "meps21", "fb1", "fb2", "blog", "bio", "star", "community", "bike")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "checkpoint"), default="train")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2000, 2030)))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        for seed in args.seeds:
            result = ROOT / "artifacts" / "results" / args.mode / f"{dataset}_{seed}.json"
            if args.skip_existing and result.exists():
                print(f"Skipping {dataset} seed {seed}")
                continue
            command = [
                sys.executable,
                str(ROOT / "run.py"),
                "--mode", args.mode,
                "--dataset", dataset,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--device", args.device,
                "--threads", str(args.threads),
            ]
            if args.methods:
                command.extend(["--methods", *args.methods])
            print(f"Running {dataset} seed {seed}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
