#!/usr/bin/env python3
"""Run every synthetic experiment reported in the manuscript."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(arguments: list[str]) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> None:
    run(
        [
            "scripts/run_synthetic.py",
            "oracle",
            "--seeds",
            "30",
            "--Ks",
            "20,30,50,99",
            "--n",
            "2000",
            "--interp",
            "flin",
        ]
    )
    run(
        [
            "scripts/run_synthetic.py",
            "e2e",
            "--seeds",
            "10",
            "--seed-start",
            "2000",
            "--n-eval",
            "1000",
            "--interp",
            "flin",
            "--threads",
            "1",
        ]
    )
    configurations = ((3000, 30), (12000, 30), (48000, 30), (3000, 20), (48000, 45))
    for training_size, knots in configurations:
        for split in range(5):
            run(
                [
                    "scripts/run_synthetic_convergence.py",
                    str(training_size),
                    str(knots),
                    str(split),
                ]
            )


if __name__ == "__main__":
    main()
