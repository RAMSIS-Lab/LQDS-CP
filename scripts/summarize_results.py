#!/usr/bin/env python3
"""Aggregate split-level result files into manuscript-ready CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("meps19", "meps20", "meps21", "fb1", "fb2", "blog", "bio", "star", "community", "bike")
METHODS = ("cti", "spice_nd", "lqds_cp", "cir_nu", "hpd_split", "spice_hpd", "lqds_hpd", "cir_fast", "cir_plus_fast", "cqr", "dcp", "dcp_cqr", "dist_split")
METRICS = ("mean_width", "coverage", "mean_components", "worst_slab_coverage", "stage1_seconds", "stage2_seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2000, 2030)))
    parser.add_argument("--mode", choices=("train", "checkpoint"), default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = ROOT / "artifacts" / "results" / args.mode
    output = ROOT / "artifacts" / "summary"
    metrics = tuple(metric for metric in METRICS if args.mode == "train" or metric != "stage1_seconds")
    output.mkdir(parents=True, exist_ok=True)
    split_values: dict[str, dict[str, dict[str, list[float]]]] = {
        metric: {dataset: {method: [] for method in METHODS} for dataset in DATASETS}
        for metric in metrics
    }
    missing = []
    for dataset in DATASETS:
        for seed in args.seeds:
            path = source / f"{dataset}_{seed}.json"
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            for method in METHODS:
                if method not in record["methods"]:
                    missing.append(f"{path.relative_to(ROOT)}:{method}")
                    continue
                for metric in metrics:
                    split_values[metric][dataset][method].append(record["methods"][method][metric])
    if missing:
        raise SystemExit("Missing results:\n" + "\n".join(missing))

    with (output / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "method", "metric", "mean", "standard_deviation", "splits"))
        for metric in metrics:
            for dataset in DATASETS:
                for method in METHODS:
                    values = np.asarray(split_values[metric][dataset][method], dtype=float)
                    writer.writerow((dataset, method, metric, values.mean(), values.std(ddof=1), len(values)))
    (output / "benchmark_splits.json").write_text(
        json.dumps(split_values, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output / 'benchmark_summary.csv'}")
    print(f"Wrote {output / 'benchmark_splits.json'}")


if __name__ == "__main__":
    main()
