#!/usr/bin/env python3
"""Run the methods reported in the paper from checkpoints or from scratch."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import make_splits
from src.intervals import PredictionSets
from src.lqdscp import LQDSCP
from src.metrics import evaluate
from src.methods import (
    ArrayQuantilePredictor,
    run_cir_nu,
    run_cti,
    run_density_level_set,
    run_quantile_method,
)
from src.models import (
    GMMNet,
    LinearSplineNet,
    QuantileNet,
    TrainConfig,
    predict_quantiles,
    set_seed,
    train_model,
)

DATASETS = (
    "meps19",
    "meps20",
    "meps21",
    "fb1",
    "fb2",
    "blog",
    "bio",
    "star",
    "community",
    "bike",
)
METHODS = (
    "cti",
    "spice_nd",
    "lqds_cp",
    "cir_nu",
    "hpd_split",
    "spice_hpd",
    "lqds_hpd",
    "cir_fast",
    "cir_plus_fast",
    "cqr",
    "dcp",
    "dcp_cqr",
    "dist_split",
)
QUANTILE = {
    "cti",
    "cir_nu",
    "cir_fast",
    "cir_plus_fast",
    "cqr",
    "dcp",
    "dcp_cqr",
    "dist_split",
}
SPLINE = {"spice_nd", "spice_hpd"}
GMM = {"hpd_split"}
LQDS = {"lqds_cp", "lqds_hpd"}


def build_models(input_dim: int, seed: int) -> dict[str, torch.nn.Module]:
    taus = np.linspace(0.01, 0.99, 99)
    set_seed(seed)
    quantile = QuantileNet(input_dim, taus, 64, 0.10)
    set_seed(seed)
    gmm = GMMNet(input_dim, 10, 64, 0.10)
    set_seed(seed)
    spline = LinearSplineNet(input_dim, 21, 64, 0.10)
    return {"quantile": quantile, "gmm": gmm, "spline": spline}


def fit_lqds(split, seed: int, epochs: int, device: str) -> LQDSCP:
    model = LQDSCP(
        alpha=0.10,
        hidden_grid=(64,),
        k_grid=(30,),
        smooth=(0.1,),
        dropout=0.10,
        epochs=epochs,
        lr=1e-3,
        patience=40,
        early_stop_min_delta=1e-7,
        batch_size=512,
        fixed_post_kernels={"mw": 0, "hpd": 0},
        knot_anchor="y",
        loss_at="ycrps",
        nll_weight=1.0,
        y_top_q=0.995,
        y_affine="qshift",
        interp="flin",
        device=device,
    )
    model.fit(
        split.X_train,
        split.inverse_y(split.y_train),
        split.X_val,
        split.inverse_y(split.y_val),
        seed=seed,
    )
    return model


def load_or_train(args, split):
    requested = set(args.methods)
    families = build_models(split.X_train.shape[1], args.seed)
    lqds = None
    training_seconds = {}
    checkpoint_dir = ROOT / "artifacts" / "checkpoints" / args.dataset / str(args.seed)
    needed_by_family = {"quantile": QUANTILE, "gmm": GMM, "spline": SPLINE}
    if args.mode == "checkpoint":
        for name in ("quantile", "gmm", "spline"):
            if needed_by_family[name] & requested:
                path = checkpoint_dir / f"{name}.pt"
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing {path}. Generate it with the same command and --mode train."
                    )
                families[name].load_state_dict(
                    torch.load(path, map_location="cpu", weights_only=True)
                )
                families[name].eval()
                training_seconds[name] = None
        if requested & LQDS:
            path = checkpoint_dir / "lqds.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Generate it with the same command and --mode train."
                )
            lqds = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            lqds.device = "cpu"
            training_seconds["lqds"] = None
    else:
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=256,
            learning_rate=1e-3,
            weight_decay=0.0,
            patience=40,
            device=args.device,
        )
        for name, needed in (("quantile", QUANTILE), ("gmm", GMM), ("spline", SPLINE)):
            if requested & needed:
                start = time.perf_counter()
                train_model(
                    families[name],
                    split.X_train,
                    split.y_train,
                    split.X_val,
                    split.y_val,
                    config,
                    args.seed,
                )
                training_seconds[name] = time.perf_counter() - start
        if requested & LQDS:
            start = time.perf_counter()
            lqds = fit_lqds(split, args.seed, args.epochs, args.device)
            training_seconds["lqds"] = time.perf_counter() - start
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for name, model in families.items():
            if requested & needed_by_family[name]:
                torch.save(model.state_dict(), checkpoint_dir / f"{name}.pt")
        if lqds is not None:
            torch.save(lqds, checkpoint_dir / "lqds.pt")
    return families, lqds, training_seconds


def run(args) -> dict:
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    split = make_splits(args.dataset, ROOT / "datasets", args.seed)
    families, lqds, training_seconds = load_or_train(args, split)
    qpredictor = ArrayQuantilePredictor(
        families["quantile"], np.linspace(0.01, 0.99, 99), predict_quantiles
    )
    y_test = split.inverse_y(split.y_test)
    records = {}
    for method in args.methods:
        start = time.perf_counter()
        if method == "cti":
            normalized = run_cti(
                qpredictor, split.X_cal, split.y_cal, split.X_test, 0.10
            )
        elif method == "cir_nu":
            normalized = run_cir_nu(
                qpredictor, split.X_cal, split.y_cal, split.X_test, 0.10
            )
        elif method in QUANTILE:
            normalized = run_quantile_method(
                method, qpredictor, split.X_cal, split.y_cal, split.X_test, 0.10, 400
            )
        elif method in SPLINE:
            normalized = run_density_level_set(
                method,
                families["spline"],
                split.X_cal,
                split.y_cal,
                split.X_test,
                0.10,
                400,
            )
        elif method in GMM:
            normalized = run_density_level_set(
                method,
                families["gmm"],
                split.X_cal,
                split.y_cal,
                split.X_test,
                0.10,
                400,
                100,
            )
        else:
            lqds.score = "mw" if method == "lqds_cp" else "hpd"
            lqds.calibrate(split.X_cal, split.inverse_y(split.y_cal))
            predictions = PredictionSets(lqds.predict(split.X_test))
            records[method] = {
                **evaluate(predictions, y_test, 0.10, split.X_test, args.seed),
                "stage1_seconds": training_seconds["lqds"],
                "stage2_seconds": time.perf_counter() - start,
            }
            continue
        predictions = normalized.affine(split.y_scale, split.y_offset)
        records[method] = {
            **evaluate(predictions, y_test, 0.10, split.X_test, args.seed),
            "stage1_seconds": training_seconds[
                "quantile" if method in QUANTILE else "spline" if method in SPLINE else "gmm"
            ],
            "stage2_seconds": time.perf_counter() - start,
        }
    result = {
        "dataset": args.dataset,
        "seed": args.seed,
        "mode": args.mode,
        "alpha": 0.10,
        "methods": records,
    }
    output = ROOT / "artifacts" / "results" / args.mode
    output.mkdir(parents=True, exist_ok=True)
    with open(
        output / f"{args.dataset}_{args.seed}.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("checkpoint", "train"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=range(2000, 2030), required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="Training budget; 600 reproduces the paper protocol.",
    )
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    print(json.dumps(run(parsed), indent=2))
