"""Fit the four model families used in the learned synthetic experiment."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .data import make_splits
from .intervals import PredictionSets
from .lqdscp import LQDSCP
from .methods import ArrayQuantilePredictor, run_cti, run_density_level_set, run_quantile_method
from .models import (
    GMMNet,
    LinearSplineNet,
    QuantileNet,
    TrainConfig,
    predict_quantiles,
    set_seed,
    train_model,
)


METHODS = (
    "lqds_cp",
    "cti",
    "spice_nd",
    "lqds_hpd",
    "spice_hpd",
    "hpd_split",
    "cqr",
)


def run_synthetic_details(seed: int, threads: int = 1) -> SimpleNamespace:
    torch.set_num_threads(threads)
    set_seed(seed)
    split = make_splits("synth_hard", ".", seed)
    input_dim = split.X_train.shape[1]
    taus = np.linspace(0.01, 0.99, 99)
    config = TrainConfig(
        epochs=600,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=40,
        device="cpu",
    )
    artifacts = {}
    training_times = {}

    for name, make_model in (
        ("quantile", lambda: QuantileNet(input_dim, taus, 64, 0.10)),
        ("gmm", lambda: GMMNet(input_dim, 10, 64, 0.10)),
        ("spline", lambda: LinearSplineNet(input_dim, 21, 64, 0.10)),
    ):
        set_seed(seed)
        model = make_model()
        started = time.perf_counter()
        train_model(
            model,
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
            config,
            seed,
        )
        training_times[name] = time.perf_counter() - started
        artifacts[name] = model

    started = time.perf_counter()
    lqds = LQDSCP(
        alpha=0.10,
        hidden_grid=(64,),
        k_grid=(30,),
        smooth=(0.1,),
        dropout=0.10,
        epochs=600,
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
        device="cpu",
    )
    lqds.fit(
        split.X_train,
        split.inverse_y(split.y_train),
        split.X_val,
        split.inverse_y(split.y_val),
        seed=seed,
    )
    training_times["lqds"] = time.perf_counter() - started
    artifacts.update(lqds=lqds, taus=taus)

    predictor = ArrayQuantilePredictor(artifacts["quantile"], taus, predict_quantiles)
    prediction_sets = {}
    rows = []
    families = {
        "cti": "quantile",
        "cqr": "quantile",
        "spice_nd": "spline",
        "spice_hpd": "spline",
        "hpd_split": "gmm",
        "lqds_cp": "lqds",
        "lqds_hpd": "lqds",
    }
    for method in METHODS:
        if method == "cti":
            normalized = run_cti(
                predictor, split.X_cal, split.y_cal, split.X_test, 0.10
            )
            sets = normalized.affine(split.y_scale, split.y_offset)
        elif method == "cqr":
            normalized = run_quantile_method(
                method, predictor, split.X_cal, split.y_cal, split.X_test, 0.10, 400
            )
            sets = normalized.affine(split.y_scale, split.y_offset)
        elif method in {"spice_nd", "spice_hpd"}:
            normalized = run_density_level_set(
                method,
                artifacts["spline"],
                split.X_cal,
                split.y_cal,
                split.X_test,
                0.10,
                400,
            )
            sets = normalized.affine(split.y_scale, split.y_offset)
        elif method == "hpd_split":
            normalized = run_density_level_set(
                method,
                artifacts["gmm"],
                split.X_cal,
                split.y_cal,
                split.X_test,
                0.10,
                400,
                100,
            )
            sets = normalized.affine(split.y_scale, split.y_offset)
        else:
            lqds.score = "mw" if method == "lqds_cp" else "hpd"
            lqds.calibrate(split.X_cal, split.inverse_y(split.y_cal))
            sets = PredictionSets(lqds.predict(split.X_test))
        prediction_sets[method] = sets
        family = families[method]
        rows.append(
            {
                "method": method,
                "training_seconds": training_times[family],
            }
        )
    return SimpleNamespace(
        split=split,
        artifacts=artifacts,
        training_times=training_times,
        prediction_sets=prediction_sets,
        results=pd.DataFrame(rows),
    )
