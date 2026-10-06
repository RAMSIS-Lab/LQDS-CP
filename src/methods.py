from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .intervals import (
    PredictionSets,
    conformal_quantile,
    lower_conformal_quantile,
    mask_to_intervals,
    merge_intervals,
)
from .models import (
    GMMNet,
    LinearSplineNet,
    density_at,
    density_grid,
    predict_spline_knots,
)
from .quantile_baselines import CQR, CQR2, DCP, DistSplit


class ArrayQuantilePredictor:
    """Adapter expected by the published quantile-based methods."""

    def __init__(self, model, taus: np.ndarray, predict_fn):
        self.model = model
        self.quantiles = np.asarray(taus, dtype=float)
        self._predict_fn = predict_fn

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_fn(self.model, np.asarray(X, dtype=np.float32))

    def get_quantiles(self) -> np.ndarray:
        return self.quantiles


from . import cir_fast as _cir_fast


def run_quantile_method(name, predictor, X_cal, y_cal, X_test, alpha, grid_size=1000):
    constructors = {
        "cqr": lambda: CQR(predictor),
        "dist_split": lambda: DistSplit(predictor, ymin=-0.1, ymax=1.1),
        "dcp": lambda: DCP(predictor, ymin=-0.1, ymax=1.1),
        "dcp_cqr": lambda: CQR2(predictor),
        "cir_fast": lambda: _cir_fast.CIRFast(predictor, ymin=-0.1, ymax=1.1),
        "cir_plus_fast": lambda: _cir_fast.CIRRankFast(predictor, ymin=-0.1, ymax=1.1),
    }
    if name not in constructors:
        raise ValueError(name)
    method = constructors[name]()
    method.calibrate(X_cal, y_cal, alpha)
    bounds = np.asarray(method.predict(X_test), dtype=float).reshape(-1, 2)
    bounds.sort(axis=1)
    return PredictionSets.from_bounds(bounds)


def _grid_geometry(grid_size: int) -> tuple[np.ndarray, np.ndarray, float]:
    edges = np.linspace(0.0, 1.0, grid_size + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return (edges, centers, 1.0 / grid_size)


def _normalize_mass(density: np.ndarray, dx: float) -> np.ndarray:
    mass = np.maximum(density, 0) * dx
    return mass / np.maximum(mass.sum(axis=1, keepdims=True), 1e-12)


def _sets_from_masks(masks: np.ndarray, edges: np.ndarray) -> PredictionSets:
    return PredictionSets([mask_to_intervals(row, edges) for row in masks])


def _hpd_scores(
    density: np.ndarray, mass: np.ndarray, observed_density: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            row_mass[row_density <= threshold + 1e-12].sum()
            for (row_density, row_mass, threshold) in zip(
                density, mass, observed_density
            )
        ]
    )


def _linear_intervals(
    positions: np.ndarray, heights: np.ndarray, cutoffs: np.ndarray | float
) -> list[np.ndarray]:
    """Exact roots of a conditional piecewise-linear density above a cutoff."""
    cutoffs = np.broadcast_to(np.asarray(cutoffs, dtype=float), (len(positions),))
    output: list[np.ndarray] = []
    for (p, h, cutoff) in zip(positions, heights, cutoffs):
        pieces = []
        for (left, right, hl, hr) in zip(p[:-1], p[1:], h[:-1], h[1:]):
            if hl >= cutoff and hr >= cutoff:
                pieces.append((left, right))
            elif hl < cutoff and hr < cutoff:
                continue
            else:
                crossing = left + (cutoff - hl) * (right - left) / (hr - hl)
                pieces.append((crossing, right) if hr >= cutoff else (left, crossing))
        output.append(merge_intervals(np.asarray(pieces, dtype=float).reshape(-1, 2)))
    return output


def _linear_mass_below(
    positions: np.ndarray, heights: np.ndarray, cutoffs: np.ndarray | float
) -> np.ndarray:
    """Exact H(c|x)=integral 1{f(y|x)<=c} f(y|x)dy for SPICE n=1.

    Vectorised over rows and segments; agrees with the per-row loop of the
    authors' implementation to round-off (max abs difference 3e-16 over 200
    randomised cases, the two summation orders differing only in the last
    bit).  Every segment is integrated separately,
    as in the original: merging touching selected segments before trapezoidal
    integration would erase their knots.
    """
    cutoffs = np.broadcast_to(np.asarray(cutoffs, dtype=float), (len(positions),))[
        :, None
    ]
    (left, right) = (positions[:, :-1], positions[:, 1:])
    (hl, hr) = (heights[:, :-1], heights[:, 1:])
    span = right - left
    dh = hr - hl
    cross = left + (cutoffs - hl) * span / np.where(dh == 0.0, 1.0, dh)
    sel_left = np.where(hl < cutoffs, cross, left)
    sel_right = np.where((hl >= cutoffs) & (hr < cutoffs), cross, right)
    sel_hl = hl + dh * (sel_left - left) / span
    sel_hr = hl + dh * (sel_right - left) / span
    skip = (hl < cutoffs) & (hr < cutoffs)
    contrib = np.where(skip, 0.0, 0.5 * (sel_right - sel_left) * (sel_hl + sel_hr))
    return np.clip(1.0 - contrib.sum(axis=1), 0.0, 1.0)


def _run_spice_exact(
    name: str,
    model: LinearSplineNet,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> PredictionSets:
    """Analytical SPICE-n1 calibration and inversion from the pinned source."""
    (p_cal, h_cal) = predict_spline_knots(model, X_cal)
    (p_test, h_test) = predict_spline_knots(model, X_test)
    at_cal = density_at(model, X_cal, y_cal)
    if name == "spice_nd":
        threshold = -conformal_quantile(-at_cal, alpha)
        return PredictionSets(_linear_intervals(p_test, h_test, threshold))
    scores = _linear_mass_below(p_cal, h_cal, at_cal)
    threshold = -conformal_quantile(-scores, alpha)
    lower = np.zeros(len(X_test), dtype=float)
    upper = h_test.max(axis=1)
    mid = (lower + upper) / 2
    for _ in range(15):
        mid = (lower + upper) / 2
        mass = _linear_mass_below(p_test, h_test, mid)
        below = mass < threshold
        lower = np.where(below, mid, lower)
        upper = np.where(below, upper, mid)
    return PredictionSets(_linear_intervals(p_test, h_test, mid))


def run_density_level_set(
    name, model, X_cal, y_cal, X_test, alpha, grid_size, neighbors=100
):
    if name in {"spice_nd", "spice_hpd"}:
        if not isinstance(model, LinearSplineNet):
            raise TypeError(f"{name} requires LinearSplineNet")
        return _run_spice_exact(name, model, X_cal, y_cal, X_test, alpha)
    if name != "hpd_split" or not isinstance(model, GMMNet):
        raise ValueError(name)
    (edges, centers, dx) = _grid_geometry(grid_size)
    d_cal = density_grid(model, X_cal, centers)
    d_test = density_grid(model, X_test, centers)
    at_cal = density_at(model, X_cal, y_cal)
    mass_cal = _normalize_mass(d_cal, dx)
    mass_test = _normalize_mass(d_test, dx)
    scores = _hpd_scores(d_cal, mass_cal, at_cal)
    cutoff = lower_conformal_quantile(scores, alpha)
    masks = np.empty_like(d_test, dtype=bool)
    for (i, (density, mass)) in enumerate(zip(d_test, mass_test)):
        order = np.argsort(density)
        rank_mass = np.empty_like(density)
        rank_mass[order] = np.cumsum(mass[order])
        masks[i] = rank_mass >= cutoff
    return _sets_from_masks(masks, edges)


def run_cti(
    predictor: ArrayQuantilePredictor,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> PredictionSets:
    """Conformal Thresholded Intervals (Luo & Zhou, AAAI 2025).

    Equal-mass interquantile cells from the backbone's own quantile grid; the
    conformity score is the RAW length of the cell the response falls into, and
    the prediction set is the union of all cells whose length is at most the
    calibrated global threshold.

    Run on the method's published backbone (multi-output quantile regression),
    not on our spline -- the quantile grid is whatever the predictor exposes.
    """
    qc = np.asarray(predictor.predict(X_cal), dtype=float)
    qt = np.asarray(predictor.predict(X_test), dtype=float)
    qc = np.sort(qc, axis=1)
    qt = np.sort(qt, axis=1)
    len_cal = np.diff(qc, axis=1)
    len_test = np.diff(qt, axis=1)
    (lo, hi) = (qc[:, :-1], qc[:, 1:])
    inside = (y_cal[:, None] >= lo) & (y_cal[:, None] <= hi)
    scores = np.where(
        inside.any(axis=1), np.where(inside, len_cal, np.inf).min(axis=1), np.inf
    )
    threshold = conformal_quantile(scores, alpha)
    sets = []
    for i in range(len(qt)):
        keep = np.flatnonzero(len_test[i] <= threshold)
        if len(keep) == 0:
            keep = np.array([int(np.argmin(len_test[i]))])
        sets.append(merge_intervals(np.column_stack((qt[i, keep], qt[i, keep + 1]))))
    return PredictionSets(sets)


def run_cir_nu_from_quantiles(
    q_cal: np.ndarray,
    y_cal: np.ndarray,
    q_test: np.ndarray,
    alpha: float,
    ymin: float = -0.1,
    ymax: float = 1.1,
) -> PredictionSets:
    """Non-unimodal CIR: conformally select within-input width ranks.

    The predicted quantiles, together with the same two endpoint pads used by
    CIR/CIR+, define equal-probability interquantile bins.  For each input the
    bins are ranked from shortest to longest.  A calibration response scores
    as the rank of its containing bin, and prediction returns the union of all
    bins at or below the split-conformal cutoff rank.

    Stable sorting gives deterministic, distinct ranks when fitted quantiles
    tie.  Since the score and set use that same ordering, ties do not affect
    conformal score/set consistency.
    """
    q_cal = np.sort(np.asarray(q_cal, dtype=float), axis=1)
    q_test = np.sort(np.asarray(q_test, dtype=float), axis=1)
    y_cal = np.asarray(y_cal, dtype=float).reshape(-1)
    if q_cal.ndim != 2 or q_test.ndim != 2 or q_cal.shape[1] != q_test.shape[1]:
        raise ValueError(
            "calibration and test quantiles must be 2D with the same number of columns"
        )
    if len(q_cal) != len(y_cal):
        raise ValueError("q_cal and y_cal have different lengths")
    q_cal = np.pad(
        np.clip(q_cal, ymin, ymax), ((0, 0), (1, 1)), constant_values=(ymin, ymax)
    )
    q_test = np.pad(
        np.clip(q_test, ymin, ymax), ((0, 0), (1, 1)), constant_values=(ymin, ymax)
    )

    def ranks(q: np.ndarray) -> np.ndarray:
        order = np.argsort(np.diff(q, axis=1), axis=1, kind="stable")
        out = np.empty_like(order)
        out[np.arange(len(q))[:, None], order] = np.arange(1, q.shape[1])
        return out

    cal_ranks = ranks(q_cal)
    cal_bin = np.sum(y_cal[:, None] >= q_cal[:, 1:], axis=1)
    cal_bin = np.clip(cal_bin, 0, q_cal.shape[1] - 2)
    scores = cal_ranks[np.arange(len(y_cal)), cal_bin]
    cutoff = conformal_quantile(scores, alpha)
    test_ranks = ranks(q_test)
    sets = []
    for (edges, row_ranks) in zip(q_test, test_ranks):
        keep = row_ranks <= cutoff
        sets.append(mask_to_intervals(keep, edges))
    return PredictionSets(sets)


def run_cir_nu(
    predictor: ArrayQuantilePredictor,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> PredictionSets:
    """Run non-unimodal CIR from a fitted quantile predictor."""
    return run_cir_nu_from_quantiles(
        predictor.predict(X_cal), y_cal, predictor.predict(X_test), alpha
    )
