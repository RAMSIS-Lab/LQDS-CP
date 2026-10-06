from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .intervals import PredictionSets


def _worst_group_coverage(
    values: np.ndarray, covered: np.ndarray, groups: int = 5
) -> float:
    cuts = np.unique(np.quantile(values, np.linspace(0, 1, groups + 1)))
    if len(cuts) < 3:
        return float(covered.mean())
    labels = np.clip(np.digitize(values, cuts[1:-1]), 0, len(cuts) - 2)
    rates = [
        covered[labels == i].mean() for i in range(len(cuts) - 1) if np.any(labels == i)
    ]
    return float(min(rates))


def _worst_slab_coverage(X: np.ndarray, covered: np.ndarray, seed: int,) -> float:
    """Compute worst-slab coverage on a held-out logistic direction."""
    X, covered = np.asarray(X), np.asarray(covered, dtype=bool)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(covered))
    fit, test = order[: len(order) // 2], order[len(order) // 2 :]
    miss_fit = (~covered[fit]).astype(int)
    if len(fit) < 10 or miss_fit.min() == miss_fit.max():
        return float(covered.mean())
    classifier = LogisticRegression(max_iter=200).fit(X[fit], miss_fit)
    projection_order = np.argsort(X[test] @ classifier.coef_.ravel())
    sorted_coverage = covered[test][projection_order].astype(float)
    cumulative = np.r_[0.0, np.cumsum(sorted_coverage)]
    rates = []
    for fraction in (0.10, 0.20, 0.30, 0.50):
        width = max(int(np.ceil(fraction * len(test))), 1)
        if width <= len(test):
            rates.extend((cumulative[width:] - cumulative[:-width]) / width)
    return float(np.min(rates))


def _generalized_winkler(
    sets: PredictionSets, y: np.ndarray, alpha: float
) -> np.ndarray:
    """Interval score extended to finite unions using total width and nearest-set distance."""
    covered = sets.coverage(y)
    distances = np.zeros(len(y), dtype=float)
    for i, (row, yi, is_covered) in enumerate(zip(sets.sets, y, covered)):
        if not is_covered:
            distances[i] = (
                min(abs(yi - endpoint) for interval in row for endpoint in interval)
                if len(row)
                else float("inf")
            )
    return sets.widths() + (~covered) * (2.0 / alpha) * distances


def evaluate(
    sets: PredictionSets,
    y: np.ndarray,
    alpha: float,
    X: np.ndarray | None = None,
    seed: int = 2026,
) -> dict[str, float]:
    covered = sets.coverage(y)
    widths = sets.widths()
    label_conditional = _worst_group_coverage(np.asarray(y), covered)
    size_conditional = _worst_group_coverage(widths, covered)
    size_cuts = np.unique(np.quantile(widths, np.linspace(0, 1, 6)))
    size_labels = np.clip(
        np.digitize(widths, size_cuts[1:-1]), 0, max(len(size_cuts) - 2, 0)
    )
    size_rates = [
        covered[size_labels == i].mean()
        for i in range(max(len(size_cuts) - 1, 1))
        if np.any(size_labels == i)
    ]
    winkler = _generalized_winkler(sets, np.asarray(y), alpha)
    target = 1.0 - alpha
    return {
        "coverage": float(covered.mean()),
        "coverage_error": float(abs(covered.mean() - (1 - alpha))),
        "mean_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
        "mean_components": float(sets.components().mean()),
        "mean_winkler_score": float(winkler.mean()),
        "median_winkler_score": float(np.median(winkler)),
        "label_conditional_coverage": label_conditional,
        "label_conditional_coverage_gap": float(target - label_conditional),
        "size_stratified_coverage": size_conditional,
        "size_stratified_coverage_gap": float(
            max(abs(rate - target) for rate in size_rates)
        ),
        "worst_slab_coverage": (
            _worst_slab_coverage(X, covered, seed) if X is not None else float("nan")
        ),
    }
