from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictionSets:
    """A batch of finite unions of closed intervals."""

    sets: list[np.ndarray]

    def __post_init__(self) -> None:
        clean: list[np.ndarray] = []
        for item in self.sets:
            arr = np.asarray(item, dtype=float).reshape(-1, 2)
            if len(arr):
                arr = arr[np.argsort(arr[:, 0])]
                if np.any(arr[:, 1] < arr[:, 0]):
                    raise ValueError("Every interval must have right >= left")
            clean.append(arr)
        object.__setattr__(self, "sets", clean)

    @classmethod
    def from_bounds(cls, bounds: np.ndarray) -> "PredictionSets":
        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n, 2)")
        return cls([row.reshape(1, 2) for row in bounds])

    def __len__(self) -> int:
        return len(self.sets)

    def coverage(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != len(self):
            raise ValueError("y and prediction sets have different lengths")
        return np.asarray(
            [
                bool(np.any((intervals[:, 0] <= yi) & (yi <= intervals[:, 1])))
                for intervals, yi in zip(self.sets, y)
            ]
        )

    def widths(self) -> np.ndarray:
        return np.asarray([np.sum(x[:, 1] - x[:, 0]) for x in self.sets])

    def components(self) -> np.ndarray:
        return np.asarray([len(x) for x in self.sets], dtype=int)

    def affine(self, scale: float, offset: float) -> "PredictionSets":
        transformed = []
        for intervals in self.sets:
            arr = intervals * scale + offset
            if scale < 0:
                arr = arr[:, ::-1]
            transformed.append(arr)
        return PredictionSets(transformed)


def mask_to_intervals(mask: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Convert selected adjacent grid cells to merged intervals."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.asarray(edges, dtype=float)
    if len(edges) != len(mask) + 1:
        raise ValueError("edges must contain one more element than mask")
    padded = np.r_[False, mask, False].astype(int)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return np.column_stack((edges[starts], edges[ends]))


def merge_intervals(intervals: np.ndarray) -> np.ndarray:
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    if not len(intervals):
        return intervals
    intervals = intervals[np.argsort(intervals[:, 0])]
    merged = [intervals[0].copy()]
    for left, right in intervals[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append(np.array([left, right]))
    return np.asarray(merged)


def conformal_rank(n: int, alpha: float) -> int:
    """k = ceil((n+1)(1-alpha)): the 1-based rank of the split-conformal threshold."""
    return int(np.ceil((n + 1) * (1 - alpha)))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the finite-sample split-conformal threshold."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if not len(scores):
        raise ValueError("Calibration scores are empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    k = conformal_rank(len(scores), alpha)
    if k > len(scores):
        return float("inf")
    return float(np.sort(scores)[k - 1])


def lower_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample lower cutoff for a conformity score (larger is better)."""
    scores = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    if not len(scores):
        raise ValueError("Calibration scores are empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    kth = int(np.floor(alpha * (len(scores) + 1)))
    if kth == 0:
        return float("-inf")
    rank = kth - 1
    return float(scores[rank])
