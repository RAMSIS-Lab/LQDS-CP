"""Linear-spline operations used by the synthetic SPICE experiments."""

from __future__ import annotations

import numpy as np

from src.intervals import merge_intervals


def spline_at(positions: np.ndarray, heights: np.ndarray, y: np.ndarray) -> np.ndarray:
    index = np.clip(
        (positions < y[:, None]).sum(1) - 1, 0, positions.shape[1] - 2
    )
    row = np.arange(len(y))
    left, right = positions[row, index], positions[row, index + 1]
    h_left, h_right = heights[row, index], heights[row, index + 1]
    inside = (y >= positions[:, 0]) & (y <= positions[:, -1])
    value = h_left + (h_right - h_left) * (y - left) / np.maximum(
        right - left, 1e-300
    )
    return np.where(inside, value, 0.0)


def spline_intervals(
    positions: np.ndarray, heights: np.ndarray, cutoff: float | np.ndarray
) -> list[np.ndarray]:
    cutoff = np.asarray(cutoff, float).reshape(-1, 1)
    left, right = positions[:, :-1], positions[:, 1:]
    h_left, h_right = heights[:, :-1], heights[:, 1:]
    both = (h_left >= cutoff) & (h_right >= cutoff)
    neither = (h_left < cutoff) & (h_right < cutoff)
    crossing = left + (cutoff - h_left) * (right - left) / np.where(
        np.abs(h_right - h_left) < 1e-300, 1e-300, h_right - h_left
    )
    starts = np.where(both, left, np.where(h_right >= cutoff, crossing, left))
    ends = np.where(both, right, np.where(h_right >= cutoff, right, crossing))
    rows = []
    for index in range(len(positions)):
        keep = ~neither[index]
        rows.append(
            merge_intervals(np.column_stack((starts[index, keep], ends[index, keep])))
            if keep.any()
            else np.zeros((0, 2))
        )
    return rows


def spline_mass_below(
    positions: np.ndarray, heights: np.ndarray, cutoff: float | np.ndarray
) -> np.ndarray:
    cutoff = np.asarray(cutoff, float).reshape(-1, 1)
    left, right = positions[:, :-1], positions[:, 1:]
    h_left, h_right = heights[:, :-1], heights[:, 1:]
    delta = np.where(
        np.abs(h_right - h_left) < 1e-300, 1e-300, h_right - h_left
    )
    crossing = left + (cutoff - h_left) * (right - left) / delta
    start = np.where(h_left < cutoff, crossing, left)
    end = np.where(h_right < cutoff, crossing, right)
    neither = (h_left < cutoff) & (h_right < cutoff)
    start_height = h_left + (h_right - h_left) * (start - left) / np.maximum(
        right - left, 1e-300
    )
    end_height = h_left + (h_right - h_left) * (end - left) / np.maximum(
        right - left, 1e-300
    )
    mass_above = np.where(
        neither, 0.0, 0.5 * (end - start) * (start_height + end_height)
    ).sum(1)
    return np.clip(1.0 - mass_above, 0.0, 1.0)
