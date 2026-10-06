"""Vectorised CIR and CIR+ (Guo, Luo & Zhou), reproducing the authors' output.

The published implementation calibrates with a per-sample Python for/while
loop and, for CIR+, predicts with a per-row loop.  Timing that against
vectorised implementations of other methods measures loop overhead, not the
algorithm.  These versions advance every calibration sample one greedy
expansion per iteration in lockstep (at most n_quantiles iterations of numpy
array ops) and are checked to give the identical threshold and intervals.

Algorithm (CIR):  start from the single shortest inter-quantile bin; expand the
side whose neighbouring bin is narrower until y is covered; the conformity
score is the number of expansions.  Predict: the shortest window spanning
`threshold` bins.  CIR+ additionally carries a fractional part of the score --
the width of the final expansion, normalised -- and uses it to decide whether
to widen the test interval by one more bin.
"""
import numpy as np
from .intervals import conformal_rank


def _pad(q, ymin, ymax):
    return np.pad(
        np.asarray(q, float), ((0, 0), (1, 1)), "constant", constant_values=(ymin, ymax)
    )


def _greedy_expand(q, y):
    """Lockstep greedy expansion for all samples at once.

    Returns (n_expansions, last_expansion_width) with n_expansions matching the
    authors' integer threshold (counted from 1) exactly."""
    n, m = q.shape
    r = np.arange(n)
    lo = np.argmin(q[:, 1:] - q[:, :-1], axis=1)
    hi = lo + 1
    steps = np.ones(n)
    last = np.zeros(n)
    done = (y >= q[r, lo]) & (y <= q[r, hi])

    done[:] = False
    for _ in range(m - 1):
        active = ~done & (steps <= m - 1)
        if not active.any():
            break
        can_lo = lo > 0
        can_hi = hi < m - 1
        wl = np.where(can_lo, q[r, lo] - q[r, np.maximum(lo - 1, 0)], np.inf)
        wh = np.where(can_hi, q[r, np.minimum(hi + 1, m - 1)] - q[r, hi], np.inf)
        both = can_lo & can_hi
        go_lo = active & ((both & (wl <= wh)) | (~can_hi & can_lo))
        go_hi = active & ((both & (wl > wh)) | (~can_lo & can_hi))
        last = np.where(go_lo, wl, np.where(go_hi, wh, last))
        lo = np.where(go_lo, lo - 1, lo)
        hi = np.where(go_hi, hi + 1, hi)
        inside = (y >= q[r, lo]) & (y <= q[r, hi])
        newly = active & inside
        steps = np.where(active, steps + 1, steps)
        done = done | newly
    return steps, last


class CIRFast:
    def __init__(self, model, ymin=-np.inf, ymax=np.inf):
        self._model, self.ymin, self.ymax = model, ymin, ymax

    def calibrate(self, x_calib, y_calib, alpha, init_rate=0.9):
        q = _pad(self._model.predict(x_calib), self.ymin, self.ymax)
        y = np.asarray(y_calib, float).reshape(-1)
        steps, _ = _greedy_expand(q, y)

        level = conformal_rank(len(y), alpha) - 1
        assert level < len(y), "too few calibration points for this alpha"
        self.threshold = sorted(steps)[level]

    def predict(self, x_test):
        q = _pad(self._model.predict(x_test), self.ymin, self.ymax)
        n = len(q)
        k = int(self.threshold)
        lo = np.argmin(q[:, k:] - q[:, :-k], axis=1)
        hi = lo + k
        r = np.arange(n)
        return np.stack((q[r, lo], q[r, hi]), axis=1)


class CIRRankFast(CIRFast):
    def calibrate(self, x_calib, y_calib, alpha, init_rate=0.9):
        q = _pad(self._model.predict(x_calib), self.ymin, self.ymax)
        y = np.asarray(y_calib, float).reshape(-1)
        steps, last = _greedy_expand(q, y)
        self.normalized_constant = np.max(y) * 100

        scores = steps - 1 + last / self.normalized_constant

        level = conformal_rank(len(y), alpha) - 1
        assert level < len(y), "too few calibration points for this alpha"
        self.threshold = sorted(scores)[level]

    def predict(self, x_test):
        q = _pad(self._model.predict(x_test), self.ymin, self.ymax)
        n, m = q.shape
        k = int(self.threshold)
        frac = self.threshold - k
        lo = np.argmin(q[:, k:] - q[:, :-k], axis=1)
        hi = lo + k
        r = np.arange(n)
        out = np.stack((q[r, lo], q[r, hi]), axis=1)
        can_lo, can_hi = lo > 0, hi < m - 1
        wl = np.where(can_lo, q[r, lo] - q[r, np.maximum(lo - 1, 0)], np.inf)
        wh = np.where(can_hi, q[r, np.minimum(hi + 1, m - 1)] - q[r, hi], np.inf)
        both = can_lo & can_hi
        pick_lo = (both & (wl < wh)) | (~can_hi & can_lo)
        pick_hi = (both & (wl >= wh)) | (~can_lo & can_hi)
        widen_lo = pick_lo & (wl / self.normalized_constant > frac)
        widen_hi = pick_hi & (wh / self.normalized_constant > frac)
        out[widen_lo, 0] = q[r[widen_lo], lo[widen_lo] - 1]
        out[widen_hi, 1] = q[r[widen_hi], hi[widen_hi] + 1]
        return out
