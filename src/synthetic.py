"""Synthetic conditional distribution with a known density, CDF and quantile
function, built to be hard for every Stage-1 family rather than to suit one:

    y | x  ~  w1(x) * (a1(x) + Gamma(k(x), 0.7))          skewed, shape varies with x
            + w2(x) * (a2(x) + 0.6 * t_3)                   heavy-tailed (Student t, 3 df)
            + w3(x) * N(0.8 x2, 0.12^2)                     narrow spike, only for x1 > 0.3

    x ~ U[-1.5, 1.5]^5 (x3..x5 are irrelevant),  k = 2.5 + x1,  a1 = -3 + sin(2 x1),
    a2 = 1.5 + 1.2 x1,  w3 = 0.3 sigmoid(6 (x1 - 0.3)),  p = sigmoid(2 x2),
    w1 = (1 - w3) p,  w2 = (1 - w3)(1 - p).

The number of modes changes from one to three across x, the right-skewed and
heavy-tailed components are not Gaussian, and the spike is ten times narrower
than the others, so neither a Gaussian mixture nor a fixed-knot spline matches
the family exactly.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import expit

DIM = 5
GAMMA_SCALE, T_SCALE, T_DF, SPIKE_SD = 0.7, 0.6, 3, 0.12


def _parts(x):
    x = np.atleast_2d(np.asarray(x, float))
    x1, x2 = x[:, 0], x[:, 1]
    w3 = 0.3 * expit(6.0 * (x1 - 0.3))
    p = expit(2.0 * x2)
    return dict(w1=(1 - w3) * p, w2=(1 - w3) * (1 - p), w3=w3,
                k=2.5 + x1, a1=-3.0 + np.sin(2.0 * x1), a2=1.5 + 1.2 * x1, m3=0.8 * x2)


def sample(n, rng):
    x = rng.uniform(-1.5, 1.5, (n, DIM))
    P = _parts(x)
    u = rng.uniform(size=n)
    c1, c2 = u < P["w1"], (u >= P["w1"]) & (u < P["w1"] + P["w2"])
    y = np.where(c1, P["a1"] + rng.gamma(P["k"], GAMMA_SCALE),
                 np.where(c2, P["a2"] + T_SCALE * rng.standard_t(T_DF, n),
                          rng.normal(P["m3"], SPIKE_SD)))
    return x, y


def density(y, x):
    """f(y|x); y is [n] or [n, m] (one row of y per x)."""
    P = _parts(x)
    y = np.asarray(y, float)
    col = (lambda v: v[:, None]) if y.ndim == 2 else (lambda v: v)
    return (col(P["w1"]) * stats.gamma.pdf(y - col(P["a1"]), col(P["k"]), scale=GAMMA_SCALE)
            + col(P["w2"]) * stats.t.pdf((y - col(P["a2"])) / T_SCALE, T_DF) / T_SCALE
            + col(P["w3"]) * stats.norm.pdf(y, col(P["m3"]), SPIKE_SD))


def cdf(y, x):
    P = _parts(x)
    y = np.asarray(y, float)
    col = (lambda v: v[:, None]) if y.ndim == 2 else (lambda v: v)
    return (col(P["w1"]) * stats.gamma.cdf(y - col(P["a1"]), col(P["k"]), scale=GAMMA_SCALE)
            + col(P["w2"]) * stats.t.cdf((y - col(P["a2"])) / T_SCALE, T_DF)
            + col(P["w3"]) * stats.norm.cdf(y, col(P["m3"]), SPIKE_SD))


def quantile(taus, x, iters=80):
    """Q(tau|x) -> [n, m] by bisection on the exact CDF (monotone, no failure modes)."""
    x = np.atleast_2d(np.asarray(x, float))
    taus = np.asarray(taus, float)
    lo = np.full((len(x), len(taus)), -200.0)
    hi = np.full((len(x), len(taus)), 200.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        below = cdf(mid, x) < taus[None, :]
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


def make_dataset(n=20_000, seed=1):
    x, y = sample(n, np.random.default_rng(seed))
    return x.astype(np.float32), y.astype(np.float32)
