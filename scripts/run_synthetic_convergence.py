"""Convergence and conditional-coverage experiments from the manuscript.

5  Convergence to the target sets (Theorem 4): LQDS fitted with n_tr in {3000, 12000, 48000}
   training points, with K = 30 fixed and with K growing with n_tr (20, 30, 45); 5 splits each.
   Reported: E_X|C_hat (sym. diff.) C_star| for MW and HPD, density L1 error, sup error eta on the
   modelled range, and the difference between the mean width and the oracle's.
6  Conditional coverage bound (Theorem 5), on the same fits: for 1000 test inputs, the exact
   conditional coverage cov(x) of LQDS-HPD and LQDS-MW (by the true conditional CDF), the score error
   Delta_H(x) = sup_{y in [a,b]} |S_HPD(x,y)/p_core - H(x,y)| with the true rank
   H(x,y) = P{f(Y|x) > f(y|x) | X=x}, and the bound Delta_H(x) + E_X Delta_H + eps_n + 1/n
   (gamma = 0.05). Every input that violates the bound is counted and kept.

Split: n_tr training points, n_tr/6 validation points (the 60/10 ratio of the benchmark),
n_cal = 4000 calibration points, 1000 test inputs; X standardised on the training sample.
usage: python scripts/run_synthetic_convergence.py 12000 30 0
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src import synthetic as SY
from src.lqdscp import LQDSCP, _invert_f, _mass_below, _to_g, _g_level, _D_at
from scripts.run_synthetic import oracle_mw_level, oracle_masks, set_metrics, lqds_law, lqds_cdf_pdf, YG, DY, G, ALPHA

OUT = ROOT / "artifacts" / "synthetic" / "convergence"; OUT.mkdir(parents=True, exist_ok=True)
N_CAL, N_TEST, GAMMA = 4000, 1000, 0.05
GRID_H = np.linspace(YG[0], YG[-1], 2401)      # y grid for Delta_H


def true_rank(x, y):
    """H(x, y) = P{f(Y|x) > f(y|x) | X=x} on a y grid, by the true density on the fine grid YG."""
    H = np.empty((len(x), len(y)))
    for s in range(0, len(x), 100):
        xb = x[s:s + 100]
        D = SY.density(np.broadcast_to(YG, (len(xb), G)), xb)
        Dy = SY.density(np.broadcast_to(y, (len(xb), len(y))), xb)
        for r in range(len(xb)):
            srt = np.sort(D[r]); cum = np.cumsum(srt[::-1])[::-1] * DY     # mass of f >= srt[i]
            j = np.searchsorted(srt, Dy[r], side="right")                    # first index with srt > f(y)
            H[s + r] = np.where(j < G, cum[np.minimum(j, G - 1)], 0.0)
    return H


def fitted_rank(lq, X, y):
    """S_HPD(x, y) / p_core on the grid y (inf outside the row's modelled range), in batches."""
    out = np.empty((len(X), len(y)))
    q, d, dt = lq._params(X)
    p_core = float(lq.net_.tau_span)
    yi = (y - lq.y_off_) / lq.y_scl_
    for r in range(len(X)):
        qq = np.broadcast_to(q[r], (len(y), q.shape[1])); dd = np.broadcast_to(d[r], qq.shape)
        tt = np.broadcast_to(dt[r], (len(y), dt.shape[1]))
        lo, hi = q[r, 0], q[r, -1]; inside = (yi >= lo) & (yi <= hi)
        i, s = _invert_f(qq, dd, tt, np.clip(yi, lo, hi))
        Dy = _D_at(dd, i, s, "flin")
        m = _mass_below(_to_g(dd), tt, _g_level(Dy), strict=True)
        out[r] = np.where(inside, m / p_core, np.nan)
    return out


def main(n_tr, K, split):
    torch.set_num_threads(1)
    tag = f"{n_tr}_{K}_{split}"
    if (OUT / f"{tag}.json").exists():
        print(tag, "exists"); return
    rng = np.random.default_rng(10_000 + split)
    xtr, ytr = SY.sample(n_tr, rng); xva, yva = SY.sample(max(n_tr // 6, 500), rng)
    xc, yc = SY.sample(N_CAL, rng); xt, yt = SY.sample(N_TEST, rng)
    mu, sd = xtr.mean(0), xtr.std(0)
    Z = lambda x: ((x - mu) / sd).astype(np.float32)
    lq = LQDSCP(alpha=ALPHA, hidden_grid=(64,), k_grid=(K,), smooth=(0.1,), dropout=0.1, epochs=600, lr=1e-3,
                patience=40, early_stop_min_delta=1e-7, batch_size=512,
                fixed_post_kernels={"mw": 0, "hpd": 0}, device="cpu", knot_anchor="y",
                loss_at="ycrps", nll_weight=1.0, y_top_q=0.995, y_affine="qshift", interp="flin")
    t0 = time.perf_counter()
    lq.fit(Z(xtr), ytr, Z(xva), yva, seed=split)
    fit_s = time.perf_counter() - t0
    c_mw = oracle_mw_level()
    O_mw, O_hpd = oracle_masks(xt, c_mw)
    rec = {"n_tr": n_tr, "K": K, "split": split, "fit_seconds": fit_s, "epochs": int(getattr(lq.net_, "epochs_", -1)),
           "n_cal": N_CAL, "n_test": N_TEST, "gamma": GAMMA}
    save = {}
    for op, O in (("mw", O_mw), ("hpd", O_hpd)):
        lq.score = op; lq.calibrate(Z(xc), yc)
        rows = lq.predict(Z(xt))
        m = set_metrics(rows, xt, yt, O)
        rec[op] = {"symdiff": m["symdiff"], "width": m["width"], "oracle_width": float(O.sum(1).mean() * DY),
                   "coverage": m["coverage"], "threshold": float(lq.threshold_)}
        save[f"{op}_cond_cov"] = m["cond_cov"]
    # density error (with the exponential tails of the training likelihood outside the knots)
    law = lqds_law(lq, Z(xt))
    _, fhat = lqds_cdf_pdf(law, "flin", YG)
    ftrue = SY.density(np.broadcast_to(YG, (len(xt), G)), xt)
    rec["density_l1"] = float((np.abs(fhat - ftrue).sum(1) * DY).mean())
    q, tlo, thi = law[0], law[3], law[4]
    p_core = thi - tlo
    inside = (YG[None, :] >= q[:, :1]) & (YG[None, :] <= q[:, -1:])
    eta = np.where(inside, np.abs(fhat / p_core - ftrue), 0.0).max(1)
    rec["eta_mean"] = float(eta.mean()); save["eta"] = eta
    rec["mass_outside_range"] = float(np.mean([SY.cdf(q[r, :1], xt[[r]])[0] + 1 - SY.cdf(q[r, -1:], xt[[r]])[0] for r in range(len(xt))]))
    # experiment 6: score error and the bound of Theorem 5
    lq.score = "hpd"; lq.calibrate(Z(xc), yc)
    S = fitted_rank(lq, Z(xt), GRID_H)
    H = true_rank(xt, GRID_H)
    Delta = np.nanmax(np.abs(S - H), axis=1)
    eps = np.sqrt(np.log(2 / GAMMA) / (2 * N_CAL))
    rhs = Delta + Delta.mean() + eps + 1 / N_CAL
    err = np.abs(save["hpd_cond_cov"] - (1 - ALPHA))
    save.update(Delta_H=Delta, bound=rhs, hpd_cov_error=err)
    rec["exp6"] = {"Delta_mean": float(Delta.mean()), "Delta_median": float(np.median(Delta)), "eps_n": float(eps),
                   "violations": int((err > rhs).sum()), "violation_rate": float((err > rhs).mean()),
                   "hpd_cov_error_mean": float(err.mean()), "hpd_cov_q05": float(np.quantile(save["hpd_cond_cov"], 0.05)),
                   "mw_cov_q05": float(np.quantile(save["mw_cond_cov"], 0.05)),
                   "hpd_cov_sd": float(save["hpd_cond_cov"].std()), "mw_cov_sd": float(save["mw_cond_cov"].std()),
                   "note": "Theorem 5 assumes the conditional support lies in [a,b]; the mass outside the modelled range is reported as mass_outside_range"}
    np.savez_compressed(OUT / f"{tag}.npz", xt=xt, yt=yt, **save)
    lq.save(OUT / f"{tag}.pt")
    json.dump(rec, open(OUT / f"{tag}.json", "w"), indent=1)
    print(tag, f"fit {fit_s:.0f}s", "MW sd", round(rec["mw"]["symdiff"], 3), "HPD sd", round(rec["hpd"]["symdiff"], 3),
          "L1", round(rec["density_l1"], 3), "Delta", round(rec["exp6"]["Delta_mean"], 3),
          "violations", rec["exp6"]["violations"], flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
