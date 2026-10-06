"""Synthetic study on the known conditional law used in the manuscript,
separated into the three questions the paper asks.

  oracle  Stage 2 only.  Every method receives the EXACT conditional law at the same
          resolution (K knots / quantiles / spline knots) and runs its own Stage 2.
          Reported separately: the representation error (sets at the exact target
          level, no conformal step) and the error after split-conformal calibration.
  e2e     Stage 1 and end to end, from one set of fits per split (benchmark protocol):
            (a) Stage-1 accuracy of each family's fitted conditional law against the
                truth: CRPS excess int (F_hat - F)^2 dy, quantile error, density L1,
                test NLL, parameters, training time;
            (b) end-to-end sets: coverage, width, conditional coverage, symmetric
                difference to the oracle set of the method's own target;
            (c) cross ablation: the baselines' Stage 2 (CTI, SPICE, HPD-split) run on
                the LQDS Stage-1 fit, and the LQDS Stage 2 run on the QuantileNet and
                SPICE Stage-1 fits, so a gain can be attributed to one stage.

usage:  python scripts/run_synthetic.py oracle --seeds 30 --Ks 20,30,50,99
        python scripts/run_synthetic.py e2e --seeds 10
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src import synthetic as SY
from src.lqdscp import (_invert, _invert_f, _D_at, _sets, _mass_below, _hpd_threshold,
                             _to_g, _g_level)
from src.intervals import conformal_quantile, lower_conformal_quantile, merge_intervals, mask_to_intervals
from src.synthetic_runner import run_synthetic_details
from scripts.synthetic_helpers import spline_at, spline_intervals, spline_mass_below

ALPHA = 0.10
YLO, YHI, G = -9.0, 15.0, 9601
YG = np.linspace(YLO, YHI, G); DY = YG[1] - YG[0]
EDGES = np.r_[YG - DY / 2, YG[-1] + DY / 2]
RES = ROOT / "artifacts" / "synthetic"
RES.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ oracle targets
def oracle_mw_level(n=1_000_000, seed=12345):
    x, y = SY.sample(n, np.random.default_rng(seed))
    return float(np.quantile(SY.density(y, x), ALPHA))


def oracle_masks(x, c_mw, batch=500):
    """MW: {f >= c}; HPD: the per-x highest-density region of mass 1 - alpha."""
    MW, HPD = [], []
    for s in range(0, len(x), batch):
        D = SY.density(np.broadcast_to(YG, (len(x[s:s + batch]), G)), x[s:s + batch])
        MW.append(D >= c_mw)
        o = np.argsort(-D, 1); cum = np.cumsum(np.take_along_axis(D, o, 1), 1) * DY
        k = np.minimum((cum < 1 - ALPHA).sum(1), G - 1)
        c = D[np.arange(len(D)), o[np.arange(len(D)), k]]
        HPD.append(D >= c[:, None])
    return np.vstack(MW), np.vstack(HPD)


def mask_of(rows):
    M = np.zeros((len(rows), G), bool)
    for i, r in enumerate(rows):
        for a, b in np.asarray(r).reshape(-1, 2):
            M[i, (YG >= a) & (YG <= b)] = True
    return M


def set_metrics(rows, x, y, mask_oracle):
    rows = [np.asarray(r, float).reshape(-1, 2) for r in rows]
    cov = np.array([bool(len(r)) and bool(((y[i] >= r[:, 0]) & (y[i] <= r[:, 1])).any()) for i, r in enumerate(rows)])
    w = np.array([(r[:, 1] - r[:, 0]).sum() for r in rows])
    ccov = np.array([(SY.cdf(r[:, 1], np.repeat(x[[i]], len(r), 0)) - SY.cdf(r[:, 0], np.repeat(x[[i]], len(r), 0))).sum()
                     if len(r) else 0.0 for i, r in enumerate(rows)])
    sd = (mask_of(rows) ^ mask_oracle).sum(1) * DY
    return dict(coverage=float(cov.mean()), width=float(w.mean()), comps=float(np.mean([len(r) for r in rows])),
                symdiff=float(sd.mean()), cond_cov=ccov)


def best_time(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t0)
    return min(ts), r


# ---------------------------------------------------- Stage-2 readers (generic)
def lqds_rep_from_knots(q, f, interp):
    """(q, d, dtau) for the LQDS reading of knot values q_k with density f_k:
    levels by the representation's own consistency rule (no renormalisation)."""
    d = 1.0 / np.maximum(f, 1e-300)
    if interp == "flin":
        f0, f1 = f[:, :-1], f[:, 1:]; df = f1 - f0
        ok = np.abs(df) > 1e-9 * np.maximum(f0, 1e-300)
        lm = np.where(ok, df / np.where(ok, np.log(np.maximum(f1, 1e-300) / np.maximum(f0, 1e-300)), 1.0), 0.5 * (f0 + f1))
        dt = np.diff(q, axis=1) * lm
    else:
        dt = 2.0 * np.diff(q, axis=1) / (d[:, :-1] + d[:, 1:])
    return q, d, dt


def lqds_stage2(cal, test, ycal, hpd, interp, exact_level=None):
    """LQDS Stage 2 on (q, d, dtau).  exact_level: skip calibration and use the
    given target (MW: a density level; HPD: a mass) -- representation error."""
    def scores(q, d, dt, y):
        lo, hi = q[:, 0], q[:, -1]; inside = (y >= lo) & (y <= hi)
        i, s = (_invert_f if interp == "flin" else _invert)(q, d, dt, np.clip(y, lo, hi))
        Dy = _D_at(d, i, s, interp)
        if not hpd:
            return np.where(inside, Dy, np.inf)
        if interp == "flin":
            return np.where(inside, _mass_below(_to_g(d), dt, _g_level(Dy), strict=True), np.inf)
        return np.where(inside, _mass_below(d, dt, Dy, strict=True), np.inf)

    def build(q, d, dt, t):
        if not hpd:
            return _sets(q, d, dt, np.full(len(q), t), interp)
        if interp == "flin":
            cg = _hpd_threshold(_to_g(d), dt, t)
            return _sets(q, d, dt, np.where(cg < 0, -1.0 / np.minimum(cg, -1e-300), np.inf), "flin")
        return _sets(q, d, dt, _hpd_threshold(d, dt, t))

    q, d, dt = test
    if exact_level is not None:
        t = (1.0 / exact_level) if not hpd else exact_level      # HPD: true-mass units
        return 0.0, 0.0, build(q, d, dt, t)
    qc, dc, dtc = cal
    t_cal, t = best_time(lambda: conformal_quantile(scores(qc, dc, dtc, ycal), ALPHA))
    t_test, rows = best_time(lambda: build(q, d, dt, t))
    return t_cal, t_test, rows


def spice_stage2(cal, test, ycal, hpd, exact_level=None):
    Pt, Ht = test
    def hpd_rows(P, H, thr):
        lo = np.zeros(len(P)); hi = H.max(1)
        for _ in range(30):
            mid = (lo + hi) / 2
            below = spline_mass_below(P, H, mid) < thr
            lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
        return spline_intervals(P, H, (lo + hi) / 2)
    if exact_level is not None:
        return 0.0, 0.0, (spline_intervals(Pt, Ht, exact_level) if not hpd else hpd_rows(Pt, Ht, 1 - exact_level))
    Pc, Hc = cal
    if not hpd:
        t_cal, thr = best_time(lambda: -conformal_quantile(-spline_at(Pc, Hc, ycal), ALPHA))
        t_test, rows = best_time(lambda: spline_intervals(Pt, Ht, thr))
        return t_cal, t_test, rows
    t_cal, thr = best_time(lambda: -conformal_quantile(-spline_mass_below(Pc, Hc, spline_at(Pc, Hc, ycal)), ALPHA))
    t_test, rows = best_time(lambda: hpd_rows(Pt, Ht, thr))
    return t_cal, t_test, rows


def cti_stage2(Qc, Qt, ycal, exact_level=None):
    """exact_level: the oracle density level c*; cells of normalised width <= 1/c* are kept."""
    if exact_level is not None:
        L = np.diff(Qt, axis=1) / np.diff(np.linspace(0.01, 0.99, Qt.shape[1]))[None, :]
        keep = L <= 1.0 / exact_level
        return 0.0, 0.0, [merge_intervals(np.column_stack((Qt[i, :-1][keep[i]], Qt[i, 1:][keep[i]]))) if keep[i].any()
                          else np.zeros((0, 2)) for i in range(len(Qt))]
    def cal():
        L = np.diff(Qc, axis=1); inside = (ycal[:, None] >= Qc[:, :-1]) & (ycal[:, None] <= Qc[:, 1:])
        return conformal_quantile(np.where(inside.any(1), np.where(inside, L, np.inf).min(1), np.inf), ALPHA)
    t_cal, thr = best_time(cal)
    def build():
        keep = np.diff(Qt, axis=1) <= thr
        return [merge_intervals(np.column_stack((Qt[i, :-1][keep[i]], Qt[i, 1:][keep[i]]))) if keep[i].any()
                else np.zeros((0, 2)) for i in range(len(Qt))]
    t_test, rows = best_time(build)
    return t_cal, t_test, rows


def hpd_split_stage2(Dc, Dt, edges, ycal, fy_cal, exact_level=None):
    """HPD-split on densities tabulated on a grid (cell masses normalised per row).
    exact_level: keep the densest cells up to that grid mass (no calibration)."""
    w = np.diff(edges)
    if exact_level is not None:
        Mt = Dt * w; Mt = Mt / Mt.sum(1, keepdims=True)
        o = np.argsort(-Dt, 1); cm = np.cumsum(np.take_along_axis(Mt, o, 1), 1)
        k = (cm < exact_level).sum(1)
        rows = []
        for i in range(len(Dt)):
            keep = np.zeros(len(w), bool); keep[o[i, :k[i] + 1]] = True
            rows.append(mask_to_intervals(keep, edges))
        return 0.0, 0.0, rows
    Mc = Dc * w; Mc = Mc / Mc.sum(1, keepdims=True)
    Mt = Dt * w; Mt = Mt / Mt.sum(1, keepdims=True)
    t_cal, cut = best_time(lambda: lower_conformal_quantile((Mc * (Dc <= fy_cal[:, None] + 1e-12)).sum(1), ALPHA))
    def build():
        rows = []
        for i in range(len(Dt)):
            keep = (Mt[i] * (Dt[i] <= Dt[i][:, None] + 1e-12)).sum(1) >= cut
            rows.append(mask_to_intervals(keep, edges) if keep.any() else np.zeros((0, 2)))
        return rows
    t_test, rows = best_time(build)
    return t_cal, t_test, rows


# ============================================================ Stage 2: oracle
def run_oracle(a):
    c_mw = oracle_mw_level()
    Ks = [int(k) for k in str(a.Ks).split(",")] if a.Ks else [a.K]
    tq99 = np.linspace(0.01, 0.99, 99)
    R = {}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        xc, yc = SY.sample(a.n, rng); xt, yt = SY.sample(a.n, rng)
        O_mw, O_hpd = oracle_masks(xt, c_mw)
        for K in Ks:
            oracle_one_K(a, K, len(Ks) > 1, xc, yc, xt, yt, O_mw, O_hpd, c_mw, tq99, R)
        print(f"seed {seed}: " + "  ".join(f"{k} {np.mean(v['symdiff']):.3f}" for k, v in R.items()), flush=True)
    finish_oracle(a, Ks, c_mw, R)


def oracle_one_K(a, K, sweep, xc, yc, xt, yt, O_mw, O_hpd, c_mw, tq99, R):
        """Every method gets the exact law at resolution K (same knots for the knot-based
        readers); the native-resolution baselines (CTI with 99 quantiles, HPD-Split on
        the benchmark's 400-point grid) are run once, at the first K."""
        tauK = np.linspace(0.001, 0.999, K); tq = np.linspace(0.01, 0.99, K)
        Qc, Qt = SY.quantile(tauK, xc), SY.quantile(tauK, xt)
        fc, ft = SY.density(Qc, xc), SY.density(Qt, xt)
        runs = {}
        for interp in ("dlin", "flin"):
            Lc, Lt = lqds_rep_from_knots(Qc, fc, interp), lqds_rep_from_knots(Qt, ft, interp)
            tag = "" if interp == a.interp else f" [{interp}]"
            runs[f"LQDS-MW{tag}"] = ("mw", lambda L=(Lc, Lt), i=interp, e=None: lqds_stage2(L[0], L[1], yc, False, i, e))
            runs[f"LQDS-HPD{tag}"] = ("hpd", lambda L=(Lc, Lt), i=interp, e=None: lqds_stage2(L[0], L[1], yc, True, i, e))
        runs["SPICE-ND"] = ("mw", lambda e=None: spice_stage2((Qc, fc), (Qt, ft), yc, False, e))
        runs["SPICE-HPD"] = ("hpd", lambda e=None: spice_stage2((Qc, fc), (Qt, ft), yc, True, e))
        Tc, Tt = SY.quantile(tq, xc), SY.quantile(tq, xt)
        runs["CTI"] = ("mw", lambda e=None: cti_stage2(Tc, Tt, yc, e))
        edges = np.linspace(YLO, YHI, K + 1); cen = (edges[:-1] + edges[1:]) / 2
        Hc = SY.density(np.broadcast_to(cen, (len(xc), K)), xc); Ht = SY.density(np.broadcast_to(cen, (len(xt), K)), xt)
        runs["HPD-Split"] = ("hpd", lambda e=None, H=(Hc, Ht, edges): hpd_split_stage2(H[0], H[1], H[2], yc, SY.density(yc, xc), e))
        runs = {f"{k} | K={K}": v for k, v in runs.items()}
        if K == (int(str(a.Ks).split(",")[0]) if a.Ks else a.K):
            T9c, T9t = SY.quantile(tq99, xc), SY.quantile(tq99, xt)
            runs["CTI (native T=99)"] = ("mw", lambda e=None: cti_stage2(T9c, T9t, yc, e))
            e4 = np.linspace(YLO, YHI, 401); c4 = (e4[:-1] + e4[1:]) / 2
            H4c = SY.density(np.broadcast_to(c4, (len(xc), 400)), xc); H4t = SY.density(np.broadcast_to(c4, (len(xt), 400)), xt)
            runs["HPD-Split (native grid 400)"] = ("hpd", lambda e=None: hpd_split_stage2(H4c, H4t, e4, yc, SY.density(yc, xc), e))
        for name, (tgt, fn) in runs.items():
            O = O_mw if tgt == "mw" else O_hpd
            t1, t2, rows = fn()
            m = set_metrics(rows, xt, yt, O)
            e = R.setdefault(name, dict(target=tgt, t_cal=[], t_test=[], cov=[], width=[], comps=[], symdiff=[], repr_symdiff=[]))
            e["t_cal"].append(t1); e["t_test"].append(t2); e["cov"].append(m["coverage"]); e["width"].append(m["width"])
            e["comps"].append(m["comps"]); e["symdiff"].append(m["symdiff"])
            ex = fn(e=(c_mw if tgt == "mw" else 1 - ALPHA))           # representation error at the exact target
            if ex is not None:
                e["repr_symdiff"].append(set_metrics(ex[2], xt, yt, O)["symdiff"])


def finish_oracle(a, Ks, c_mw, R):
    out = dict(Ks=Ks, n=a.n, seeds=a.seeds, interp=a.interp, c_mw=c_mw,
               methods={k: dict(target=v["target"], **{m: ([float(np.mean(v[m])), float(np.std(v[m]))] if v[m] else None)
                                                    for m in ("t_cal", "t_test", "cov", "width", "comps", "symdiff", "repr_symdiff")})
                        for k, v in R.items()})
    out["per_seed"] = {k: {m: [float(x) for x in v[m]] for m in ("t_cal", "t_test", "cov", "width", "comps", "symdiff", "repr_symdiff")}
                       for k, v in R.items()}                                  # raw values per seed
    json.dump(out, open(a.out or RES / "synth_oracle.json", "w"), indent=1)
    print(f"\n{'method':<36}{'target':>7}{'repr sd':>9}{'sd':>8}{'cov':>8}{'width':>8}{'cal ms':>9}{'test ms':>9}")
    for k, v in out["methods"].items():
        r = v["repr_symdiff"][0] if v["repr_symdiff"] else float("nan")
        print(f"{k:<36}{v['target']:>7}{r:>9.3f}{v['symdiff'][0]:>8.3f}{v['cov'][0]:>8.3f}{v['width'][0]:>8.3f}"
              f"{1e3 * v['t_cal'][0]:>9.2f}{1e3 * v['t_test'][0]:>9.2f}")


# ================================================= Stage 1 + end to end + cross
def lqds_law(lq, X):
    """(q, d, dt, tau_lo, tau_hi) of a fitted LQDSCP in ORIGINAL response units."""
    q, d, dt = lq._params(X)
    return q * lq.y_scl_ + lq.y_off_, d * lq.y_scl_, dt, lq.net_.tau_lo, lq.net_.tau_lo + lq.net_.tau_span


def lqds_cdf_pdf(law, interp, grid):
    """F and f of the LQDS representation on a y-grid, with the exponential tails
    of the training likelihood outside the knot range."""
    q, d, dt, tlo, thi = law
    n = len(q); F = np.empty((n, len(grid))); f = np.empty((n, len(grid)))
    T = tlo + np.concatenate([np.zeros((n, 1)), np.cumsum(dt, 1)], 1)
    for r in range(n):
        y = grid; qq = np.broadcast_to(q[r], (len(y), q.shape[1])); dd = np.broadcast_to(d[r], qq.shape)
        tt = np.broadcast_to(dt[r], (len(y), dt.shape[1]))
        lo, hi = q[r, 0], q[r, -1]
        i, s = (_invert_f if interp == "flin" else _invert)(qq, dd, tt, np.clip(y, lo, hi))
        Dy = _D_at(dd, i, s, interp)
        Fin = T[r, i] + s * dt[r, i]            # s is the tau-fraction of the segment in both readings
        m_lo, m_hi = tlo, 1 - thi
        d0, dK = max(d[r, 0], 1e-12), max(d[r, -1], 1e-12)
        Flo = m_lo * np.exp(-(lo - y) / (m_lo * d0)); Fhi = 1 - m_hi * np.exp(-(y - hi) / (m_hi * dK))
        F[r] = np.where(y < lo, Flo, np.where(y > hi, Fhi, Fin))
        f[r] = np.where(y < lo, Flo / (m_lo * d0), np.where(y > hi, (1 - Fhi) / (m_hi * dK), 1.0 / np.maximum(Dy, 1e-300)))
    return F, f


def quantile_cdf_pdf(Q, taus, grid):
    """CDF by linear interpolation of the fitted quantiles (tails extrapolated
    linearly to 0 and 1), density = the implied piecewise-constant 1/(dQ/dtau)."""
    Q = np.sort(Q, 1); n = len(Q)
    F = np.empty((n, len(grid))); f = np.zeros((n, len(grid)))
    for r in range(n):
        qq = Q[r]; s0 = (taus[1] - taus[0]) / max(qq[1] - qq[0], 1e-12); s1 = (taus[-1] - taus[-2]) / max(qq[-1] - qq[-2], 1e-12)
        Fi = np.interp(grid, qq, taus)
        Fi = np.where(grid < qq[0], taus[0] - (qq[0] - grid) * s0, np.where(grid > qq[-1], taus[-1] + (grid - qq[-1]) * s1, Fi))
        F[r] = np.clip(Fi, 0, 1)
        f[r] = np.gradient(F[r], grid)
    return F, f


def density_cdf(f, grid):
    F = np.concatenate([np.zeros((len(f), 1)), np.cumsum((f[:, 1:] + f[:, :-1]) / 2 * np.diff(grid), 1)], 1)
    return F


def stage1_metrics(F, f, x, y_at_f, tau_eval=np.linspace(0.05, 0.95, 19)):
    Ft = SY.cdf(np.broadcast_to(YG, (len(x), G)), x); ft = SY.density(np.broadcast_to(YG, (len(x), G)), x)
    crps_ex = ((F - Ft) ** 2).sum(1) * DY
    dl1 = np.abs(f - ft).sum(1) * DY
    Qt = SY.quantile(tau_eval, x)
    Qh = np.array([np.interp(tau_eval, np.maximum.accumulate(F[r]) + np.arange(G) * 1e-15, YG) for r in range(len(F))])
    return dict(crps_excess=float(crps_ex.mean()), density_l1=float(dl1.mean()),
                quantile_mae=float(np.abs(Qh - Qt).mean()), nll=float(-np.log(np.maximum(y_at_f, 1e-12)).mean()))


def run_e2e(a):
    import torch
    from src.models import predict_quantiles, density_grid, density_at
    TARGET = {"lqds_cp": "mw", "cti": "mw", "spice_nd": "mw", "lqds_hpd": "hpd", "spice_hpd": "hpd", "hpd_split": "hpd", "cqr": "interval"}
    interp = a.interp or "flin"
    c_mw = oracle_mw_level()
    S1, E2E, X2 = {}, {}, {}
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        torch.set_num_threads(a.threads)
        run = run_synthetic_details(seed, a.threads)
        sp = run.split; A = run.artifacts
        unx = lambda X: X * sp.x_scaler.scale_ + sp.x_scaler.mean_
        rng = np.random.default_rng(seed); ev = np.sort(rng.choice(len(sp.X_test), min(a.n_eval, len(sp.X_test)), replace=False))
        Xt, xt, yt = sp.X_test[ev], unx(sp.X_test[ev]), sp.inverse_y(sp.y_test[ev]).astype(float)
        Xc, xc, yc = sp.X_cal, unx(sp.X_cal), sp.inverse_y(sp.y_cal).astype(float)
        O_mw, O_hpd = oracle_masks(xt, c_mw)
        O = {"mw": O_mw, "hpd": O_hpd}
        scale = float(sp.y_scaler.data_range_[0]); lo_n = float(sp.y_scaler.data_min_[0])
        g_n = (YG - lo_n) / scale
        # ---------------- (a) Stage-1 accuracy
        lq = A["lqds"]; lawt = lqds_law(lq, Xt)
        if a.save_models:
            md = Path(a.save_models); md.mkdir(parents=True, exist_ok=True)
            lq.save(md / f"lqds_{seed}.pt")
            torch.save({k: v.state_dict() for k, v in A.items() if isinstance(v, torch.nn.Module)}, md / f"baselines_{seed}.pt")
        F, f = lqds_cdf_pdf(lawt, interp, YG)
        lawy = lqds_law(lq, Xt)
        fy = np.array([np.interp(yt[r], YG, f[r]) for r in range(len(yt))])
        fams = {"LQDS (ours)": (F, f, fy)}
        taus = A["taus"]; Qt = sp.inverse_y(predict_quantiles(A["quantile"], Xt).reshape(-1)).reshape(len(Xt), -1)
        Fq, fq = quantile_cdf_pdf(Qt, taus, YG)
        fams["QuantileNet (99)"] = (Fq, fq, np.array([np.interp(yt[r], YG, fq[r]) for r in range(len(yt))]))
        fg = density_grid(A["gmm"], Xt, g_n.astype(np.float32)) / scale
        fams["MDN (GMM, 10)"] = (density_cdf(fg, YG), fg, density_at(A["gmm"], Xt, sp.y_test[ev]) / scale)
        fs = density_grid(A["spline"], Xt, g_n.astype(np.float32)); fs = np.where((g_n >= 0) & (g_n <= 1), fs, 0.0) / scale
        fams["SPICE net (21)"] = (density_cdf(fs, YG), fs, density_at(A["spline"], Xt, sp.y_test[ev]) / scale)
        fam_key = {"LQDS (ours)": "lqds", "QuantileNet (99)": "quantile", "MDN (GMM, 10)": "gmm", "SPICE net (21)": "spline"}
        for name, (F_, f_, fy_) in fams.items():
            m = stage1_metrics(F_, f_, xt, fy_)
            fk = fam_key[name]
            m["train_s"] = float(run.training_times.get(fk, np.nan)) if hasattr(run, "training_times") else float("nan")
            mod = A[fk].net_ if fk == "lqds" else A[fk]
            m["params"] = float(sum(p.numel() for p in mod.parameters()))
            e = S1.setdefault(name, {})
            for k, v in m.items(): e.setdefault(k, []).append(v)
        # ---------------- (b) end to end
        for _, row in run.results.iterrows():
            name = row["method"]; rows = [np.asarray(run.prediction_sets[name].sets[i]) for i in ev]
            tgt = TARGET[name]
            m = set_metrics(rows, xt, yt, O[tgt] if tgt != "interval" else O["mw"])
            e = E2E.setdefault(name, dict(target=tgt))
            for k in ("coverage", "width", "comps", "symdiff"): e.setdefault(k, []).append(m[k])
            e.setdefault("min_cc", []).append(float(np.quantile(m["cond_cov"], 0.05)))
            e.setdefault("train_s", []).append(float(row["training_seconds"]))
        # ---------------- (c) cross ablation (same splits, same fits)
        lawc = lqds_law(lq, Xc)
        qc_, dc_, dtc_ = lawc[:3]; qt_, dt_d, dtt_ = lawt[:3]
        cross = {}
        # baselines' Stage 2 on the LQDS Stage-1 fit
        fk_c, fk_t = 1.0 / np.maximum(dc_, 1e-300), 1.0 / np.maximum(dt_d, 1e-300)
        cross["SPICE-ND reading | LQDS fit"] = ("mw", spice_stage2((qc_, fk_c), (qt_, fk_t), yc, False))
        cross["SPICE-HPD reading | LQDS fit"] = ("hpd", spice_stage2((qc_, fk_c), (qt_, fk_t), yc, True))
        tq = np.linspace(0.01, 0.99, 99)
        def lq_quant(law):
            Fg, _ = lqds_cdf_pdf(law, interp, YG)
            return np.array([np.interp(tq, np.maximum.accumulate(Fg[r]) + np.arange(G) * 1e-15, YG) for r in range(len(Fg))])
        cross["CTI (99) | LQDS fit"] = ("mw", cti_stage2(lq_quant(lawc), lq_quant(lawt), yc))
        edges = np.linspace(YLO, YHI, 401); cen = (edges[:-1] + edges[1:]) / 2
        def lq_dens(law, pts):
            _, fg_ = lqds_cdf_pdf(law, interp, pts); return fg_
        Dc_ = lq_dens(lawc, cen); Dt_ = lq_dens(lawt, cen)
        fyc = np.array([np.interp(yc[r], cen, Dc_[r]) for r in range(len(yc))])
        cross["HPD-Split | LQDS fit"] = ("hpd", hpd_split_stage2(Dc_, Dt_, edges, yc, fyc))
        # the LQDS Stage 2 on the baselines' Stage-1 fits
        Qc = sp.inverse_y(predict_quantiles(A["quantile"], Xc).reshape(-1)).reshape(len(Xc), -1)
        def from_quantiles(Q):     # knots at the fitted quantiles, density = implied central difference
            Q = np.maximum.accumulate(np.sort(Q, 1), 1) + np.arange(Q.shape[1]) * 1e-9
            f_ = np.gradient(taus, axis=0)[None, :] / np.maximum(np.gradient(Q, axis=1), 1e-12)
            return lqds_rep_from_knots(Q, f_, interp)
        cross["LQDS-MW reading | QuantileNet fit"] = ("mw", lqds_stage2(from_quantiles(Qc), from_quantiles(Qt), yc, False, interp))
        cross["LQDS-HPD reading | QuantileNet fit"] = ("hpd", lqds_stage2(from_quantiles(Qc), from_quantiles(Qt), yc, True, interp))
        for name, (tgt, (t1, t2, rows)) in cross.items():
            m = set_metrics(rows, xt, yt, O[tgt])
            e = X2.setdefault(name, dict(target=tgt))
            for k in ("coverage", "width", "comps", "symdiff"): e.setdefault(k, []).append(m[k])
            e.setdefault("stage2_s", []).append(t1 + t2)
        print(f"seed {seed}: S1 " + "  ".join(f"{k} {np.mean(v['crps_excess']):.4f}" for k, v in S1.items())
              + " | e2e " + "  ".join(f"{k} {np.mean(v['symdiff']):.3f}" for k, v in E2E.items()), flush=True)
    agg = lambda D: {k: {m: ([float(np.mean(v)), float(np.std(v))] if isinstance(v, list) else v) for m, v in e.items()} for k, e in D.items()}
    out = dict(seeds=a.seeds, seed_start=a.seed_start, n_eval=a.n_eval, interp=interp, stage1=agg(S1), e2e=agg(E2E), cross=agg(X2),
               raw=dict(stage1=S1, e2e=E2E, cross=X2))
    json.dump(out, open(a.out or RES / "synth_e2e.json", "w"), indent=1)
    print("\nStage 1 (vs truth)"); print(f"{'family':<20}{'CRPS exc':>10}{'Q MAE':>9}{'dens L1':>9}{'NLL':>8}{'params':>8}{'train s':>9}")
    for k, v in out["stage1"].items():
        print(f"{k:<20}{v['crps_excess'][0]:>10.4f}{v['quantile_mae'][0]:>9.4f}{v['density_l1'][0]:>9.3f}{v['nll'][0]:>8.3f}{v['params'][0]:>8.0f}{v['train_s'][0]:>9.1f}")
    print("\nEnd to end"); print(f"{'method':<12}{'target':>9}{'cov':>8}{'width':>8}{'q05 cc':>8}{'sd':>8}")
    for k, v in out["e2e"].items():
        print(f"{k:<12}{v['target']:>9}{v['coverage'][0]:>8.3f}{v['width'][0]:>8.3f}{v['min_cc'][0]:>8.3f}{v['symdiff'][0]:>8.3f}")
    print("\nCross ablation"); print(f"{'reading | fit':<36}{'target':>7}{'cov':>8}{'width':>8}{'sd':>8}")
    for k, v in out["cross"].items():
        print(f"{k:<36}{v['target']:>7}{v['coverage'][0]:>8.3f}{v['width'][0]:>8.3f}{v['symdiff'][0]:>8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["oracle", "e2e"])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--Ks", default=None, help="comma-separated resolution sweep, e.g. 20,30,50,99")
    ap.add_argument("--n-eval", dest="n_eval", type=int, default=1000)
    ap.add_argument("--interp", default=None, choices=[None, "dlin", "flin"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed-start", dest="seed_start", type=int, default=2000)
    ap.add_argument("--save-models", dest="save_models", default=None)
    a = ap.parse_args()
    if a.part == "oracle":
        a.seeds = a.seeds or 30; a.interp = a.interp or "flin"; run_oracle(a)
    else:
        a.seeds = a.seeds or 10; run_e2e(a)
