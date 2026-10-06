"""LQDS-CP: Linear Quantile-Derivative Score conformal prediction.

The network emits the conditional quantile function's DERIVATIVE at K knots;
Q is its exact trapezoid integral.  Because Q' = 1/f, thresholding the
derivative is a density level set -- computed in closed form.

    d_k(x) = d_min + softplus(v_k(x))  >= 0         network output
    D(tau) = (1-s) d_k + s d_{k+1}                     piecewise LINEAR
    Q(tau) = q_k + dtau_k [ d_k s + (d_{k+1}-d_k) s^2/2 ]   piecewise QUADRATIC
    q_{k+1} = q_k + dtau_k (d_k + d_{k+1})/2           trapezoid

Why emit d rather than q: deriving d from predicted q needs the recursion
d_{k+1} = 2 s_k - d_k, whose error alternates without decaying and cannot be
absorbed by its single free parameter.  Measured on star, oscillation rose
0.73 -> 1.65 and width 810 -> 1111 as K went 12 -> 80; emitting d reversed
both (0.45 -> 0.28, 783 -> 806).  That stability is what lets K be fixed a
priori instead of selected.

Every hyperparameter is fixed or derived from the training/validation data:
    K       = clip(round(sqrt(n_train)), 8, 60), fixed before calibration
    encoder = identical to the multi-output quantile baselines (2 layers, GELU)
    anchor  = auto, iff >= 2% of y_train sits at the support floor
Split-conformal validity therefore holds exactly: nothing touches the
calibration set before calibration.
"""
from __future__ import annotations
from .intervals import conformal_quantile
import copy
import numpy as np
import torch
import torch.nn as nn

__all__ = ["LQDSCP"]


class _Net(nn.Module):
    def __init__(
        self,
        input_dim,
        y_train,
        n_knots,
        hidden_dim,
        n_layers,
        dropout,
        tau_eps,
        anchor,
        head="softplus",
        tail_weight=True,
        loss_at="sampled",
        smooth=0.0,
        kernel=0,
        knot_rule="arclength",
        learn_knots=False,
        knot_floor=0.1,
        smooth_edge=0,
        nll_weight=0.0,
        knot_anchor="tau",
        n_blocks=3,
        y_top_q=1.0,
        y_affine=False,
    ):
        super().__init__()

        self.nll_weight = float(nll_weight)
        self.smooth_edge = int(smooth_edge)
        self.head = head
        self.smooth = float(smooth)

        self.kernel = int(kernel)
        self.tail_weight = tail_weight
        self.loss_at = loss_at

        tau = (
            _arclength_knots(y_train, n_knots, tau_eps)
            if knot_rule == "arclength"
            else _curvature_knots(y_train, n_knots, tau_eps)
            if knot_rule == "curvature"
            else np.linspace(tau_eps, 1.0 - tau_eps, n_knots)
        )
        init_q = np.quantile(y_train, tau)
        self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
        self.register_buffer("dtau", torch.as_tensor(np.diff(tau), dtype=torch.float32))
        self.anchor = bool(anchor)
        self.y_floor = float(np.min(y_train))

        self.c = 0 if self.anchor else int(np.argmin(np.abs(tau - 0.5)))

        self.min_inc = 0.0 if self.anchor else 1e-6

        layers, prev = [], input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.q0_head = nn.Linear(hidden_dim, 1)
        self.d_head = nn.Linear(hidden_dim, len(tau))

        self.learn_knots = bool(learn_knots)
        self.tau_lo, self.tau_span = float(tau[0]), float(tau[-1] - tau[0])

        self.knot_floor = float(knot_floor) / (len(tau) - 1)
        if self.learn_knots:
            self.tau_head = nn.Linear(hidden_dim, len(tau) - 1)
            frac = np.diff(tau) / self.tau_span
            share = np.maximum(frac - self.knot_floor, 1e-3 * self.knot_floor)
            share = share / share.sum()
            nn.init.normal_(self.tau_head.weight, std=1e-3)
            with torch.no_grad():
                self.tau_head.bias.copy_(
                    torch.as_tensor(np.log(share), dtype=torch.float32)
                )
        nn.init.normal_(self.q0_head.weight, std=1e-3)
        nn.init.constant_(self.q0_head.bias, float(init_q[self.c]))
        slopes = np.maximum(np.gradient(init_q, tau), 1e-3)
        nn.init.normal_(self.d_head.weight, std=1e-3)
        with torch.no_grad():
            if head == "exp":
                self.d_head.bias.copy_(
                    torch.as_tensor(np.log(slopes), dtype=torch.float32)
                )
            elif head == "relu":

                self.d_head.bias.copy_(torch.as_tensor(slopes, dtype=torch.float32))
            else:
                self.d_head.bias.copy_(
                    torch.as_tensor(
                        slopes + np.log(-np.expm1(-np.minimum(slopes, 30.0))),
                        dtype=torch.float32,
                    )
                )

        if tail_weight:
            left = np.geomspace(tau_eps, 0.05, 40)
            right = 1.0 - left[::-1]
            lt = np.r_[left, np.linspace(0.05, 0.95, 160), right, tau]
        else:

            lt = np.r_[np.linspace(tau_eps, 1.0 - tau_eps, 200), tau]
        self.register_buffer(
            "loss_taus", torch.as_tensor(np.unique(lt).astype(np.float32))
        )

        self.nll_scale = 1.0

        self.y_anchor = knot_anchor == "y"
        if self.y_anchor:
            yt = np.asarray(y_train, float).reshape(-1)
            cont = yt[yt > self.y_floor + 1e-9] if self.anchor else yt

            u = np.unique(np.round(cont, 9))
            if len(u) > 1 and len(u) < 0.5 * len(cont):
                step = float(np.median(np.diff(u)[:50]))
                cont = cont + (
                    np.random.default_rng(0).uniform(-0.5, 0.5, len(cont)) * step
                )
            lo_c = self.y_floor if self.anchor else float(np.min(yt))
            G = np.quantile(cont, np.linspace(0.0, 1.0, n_knots))

            top = (
                float(np.max(yt)) if y_top_q >= 1.0 else float(np.quantile(yt, y_top_q))
            )
            G[0], G[-1] = (
                min(G[0], lo_c) if not self.anchor else lo_c,
                (max(G[-1], top) if y_top_q >= 1.0 else top),
            )
            G = np.maximum.accumulate(G)
            G = G + np.arange(len(G)) * 1e-9
            self.register_buffer("y_grid", torch.as_tensor(G, dtype=torch.float32))

            self.y_affine_loc_only = y_affine in ("loc", "locpos", "qshift", "yshift")
            self.y_affine_mode = y_affine if isinstance(y_affine, str) else None

            if y_affine == "yshift":

                lo_g, hi_g = float(G[0]), float(G[-1])
                zg = np.clip((G[1:-1] - lo_g) / (hi_g - lo_g), 1e-6, 1 - 1e-6)
                self.ys_lo, self.ys_span = lo_g, hi_g - lo_g
                self.register_buffer(
                    "ys_logit",
                    torch.as_tensor(np.log(zg) - np.log1p(-zg), dtype=torch.float32),
                )
            if y_affine == "qshift":
                uf = np.linspace(0.0, 1.0, 4001)
                self.register_buffer("qm_u", torch.as_tensor(uf, dtype=torch.float32))
                qm = np.quantile(cont, uf)
                qm = np.maximum.accumulate(qm) + np.arange(len(qm)) * 1e-10
                self.register_buffer("qm_y", torch.as_tensor(qm, dtype=torch.float32))

                Fg = np.clip(np.interp(G, qm, uf), 1e-6, 1 - 1e-6)
                self.qs_top = float(Fg[-1])
                self.register_buffer(
                    "qs_logit",
                    torch.as_tensor(
                        np.log(Fg[1:-1] / self.qs_top)
                        - np.log1p(-Fg[1:-1] / self.qs_top),
                        dtype=torch.float32,
                    ),
                )
            self.y_affine = bool(y_affine)
            if self.y_affine:
                self.affine_head = nn.Linear(hidden_dim, 2)
                nn.init.normal_(self.affine_head.weight, std=1e-3)
                with torch.no_grad():
                    self.affine_head.bias.copy_(
                        torch.tensor(
                            [
                                -4.0 if y_affine == "locpos" else 0.0,
                                float(np.log(np.expm1(1.0 - 1e-3))),
                            ]
                        )
                    )
                self.affine_c0 = (
                    float(self.y_floor) if self.anchor else float(np.median(yt))
                )

            n_d = len(G)
            n_int = len(G) + 2 if self.anchor else len(G)
            self.d_head = nn.Linear(hidden_dim, n_d)
            dG = np.diff(G)
            loc = np.r_[dG[0], (dG[:-1] + dG[1:]) / 2.0, dG[-1]]
            init = np.maximum(loc / loc.mean(), 1e-3)
            nn.init.normal_(self.d_head.weight, std=1e-3)
            with torch.no_grad():
                self.d_head.bias.copy_(
                    torch.as_tensor(
                        init + np.log(-np.expm1(-np.minimum(init, 30.0))),
                        dtype=torch.float32,
                    )
                )
            if self.anchor:
                self.atom_head = nn.Linear(hidden_dim, 1)
                am = float(np.clip(np.mean(yt <= self.y_floor + 1e-9), 1e-3, 0.97))
                nn.init.normal_(self.atom_head.weight, std=1e-3)
                nn.init.constant_(self.atom_head.bias, float(np.log(am / (1 - am))))

            ph = np.linspace(self.tau_lo, self.tau_lo + self.tau_span, n_int)
            self.tau = torch.as_tensor(ph, dtype=torch.float32)
            self.dtau = torch.as_tensor(np.diff(ph), dtype=torch.float32)
            self.c = 0

        self.split = knot_anchor == "split"
        if self.split:
            K = len(self.tau)
            self.split_K1 = 2 if self.anchor else K // 2
            self.split_head = nn.Linear(hidden_dim, 1)
            nn.init.normal_(self.split_head.weight, std=1e-3)
            yt = np.asarray(y_train, float).reshape(-1)
            w0 = (
                float(np.clip(np.mean(yt <= self.y_floor + 1e-9), 1e-3, 0.97))
                if self.anchor
                else 0.5
            )
            nn.init.constant_(self.split_head.bias, float(np.log(w0 / (1 - w0))))
            if not self.anchor:
                self.c = self.split_K1

            K1, K2 = self.split_K1, K - self.split_K1
            t1 = self.tau_lo + np.linspace(0.0, w0 * self.tau_span, K1)
            t2 = self.tau_lo + np.linspace(w0 * self.tau_span, self.tau_span, K2)
            tk = np.r_[t1, t2 + 1e-9]
            mq = np.quantile(yt, np.clip(tk, 0, 1))
            sl = np.maximum(np.gradient(mq, tk), 1e-3)
            with torch.no_grad():
                self.d_head.bias.copy_(
                    torch.as_tensor(
                        sl + np.log(-np.expm1(-np.minimum(sl, 30.0))),
                        dtype=torch.float32,
                    )
                )
                if not self.anchor:
                    self.q0_head.bias.fill_(float(mq[self.split_K1]))

        self.blocks = knot_anchor == "blocks"
        if self.blocks:
            K = len(self.tau)
            B = int(n_blocks)
            yt = np.asarray(y_train, float).reshape(-1)
            if self.anchor:
                cont = B - 1
                rest = K - 2
                nb = [2] + [
                    rest // cont + (1 if i < rest % cont else 0) for i in range(cont)
                ]
            else:
                nb = [K // B + (1 if i < K % B else 0) for i in range(B)]
            assert min(nb) >= 2, nb
            self.blk_n = nb
            self.blk_base = list(np.cumsum([0] + [n + 1 - 1 + 1 for n in nb[:-1]]))

            base, bases = 0, []
            for b, n_ in enumerate(nb):
                bases.append(base)
                base += (n_ - 1) + (1 if b < B - 1 else 0)
            self.blk_base = bases
            n_logits = B - 1 if self.anchor else B
            self.blk_head = nn.Linear(hidden_dim, n_logits)
            nn.init.normal_(self.blk_head.weight, std=1e-3)
            nn.init.zeros_(self.blk_head.bias)
            if self.anchor:
                self.atom_head = nn.Linear(hidden_dim, 1)
                am = float(np.clip(np.mean(yt <= self.y_floor + 1e-9), 1e-3, 0.97))
                nn.init.normal_(self.atom_head.weight, std=1e-3)
                nn.init.constant_(self.atom_head.bias, float(np.log(am / (1 - am))))
                m0 = np.r_[am, np.full(B - 1, (1 - am) / (B - 1))]
            else:
                m0 = np.full(B, 1.0 / B)
            self.blk_floor = 0.2
            self.c = 0 if self.anchor else nb[0]

            tks, st = [], 0.0
            for b, n_ in enumerate(nb):
                tks.append(
                    self.tau_lo
                    + st * self.tau_span
                    + np.linspace(0.0, m0[b] * self.tau_span, n_)
                )
                st += m0[b]
            tk = np.concatenate(tks) + np.arange(K) * 1e-9
            mq = np.quantile(yt, np.clip(tk, 0, 1))
            sl = np.maximum(np.gradient(mq, tk), 1e-3)
            with torch.no_grad():
                self.d_head.bias.copy_(
                    torch.as_tensor(
                        sl + np.log(-np.expm1(-np.minimum(sl, 30.0))),
                        dtype=torch.float32,
                    )
                )
                if not self.anchor:
                    self.q0_head.bias.fill_(float(mq[self.c]))

    def params(self, x):
        """(q, d): knot values and knot derivatives.  Q interpolates q exactly."""
        q, d, _ = self.params3(x)
        return q, d

    def block_masses(self, h):
        """[n, B] tau-mass of each block (links excluded)."""
        B = len(self.blk_n)
        EPS = 1e-9
        avail = self.tau_span - (B - 1) * EPS
        if self.anchor:
            p0 = (avail - 1e-6) * 0.98 * torch.sigmoid(self.atom_head(h))
            fl = self.blk_floor / (B - 1)
            w = fl + (1.0 - fl * (B - 1)) * torch.softmax(self.blk_head(h), dim=1)
            return torch.cat([p0, (avail - p0) * w], dim=1)
        fl = self.blk_floor / B
        return avail * (fl + (1.0 - fl * B) * torch.softmax(self.blk_head(h), dim=1))

    def spacing(self, h):
        """[n, K-1] knot spacings in tau (the fixed ones, repeated, unless learned)."""
        if getattr(self, "blocks", False):
            M = self.block_masses(h)
            B = len(self.blk_n)
            EPS = 1e-9
            parts = []
            for b, n_ in enumerate(self.blk_n):
                parts.append((M[:, b : b + 1] / (n_ - 1)).expand(-1, n_ - 1))
                if b < B - 1:
                    parts.append(torch.full_like(M[:, :1], EPS))
            return torch.cat(parts, dim=1)
        if getattr(self, "split", False):
            K = self.dtau.shape[0] + 1
            K1 = self.split_K1
            K2 = K - K1
            EPS = 1e-9
            w = (self.tau_span - 2 * EPS) * torch.sigmoid(self.split_head(h)) + EPS
            b1 = (w / (K1 - 1)).expand(-1, K1 - 1)
            link = torch.full_like(w, EPS)
            b2 = ((self.tau_span - w - EPS) / (K2 - 1)).expand(-1, K2 - 1)
            return torch.cat([b1, link, b2], dim=1)
        if not self.learn_knots:
            return self.dtau.expand(h.shape[0], -1)
        K1 = self.dtau.shape[0]
        w = self.knot_floor + (1.0 - K1 * self.knot_floor) * torch.softmax(
            self.tau_head(h), dim=1
        )
        return self.tau_span * w

    def params3(self, x):
        """(q, d, dt): knot values, knot derivatives and per-row knot spacings."""
        if getattr(self, "y_anchor", False):
            return self._params_y(x)
        h = self.encoder(x)
        z = self.d_head(h)
        if self.kernel > 0:
            k = self.kernel
            zp = torch.nn.functional.pad(z[:, None, :], (k, k), mode="replicate")
            z = torch.nn.functional.avg_pool1d(zp, 2 * k + 1, stride=1)[:, 0, :]
        if self.head == "relu":

            d = self.min_inc + torch.relu(z)
        elif self.head == "exp":

            d = self.min_inc + torch.exp(z.clamp(max=30.0))
        else:
            d = self.min_inc + nn.functional.softplus(z)
        if (
            getattr(self, "split", False) or getattr(self, "blocks", False)
        ) and self.anchor:
            d = torch.cat([torch.zeros_like(d[:, :2]), d[:, 2:]], dim=1)
        q0 = (
            torch.full((x.shape[0], 1), self.y_floor, dtype=d.dtype, device=d.device)
            if self.anchor
            else self.q0_head(h)
        )
        dt = self.spacing(h)
        step = dt * (d[:, :-1] + d[:, 1:]) / 2.0
        c = self.c
        up = q0 + torch.cumsum(step[:, c:], dim=1)
        if c > 0:
            down = q0 - torch.flip(
                torch.cumsum(torch.flip(step[:, :c], [1]), dim=1), [1]
            )
            q = torch.cat([down, q0, up], dim=1)
        else:
            q = torch.cat([q0, up], dim=1)
        return q, d, dt

    def _params_y(self, x):
        h = self.encoder(x)
        dr = nn.functional.softplus(self.d_head(h)) + 1e-6
        G = self.y_grid
        n = x.shape[0]
        if getattr(self, "y_affine_mode", None) == "yshift":
            delta = self.affine_head(h)[:, :1]
            inner = self.ys_lo + self.ys_span * torch.sigmoid(
                self.ys_logit[None, :] + delta
            )
            Gx = torch.cat([G[:1].expand(n, 1), inner, G[-1:].expand(n, 1)], dim=1)
            return self._params_y_rows(h, dr, Gx, Gx[:, 1:] - Gx[:, :-1], n)
        if getattr(self, "y_affine_mode", None) == "qshift":
            delta = self.affine_head(h)[:, :1]
            v = self.qs_top * torch.sigmoid(self.qs_logit[None, :] + delta)

            uf, yf = self.qm_u, self.qm_y
            j = torch.bucketize(v.detach().contiguous(), uf[1:-1].contiguous())
            u0, u1, y0, y1 = uf[j], uf[j + 1], yf[j], yf[j + 1]
            inner = y0 + (y1 - y0) * (v - u0) / (u1 - u0)
            Gx = torch.cat([G[:1].expand(n, 1), inner, G[-1:].expand(n, 1)], dim=1)
            dG = Gx[:, 1:] - Gx[:, :-1]
            return self._params_y_rows(h, dr, Gx, dG, n)
        dG = G[1:] - G[:-1]
        if getattr(self, "interp", "dlin") == "flin" and not getattr(
            self, "y_affine", False
        ):

            return self._params_y_rows(h, dr, G.expand(n, -1), dG.expand(n, -1), n)
        if self.anchor:
            EPS = 1e-9
            p0 = self.tau_span * 0.98 * torch.sigmoid(self.atom_head(h))
            raw = 2.0 * dG / (dr[:, :-1] + dr[:, 1:])
            s = raw.sum(1, keepdim=True) / (self.tau_span - p0 - EPS)
            link = torch.full_like(p0, EPS)
            dt = torch.cat([p0, link, raw / s], dim=1)
            zero = torch.zeros(n, 2, dtype=dr.dtype, device=dr.device)
            d = torch.cat([zero, s * dr], dim=1)

            step = dt * (d[:, :-1] + d[:, 1:]) / 2.0
            q = G[0] + torch.cat(
                [
                    torch.zeros(n, 1, dtype=d.dtype, device=d.device),
                    torch.cumsum(step, dim=1),
                ],
                dim=1,
            )
        else:
            raw = 2.0 * dG / (dr[:, :-1] + dr[:, 1:])
            s = raw.sum(1, keepdim=True) / self.tau_span
            dt, d, q = raw / s, s * dr, G.expand(n, -1)
        if getattr(self, "y_affine", False):

            a = self.affine_head(h)
            sig = (
                torch.ones_like(a[:, 1:2])
                if getattr(self, "y_affine_loc_only", False)
                else nn.functional.softplus(a[:, 1:2]) + 1e-3
            )
            if self.anchor:
                mu = torch.zeros_like(sig)
            elif getattr(self, "y_affine_mode", None) == "locpos":

                mu = nn.functional.softplus(a[:, :1])
            else:
                mu = a[:, :1]
            c0 = self.affine_c0
            q = c0 + mu + sig * (q - c0)
            d = sig * d
        return q.contiguous(), d, dt

    def _params_y_rows(self, h, dr, Gx, dG, n):
        """_params_y with a per-row grid Gx [n, K] (qshift)."""
        flin = getattr(self, "interp", "dlin") == "flin"
        if flin:

            raw = (
                dG
                / dr[:, :-1]
                * _expm1_over(torch.log(dr[:, :-1]) - torch.log(dr[:, 1:]))
            )
        else:
            raw = 2.0 * dG / (dr[:, :-1] + dr[:, 1:])
        if self.anchor:
            EPS = 1e-9
            p0 = self.tau_span * 0.98 * torch.sigmoid(self.atom_head(h))
            s = raw.sum(1, keepdim=True) / (self.tau_span - p0 - EPS)
            dt = torch.cat([p0, torch.full_like(p0, EPS), raw / s], dim=1)
            zero = torch.zeros(n, 2, dtype=dr.dtype, device=dr.device)
            d = torch.cat([zero, s * dr], dim=1)
            step = dt * (d[:, :-1] + d[:, 1:]) / 2.0
            if flin:

                step = torch.cat([step[:, :2], dG], dim=1)
            q = Gx[:, :1] + torch.cat(
                [
                    torch.zeros(n, 1, dtype=d.dtype, device=d.device),
                    torch.cumsum(step, dim=1),
                ],
                dim=1,
            )
        else:
            s = raw.sum(1, keepdim=True) / self.tau_span
            dt, d, q = raw / s, s * dr, Gx
        return q.contiguous(), d, dt

    def quantile(self, x, tau, qd=None):
        q, d, dt = self.params3(x) if qd is None else qd
        if (
            not self.learn_knots
            and not getattr(self, "y_anchor", False)
            and not getattr(self, "split", False)
            and not getattr(self, "blocks", False)
        ):

            tk = self.tau
            tau = tau.clamp(float(tk[0]), float(tk[-1]))
            i = torch.bucketize(tau, tk[1:-1].contiguous())
            dt = self.dtau[i]
            s = (tau - tk[i]) / dt
            dk, dk1 = d[:, i], d[:, i + 1]
            return q[:, i] + dt * (dk * s + (dk1 - dk) * s * s / 2.0)
        n = q.shape[0]
        if getattr(self, "blocks", False):
            B = len(self.blk_n)
            EPS = 1e-9
            nb = torch.tensor(self.blk_n, device=dt.device)
            base = torch.tensor(self.blk_base, device=dt.device)
            hb = torch.stack([dt[:, self.blk_base[b]] for b in range(B)], dim=1)
            Mb = hb * (nb - 1).to(dt.dtype)
            starts = torch.cat(
                [torch.zeros_like(Mb[:, :1]), torch.cumsum(Mb + EPS, dim=1)[:, :-1]],
                dim=1,
            )
            r = (
                tau.to(dt.dtype).clamp(self.tau_lo, self.tau_lo + self.tau_span)[
                    None, :
                ]
                - self.tau_lo
            )
            j = (r[:, :, None] >= starts[:, None, 1:]).sum(-1)
            S_ = starts.gather(1, j)
            h_ = hb.gather(1, j)
            u = (r - S_) / h_
            fl = torch.minimum(torch.floor(u).clamp(min=0), (nb[j] - 2).to(dt.dtype))
            i = (base[j] + fl).long()
            s = (u - fl).clamp(0.0, 1.0)
            dk, dk1 = d.gather(1, i), d.gather(1, i + 1)
            return q.gather(1, i) + h_ * s * (dk + (dk1 - dk) * s / 2.0)
        if getattr(self, "split", False):

            K1 = self.split_K1
            K = q.shape[1]
            K2 = K - K1
            h1, lk, h2 = dt[:, :1], dt[:, K1 - 1 : K1], dt[:, K1 : K1 + 1]
            w = h1 * (K1 - 1)
            r = (
                tau.to(dt.dtype).clamp(self.tau_lo, self.tau_lo + self.tau_span)[
                    None, :
                ]
                - self.tau_lo
            )
            in1 = r < w

            off = torch.where(in1, torch.zeros_like(w), w + lk)
            h = torch.where(in1, h1, h2)
            u = (r - off) / h
            fl = torch.minimum(
                torch.floor(u).clamp(min=0),
                torch.where(in1, float(K1 - 2), float(K2 - 2)),
            )
            i = (fl + torch.where(in1, 0.0, float(K1))).long()
            s = (u - fl).clamp(0.0, 1.0)
            dk, dk1 = d.gather(1, i), d.gather(1, i + 1)
            return q.gather(1, i) + h * s * (dk + (dk1 - dk) * s / 2.0)
        tk = torch.cat(
            [
                torch.full((n, 1), self.tau_lo, dtype=dt.dtype, device=dt.device),
                self.tau_lo + torch.cumsum(dt, dim=1),
            ],
            dim=1,
        )
        tau = tau.to(dt.dtype).clamp(self.tau_lo, self.tau_lo + self.tau_span)
        t = tau[None, :].expand(n, -1).contiguous()
        i = torch.searchsorted(tk[:, 1:-1].contiguous(), t)
        dti = dt.gather(1, i)
        s = (t - tk.gather(1, i)) / dti
        dk, dk1 = d.gather(1, i), d.gather(1, i + 1)
        return q.gather(1, i) + dti * (dk * s + (dk1 - dk) * s * s / 2.0)

    def _penalty(self, d):

        ld = torch.log(d + 1e-8)
        c2 = (ld[:, 2:] - 2 * ld[:, 1:-1] + ld[:, :-2]) ** 2

        ok = (d > 1e-4).detach()
        ok = ok[:, 2:] & ok[:, 1:-1] & ok[:, :-2]
        if getattr(self, "split", False):

            L = self.split_K1 - 1
            ok = ok.clone()
            ok[:, max(L - 1, 0) : L + 1] = False
        if getattr(self, "blocks", False):
            ok = ok.clone()
            for b in range(len(self.blk_n) - 1):
                L = self.blk_base[b] + self.blk_n[b] - 1
                ok[:, max(L - 1, 0) : L + 1] = False
        if self.smooth_edge > 0:

            e = self.smooth_edge
            ok = ok.clone()
            ok[:, :e] = False
            ok[:, -e:] = False
        return self.smooth * (c2 * ok).sum() / ok.sum().clamp(min=1)

    def loss(self, x, y):
        main, nll = self.loss_parts(x, y)
        main = getattr(self, "crps_weight", 1.0) * main
        return main if nll is None else main + self.nll_weight * self.nll_scale * nll

    def loss_parts(self, x, y):
        """(pinball [+ curvature penalty], likelihood term or None)"""
        if self.loss_at == "ycrps":

            qd = self.params3(x)
            q, d, dt = qd
            F = self.tau_lo + torch.cat(
                [torch.zeros_like(dt[:, :1]), torch.cumsum(dt, dim=1)], dim=1
            )
            dq = q[:, 1:] - q[:, :-1]
            w = torch.cat([dq[:, :1], dq[:, :-1] + dq[:, 1:], dq[:, -1:]], dim=1) / 2.0
            ind = (y[:, None] <= q).to(F.dtype)
            main = ((F - ind) ** 2 * w).sum(1)

            main = main + torch.relu(q[:, 0] - y) + torch.relu(y - q[:, -1])
            main = main.mean()
            if self.smooth > 0 and self.training:
                main = main + self._penalty(d)
            nll = None
            if self.nll_weight > 0:
                qn_, yy = qd, y
                r_ = getattr(self, "nll_rows", 0)
                if self.training and 0 < r_ < len(y):

                    sel = torch.randperm(len(y), device=y.device)[:r_]
                    qn_, yy = (q[sel], d[sel], dt[sel]), y[sel]
                step = getattr(self, "dq_step", 0.0)
                yn = yy
                if step > 0:
                    u = (
                        torch.rand_like(yy)
                        if self.training
                        else torch.full_like(yy, 0.5)
                    )
                    yn = torch.where(
                        yy > self.y_floor + 1e-9, yy + (u - 0.5) * step, yy
                    )
                nll = _nll(self, yn, qn_)
            return main, nll
        if self.loss_at == "rowknots":

            qd = self.params3(x)
            q, d, dt = qd
            tk = self.tau_lo + torch.cat(
                [torch.zeros_like(dt[:, :1]), torch.cumsum(dt, dim=1)], dim=1
            )
            e = y[:, None] - q
            main = torch.maximum(tk * e, (tk - 1.0) * e).mean()
            nll = None
            if self.nll_weight > 0:
                step = getattr(self, "dq_step", 0.0)
                yn = y
                if step > 0:
                    u = torch.rand_like(y) if self.training else torch.full_like(y, 0.5)
                    yn = torch.where(y > self.y_floor + 1e-9, y + (u - 0.5) * step, y)
                nll = _nll(self, yn, qd)
            return main, nll
        if self.loss_at == "knots":

            q, _ = self.params(x)
            e = y[:, None] - q
            t = self.tau[None, :]
            return torch.maximum(t * e, (t - 1.0) * e).mean(), None
        taus = self.loss_taus
        if self.loss_at == "full":
            pass
        elif self.training and not self.tail_weight:
            taus = taus[torch.randint(len(taus), (96,), device=taus.device)]
        elif self.training:
            g = (
                torch.where(taus < 0.05)[0],
                torch.where((taus >= 0.05) & (taus <= 0.95))[0],
                torch.where(taus > 0.95)[0],
            )
            taus = taus[
                torch.sort(
                    torch.cat(
                        [
                            gi[torch.randint(len(gi), (n,), device=gi.device)]
                            for gi, n in zip(g, getattr(self, "tau_draw", (16, 64, 16)))
                        ]
                    )
                ).values
            ]
        qd = self.params3(x)
        e = y[:, None] - self.quantile(x, taus, qd)
        t = taus[None, :]
        loss = torch.maximum(t * e, (t - 1.0) * e).mean()
        nll = None
        if self.nll_weight > 0:

            step = getattr(self, "dq_step", 0.0)
            yn = y
            if step > 0:
                u = torch.rand_like(y) if self.training else torch.full_like(y, 0.5)
                yn = torch.where(y > self.y_floor + 1e-9, y + (u - 0.5) * step, y)
            nll = _nll(self, yn, qd)
        if self.smooth > 0 and self.training:
            loss = loss + self._penalty(qd[1])
        return loss, nll


def _nll(net, y, qd):
    """-log f(y|x).  Inside [Q(tau_0), Q(tau_K)]: log D(tau_y), with tau_y from
    the same exact quadratic inversion Stage 2 uses (differentiable through q,
    d and the knot spacings).  Outside: the density is completed by an
    exponential tail carrying exactly the remaining mass and continuous with
    the boundary density, -log f = log D_end + |y - Q_end| / (m_end D_end).
    Dropping those points instead lets the fit shrink its support to exclude
    hard observations (bio: coverage 0.90 -> 0.003).  Point-mass observations
    of an anchored fit (D = 0: a mass, not a density) are left to pinball."""
    q, d, dt = qd
    K = q.shape[1]
    lo, hi = q[:, 0], q[:, -1]
    with torch.no_grad():
        i = (
            torch.searchsorted(
                q.detach().contiguous(), y[:, None].contiguous()
            ).squeeze(1)
            - 1
        ).clamp(0, K - 2)
        inside, below = (y >= lo) & (y <= hi), y < lo
        keep = ~(net.anchor & (y <= lo + 1e-6))
    ii = i[:, None]
    qk, dk, dk1, dti = (
        q.gather(1, ii)[:, 0],
        d.gather(1, ii)[:, 0],
        d.gather(1, ii + 1)[:, 0],
        dt.gather(1, ii)[:, 0],
    )
    target = ((torch.minimum(torch.maximum(y, lo), hi) - qk) / dti).clamp(min=0.0)
    dd = dk1 - dk
    root = torch.sqrt((dk * dk + 2.0 * dd * target).clamp(min=1e-12))
    s = (2.0 * target / (dk + root).clamp(min=1e-12)).clamp(0.0, 1.0)
    ll_in = torch.log((dk + s * dd).clamp(min=1e-8))
    if getattr(net, "interp", "dlin") == "flin":

        pos = (dk > 1e-12) & (dk1 > 1e-12)
        dks, dk1s = (
            torch.where(pos, dk, torch.ones_like(dk)),
            torch.where(pos, dk1, torch.ones_like(dk1)),
        )
        dts = torch.where(pos, dti, torch.ones_like(dti))
        off = (torch.minimum(torch.maximum(y, lo), hi) - qk).clamp(min=0.0)
        ll_f = (
            torch.log(dks)
            - (1.0 / dk1s - 1.0 / dks)
            * torch.where(pos, off, torch.zeros_like(off))
            / dts
        )
        ll_in = torch.where(pos, ll_f, ll_in)
    m_lo = net.tau_lo
    m_hi = 1.0 - (net.tau_lo + net.tau_span)
    d0, dK = d[:, 0].clamp(min=1e-8), d[:, -1].clamp(min=1e-8)
    ll_lo = torch.log(d0) + (lo - y).clamp(min=0.0) / (m_lo * d0)
    ll_hi = torch.log(dK) + (y - hi).clamp(min=0.0) / (m_hi * dK)
    nll = torch.where(inside, ll_in, torch.where(below, ll_lo, ll_hi))
    return (nll * keep).sum() / keep.sum().clamp(min=1)


def _curvature_knots(y, n_knots, tau_eps, n_fine=400, floor=0.3):
    """Knots that minimise the interpolation error of D = Q'.  D is linear
    between knots, so a segment of width h errs by ~h^2 |D''| / 8 and the
    optimal knot density is proportional to |D''|^(1/2).  D is estimated from
    the marginal training quantile function, Gaussian-smoothed in tau so that
    sampling noise does not create spurious curvature; a uniform floor (a
    fraction ``floor`` of the mean weight) keeps every region resolved."""
    y = np.asarray(y, float).reshape(-1)
    ft = np.linspace(tau_eps, 1.0 - tau_eps, n_fine)
    fq = np.quantile(y, ft)
    fq = (fq - fq.min()) / (float(np.ptp(fq)) or 1.0)

    def smooth(v, sig=n_fine / 40):
        k = np.exp(-0.5 * (np.arange(-int(4 * sig), int(4 * sig) + 1) / sig) ** 2)
        k /= k.sum()
        return np.convolve(np.pad(v, len(k) // 2, mode="edge"), k, mode="valid")

    D = smooth(np.gradient(smooth(fq), ft))
    w = np.sqrt(np.abs(np.gradient(np.gradient(D, ft), ft)))
    w = w + floor * w.mean() + 1e-12
    cw = np.r_[0.0, np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(ft))]
    k = np.interp(np.linspace(0.0, cw[-1], n_knots), cw, ft)
    k[0], k[-1] = tau_eps, 1.0 - tau_eps
    return np.unique(k)


def _arclength_knots(y, n_knots, tau_eps, n_fine=2000):
    """Knots at equal arc length along the empirical quantile curve, so
    resolution goes where Q moves (the tails) rather than uniformly in tau."""
    y = np.asarray(y, float).reshape(-1)
    ft = np.linspace(tau_eps, 1.0 - tau_eps, n_fine)
    fq = np.quantile(y, ft)
    tn = (ft - ft[0]) / (ft[-1] - ft[0])
    rng = float(np.ptp(fq)) or 1.0
    qn = (fq - fq.min()) / rng
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(tn), np.diff(qn)))]
    k = np.interp(np.linspace(0.0, arc[-1], n_knots), arc, ft)
    k[0], k[-1] = tau_eps, 1.0 - tau_eps
    return np.unique(k)


def _expm1_over(a):
    """expm1(a) / a, stable at a -> 0 (torch).  With a = log d_k - log d_k+1,
    logmean(1/d_k, 1/d_k+1) = expm1(a) / (a d_k).  (Forming d_k/d_k+1 - 1 first
    rounds to -1 in float32 once the slopes differ by ~1e7: NaN on fb1.)"""
    small = a.abs() < 1e-4
    a_s = torch.where(small, torch.ones_like(a), a)
    return torch.where(small, 1.0 + a / 2.0 + a * a / 6.0, torch.expm1(a_s) / a_s)


def _to_g(d):
    return -1.0 / np.maximum(d, 1e-300)


def _g_level(c):
    return -1.0 / np.maximum(np.asarray(c, float), 1e-300)


def _flin_parts(d):
    dk, dk1 = d[:, :-1], d[:, 1:]
    pos = (dk > 0) & (dk1 > 0)
    rho = np.where(pos, dk / np.where(pos, dk1, 1.0) - 1.0, 0.0)
    return dk, dk1, pos, rho


def _q_at_s_f(q, d, dtau, s):
    dk, dk1, pos, rho = _flin_parts(d)
    small = np.abs(rho) < 1e-8
    L = np.where(
        small, s - rho * s * s / 2.0, np.log1p(rho * s) / np.where(small, 1.0, rho)
    )
    lin = q[:, :-1] + dtau * (dk * s + (dk1 - dk) * s * s / 2.0)
    return np.where(pos, q[:, :-1] + dtau * dk * L, lin)


def _invert_f(q, d, dtau, y):
    """tau_y under the f-linear reading: s = expm1(rho u) / rho.  Exact."""
    K = q.shape[1]
    dtau = np.broadcast_to(dtau, (len(q), K - 1))
    i0, s0 = _invert(q, d, dtau, y)
    r = np.arange(len(y))
    dk, dk1, dt = d[r, i0], d[r, i0 + 1], dtau[r, i0]
    pos = (dk > 0) & (dk1 > 0)
    rho = np.where(pos, dk / np.where(pos, dk1, 1.0) - 1.0, 0.0)
    u = (y - q[r, i0]) / np.where(pos, dt * dk, 1.0)
    small = np.abs(rho) < 1e-8
    sf = np.where(
        small,
        u * (1 + rho * u / 2.0),
        np.expm1(np.clip(rho * u, -700, 700)) / np.where(small, 1.0, rho),
    )
    return i0, np.where(pos, np.clip(sf, 0.0, 1.0), s0)


def _D_at(d, i, s, interp):
    r = np.arange(len(i))
    dk, dk1 = d[r, i], d[r, i + 1]
    if interp != "flin":
        return (1 - s) * dk + s * dk1
    pos = (dk > 0) & (dk1 > 0)
    return np.where(
        pos, dk * dk1 / np.where(pos, dk1 + s * (dk - dk1), 1.0), (1 - s) * dk + s * dk1
    )


def _invert(q, d, dtau, y):
    """tau_y = Q^-1(y): locate the segment, then ONE quadratic root.  Exact."""
    K = q.shape[1]
    dtau = np.broadcast_to(dtau, (len(q), K - 1))
    i = np.clip((q < y[:, None]).sum(1) - 1, 0, K - 2)
    r = np.arange(len(y))
    qk, dk, dk1, dt = q[r, i], d[r, i], d[r, i + 1], dtau[r, i]
    target = (y - qk) / dt
    dd = dk1 - dk
    lin = np.abs(dd) < 1e-12
    disc = np.maximum(dk * dk + 2.0 * dd * target, 0.0)
    s = np.where(
        lin,
        target / np.maximum(dk, 1e-300),
        (-dk + np.sqrt(disc)) / np.where(lin, 1.0, dd),
    )
    return i, np.clip(s, 0.0, 1.0)


def _level_set(d, dtau, thr):
    """{tau : D(tau) <= thr} per segment.  D is linear, so at most one crossing:
    a comparison and a division over the whole [n, K-1] array."""
    dk, dk1 = d[:, :-1], d[:, 1:]
    thr = np.asarray(thr, float).reshape(-1, 1)
    lo_in, hi_in = dk <= thr, dk1 <= thr
    dd = dk1 - dk
    with np.errstate(over="ignore", invalid="ignore"):
        cross = np.clip(
            (thr - dk) / np.where(np.abs(dd) < 1e-300, 1e-300, dd), 0.0, 1.0
        )
    s0 = np.where(lo_in, 0.0, cross)
    s1 = np.where(hi_in, 1.0, cross)
    empty = ((~lo_in) & (~hi_in)) | (s1 <= s0)
    return np.where(empty, np.nan, s0), np.where(empty, np.nan, s1)


def _q_at_s(q, d, dtau, s):
    dk, dk1 = d[:, :-1], d[:, 1:]
    return q[:, :-1] + dtau * (dk * s + (dk1 - dk) * s * s / 2.0)


def _sets(q, d, dtau, thr, interp="dlin"):
    """Sets {D <= thr}.  With interp="flin", d is still D at the knots and thr
    a D-level; the level set is read on g = -1/D, which is linear in tau."""
    if interp == "flin":
        s0, s1 = _level_set(_to_g(d), dtau, _g_level(thr))
        qs = _q_at_s_f
    else:
        s0, s1 = _level_set(d, dtau, thr)
        qs = _q_at_s
    y0 = np.where(np.isnan(s0), np.nan, qs(q, d, dtau, np.nan_to_num(s0)))
    y1 = np.where(np.isnan(s1), np.nan, qs(q, d, dtau, np.nan_to_num(s1)))

    y1 = np.where(np.isnan(y1), np.nan, np.maximum(y0, y1))
    rows = []
    for r in range(len(q)):
        m = ~np.isnan(y0[r])
        if not m.any():
            rows.append(np.zeros((0, 2)))
            continue
        a, b = y0[r][m], y1[r][m]
        out = [[a[0], b[0]]]
        for lo, hi in zip(a[1:], b[1:]):
            if lo <= out[-1][1] + 1e-12:
                out[-1][1] = max(out[-1][1], hi)
            else:
                out.append([lo, hi])
        rows.append(np.asarray(out))
    return rows


def _mass_below(d, dtau, c, strict=False):
    """m(c|x) = |{tau : D(tau) <= c}| (or |{D < c}| with strict=True), exact
    for piecewise-linear D.

    On segment k, D runs linearly from d_k to d_{k+1}; the sub-length with
    D <= c is a clipped ratio.  The two versions differ only on flat
    segments (lo == hi), where strict excludes the level itself.  c may be a
    scalar, [n] or [n, m] (levels)."""

    def ge(cc, ll):
        tol = 1e-12 * np.maximum(np.abs(ll), 1e-300)
        return (cc > ll + tol) if strict else (cc >= ll - tol)

    lo = np.minimum(d[:, :-1], d[:, 1:])
    hi = np.maximum(d[:, :-1], d[:, 1:])
    dtau = np.broadcast_to(dtau, lo.shape)
    c = np.asarray(c, float)
    if c.ndim <= 1:
        c = c.reshape(-1, 1)
        span = hi - lo
        frac = np.where(
            span > 1e-300,
            np.clip((c - lo) / np.where(span > 1e-300, span, 1.0), 0.0, 1.0),
            ge(c, lo).astype(float),
        )
        return (dtau * frac).sum(1)

    lo, hi, dt = lo[:, None, :], hi[:, None, :], dtau[:, None, :]
    span = hi - lo
    frac = np.where(
        span > 1e-300,
        np.clip((c[:, :, None] - lo) / np.where(span > 1e-300, span, 1.0), 0.0, 1.0),
        ge(c[:, :, None], lo).astype(float),
    )
    return (dt * frac).sum(2)


def _hpd_threshold_exact(d, dtau, t):
    """c(x) = min{c : m(c|x) >= t}, evaluating m at every knot value (O(K^2)
    per row, no cancellation).  Used as the fallback of the sweep.

    Between consecutive knot values m is linear; it can jump only AT a knot
    value, where a flat segment of D sits.  So on (C_{j-1}, C_j) m runs
    linearly from m(C_{j-1}) to the left limit m^<(C_j), and then jumps to
    m(C_j).  Interpolating across the jump returned a level inside the bracket
    where m < t (consistency test with flat segments)."""
    C = np.sort(d, axis=1)
    M = _mass_below(d, dtau, C)
    Ml = _mass_below(d, dtau, C, strict=True)
    n, K = d.shape
    r = np.arange(n)
    j = np.clip((M < t).sum(1), 0, K - 1)
    jm = np.maximum(j - 1, 0)
    c0, c1, m0, ml1 = C[r, jm], C[r, j], M[r, jm], Ml[r, j]
    on_ramp = (j > 0) & (ml1 >= t) & (ml1 - m0 > 1e-300)
    w = np.clip((t - m0) / np.where(ml1 - m0 > 1e-300, ml1 - m0, 1.0), 0.0, 1.0)
    c = np.where(on_ramp, c0 + w * (c1 - c0), c1)
    return np.where(t > M[:, -1], C[:, -1], c)


FALLBACK_COUNT = {"rows": 0, "fallback_rows": 0}


def _hpd_threshold(d, dtau, t):
    """c(x) with m(c|x) = t, where m(c) = |{tau : D(tau) <= c}|.

    m is piecewise linear in c: segment k contributes a ramp of slope
    a_k = dtau_k/(hi_k-lo_k) between lo_k and hi_k and the constant dtau_k
    beyond.  Sweep the 2(K-1) ramp end-points in sorted order, accumulating
    slope and intercept (O(K log K) per row, vectorised over rows), locate the
    bracket containing t, and interpolate.  The sweep's cumulative sums can
    cancel catastrophically when two adjacent slopes are nearly equal (the atom
    run of an anchored fit), so the result is verified exactly and any row
    that fails is recomputed by _hpd_threshold_exact."""
    lo = np.minimum(d[:, :-1], d[:, 1:])
    hi = np.maximum(d[:, :-1], d[:, 1:])
    dtau = np.broadcast_to(dtau, lo.shape)
    span = hi - lo
    degenerate = span <= 1e-9 * np.maximum(np.abs(hi), 1e-300)
    a = np.where(degenerate, 0.0, dtau / np.where(degenerate, 1.0, span))
    pos = np.concatenate([lo, hi], 1)
    dS = np.concatenate([a, -a], 1)
    dI = np.concatenate([-a * lo, np.where(degenerate, dtau, a * hi)], 1)
    order = np.argsort(pos, 1, kind="stable")
    pos = np.take_along_axis(pos, order, 1)
    S = np.cumsum(np.take_along_axis(dS, order, 1), 1)
    I = np.cumsum(np.take_along_axis(dI, order, 1), 1)
    M = S * pos + I
    n, E = pos.shape
    j = np.clip((M <= t).sum(1) - 1, 0, E - 1)
    r = np.arange(n)
    Sj, Mj, pj = S[r, j], M[r, j], pos[r, j]
    c = np.where(Sj > 1e-300, pj + (t - Mj) / np.where(Sj > 1e-300, Sj, 1.0), pj)
    c = np.minimum(c, pos[r, np.minimum(j + 1, E - 1)])
    c = np.where(t >= M[:, -1], pos[:, -1], c)

    total = dtau.sum(1)
    ok = (_mass_below(d, dtau, c) >= np.minimum(t, total) - 1e-9) & (
        _mass_below(d, dtau, c - 1e-12 * np.abs(c) - 1e-300) <= t + 1e-9
    )
    FALLBACK_COUNT["rows"] += int(len(ok))
    FALLBACK_COUNT["fallback_rows"] += int((~ok).sum())
    if not ok.all():
        bad = np.flatnonzero(~ok)
        c[bad] = _hpd_threshold_exact(d[bad], dtau[bad], t)

    return c


class LQDSCP:
    """fit -> calibrate -> predict.

    The paper configuration derives K from training-set size, uses fewer outputs
    than the 99-quantile baselines, and makes one fit. Pass ``k_grid`` only for
    controlled K-ablation studies.
    """

    K_CAP = 60

    @staticmethod
    def default_k(n_train, cap=60):
        """Training-only resolution rule, fixed before calibration."""
        return int(np.clip(round(np.sqrt(n_train)), 8, cap))

    def __init__(
        self,
        alpha=0.10,
        k_grid=None,
        hidden_grid=(64,),
        n_layers=2,
        dropout=0.1,
        tau_eps=1e-3,
        atom_threshold=0.02,
        knot_rule="arclength",
        learn_knots=False,
        knot_floor=0.1,
        smooth_edge=0,
        nll_weight=0.0,
        knot_anchor="tau",
        n_blocks=3,
        y_top_q=1.0,
        y_affine=False,
        nll_rows=0,
        crps_weight=1.0,
        interp="dlin",
        epochs=600,
        lr=1e-3,
        patience=40,
        batch_size=256,
        device="cpu",
        weight_decay=0.0,
        early_stop_min_delta=1e-7,
        head="softplus",
        tail_weight=True,
        loss_at="sampled",
        train_mode="match",
        smooth=0.0,
        score="mw",
        kernel=0,
        post_kernel_grid=(0, 1, 2, 3, 4),
        model_grid=None,
        fixed_post_kernels=None,
    ):
        self.kernel = int(kernel)

        self.fixed_post_kernels = (
            None
            if fixed_post_kernels is None
            else {str(op): int(k) for op, k in dict(fixed_post_kernels).items()}
        )

        self.post_kernel_grid = tuple(int(k) for k in post_kernel_grid)
        self.post_kernel = 0

        self.score = score
        self.head = head

        self.smooth_grid = (
            (float(smooth),) if np.isscalar(smooth) else tuple(float(v) for v in smooth)
        )
        self.smooth = self.smooth_grid[0]
        self.tail_weight = tail_weight
        self.loss_at = loss_at
        self.train_mode = train_mode
        self.alpha = float(alpha)
        self.k_grid = None if k_grid is None else tuple(k_grid)
        self.hidden_grid = tuple(hidden_grid)

        self.model_grid = (
            None
            if model_grid is None
            else tuple((None if k is None else int(k), int(h)) for k, h in model_grid)
        )
        self.cfg = dict(
            n_layers=n_layers,
            dropout=dropout,
            tau_eps=tau_eps,
            knot_rule=knot_rule,
            learn_knots=bool(learn_knots),
            knot_floor=float(knot_floor),
            smooth_edge=int(smooth_edge),
            nll_weight=float(nll_weight),
            knot_anchor=str(knot_anchor),
            n_blocks=int(n_blocks),
            y_top_q=float(y_top_q),
            y_affine=(
                y_affine
                if y_affine in ("loc", "locpos", "qshift", "yshift")
                else bool(y_affine)
            ),
            nll_rows=int(nll_rows),
            crps_weight=float(crps_weight),
            interp=str(interp),
            epochs=epochs,
            lr=lr,
            patience=patience,
            batch_size=batch_size,
            weight_decay=weight_decay,
            early_stop_min_delta=early_stop_min_delta,
        )
        self.atom_threshold = float(atom_threshold)
        self.device = device
        self.net_ = None
        self.threshold_ = None
        self.thresholds_ = {}
        self.models_ = None

    def _train_one(self, net, Xtr, ytr, Xva, yva, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        net.to(self.device)
        Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=self.device)
        ytr = torch.as_tensor(ytr, dtype=torch.float32, device=self.device)
        Xva = torch.as_tensor(Xva, dtype=torch.float32, device=self.device)
        yva = torch.as_tensor(yva, dtype=torch.float32, device=self.device)

        legacy = self.train_mode == "legacy"
        opt = (torch.optim.Adam if legacy else torch.optim.AdamW)(
            net.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg.get("weight_decay", 0.0),
        )
        sched = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.5, patience=8, min_lr=1e-5
            )
            if legacy
            else None
        )
        tol = 1e-4 if legacy else float(self.cfg.get("early_stop_min_delta", 1e-7))
        n = len(ytr)
        bs = min(self.cfg["batch_size"], n)
        if net.nll_weight > 0 and getattr(net, "crps_weight", 1.0) == 0:
            net.nll_scale = 1.0
        elif net.nll_weight > 0:

            idx = torch.randperm(n, device=self.device)[: min(n, 2048)]
            net.train()
            main, nll = net.loss_parts(Xtr[idx], ytr[idx])
            params = [p for p in net.parameters() if p.requires_grad]
            gm = torch.autograd.grad(main, params, retain_graph=True, allow_unused=True)
            gn = torch.autograd.grad(nll, params, allow_unused=True)
            norm = lambda gs: float(
                torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None))
            )
            net.nll_scale = norm(gm) / max(norm(gn), 1e-12)
        best, state, bad = np.inf, copy.deepcopy(net.state_dict()), 0
        for ep in range(self.cfg["epochs"]):
            net.train()
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, bs):
                idx = perm[s : s + bs]
                if len(idx) < 8:
                    continue
                loss = net.loss(Xtr[idx], ytr[idx])
                opt.zero_grad()
                loss.backward()
                if legacy:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(net.loss(Xva, yva))
            if sched is not None:
                sched.step(v)
            if v < best - tol:
                best, state, bad = v, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if legacy and ep < 15:
                    continue
                if bad >= self.cfg["patience"]:
                    break
        net.load_state_dict(state)
        net.eval()
        net.epochs_ = ep + 1
        return net

    def fit(self, X_train, y_train, X_val, y_val, seed=0):
        score_in = self.score
        y_train = np.asarray(y_train, float).reshape(-1)
        y_val = np.asarray(y_val, float).reshape(-1)

        self.y_off_ = float(np.min(y_train))
        self.y_scl_ = float(np.max(y_train) - np.min(y_train)) or 1.0
        y_train = (y_train - self.y_off_) / self.y_scl_
        y_val = (y_val - self.y_off_) / self.y_scl_
        floor = float(np.min(y_train))
        self.atom_mass_ = float(np.mean(np.isclose(y_train, floor)))

        self.anchor_ = self.atom_mass_ >= self.atom_threshold

        u = np.unique(np.round(y_train, 9))
        st = float(np.median(np.diff(u)[:50])) if len(u) > 1 else 0.0
        on = (
            st > 0
            and len(u) < 0.5 * len(y_train)
            and np.max(np.abs((u - u[0]) / st - np.round((u - u[0]) / st))) < 0.05
        )
        self.dq_step_ = st if on else 0.0
        default_k = self.default_k(len(y_train), self.K_CAP)
        if self.k_grid is None:
            self.k_grid = (default_k,)
        half = len(y_val) // 2
        self._Xval, self._yval = np.asarray(X_val), y_val
        specs = (
            [(default_k if k is None else k, h) for k, h in self.model_grid]
            if self.model_grid is not None
            else [(K, h) for h in self.hidden_grid for K in self.k_grid]
        )
        if any(K > self.K_CAP for K, _ in specs):
            raise ValueError(
                f"LQDS K must be <= {self.K_CAP}; got {sorted({K for K, _ in specs if K > self.K_CAP})}"
            )
        candidates = []
        for K, hidden in specs:
            for smooth in self.smooth_grid:

                torch.manual_seed(seed)
                np.random.seed(seed)
                net = _Net(
                    X_train.shape[1],
                    y_train,
                    K,
                    hidden,
                    self.cfg["n_layers"],
                    self.cfg["dropout"],
                    self.cfg["tau_eps"],
                    self.anchor_,
                    self.head,
                    self.tail_weight,
                    self.loss_at,
                    smooth,
                    self.kernel,
                    self.cfg.get("knot_rule", "arclength"),
                    self.cfg.get("learn_knots", False),
                    self.cfg.get("knot_floor", 0.1),
                    self.cfg.get("smooth_edge", 0),
                    self.cfg.get("nll_weight", 0.0),
                    self.cfg.get("knot_anchor", "tau"),
                    self.cfg.get("n_blocks", 3),
                    self.cfg.get("y_top_q", 1.0),
                    self.cfg.get("y_affine", False),
                )
                net.dq_step = self.dq_step_
                net.nll_rows = self.cfg.get("nll_rows", 0)
                net.crps_weight = self.cfg.get("crps_weight", 1.0)
                net.interp = self.cfg.get("interp", "dlin")
                net = self._train_one(net, X_train, y_train, X_val, y_val, seed)
                candidates.append((net, (K, hidden, smooth)))

        selected = {}
        if self.fixed_post_kernels is not None:
            if len(candidates) != 1:
                raise ValueError(
                    "fixed_post_kernels requires exactly one (K, hidden, smooth) candidate"
                )
            net, spec = candidates[0]
            for sc in ("mw", "hpd"):
                k = self.fixed_post_kernels[sc]
                selected[sc] = dict(
                    net=net,
                    selected=spec + (k,),
                    post_kernel=k,
                    val_coverage=None,
                    val_width=None,
                )
            self.models_ = selected
            self.post_kernels_ = dict(self.fixed_post_kernels)
            self.score = score_in
            self._activate_score()
            return self
        for sc in ("mw", "hpd"):
            eligible, fallback = [], []
            for net, spec in candidates:
                kernel, cov, width = self._select_post_kernel_for(net, sc, half)
                row = (width, net, spec, kernel, cov)
                fallback.append(row)
                if cov >= 1 - self.alpha - 0.02:
                    eligible.append(row)
            choice = min(eligible or fallback, key=lambda row: row[0])
            selected[sc] = dict(
                net=choice[1],
                selected=choice[2] + (choice[3],),
                post_kernel=choice[3],
                val_coverage=choice[4],
                val_width=choice[0],
            )
        self.models_ = selected
        self.post_kernels_ = {sc: row["post_kernel"] for sc, row in selected.items()}
        self.score = score_in
        self._activate_score()
        return self

    def save(self, path):
        """Portable checkpoint: network weights plus everything needed to
        rebuild the object (scaling, knots, anchor, threshold, selection)."""
        n = self.net_
        payload = dict(
            checkpoint_version=1,
            family="lqds",
            model_class="LQDSCP",
            state_dict={k: v.detach().cpu() for k, v in n.state_dict().items()},
            input_dim=int(n.encoder[0].in_features),
            hidden_dim=int(n.encoder[0].out_features),
            n_knots=int(len(n.tau)),
            tau=n.tau.detach().cpu(),
            y_floor=n.y_floor,
            anchor=n.anchor,
            head=n.head,
            net_attrs={
                k: v
                for k, v in vars(n).items()
                if isinstance(v, (bool, int, float, str))
                and not k.startswith("_")
                and k != "training"
            },
            tail_weight=n.tail_weight,
            loss_at=n.loss_at,
            smooth=n.smooth,
            smooth_grid=self.smooth_grid,
            kernel=n.kernel,
            alpha=self.alpha,
            cfg=dict(self.cfg),
            train_mode=self.train_mode,
            score=self.score,
            atom_threshold=self.atom_threshold,
            y_off=self.y_off_,
            y_scl=self.y_scl_,
            atom_mass=self.atom_mass_,
            selected=tuple(self.selected_),
            threshold=self.threshold_,
            post_kernel=getattr(self, "post_kernel", 0),
            kernel_score=getattr(self, "_kernel_score", "mw"),
            post_kernels=getattr(self, "post_kernels_", None),
            X_val=getattr(self, "_Xval", None),
            y_val=getattr(self, "_yval", None),
        )
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path, device="cpu"):
        pl = torch.load(path, map_location="cpu", weights_only=False)
        cfg = pl["cfg"]
        m = cls(
            alpha=pl["alpha"],
            hidden_grid=(pl["hidden_dim"],),
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            tau_eps=cfg["tau_eps"],
            atom_threshold=pl["atom_threshold"],
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            patience=cfg["patience"],
            batch_size=cfg["batch_size"],
            device=device,
            weight_decay=cfg.get("weight_decay", 0.0),
            early_stop_min_delta=cfg.get("early_stop_min_delta", 1e-7),
            head=pl["head"],
            tail_weight=pl["tail_weight"],
            loss_at=pl["loss_at"],
            train_mode=pl["train_mode"],
            smooth=pl.get("smooth_grid", pl.get("smooth", 0.0)),
            score=pl.get("score", "mw"),
            kernel=pl.get("kernel", 0),
            knot_rule=cfg.get("knot_rule", "arclength"),
            learn_knots=cfg.get("learn_knots", False),
            knot_floor=cfg.get("knot_floor", 0.1),
            smooth_edge=cfg.get("smooth_edge", 0),
            nll_weight=cfg.get("nll_weight", 0.0),
            knot_anchor=cfg.get("knot_anchor", "tau"),
            n_blocks=cfg.get("n_blocks", 3),
            y_top_q=cfg.get("y_top_q", 1.0),
            y_affine=cfg.get("y_affine", False),
            nll_rows=cfg.get("nll_rows", 0),
            crps_weight=cfg.get("crps_weight", 1.0),
            interp=cfg.get("interp", "dlin"),
        )
        tau = pl["tau"].numpy()

        surrogate = np.linspace(pl["y_floor"], pl["y_floor"] + 1.0, 64)

        n_knots = (
            int(pl["state_dict"]["y_grid"].shape[0])
            if "y_grid" in pl["state_dict"]
            else pl["n_knots"]
        )
        net = _Net(
            pl["input_dim"],
            surrogate,
            n_knots,
            pl["hidden_dim"],
            cfg["n_layers"],
            cfg["dropout"],
            cfg["tau_eps"],
            pl["anchor"],
            pl["head"],
            pl["tail_weight"],
            pl["loss_at"],
            pl.get("smooth", 0.0),
            pl.get("kernel", 0),
            cfg.get("knot_rule", "arclength"),
            cfg.get("learn_knots", False),
            cfg.get("knot_floor", 0.1),
            cfg.get("smooth_edge", 0),
            cfg.get("nll_weight", 0.0),
            cfg.get("knot_anchor", "tau"),
            cfg.get("n_blocks", 3),
            cfg.get("y_top_q", 1.0),
            cfg.get("y_affine", False),
        )
        net.interp = cfg.get("interp", "dlin")
        net.tau = torch.as_tensor(tau, dtype=torch.float32)
        net.dtau = torch.as_tensor(np.diff(tau), dtype=torch.float32)
        net.tau_lo, net.tau_span = float(tau[0]), float(tau[-1] - tau[0])
        net.c = (
            0
            if pl["anchor"]
            else (
                net.split_K1
                if getattr(net, "split", False)
                else int(np.argmin(np.abs(tau - 0.5)))
            )
        )
        net.y_floor = pl["y_floor"]
        net.load_state_dict(pl["state_dict"])
        if pl.get("net_attrs"):
            net.__dict__.update(pl["net_attrs"])
        elif getattr(net, "y_affine_mode", None) == "qshift":

            net.qs_top = float(
                np.clip(
                    np.interp(
                        float(net.y_grid[-1]), net.qm_y.numpy(), net.qm_u.numpy()
                    ),
                    1e-6,
                    1 - 1e-6,
                )
            )
        net.to(device).eval()
        m.net_ = net
        m.y_off_, m.y_scl_, m.atom_mass_ = pl["y_off"], pl["y_scl"], pl["atom_mass"]
        m.anchor_, m.selected_, m.threshold_ = (
            pl["anchor"],
            tuple(pl["selected"]),
            pl["threshold"],
        )
        m.smooth = pl.get("smooth", 0.0)
        m.post_kernel = pl.get("post_kernel", 0)
        m._kernel_score = pl.get("kernel_score", "mw")
        if pl.get("post_kernels") is not None:
            m.post_kernels_ = dict(pl["post_kernels"])
        if pl.get("X_val") is not None:
            m._Xval, m._yval = pl["X_val"], pl["y_val"]
        m.k_grid = (pl["n_knots"],)
        return m

    def _net64(self):
        """float64 copy of the fitted network for Stage 2: training runs in
        float32 for speed, but level-set crossings are read off D, and float32
        rounding in the slopes creates measure-zero splinters where D sits at
        the threshold."""
        key = id(self.net_)
        if getattr(self, "_net64_key", None) != key:
            self._net64_cache = copy.deepcopy(self.net_).double().eval()
            self._net64_key = key
        return self._net64_cache

    def _params(self, X, batch=512):
        """(q, d, dt) in float64: knot values, slopes and per-row knot spacings."""
        Q, D, T = [], [], []
        net = self._net64()
        for s in range(0, len(X), batch):
            xb = torch.as_tensor(
                np.asarray(X[s : s + batch], np.float64), device=self.device
            )
            with torch.no_grad():
                q, d, dt = net.params3(xb)
            Q.append(q.cpu().numpy())
            D.append(d.cpu().numpy())
            T.append(dt.cpu().numpy())
        q, d, dt = np.vstack(Q), np.vstack(D), np.vstack(T)
        if getattr(self, "post_kernel", 0) > 0:
            q, d = self._smooth(q, d, dt, self.post_kernel)
        return q, d, dt

    def _smooth(self, q, d, dtau, k):
        """Post-fit smoothing of the slopes: moving average over 2k+1 knots
        (edge-replicated), exact zeros of an anchored fit preserved, and the
        quantile function re-integrated from the anchor so the representation
        stays consistent.  A fixed transform of the score function -- chosen on
        validation, applied identically to calibration and test -- so the
        conformal guarantee is unaffected and no retraining is involved."""

        zero = (
            (d <= 1e-4 * np.median(d, axis=1, keepdims=True))
            if self.anchor_
            else np.zeros_like(d, bool)
        )
        dp = np.pad(d, ((0, 0), (k, k)), mode="edge")
        ds = np.mean([dp[:, i : i + d.shape[1]] for i in range(2 * k + 1)], axis=0)
        ds[zero] = 0.0
        c = self.net_.c
        step = dtau * (ds[:, :-1] + ds[:, 1:]) / 2.0
        up = q[:, [c]] + np.cumsum(step[:, c:], axis=1)
        down = q[:, [c]] - np.flip(
            np.cumsum(np.flip(step[:, :c], axis=1), axis=1), axis=1
        )
        return np.concatenate([down, q[:, [c]], up], axis=1), ds

    def _scores(self, X, y):
        q, d, dtau = self._params(X)
        y = np.asarray(y, float).reshape(-1)
        lo, hi = q[:, 0], q[:, -1]
        inside = (y >= lo) & (y <= hi)
        flin = self.cfg.get("interp", "dlin") == "flin"
        i, s = (_invert_f if flin else _invert)(q, d, dtau, np.clip(y, lo, hi))

        Dy = _D_at(d, i, s, "flin" if flin else "dlin")
        if self.score == "hpd":
            if flin:
                return np.where(
                    inside,
                    _mass_below(_to_g(d), dtau, _g_level(Dy), strict=True),
                    np.inf,
                )
            return np.where(inside, _mass_below(d, dtau, Dy, strict=True), np.inf)
        return np.where(inside, Dy, np.inf)

    def _val_score(self, net, Xc, yc, Xt, yt):
        keep, self.net_ = self.net_, net
        try:
            sc = self._scores(Xc, yc)
            thr = conformal_quantile(sc, self.alpha)
            q, d, dtau = self._params(Xt)
            rows = self._build(q, d, dtau, thr)
            st = self.summarise(rows, yt)
            return st["coverage"], st["mean_width"]
        finally:
            self.net_ = keep

    def _build(self, q, d, dtau, t):
        """Prediction sets from the calibrated threshold t (MW: a D-level;
        HPD: a mass level, converted to the per-input level c(x))."""
        if not np.isfinite(t):

            return [np.array([[-np.inf, np.inf]]) for _ in range(len(q))]
        if self.cfg.get("interp", "dlin") == "flin":
            if self.score == "hpd":
                cg = _hpd_threshold(_to_g(d), dtau, t)
                c = np.where(cg < 0, -1.0 / np.minimum(cg, -1e-300), np.inf)
            else:
                c = np.full(len(q), t)
            return _sets(q, d, dtau, c, "flin")
        c = _hpd_threshold(d, dtau, t) if self.score == "hpd" else np.full(len(q), t)
        return _sets(q, d, dtau, c)

    def _to_internal(self, y):
        return (np.asarray(y, float).reshape(-1) - self.y_off_) / self.y_scl_

    def _activate_score(self):
        """Activate the validation-selected representation for ``self.score``."""
        if self.models_ is not None and self.score in self.models_:
            row = self.models_[self.score]
            self.net_ = row["net"]
            self.selected_ = tuple(row["selected"])
            self.post_kernel = int(row["post_kernel"])
            self.smooth = float(self.selected_[2])

    def _select_post_kernel_for(self, net, score, half=None):
        keep = (self.net_, self.score, self.post_kernel)
        self.net_, self.score = net, score
        half = len(self._yval) // 2 if half is None else half
        choices = []
        try:
            for k in self.post_kernel_grid:
                self.post_kernel = k
                a = self._val_score(
                    net,
                    self._Xval[:half],
                    self._yval[:half],
                    self._Xval[half:],
                    self._yval[half:],
                )
                b = self._val_score(
                    net,
                    self._Xval[half:],
                    self._yval[half:],
                    self._Xval[:half],
                    self._yval[:half],
                )
                cov, width = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                choices.append((width, k, cov))
            eligible = [row for row in choices if row[2] >= 1 - self.alpha - 0.02]
            width, kernel, cov = min(eligible or choices, key=lambda row: row[0])
            return kernel, cov, width
        finally:
            self.net_, self.score, self.post_kernel = keep

    def _select_post_kernel(self):
        self.post_kernel = self._select_post_kernel_for(self.net_, self.score)[0]
        self._kernel_score = self.score

    def calibrate(self, X_cal, y_cal):
        self._activate_score()
        pk = getattr(self, "post_kernels_", None)
        if pk is not None and self.score in pk:
            self.post_kernel = pk[self.score]
        elif getattr(self, "_kernel_score", None) not in (None, self.score) and hasattr(
            self, "_Xval"
        ):
            self._select_post_kernel()
        sc = self._scores(X_cal, self._to_internal(y_cal))
        self.threshold_ = conformal_quantile(sc, self.alpha)
        self.thresholds_[self.score] = self.threshold_
        return self

    def predict(self, X):
        self._activate_score()
        if self.score in self.thresholds_:
            self.threshold_ = self.thresholds_[self.score]
        if self.threshold_ is None:
            raise RuntimeError("call calibrate() before predict()")
        q, d, dtau = self._params(X)
        rows = self._build(q, d, dtau, self.threshold_)
        return [r * self.y_scl_ + self.y_off_ for r in rows]

    @staticmethod
    def summarise(rows, y):
        y = np.asarray(y, float).reshape(-1)
        cov = np.array(
            [
                bool(len(r)) and bool(((y[i] >= r[:, 0]) & (y[i] <= r[:, 1])).any())
                for i, r in enumerate(rows)
            ]
        )
        w = np.array([(r[:, 1] - r[:, 0]).sum() if len(r) else 0.0 for r in rows])
        return dict(
            coverage=float(cov.mean()),
            mean_width=float(w.mean()),
            median_width=float(np.median(w)),
            components=float(np.mean([len(r) for r in rows])),
        )
