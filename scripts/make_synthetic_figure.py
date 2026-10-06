#!/usr/bin/env python3
"""Generate the four-panel synthetic figure from locally generated results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "synthetic"
KS = (20, 30, 50, 99)
COLORS = {"LQDS": "#2a78d6", "CTI": "#eb6834", "SPICE": "#1baf7a", "HPD": "#4a3aa7"}


def load_convergence() -> dict[tuple[int, int], list[dict]]:
    output = {}
    for path in sorted((RESULTS / "convergence").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        output.setdefault((record["n_tr"], record["K"]), []).append(record)
    return output


def main() -> None:
    oracle = json.loads((RESULTS / "synth_oracle.json").read_text(encoding="utf-8"))
    convergence = load_convergence()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Computer Modern Roman"], "text.usetex": True, "font.size": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(486 / 72, 95 / 72))
    panels = (
        (("LQDS-MW", "LQDS", "s"), ("CTI", "CTI", "o"), ("SPICE-ND", "SPICE", "^")),
        (("LQDS-HPD", "LQDS", "s"), ("HPD-Split", "HPD", "D"), ("SPICE-HPD", "SPICE", "^")),
    )
    for axis, series in zip(axes[:2], panels):
        for method, color, marker in series:
            means, errors = [], []
            for knots in KS:
                values = oracle["per_seed"][f"{method} | K={knots}"]["symdiff"]
                means.append(np.mean(values))
                errors.append(2 * np.std(values) / math.sqrt(len(values)))
            axis.errorbar(KS, means, yerr=errors, label=method.replace("-Split", "-split"), color=COLORS[color], marker=marker, lw=1, ms=3.2, capsize=1.5)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(KS)
        axis.set_xticklabels(KS)
        axis.set_xlabel("$K$")
        axis.legend(frameon=False, fontsize=6.5)
    axes[0].set_ylabel(r"$|\widehat{C}\,\triangle\,C^\star|$")

    grown = ((3000, 20), (12000, 30), (48000, 45))
    sizes = [item[0] for item in grown]
    for target, label, face in (("mw", "LQDS-MW", COLORS["LQDS"]), ("hpd", "LQDS-HPD", "white")):
        means, errors, coverage_sd = [], [], []
        for configuration in grown:
            records = convergence[configuration]
            values = [record[target]["symdiff"] for record in records]
            means.append(np.mean(values))
            errors.append(2 * np.std(values) / math.sqrt(len(values)))
            coverage_sd.append(np.mean([record["exp6"][f"{target}_cov_sd"] for record in records]))
        style = dict(color=COLORS["LQDS"], marker="s", markerfacecolor=face, markeredgewidth=0.8, lw=1, ms=3.5)
        axes[2].errorbar(sizes, means, yerr=errors, label=label, capsize=1.5, **style)
        axes[3].plot(sizes, coverage_sd, label=label, **style)
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$|\widehat{C}\,\triangle\,C^\star|$")
    axes[3].set_ylabel(r"$\mathrm{sd}\{\mathrm{cov}(X)\}$")
    for axis in axes[2:]:
        axis.set_xscale("log")
        axis.set_xticks(sizes)
        axis.set_xticklabels([f"{size:,}" for size in sizes])
        axis.set_xlabel(r"$n_{\mathrm{tr}}$")
        axis.legend(frameon=False, fontsize=6.5)
    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes):
        axis.text(0, 1.03, label, transform=axis.transAxes)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.minorticks_off()
    fig.tight_layout(pad=0.3, w_pad=1.0)
    output = RESULTS / "synthetic_figure.pdf"
    fig.savefig(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
