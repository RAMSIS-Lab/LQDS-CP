"""Generate the three appendix width boxplots from split-level results."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
split_widths = json.loads(
    (ROOT / "artifacts" / "summary" / "benchmark_splits.json").read_text()
)

DATASETS = ["meps19", "meps20", "meps21", "fb1", "fb2", "blog", "bio", "star", "community", "bike"]
DATASET_NAMES = {
    "meps19": "MEPS 19", "meps20": "MEPS 20", "meps21": "MEPS 21",
    "fb1": "Facebook 1", "fb2": "Facebook 2", "blog": "Blog", "bio": "CASP",
    "star": "STAR", "community": "Communities", "bike": "Bike",
}
GROUPS = [
    ("mw", [("cti", "CTI"), ("spice_nd", "SPICE-ND"), ("lqds_cp", "LQDS-MW")]),
    ("hpd", [("cir_nu", "CIR-NU"), ("hpd_split", "HPD-split"),
             ("spice_hpd", "SPICE-HPD"), ("lqds_hpd", "LQDS-HPD")]),
    ("interval", [("cir_fast", "CIR"), ("cir_plus_fast", r"CIR$^{+}$"),
                  ("cqr", "CQR"), ("dcp", "DCP"), ("dcp_cqr", "DCP-CQR"),
                  ("dist_split", "Dist-split")]),
]

# Okabe--Ito-inspired colours, with the two proposed methods consistently blue.
COLORS = {
    "cti": "#E69F00", "spice_nd": "#009E73", "lqds_cp": "#2A78D6",
    "cir_nu": "#E69F00", "hpd_split": "#CC79A7", "spice_hpd": "#009E73",
    "lqds_hpd": "#2A78D6", "cir_fast": "#E69F00", "cir_plus_fast": "#56B4E9",
    "cqr": "#009E73", "dcp": "#CC79A7", "dcp_cqr": "#D55E00",
    "dist_split": "#7A7A7A",
}


def load_widths():
    """Return widths[dataset][method], with one value for each of the 30 splits."""
    widths = {ds: {} for ds in DATASETS}
    methods = {m for _, group in GROUPS for m, _ in group}
    for ds in DATASETS:
        for method in methods:
            arr = np.asarray(split_widths["mean_width"][ds][method], dtype=float)
            assert len(arr) == 30 and np.isfinite(arr).all(), (ds, method)
            widths[ds][method] = arr
    return widths


def style_axis(ax, panel_index):
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=6.2, length=2, width=0.45, pad=1.5)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9A9A9A")
        ax.spines[side].set_linewidth(0.45)
    if panel_index % 5 == 0:
        ax.set_ylabel("Mean total width", fontsize=6.7, labelpad=2)


def make_figure(widths, slug, methods):
    # 6.75 in is exactly \textwidth in aistats2027.sty.
    fig, axes = plt.subplots(2, 5, figsize=(6.75, 3.02))
    rng = np.random.default_rng(2027)
    positions = np.arange(1, len(methods) + 1)
    for panel_index, (ax, ds) in enumerate(zip(axes.flat, DATASETS)):
        values = [widths[ds][method] for method, _ in methods]
        bp = ax.boxplot(
            values, positions=positions, widths=0.62, patch_artist=True,
            showfliers=False, whis=(5, 95), medianprops={"linewidth": 0.9},
            boxprops={"linewidth": 0.65}, whiskerprops={"linewidth": 0.55},
            capprops={"linewidth": 0.55},
        )
        for i, (method, _) in enumerate(methods):
            color = COLORS[method]
            bp["boxes"][i].set(facecolor=color, edgecolor=color, alpha=0.88)
            bp["medians"][i].set(color="white")
            for artist in bp["whiskers"][2 * i:2 * i + 2] + bp["caps"][2 * i:2 * i + 2]:
                artist.set(color=color)
            jitter = rng.uniform(-0.17, 0.17, size=len(values[i]))
            ax.scatter(np.full(len(values[i]), positions[i]) + jitter, values[i], s=2.6,
                       color="#202020", alpha=0.28, linewidths=0, zorder=3)
        ax.set_title(DATASET_NAMES[ds], fontsize=7.2, pad=2.5)
        ax.set_xlim(0.45, len(methods) + 0.55)
        style_axis(ax, panel_index)

    handles = [Patch(facecolor=COLORS[m], edgecolor=COLORS[m], label=label) for m, label in methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(methods), frameon=False,
               fontsize=6.5, handlelength=1.1, handleheight=0.7,
               columnspacing=1.25, borderaxespad=0)
    fig.subplots_adjust(left=0.073, right=0.995, top=0.925, bottom=0.145, wspace=0.46, hspace=0.48)
    output = ROOT / "artifacts" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    out = output / f"width_box_{slug}.pdf"
    fig.savefig(out, bbox_inches=None)
    plt.close(fig)
    print(f"wrote {out.name}")


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "text.usetex": True, "font.size": 7, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "pdf.fonttype": 42, "savefig.transparent": False,
    })
    widths = load_widths()
    for slug, methods in GROUPS:
        make_figure(widths, slug, methods)


if __name__ == "__main__":
    main()
