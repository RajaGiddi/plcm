"""
Figure 1 — the paper's thesis in two panels.  BUILD SPEC REV 2.

LOCK (Deck-Close Contract, Item 4): every number here traces to a row in
`docs/CLAIM_LEDGER.md`. If a number is not in the ledger at build time it does
not go in the figure.

  Ledger trace: 0.9313 / 0.2807 / 0.9447 (E14) · 66-88 (E6b, LSTM/HAR, 4 arms) ·
  82.5 [74.1, 90.9] (E17) · 91.8 [90.9, 92.7] (E12) · 98.8 [98.25, 99.38] (E14) ·
  97.9 (H-V3, CAPTION ONLY).

REV 1 SHIPPED A NUMBER THAT EXISTS IN NO LEDGER ROW. Panel (b) drew a point at
77.0% for the LSTM/MLP row — an average across heterogeneous arms, invented by
the figure to give the range a centre. It is removed. That row is an INTERVAL
and carries no point estimate, because none was ever measured.

REV 2 PRINCIPLE — each panel carries exactly one sentence of ink beyond the
data: (a) two bars against two references; (b) a monotone arc of three rows.
Everything qualified, corroborative, or mechanistic moves to the caption, which
this script emits alongside the figure so the two cannot drift apart.

Usage:  python scripts/figure1.py --out figures/figure1.pdf
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- palette: validated, six checks PASS, worst adjacent CVD dE 24.7 --------
BLUE = "#2a78d6"      # refit / measured shares
ORANGE = "#eb6834"    # deployed
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e3e2dd"
SURFACE = "#ffffff"

# ---- ledger-traced values ---------------------------------------------------
CEILING, DEPLOYED, REFIT = 0.9447, 0.2807, 0.9313      # E14
CHANCE = 0.20

# (label, point, lo, hi, kind). `None` point = an interval with no measured
# centre. The distinction is load-bearing and is DRAWN, not annotated.
ARC = [
    ("LSTM, 4 arms\n(HAR, scratch)", None, 66.0, 88.0, "range"),   # E6b
    ("MLP\n(MNIST, scratch)", 82.5, 74.1, 90.9, "ci"),             # E17
    ("ViT-B/16\npretrained", 91.8, 90.9, 92.7, "ci"),              # E12
    ("ResNet-50\npretrained", 98.8, 98.25, 99.38, "ci"),           # E14
]

CAPTION = (
    "Figure 1. (a) Same features, two readers: refitting the readout recovers "
    "task 0 to 0.93 against its original ceiling of 0.94, while the deployed "
    "readout scores 0.28. Encoder damage F_enc = 0.0076. (b) The readout share "
    "of forgetting across four architectures: 66-88% across four LSTM arms "
    "(UCI HAR; range shown - heterogeneous arms, no pooled point), "
    "82.5% [74.1, 90.9] for a scratch-trained MLP (Permuted MNIST, 3 seeds; "
    "same-seed share floor 0.01pp), 91.8% [90.9, 92.7] for a fully fine-tuned "
    "ViT-B/16, and 98.8% [98.25, 99.38] for a pretrained ResNet-50. Scratch "
    "evidence is per-dataset; no row pools across datasets. The share rises "
    "across the four rows but is not strictly monotone - the MLP point lies "
    "inside the LSTM range. A class-incremental protocol on the ViT yields "
    "97.9% (corroborative only; shared-head recency effects addressed in "
    "Sec. 4.3). The dashed line in (a) and the 50% gridline in (b) mark the "
    "original ceiling and the majority threshold. The light interval in (b) is "
    "a range across arms; the darker whiskers are 95% confidence intervals."
)


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 7.5,
        "axes.labelpad": 20,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_a(ax):
    """Two bars, two reference lines, three printed numbers. Nothing else."""
    for lab, val, c, yy in (("refit reader", REFIT, BLUE, 1),
                            ("deployed reader", DEPLOYED, ORANGE, 0)):
        ax.barh(yy, val, height=0.38, color=c, zorder=3,
                edgecolor=SURFACE, linewidth=0.8)
        ax.text(val + 0.015, yy, f"{val:.2f}", va="center", ha="left",
                fontsize=9, color=INK, fontweight="bold", zorder=4)
        ax.text(-0.02, yy, lab, va="center", ha="right", fontsize=8.5, color=INK2)

    # Ceiling — label ABOVE the plot area, right-aligned to the line.
    ax.axvline(CEILING, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=2)
    ax.text(CEILING, 1.72, f"ceiling {CEILING:.2f}", ha="right", va="bottom",
            fontsize=8, color=INK2)
    # Chance — label BELOW the axis, aligned to the line.
    ax.axvline(CHANCE, color=MUTED, lw=0.8, ls=":", zorder=2)
    ax.annotate("chance", xy=(CHANCE, 0), xytext=(0, -21),
                xycoords=("data", "axes fraction"), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, color=MUTED)

    ax.set_xlim(0, 1.0); ax.set_ylim(-0.7, 1.72)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("task-0 accuracy after 20 tasks  (ResNet-50, 3 seeds)")
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("a   The features survive; the reader does not",
                 loc="left", fontsize=10, color=INK, fontweight="bold", pad=16)


def panel_b(ax):
    """Three rows, one axis, four printed numbers. Nothing else."""
    ys = list(range(len(ARC)))[::-1]
    for (lab, val, lo, hi, kind), yy in zip(ARC, ys):
        if kind == "range":
            # Lighter AND thicker than a CI whisker — the two are different
            # objects, and the drawing says so without a note on the plot.
            ax.plot([lo, hi], [yy, yy], color=MUTED, lw=7, alpha=0.28,
                    solid_capstyle="round", zorder=2)
            ax.text(lo - 1.0, yy, f"{lo:.0f}%", ha="right", va="center",
                    fontsize=8.5, color=INK2)
            ax.text(hi + 1.0, yy, f"{hi:.0f}%", ha="left", va="center",
                    fontsize=8.5, color=INK2)
        else:
            ax.plot([lo, hi], [yy, yy], color=BLUE, lw=1.2,
                    solid_capstyle="butt", zorder=2)
            for e in (lo, hi):        # thin caps: the CIs are 1.1-1.8pp wide,
                ax.plot([e, e], [yy - 0.07, yy + 0.07],   # so the marker would
                        color=BLUE, lw=1.2, zorder=2)     # otherwise hide them
            ax.plot(val, yy, "o", ms=5.5, color=BLUE, mec=SURFACE, mew=1.1,
                    zorder=4)
            ax.text(val, yy + 0.24, f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=9, color=INK, fontweight="bold")

    ax.axvline(50, color=GRID, lw=0.9, zorder=1)      # gridline, no on-plot word

    ax.set_xlim(45, 100); ax.set_ylim(-0.7, len(ARC) - 0.28)
    ax.set_yticks(ys)
    ax.set_yticklabels([a[0] for a in ARC], fontsize=8, color=INK2,
                       linespacing=1.35)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.set_xlabel("share of forgetting carried by the readout  (%; 50 = majority)")
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("b   One instrument, four architectures",
                 loc="left", fontsize=10, color=INK, fontweight="bold", pad=16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/figure1.pdf")
    args = ap.parse_args()
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7),
                             gridspec_kw={"width_ratios": [1.0, 1.15],
                                          "wspace": 0.30})
    panel_a(axes[0]); panel_b(axes[1])
    fig.subplots_adjust(left=0.115, right=0.985, top=0.84, bottom=0.22)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    # The caption ships WITH the figure, from the same file, so the two cannot
    # drift: every qualifier the plot dropped is carried here.
    out.with_suffix(".caption.txt").write_text(CAPTION + "\n")
    print(f"wrote {out}, {out.with_suffix('.png')}, {out.with_suffix('.caption.txt')}")


if __name__ == "__main__":
    main()
