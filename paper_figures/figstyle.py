"""Shared style of the paper-figure notebooks (2026-11-13).  One colour per entity, the same in every figure:
shared failure = orange, shared success = aqua, other / untyped tie = yellow, decided pair = neutral grey;
tie probability / our method = blue, pointwise score = violet, baselines = grey.  The five hues are the first slots of a
categorical palette checked for colour-blind separation; aqua and yellow have low contrast on white, so every figure
labels its marks directly or carries a legend."""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

C = dict(fail="#eb6834", succeed="#1baf7a", other="#eda100", decided="#c3c2b7", ours="#2a78d6", pointwise="#6250d6",
         baseline="#898781", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7")
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); FIGS = os.path.join(HERE, "figs")


def setup():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "xtick.color": C["muted"], "ytick.color": C["muted"],
        "text.color": C["ink"], "axes.titlecolor": C["ink"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": .6,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 5, "pdf.fonttype": 42,
        "axes.titleweight": "bold", "axes.titlelocation": "left"})


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    print("saved", f"figs/{name}.pdf / .png")


def placeholder(ax, title, lines):
    """an empty panel that says what will go here and what it is waiting for"""
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linestyle((0, (4, 3))); s.set_color(C["axis"])
    ax.set_title(title)
    ax.text(.5, .5, "\n".join(lines), ha="center", va="center", color=C["muted"], fontsize=8, transform=ax.transAxes, linespacing=1.6)
