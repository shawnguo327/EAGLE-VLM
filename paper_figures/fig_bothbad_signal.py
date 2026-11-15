"""fig_bothbad_signal (2026-11-08): what a forced choice delivers when both candidates failed.
Data: results/forced_choice_signal/20261108/forced_choice_signal.csv (scripts/forced_choice_signal.py; six open VLM judges,
acceptable = human rating >= scale midpoint; arena benchmarks: human votes).
  (a) share of pairs by human type: one acceptable / both acceptable / both failed (humans prefer one or tie) / other
  (b) order flips of the forced choice: pairs with one acceptable candidate vs both-failed ties (both-bad votes)
  (c) agreement between two judges on the winner, same two pair types (0.5 = chance)
  (d) accuracy against the higher-rated candidate: one acceptable vs both failed (humans still prefer one)
usage: python fig_bothbad_signal.py --csv <csv> --out <dir>"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, BLUE, ORANGE, GREY, LIGHT = "#1a1a1a", "#52514e", "#e1e0d9", "#2a78d6", "#eb6834", "#a8a7a0", "#9dc0ec"
NAME = {"imagenhub/edit": "ImagenHub edit", "genai_arena/edition": "GenAI-Arena edit", "evalmuse": "EvalMuse",
        "genaibench": "GenAI-Bench", "imagenhub/t2i": "ImagenHub t2i", "genai_arena/generation": "GenAI-Arena gen.",
        "visionarena": "VisionArena", "mllm_judge_sp": "MLLM-J (scores)"}
ORDER = ["imagenhub/edit", "genai_arena/edition", "evalmuse", "genaibench", "imagenhub/t2i", "genai_arena/generation", "visionarena", "mllm_judge_sp"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    R = [r for r in csv.DictReader(open(a.csv)) if r["tau"] in ("mid", "vote")]
    D = {(r["dataset"], r["kind"]): r for r in R}
    f = lambda ds, k, c: float(D[(ds, k)][c]) if (ds, k) in D and D[(ds, k)][c] not in ("", "nan") else np.nan
    n = lambda ds, k: int(D[(ds, k)]["pairs"]) if (ds, k) in D else 0
    ds_list = [d for d in ORDER if any(k[0] == d for k in D)]
    y = np.arange(len(ds_list))[::-1]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw=dict(width_ratios=[1.45, 1, 1, 1], wspace=.16))
    # (a) composition
    for yi, d in zip(y, ds_list):
        if d in ("genai_arena/edition", "genai_arena/generation", "visionarena"):
            one, both_ok, failed, other = n(d, "decided"), n(d, "equally_good"), n(d, "both_bad"), 0
        else:
            one, both_ok = n(d, "dec_one_ok"), n(d, "dec_both_ok") + n(d, "tie_ok")
            failed, other = n(d, "dec_none_ok") + n(d, "tie_bad"), n(d, "tie_mixed")
        tot = max(one + both_ok + failed + other, 1); left = 0
        for v, c, lab in ((one, BLUE, "one acceptable / decided vote"), (both_ok, LIGHT, "both acceptable"),
                          (failed, ORANGE, "both failed"), (other, GREY, "other")):
            ax[0].barh(yi, v / tot, left=left, color=c, height=.62, edgecolor="#fbfbf8", linewidth=1.5,
                       label=lab if yi == y[0] else None)
            if c == ORANGE and v / tot >= .08:
                ax[0].text(left + v / tot / 2, yi, f"{v / tot:.0%}", ha="center", va="center", fontsize=9, color=INK)
            left += v / tot
    ax[0].set_yticks(y); ax[0].set_yticklabels([NAME[d] for d in ds_list]); ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("share of pairs")
    ax[0].set_title("(a) How often both candidates failed", fontsize=10.5, color=INK, loc="left")
    ax[0].legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, frameon=False, fontsize=8.5)

    def dumbbell(axi, col, k_good, k_bad, title, xlab, xlim, chance=None):
        for yi, d in zip(y, ds_list):
            vote = d in ("genai_arena/edition", "genai_arena/generation", "visionarena")
            g = f(d, "decided" if vote else k_good, col); b = f(d, ("both_bad" if vote else k_bad), col)
            if g == g and b == b:
                axi.plot([g, b], [yi, yi], color="#c3c2b7", lw=2, zorder=1)
            axi.scatter([g], [yi], s=46, color=BLUE, zorder=3, edgecolor="#fbfbf8", linewidth=1.2)
            axi.scatter([b], [yi], s=46, color=ORANGE, zorder=3, edgecolor="#fbfbf8", linewidth=1.2)
        if chance is not None:
            axi.axvline(chance, color=INK2, ls=(0, (3, 2)), lw=1)
        axi.set_yticks(y); axi.set_yticklabels([]); axi.set_xlim(*xlim); axi.set_xlabel(xlab)
        axi.set_title(title, fontsize=10.5, color=INK, loc="left"); axi.grid(axis="x", color=GRID); axi.set_axisbelow(True)

    dumbbell(ax[1], "flip", "dec_one_ok", "tie_bad", "(b) Flips with the order", "share of pairs", (0, .6))
    dumbbell(ax[2], "agree", "dec_one_ok", "tie_bad", "(c) Judges agree on a winner", "agreement (dashed: chance)", (.45, 1), .5)
    dumbbell(ax[3], "acc", "dec_one_ok", "dec_none_ok", "(d) Picks the higher-rated", "accuracy (dashed: chance)", (.45, 1), .5)
    ax[1].scatter([], [], s=46, color=BLUE, label="one acceptable (decided)"); ax[1].scatter([], [], s=46, color=ORANGE, label="both failed")
    ax[1].legend(loc="upper center", bbox_to_anchor=(1.1, -.2), ncol=2, frameon=False, fontsize=8.5)
    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}/fig_bothbad_signal.{ext}", dpi=170, bbox_inches="tight", facecolor="#fbfbf8")
    print("wrote", a.out)
