"""Review 22, figures 1 and 2 (DRAFT, 2026-11-08; not in the paper until the authors decide).
Figure 1, quality plane: the human quality of the two candidates of each pair (q_A, q_B) for GenAI-Bench and ImagenHub edit,
with the predeclared acceptance threshold (scale midpoint) and the diagonal; count and share per region; inside 'both
unacceptable': the share with unequal recorded quality and the judge's accuracy at picking the higher-rated candidate there
(source-cluster bootstrap 95% interval).  Bins are aligned to the threshold (no jitter), so no point changes region.
Figure 2, forced-shipping failure: for the six rated benchmarks, the share of pairs whose forced-choice winner (one judge,
order-averaged A/B probability) is unacceptable, split into (i) no acceptable candidate: a lower bound for EVERY always-ship
policy, and (ii) an acceptable candidate existed but the unacceptable one was chosen; random choice and the always-ship
oracle as references; midpoint (main) and top-quarter thresholds; pair-uniform, source-cluster bootstrap 95% intervals.
Writes the atomic per-pair table (dataset, item_id, source_id, judge, q_A, q_B, human_relation, p_A_AB, p_A_BA, p_A_avg,
chosen) and the bootstrap summaries as CSV.  Descriptive (all saved outputs of the named judge), not a certification result.
usage (repo root): python paper_figures/fig_bothbad_planes.py --out <dir> [--judge qwen3vl8b] [--boot 2000]"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from eaglevlm.pipeline.run import load_cfg                                       # noqa: E402
from eaglevlm.pipeline.io import Store, dataset_items                            # noqa: E402
from eaglevlm.pipeline.certify import signals, source_key                       # noqa: E402
from decision_utility import ratings, SCALE                                      # noqa: E402

INK, INK2, GRIDC, BLUE, ORANGE, GREY, BG = "#1a1a1a", "#52514e", "#e1e0d9", "#2a78d6", "#eb6834", "#a8a7a0", "#fbfbf8"
CFG = {"imagenhub/edit": "newdata_certify_source", "genaibench": "certify_source", "evalmuse": "newdata_certify_source",
       "mllm_judge_sp": "certify_source", "imagenhub/t2i": "newdata_certify_source", "imagerewarddb": "certify_source"}
NAME = {"imagenhub/edit": "ImagenHub edit", "genaibench": "GenAI-Bench", "evalmuse": "EvalMuse", "mllm_judge_sp": "MLLM-J (scores)",
        "imagenhub/t2i": "ImagenHub t2i", "imagerewarddb": "ImageRewardDB"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})


def load(spec, judge):
    cfg = load_cfg(f"configs/exp/{CFG[spec]}.yaml"); P_ = cfg["paths"]
    st = Store(P_["items"], P_["outputs"], P_.get("items_override"), P_.get("snapshot")); ds, I = dataset_items(st, spec)
    S = signals(st, ds, judge, I); ids = sorted(S)
    R = np.array([ratings(spec, I[i]) for i in ids], float)
    f1 = np.array([S[i][0] for i in ids]); f2 = np.array([S[i][1] for i in ids]); p = (f1 + f2) / 2
    g = np.unique(np.array([source_key(I[i]) for i in ids]), return_inverse=True)[1]
    return ids, R, f1, f2, p, g, len(I)


def boot(g, fn, B, rng):
    """source-cluster bootstrap of a statistic fn(index array) -> float; returns the 2.5 / 97.5 percentiles"""
    ug = np.unique(g); members = [np.where(g == s)[0] for s in ug]; vals = []
    for _ in range(B):
        pick = rng.integers(0, len(ug), len(ug)); idx = np.concatenate([members[k] for k in pick]); v = fn(idx)
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > .9 * B else (np.nan, np.nan)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--judge", default="qwen3vl8b")
    ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0); D = {}; atom = []; summ = []
    for spec in CFG:
        ids, R, f1, f2, p, g, nI = load(spec, a.judge); D[spec] = (ids, R, p, g)
        ch = np.where(p > .5, 0, 1); rel = np.where(R[:, 0] > R[:, 1], "A", np.where(R[:, 1] > R[:, 0], "B", "tie"))
        for n, i in enumerate(ids):
            atom.append(dict(dataset=spec, item_id=i, source_id=int(g[n]), judge=a.judge, q_A=R[n, 0], q_B=R[n, 1], human_relation=rel[n],
                             p_A_AB=round(float(f1[n]), 6), p_A_BA=round(float(f2[n]), 6), p_A_avg=round(float(p[n]), 6), chosen="AB"[ch[n]]))
        lo, hi = SCALE[spec.split("/")[0]]
        for tname, frac in (("mid", .5), ("high", .75)):
            tau = lo + frac * (hi - lo); acc = R >= tau - 1e-9; bb = ~acc[:, 0] & ~acc[:, 1]; one = acc[:, 0] ^ acc[:, 1]
            chosen_bad = ~acc[np.arange(len(ids)), ch]; avoid = chosen_bad & ~bb
            stats = {"both_bad": lambda x: bb[x].mean(), "avoidable": lambda x: avoid[x].mean(), "forced_fail": lambda x: chosen_bad[x].mean(),
                     "random_fail": lambda x: (bb[x] + .5 * one[x]).mean()}
            row = dict(dataset=spec, judge=a.judge, tau=tname, tau_value=tau, pairs=len(ids), items=nI, sources=int(g.max() + 1))
            for k, fn in stats.items():
                row[k] = float(fn(np.arange(len(ids)))); row[k + "_lo"], row[k + "_hi"] = boot(g, fn, a.boot, rng)
            unequal = bb & (R[:, 0] != R[:, 1]); hit = (ch == np.where(R[:, 0] > R[:, 1], 0, 1))
            fn_acc = lambda x: hit[x][unequal[x]].mean() if unequal[x].sum() else np.nan
            row.update(bb_unequal_share=float(unequal.sum() / max(bb.sum(), 1)), bb_unequal_n=int(unequal.sum()),
                       bb_unequal_sources=int(len(np.unique(g[unequal]))), bb_pick_higher=float(fn_acc(np.arange(len(ids)))))
            row["bb_pick_higher_lo"], row["bb_pick_higher_hi"] = boot(g, fn_acc, a.boot, rng)
            summ.append(row); print(row, flush=True)
    for fn_, rows in (("bothbad_atomic_pairs.csv", atom), ("bothbad_decomposition.csv", summ)):
        with open(os.path.join(a.out, fn_), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    S = {(r["dataset"], r["tau"]): r for r in summ}
    # ---- Figure 1: quality planes
    cmap = LinearSegmentedColormap.from_list("seq", ["#fbfbf8", "#cfe0f6", "#7fb0ea", "#2a78d6", "#123f78"])
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw=dict(wspace=.32))
    for ax, spec in zip(axs, ("genaibench", "imagenhub/edit")):
        ids, R, p, g = D[spec]; lo, hi = SCALE[spec.split("/")[0]]; tau = lo + .5 * (hi - lo)
        w = (hi - lo) / 12 if spec == "genaibench" else .1
        edges = np.arange(lo, hi + w / 2, w)
        Hc, _, _ = np.histogram2d(R[:, 0], R[:, 1], bins=[edges, edges])
        ax.imshow(np.log1p(Hc.T), origin="lower", extent=[lo, hi, lo, hi], cmap=cmap, aspect="equal", interpolation="nearest")
        ax.axvline(tau, color=INK2, lw=1, ls=(0, (4, 2))); ax.axhline(tau, color=INK2, lw=1, ls=(0, (4, 2)))
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=1)
        acc = R >= tau - 1e-9; n = len(R)
        for (xa, ya), mask, lab in (((tau + (hi - tau) / 2, tau + (hi - tau) / 2), acc[:, 0] & acc[:, 1], "both acceptable"),
                                    ((lo + (tau - lo) / 2, lo + (tau - lo) / 2), ~acc[:, 0] & ~acc[:, 1], "both unacceptable"),
                                    ((tau + (hi - tau) / 2, lo + (tau - lo) / 2), acc[:, 0] & ~acc[:, 1], "only A"),
                                    ((lo + (tau - lo) / 2, tau + (hi - tau) / 2), ~acc[:, 0] & acc[:, 1], "only B")):
            ax.text(xa, ya, f"{lab}\n{mask.sum():,} ({mask.mean():.0%})", ha="center", va="center", fontsize=8.5, color=INK,
                    bbox=dict(boxstyle="round,pad=.25", fc="white", ec="none", alpha=.85))
        r = S[(spec, "mid")]
        ax.set_title(f"{NAME[spec]} ({n:,} pairs, {r['sources']:,} sources)", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("human quality of candidate A"); ax.set_ylabel("human quality of candidate B")
        ci = "" if r["bb_pick_higher_lo"] != r["bb_pick_higher_lo"] else f" [{r['bb_pick_higher_lo']:.2f}, {r['bb_pick_higher_hi']:.2f}]"
        ax.text(0, -.25, f"Inside both-unacceptable: {r['bb_unequal_share']:.0%} have unequal recorded quality ({r['bb_unequal_n']:,} pairs);\n"
                         f"{a.judge} picks the higher-rated one in {r['bb_pick_higher']:.2f}{ci} of them.", transform=ax.transAxes,
                fontsize=8.5, color=INK2, va="top")
    fig.savefig(os.path.join(a.out, "fig1_quality_plane.png"), dpi=170, bbox_inches="tight", facecolor=BG)
    fig.savefig(os.path.join(a.out, "fig1_quality_plane.pdf"), bbox_inches="tight", facecolor=BG)
    # ---- Figure 2: forced-shipping failure decomposition
    order = ["imagenhub/edit", "genaibench", "evalmuse", "mllm_judge_sp", "imagenhub/t2i", "imagerewarddb"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True, gridspec_kw=dict(wspace=.08))
    y = np.arange(len(order))[::-1]
    for ax, t in zip(axs, ("mid", "high")):
        for yi, spec in zip(y, order):
            r = S[(spec, t)]
            ax.barh(yi, r["both_bad"], color=ORANGE, height=.58, edgecolor=BG, linewidth=1.5, label="no acceptable candidate" if yi == y[0] else None)
            ax.barh(yi, r["avoidable"], left=r["both_bad"], color=BLUE, height=.58, edgecolor=BG, linewidth=1.5,
                    label="acceptable one existed, unacceptable chosen" if yi == y[0] else None)
            ax.errorbar(r["forced_fail"], yi, xerr=[[r["forced_fail"] - r["forced_fail_lo"]], [r["forced_fail_hi"] - r["forced_fail"]]],
                        fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
            ax.scatter([r["random_fail"]], [yi], marker="o", s=34, facecolor="white", edgecolor=INK, linewidth=1.1, zorder=4,
                       label="random choice" if yi == y[0] else None)
            ax.text(max(r["forced_fail_hi"], r["random_fail"]) + .03, yi, f"{r['forced_fail']:.0%}", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, 1.1); ax.set_xticks([0, .2, .4, .6, .8, 1.0]); ax.set_yticks(y); ax.set_yticklabels([NAME[s] for s in order]); ax.grid(axis="x", color=GRIDC); ax.set_axisbelow(True)
        ax.set_xlabel("share of pairs whose forced-choice winner is unacceptable")
        ax.set_title(f"({'a' if t == 'mid' else 'b'}) threshold: {'scale midpoint' if t == 'mid' else 'top quarter'}", fontsize=10.5, color=INK, loc="left")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.04, -.2), ncol=3, frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(a.out, "fig2_forced_ship_failure.png"), dpi=170, bbox_inches="tight", facecolor=BG)
    fig.savefig(os.path.join(a.out, "fig2_forced_ship_failure.pdf"), bbox_inches="tight", facecolor=BG)
    print("wrote", a.out)
