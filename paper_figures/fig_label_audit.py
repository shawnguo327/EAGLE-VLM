"""Review 22, figure 4, static version (DRAFT, 2026-11-08; not in the paper until the authors decide).
(a) Preference-label direction audit: the judge's forced-choice winner read as a training preference, stratified by the
    candidates' human acceptability (both acceptable / exactly one / neither; scale midpoint): share of labels whose preferred
    candidate has the higher recorded quality, equal quality (a human tie turned into a strict preference), or lower quality.
    Arrows mean 'the label ranks this candidate first', not a parameter update.  One judge (Qwen3-VL-8B), all saved pairs.
(b) Offline absolute-action utility on held-out sources at alpha .2 (scripts/absolute_actions.py; ship-or-reject, 4-class
    head, overall certificate; mean over five judges): per 100 test sources, acceptable ships, correct rejects (no acceptable
    candidate), wrong ships, wrong rejects and escalations; pair = relative prompts (4 calls), pw = single-candidate scores
    (2 calls), both (6 calls).  Correct rejects do not create an acceptable output.
(c) The same for ImagenHub edit with the null-edit anchor (scripts/anchor_nulledit.py; three judges, same protocol).
These are static audits; they report no policy training and no optimisation update.
usage (repo root): python paper_figures/fig_label_audit.py --out <dir> --atomic <bothbad_atomic_pairs.csv> --abs <absolute_actions_summary.csv>
                   --anchor <anchor_nulledit_splits.csv>"""
import argparse, collections, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from decision_utility import SCALE                                               # noqa: E402

INK, INK2, GRIDC, BG = "#1a1a1a", "#52514e", "#e1e0d9", "#fbfbf8"
C_OK, C_REJ, C_BADSHIP, C_BADREJ, C_ESC = "#2a78d6", "#9dc0ec", "#eb6834", "#f5b08f", "#d9d8d1"
C_CONS, C_TIE, C_REV = "#2a78d6", "#a8a7a0", "#eb6834"
NAME = {"imagenhub/edit": "ImagenHub edit", "genaibench": "GenAI-Bench", "evalmuse": "EvalMuse", "mllm_judge_sp": "MLLM-J (scores)",
        "imagenhub/t2i": "ImagenHub t2i", "imagerewarddb": "ImageRewardDB"}
ORDER = ["imagenhub/edit", "genaibench", "evalmuse", "mllm_judge_sp", "imagenhub/t2i", "imagerewarddb"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--out", "--atomic", "--abs", "--anchor"):
        ap.add_argument(k, required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    A = collections.defaultdict(list)
    for r in csv.DictReader(open(a.atomic)):
        A[r["dataset"]].append((float(r["q_A"]), float(r["q_B"]), r["chosen"]))
    audit = []
    for ds in ORDER:
        lo, hi = SCALE[ds.split("/")[0]]; tau = lo + .5 * (hi - lo)
        for qa, qb, ch in A[ds]:
            pref, oth = (qa, qb) if ch == "A" else (qb, qa)
            n_ok = (qa >= tau - 1e-9) + (qb >= tau - 1e-9)
            audit.append(dict(dataset=ds, stratum={2: "both acceptable", 1: "exactly one", 0: "neither"}[n_ok],
                              rel="higher" if pref > oth else "equal" if pref == oth else "lower"))
    with open(os.path.join(a.out, "label_audit.csv"), "w", newline="") as f:
        cnt = collections.Counter((x["dataset"], x["stratum"], x["rel"]) for x in audit); w = csv.writer(f)
        w.writerow(["dataset", "stratum", "preferred_higher", "preferred_equal", "preferred_lower", "pairs"])
        for ds in ORDER:
            for st in ("both acceptable", "exactly one", "neither"):
                n = sum(cnt[(ds, st, r)] for r in ("higher", "equal", "lower"))
                w.writerow([ds, st] + [cnt[(ds, st, r)] for r in ("higher", "equal", "lower")] + [n])
    S = {(r["dataset"], r["tau"], r["alpha"], r["policy"], r["cert"]): r for r in csv.DictReader(open(a.abs))}
    AN = collections.defaultdict(lambda: np.zeros(5))
    for r in csv.DictReader(open(a.anchor)):
        if r["tau"] == "mid" and r["alpha"] == "0.2" and r["head"] == "h4" and r["aset"] == "sr" and r["cert"] == "all":
            AN[(r["signal"], r["judge"])] += [float(r["ship_ok"]), float(r["rej_ok"]), float(r["ship_bad"]), float(r["rej_bad"]), float(r["sources"])]
    fig = plt.figure(figsize=(14, 8.2))
    top = fig.add_gridspec(1, 3, left=.11, right=.97, top=.95, bottom=.6, wspace=.1)
    bot = fig.add_gridspec(1, 3, left=.11, right=.97, top=.38, bottom=.1, wspace=.7)
    y = np.arange(len(ORDER))[::-1]
    for k, st in enumerate(("both acceptable", "exactly one", "neither")):
        ax = fig.add_subplot(top[0, k])
        for yi, ds in zip(y, ORDER):
            n = sum(cnt[(ds, st, r)] for r in ("higher", "equal", "lower"))
            if n == 0:
                continue
            left = 0
            for r, c in (("higher", C_CONS), ("equal", C_TIE), ("lower", C_REV)):
                v = cnt[(ds, st, r)] / n
                ax.barh(yi, v, left=left, color=c, height=.6, edgecolor=BG, linewidth=1.2)
                left += v
            ax.text(1.01, yi, f"{n:,}", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, 1.13); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks(y)
        ax.set_yticklabels([NAME[d] for d in ORDER] if k == 0 else [])
        ax.set_title(f"(a{k + 1}) {st}", fontsize=10, color=INK, loc="left"); ax.grid(axis="x", color=GRIDC); ax.set_axisbelow(True)
        ax.set_xlabel("share of labels")
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_CONS, C_TIE, C_REV)]
    fig.legend(hs, ["forced-choice winner has the higher recorded quality", "equal quality (a human tie made strict)",
                    "winner has the lower recorded quality"], loc="upper center", bbox_to_anchor=(.54, .515), ncol=3, frameon=False, fontsize=8.5)
    def stack(ax, rows, title):
        yy = np.arange(len(rows))[::-1]
        for yi, (lab, v) in zip(yy, rows):
            left = 0
            for val, c in zip(list(v) + [max(100 - sum(v), 0)], (C_OK, C_REJ, C_BADSHIP, C_BADREJ, C_ESC)):
                ax.barh(yi, val, left=left, color=c, height=.6, edgecolor=BG, linewidth=1.2); left += val
            ax.text(101, yi, f"{v[0] + v[1]:.0f} correct", va="center", fontsize=8, color=INK2)
        ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title, fontsize=10, color=INK, loc="left"); ax.grid(axis="x", color=GRIDC); ax.set_axisbelow(True)
        ax.set_xlabel("per 100 test sources")
    for k, ds in enumerate(("imagenhub/edit", "genaibench")):
        rows = []
        for sig, lab in (("pair", "relative (4 calls)"), ("pw", "pointwise (2)"), ("both", "both (6)")):
            r = S.get((ds, "mid", "0.2", f"{sig}|h4|sr", "all"))
            if r:
                rows.append((lab, [float(r["ship_ok"]), float(r["rej_ok"]), float(r["ship_bad"]), float(r["rej_bad"])]))
        stack(fig.add_subplot(bot[0, k]), rows, f"(b{k + 1}) {NAME[ds]} (5 judges)")
    rows = []
    for sig, lab in (("pair", "relative (4)"), ("anc", "anchor (1.1)"), ("pair+anc", "rel. + anchor (5.1)"), ("pw+anc", "pw + anchor (3.1)")):
        js = [v for (s_, j), v in AN.items() if s_ == sig]
        if js:
            rows.append((lab, list(np.mean([100 * v[:4] / v[4] for v in js], 0))))
    stack(fig.add_subplot(bot[0, 2]), rows, "(c) ImagenHub edit + null-edit anchor (3 judges)")
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_OK, C_REJ, C_BADSHIP, C_BADREJ, C_ESC)]
    fig.legend(hs, ["acceptable shipped", "correct reject (none acceptable)", "unacceptable shipped", "wrong reject", "escalated"],
               loc="lower center", bbox_to_anchor=(.54, -.005), ncol=5, frameon=False, fontsize=8.5)
    fig.text(.11, .435, "Ship-or-reject policies certified at alpha = .2 on held-out sources (scale-midpoint acceptability)",
             fontsize=9.5, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"fig4_label_audit_and_actions.{ext}"), dpi=170, bbox_inches="tight", facecolor=BG)
    print("wrote", a.out)
