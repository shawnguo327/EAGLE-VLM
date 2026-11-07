"""Figures for 'Should Judges Drop Ties?' (CVPR format: column 3.25 in, page 6.875 in).  Vector PDF + PNG preview.

Every number is read from the results folders (results/<experiment>/latest/summary.csv), the raw judge outputs (through
eaglevlm.pipeline.io.Store, the same loader as the analyses) or the saved SCOPE simulation; nothing is typed in by hand
except the worked example of the overview figure (the one in Sec. 6).

    python paper_figures/make_paper_figures.py --out /media/shawn/C/papers/VLM/overleaf/figs_new [--only overview,problem]

Palette (validated with the dataviz validator, light surface): methods blue #2a78d6 = graded (ours), orange #eb6834 =
SCOPE, aqua #1baf7a = SCOPE + ties, gray + hollow diamond = ToE-style (secondary encoding); claim types an ordinal blue
ramp (strict #184f95 > tie #3987e5 > weak #86b6ef) and a neutral gray for 'not a tie' {A,B}.  Text stays in ink tokens.
"""
import argparse, collections, csv, json, math, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = "/media/shawn/C/papers/VLM/results"
DATA = "/media/shawn/C/papers/VLM/judgebench_local"

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STRICT, TIE, WEAK, NONDIR = "#184f95", "#3987e5", "#86b6ef", "#d6d4cd"
BLUE_WASH, ORANGE_WASH, GRAY_WASH = "#eaf2fc", "#fdeee7", "#f4f3f0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6.2, "legend.fontsize": 6, "axes.edgecolor": AXIS,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "text.color": INK, "xtick.major.width": 0.5, "ytick.major.width": 0, "xtick.major.size": 2,
                     "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 4})

J6 = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b", "internvl35_38b"]
JN = {"qwen3vl8b": "Qwen3-VL-8B", "qwen35_9b": "Qwen3.5-9B", "internvl35_8b": "InternVL3.5-8B"}
BENCH = [("genai_arena/edition", "GenAI-Arena edit"), ("mtbench_human", "MT-Bench (text)"), ("arena55k", "Chatbot Arena (text)"),
         ("mllm_judge_sp", "MLLM-J scores"), ("genai_arena/generation", "GenAI-Arena gen."), ("genaibench", "GenAI-Bench"),
         ("mllm_judge", "MLLM-J pairs"), ("pickapic_u", "Pick-a-Pic"), ("imagerewarddb", "ImageRewardDB")]
ALPHA = 0.2


def latest(name):
    return os.path.realpath(os.path.join(RES, name, "latest"))


def summary(name, key=("dataset", "judge", "method")):
    out = {}
    for r in csv.DictReader(open(os.path.join(latest(name), "summary.csv"))):
        if abs(float(r["alpha"]) - ALPHA) < 1e-9:
            out[tuple(r[k] for k in key)] = r
    return out


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def jmean(S, ds, method, col, judges=J6):
    v = [fnum(S[(ds, j, method)][col]) for j in judges if (ds, j, method) in S]
    v = [x for x in v if x is not None]
    return (sum(v) / len(v)) if v else None


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig); print("wrote", name)


def hgrid(ax):
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)


# --------------------------------------------------------------------------------------------- Fig. 2: the problem
def fig_problem(out):
    S = summary("certify_source")
    M = [("P1", "SCOPE (as published)", ORANGE, "o", True), ("T1", "ToE-style (zero-shot)", MUTED, "D", False),
         ("P2", "SCOPE + ties", AQUA, "o", True), ("G2", "Graded (ours)", BLUE, "o", True)]
    tie = {ds: fnum(next(r for k, r in S.items() if k[0] == ds)["human_tie"]) for ds, _ in BENCH}
    fig, axs = plt.subplots(1, 3, figsize=(3.35, 2.3), sharey=True, gridspec_kw=dict(width_ratios=[0.7, 1.35, 1.25], wspace=0.16))
    y = np.arange(len(BENCH))[::-1]
    ax = axs[0]
    ax.barh(y, [tie[ds] for ds, _ in BENCH], height=0.5, color=INK2, lw=0)
    for yi, (ds, _) in zip(y, BENCH):
        ax.text(tie[ds] + 0.015, yi, f"{100 * tie[ds]:.0f}%", va="center", ha="left", fontsize=5.6, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in BENCH]); ax.set_xlim(0, 0.62); ax.set_xticks([0, 0.25, 0.5])
    ax.set_xticklabels(["0", "0.25", "0.5"]); ax.set_title("(a) Ties"); hgrid(ax)
    for k, (col, title, lim, ticks) in enumerate([("pooled_risk_source", "(b) Test risk", (0, 0.6), [0, 0.2, 0.4]),
                                                  ("source_coverage", "(c) Coverage", (-0.03, 1.03), [0, 0.5, 1])]):
        ax = axs[k + 1]
        if col == "pooled_risk_source":
            ax.axvspan(0, ALPHA, color=BLUE_WASH, lw=0, zorder=0)
            ax.axvline(ALPHA, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
            ax.text(ALPHA + 0.012, -0.55, r"$\alpha$", ha="left", va="bottom", fontsize=6.5, color=INK2)
        for code, lab, c, mk, filled in M:
            xs, ys = [], []
            for yi, (ds, _) in zip(y, BENCH):
                v = jmean(S, ds, code, col)
                if v is not None:
                    xs.append(v); ys.append(yi)
            ax.scatter(xs, ys, s=16 if filled else 11, marker=mk, facecolor=c if filled else SURF, edgecolor=SURF if filled else c,
                       linewidth=0.8 if filled else 1.0, zorder=3 + (code == "G2"), clip_on=False)
        ax.set_xlim(*lim); ax.set_xticks(ticks); ax.set_xticklabels(["%g" % t for t in ticks])
        ax.set_title(title); hgrid(ax); ax.set_ylim(-0.6, len(BENCH) - 0.4)
    handles = [Line2D([], [], ls="", marker=mk, ms=4.2 if f else 3.4, mfc=c if f else SURF, mec=SURF if f else c, mew=0.8, label=lab)
               for _, lab, c, mk, f in M]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.56, -0.11), ncol=2, frameon=False, handletextpad=0.2,
               columnspacing=0.9)
    save(fig, out, "fig_problem")


# ----------------------------------------------------------------------------- Fig. 3: where the coverage comes from
def fig_claims(out):
    S = summary("certify_source")
    order = sorted(BENCH, key=lambda b: -(jmean(S, b[0], "SA", "acceptance") or 0))
    parts = [("decisive", "strict  $\\{A\\}$", STRICT), ("tie_claim_share", "tie  $\\{T\\}$", TIE),
             ("weak_share", "weak  $\\{A,T\\}$", WEAK), ("nondirectional_share", "not a tie  $\\{A,B\\}$", NONDIR)]
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    yt, yl = [], []
    for g, (ds, name) in enumerate(order):
        base = -g * 1.0
        for k, (code, lab) in enumerate([("G2", "graded"), ("SA", "APS sets")]):
            yy = base - 0.36 * k; left = 0
            for col, _, c in parts:
                w = jmean(S, ds, code, col) or 0
                if w > 0:
                    ax.barh(yy, w, left=left, height=0.3, color=c, edgecolor=SURF, linewidth=0.6)
                left += w
            if code == "SA":
                o = jmean(S, ds, "SAo", "acceptance")
                ax.scatter([o], [yy], marker="D", s=11, facecolor=INK, edgecolor=SURF, linewidth=0.5, zorder=4)
            ax.text(-0.015, yy, lab, ha="right", va="center", fontsize=5.4, color=MUTED)
        yt.append(base - 0.18); yl.append(name)
    ax.set_yticks(yt); ax.set_yticklabels(yl); ax.tick_params(axis="y", pad=34)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("share of test pairs answered, by claim type"); hgrid(ax)
    handles = [Rectangle((0, 0), 1, 1, color=c, lw=0, label=lab) for _, lab, c in parts]
    handles.append(Line2D([], [], ls="", marker="D", ms=3.4, mfc=INK, mec=SURF, label="APS without $\\{A,B\\}$, recalibrated"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, frameon=False, handlelength=1.1,
              handletextpad=0.35, columnspacing=0.8, fontsize=5.6)
    save(fig, out, "fig_claims")


# ------------------------------------------------------------------------------------ Fig. 4: visual dependence
def auroc(score, pos):
    score = np.asarray(score, float); pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort").astype(float) + 1
    s, inv = np.unique(score, return_inverse=True)                        # average ranks for ties of the score
    ranks = np.zeros(len(s)); np.add.at(ranks, inv, r); cnt = np.bincount(inv); r = (ranks / cnt)[inv]
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def visual_stats(judges=("qwen3vl8b", "qwen35_9b"), ds="mllm_judge_sp"):
    from eaglevlm.pipeline.io import Store, dataset_items
    from eaglevlm.pipeline.certify import signals
    import yaml
    P = yaml.safe_load(open(os.path.join(ROOT, "configs", "paths.bjserver.yaml")))["paths"]
    st = Store(P["items"], P["outputs"]); _, I = dataset_items(st, ds)
    conds = {"image": None, "no image": {"bin": "bin_blind", "tri": "tri_blind"}, "wrong image": {"bin": "bin_shuf", "tri": "tri_shuf"}}
    out = {}
    for j in judges:
        sig = {c: signals(st, ds, j, I, m) for c, m in conds.items()}
        ids = sorted(set.intersection(*[set(v) for v in sig.values()]))
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids]); dec = y < 2
        for c in conds:
            p = np.array([(sig[c][i][0] + sig[c][i][1]) / 2 for i in ids])
            q = np.array([(sig[c][i][2] + sig[c][i][3]) / 2 for i in ids])
            correct = (np.where(p > .5, 0, 1) == y)
            out[(j, c)] = dict(n=len(ids), acc=float(correct[dec].mean()),
                               err_auroc=auroc(np.abs(p - .5)[dec], correct[dec]), tie_auroc=auroc(q[:, 2], y == 2))
    return out


def fig_visual(out):
    judges = ("qwen3vl8b", "qwen35_9b")
    V = visual_stats(judges)
    conds = [("image", "certify_source", "ablate_source"), ("no image", "certify_blind", "ablate_blind"),
             ("wrong image", "certify_shufimg", "ablate_shufimg")]
    for cond, crun, arun in conds:                       # certified coverage (graded) and the calibrated p(T)'s tie AUROC
        C, A = summary(crun), summary(arun)
        for j in judges:
            V[(j, cond)]["coverage"] = fnum(C[("mllm_judge_sp", j, "G2")]["source_coverage"])
            V[(j, cond)]["tie_auroc_cal"] = fnum(A[("mllm_judge_sp", j, "F02")]["tie_auroc"])
    json.dump({f"{j}|{c}": v for (j, c), v in V.items()}, open(os.path.join(out, "fig_visual.json"), "w"), indent=1)
    CC = {"image": BLUE, "no image": MUTED, "wrong image": ORANGE}; MK = {"qwen3vl8b": "o", "qwen35_9b": "s"}
    panels = [("acc", "(a) Accuracy", (0.45, 0.72), [0.5, 0.6, 0.7]),
              ("err_auroc", "(b) Error AUROC", (0.45, 0.72), [0.5, 0.6, 0.7]),
              ("tie_auroc_cal", "(c) Tie AUROC", (0.45, 0.72), [0.5, 0.6, 0.7]),
              ("coverage", "(d) Certified coverage", (-0.02, 0.45), [0, 0.2, 0.4])]
    fig, axs = plt.subplots(2, 2, figsize=(3.35, 2.25), gridspec_kw=dict(wspace=0.3, hspace=0.75))
    for ax, (key, title, lim, ticks) in zip(axs.flat, panels):
        if key != "coverage":
            ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.text(-0.3, 0.495, "chance", fontsize=5, color=INK2, ha="left", va="top")
        for j in judges:
            ys = [V[(j, c)][key] for c, *_ in conds]
            ax.plot(range(3), ys, color=AXIS, lw=0.9, zorder=2)
            for x, (c, *_) in enumerate(conds):
                ax.scatter([x], [V[(j, c)][key]], marker=MK[j], s=15, facecolor=CC[c], edgecolor=SURF, linewidth=0.6, zorder=3)
        ax.set_xlim(-0.35, 2.35); ax.set_xticks(range(3)); ax.set_xticklabels([c.replace(" ", "\n") for c, *_ in conds], fontsize=5.6, linespacing=0.95)
        ax.set_ylim(*lim); ax.set_yticks(ticks); ax.set_yticklabels(["%g" % t for t in ticks])
        ax.set_title(title, fontsize=6.3); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelsize=5.6)
    handles = [Line2D([], [], ls="-", color=AXIS, marker=MK[j], ms=3.6, mfc=INK2, mec=SURF, label=JN[j]) for j in judges]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, handlelength=1.6)
    save(fig, out, "fig_visual")


# -------------------------------------------------------------------- Supplement: SCOPE threshold without a safe one
def fig_scope(out):
    p = os.path.join(RES, "scope_counterexample", "output_20261107_se.txt")
    rows = []
    for line in open(p):
        m = re.match(r"(.*?)\s+runs accepting ([\d.]+)\s+ratio of expectations ([\d.]+) \(s\.e\. ([\d.]+)\)", line.strip())
        if m:
            rows.append((m.group(1).strip(), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    lab = {"infeasible: P(err)=.30 for every score": "no safe threshold: error .30 everywhere",
           "infeasible: P(err)=.25 for every score": "no safe threshold: error .25 everywhere",
           "tie floor: P(err)=.25+.5*s (min risk .25 > alpha)": "tie-like floor: error .25 + .5s",
           "feasible, monotone: P(err)=.4*s (risk below alpha at low s)": "safe threshold: error .4s",
           "feasible, monotone: P(err)=.05+.6*s": "safe threshold: error .05 + .6s",
           "feasible, small n=100: P(err)=.4*s": "safe threshold: error .4s, n = 100",
           "feasible, monotone: P(err)=.05+.6*s, 100,000 draws": "safe threshold: error .05 + .6s (100k draws)"}
    fig, ax = plt.subplots(figsize=(3.35, 1.7))
    y = np.arange(len(rows))[::-1]
    ax.axvline(ALPHA, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.text(ALPHA + 0.002, -0.55, r"$\alpha$", ha="left", va="bottom", fontsize=6.5, color=INK2); ax.set_ylim(-0.6, len(rows) - 0.4)
    for yi, (name, acc, r, se) in zip(y, rows):
        c = ORANGE if r > ALPHA + 2 * se else BLUE
        ax.errorbar(r, yi, xerr=2 * se, fmt="o", ms=3.6, color=c, mec=SURF, mew=0.6, elinewidth=0.8, capsize=0)
        ax.text(1.0, yi, f"answers in {100 * acc:.0f}%", va="center", ha="right", fontsize=5.4, color=MUTED,
                transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
    ax.set_yticks(y); ax.set_yticklabels([lab.get(n, n) for n, *_ in rows], fontsize=5.6)
    ax.set_xlim(0.17, 0.36); ax.set_xticks([0.2, 0.25, 0.3])
    ax.set_xticklabels(["0.20", "0.25", "0.30"]); ax.set_xlabel("ratio of expectations  E[error $\\cdot$ answer] / E[answer]  ($\\pm$2 s.e.)")
    hgrid(ax)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.6, mfc=ORANGE, mec=SURF, label="above $\\alpha$ by > 2 s.e."),
                       Line2D([], [], ls="", marker="o", ms=3.6, mfc=BLUE, mec=SURF, label="at or below $\\alpha$")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.2)
    save(fig, out, "fig_scope_counterexample")


# ------------------------------------------------------------------------------ Supplement: router same-target controls
def fig_router(out):
    S = summary("router_controls", key=("dataset", "judge", "variant", "rule"))
    V = [("pairwise graded (G2)", "graded only"), ("control: routed, no pointwise", "control (base rate)"),
         ("R1: routed, pairwise-only level head", "pairwise-only head"), ("router: logistic, random gate", "random half routed"),
         ("router: cells", "router: tie map"), ("router: logistic", "router: logistic"),
         ("router: logistic, all routed", "all pairs routed"), ("R2: routed, unified augmented head", "unified head"),
         ("router: jury pointwise", "jury pointwise")]
    fig, axs = plt.subplots(1, 2, figsize=(3.35, 1.75), sharey=True, gridspec_kw=dict(wspace=0.1))
    y = np.arange(len(V))[::-1]
    for ax, (ds, title) in zip(axs, [("genaibench", "(a) GenAI-Bench"), ("mllm_judge_sp", "(b) MLLM-J scores")]):
        for yi, (v, _) in zip(y, V):
            cv = [fnum(S[(ds, j, v, "overall")]["source_coverage"]) for j in J6 if (ds, j, v, "overall") in S]
            ab = [fnum(S[(ds, j, v, "overall+absolute")]["level_correct_share"]) for j in J6 if (ds, j, v, "overall+absolute") in S]
            c = BLUE if v in ("router: cells", "router: logistic") else MUTED
            ax.barh(yi, np.mean(cv), height=0.55, color=c, lw=0)
            ax.text(np.mean(cv) + 0.015, yi, f"{np.mean(cv):.2f}", va="center", fontsize=5.2, color=INK2)
            if ab:
                ax.scatter([np.mean(ab)], [yi], marker="|", s=26, color=INK, linewidth=1.0, zorder=4)
        ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"]); ax.set_title(title)
        hgrid(ax)
    axs[0].set_yticks(y); axs[0].set_yticklabels([lab for _, lab in V], fontsize=5.6)
    fig.text(0.56, -0.09, "bars: source-uniform coverage under the overall rule (blue: the routers of the main paper)\n"
             "|: share of test pairs with a correct absolute-level claim when those claims are certified on their own",
             ha="center", fontsize=5.3, color=INK2, linespacing=1.3)
    save(fig, out, "fig_router_controls")


# ------------------------------------------------------------------------------------------ Fig. 1: the overview
def _square(path, side=420):
    from PIL import Image
    im = Image.open(path).convert("RGB"); w, h = im.size; m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((side, side), Image.LANCZOS)
    return np.asarray(im)


def fig_overview(out):
    """Workflow: (1) human-labelled pairs with ties (real examples), (2) judge signals -> calibrated graded claims (the
    worked example of Sec. 6), (3) certification with and without ties, and the controls of the audit."""
    items = {}
    for f in ("genai_arena_full.jsonl", "mllm_judge_sp.jsonl"):
        for line in open(os.path.join(DATA, "items", f)):
            r = json.loads(line)
            if r["id"] in ("genai_arena/generation_3052", "genai_arena/edition_0366", "mllm_judge_sp/3985_5891_5892"):
                items[r["id"]] = r
    t2i, edit, vqa = items["genai_arena/generation_3052"], items["genai_arena/edition_0366"], items["mllm_judge_sp/3985_5891_5892"]
    assert str(t2i["label"]) == "tie" and t2i["meta"]["vote_type"] == "tievote"
    assert str(edit["label"]) == "tie" and edit["meta"]["vote_type"] == "bothbad_vote"
    assert str(vqa["label"]) == "tie"
    I = lambda rel: _square(os.path.join(DATA, rel))
    S = summary("certify_source")
    rng = lambda m, col: (min(jmean(S, d, m, col) for d, _ in BENCH if jmean(S, d, m, col) is not None),
                          max(jmean(S, d, m, col) for d, _ in BENCH if jmean(S, d, m, col) is not None))
    pub = [jmean(S, d, m, "risk_source") for d, _ in BENCH for m in ("P1", "T1") if jmean(S, d, m, "risk_source") is not None]
    ours, cov = rng("G2", "risk_source"), rng("G2", "source_coverage")
    f2 = lambda v: f"{v:.2f}".lstrip("0")

    W, H = 6.875, 3.02
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.set_aspect("equal")

    def box(x, y, w, h, fc, ec="none", lw=0.6, r=0.05, z=1, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z,
                                    linestyle=ls))

    def T(x, y, s, size=6.0, c=INK, ha="left", va="center", w="normal", st="normal", z=6, **kw):
        return ax.text(x, y, s, fontsize=size, color=c, ha=ha, va=va, fontweight=w, fontstyle=st, zorder=z, **kw)

    def pic(arr, x, y, s, tag=None, z=4):
        ax.imshow(arr, extent=[x, x + s, y, y + s], zorder=z, interpolation="lanczos")
        if tag:
            T(x + 0.025, y + s - 0.025, tag, size=5.2, c=INK, ha="left", va="top", w="bold", z=z + 1,
              bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none", alpha=0.85))

    def arrow(x1, y1, x2, y2, c=INK2, lw=0.8, ms=6, z=6, cs="arc3"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c, zorder=z,
                                     connectionstyle=cs, shrinkA=0, shrinkB=0))

    def chip(x, y, s, fc=INK2, tc=SURF, size=5.4, ha="left"):
        T(x, y, s, size=size, c=tc, ha=ha, z=8, bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.35", fc=fc, ec="none"))

    def header(x, y, n, s, c):
        ax.add_patch(Circle((x + 0.075, y), 0.072, fc=c, ec="none", zorder=7))
        T(x + 0.075, y - 0.003, n, size=6.2, c=SURF, ha="center", w="bold", z=8)
        T(x + 0.19, y, s, size=7.0, w="bold")

    def judge_icon(x, y, s):                      # a small VLM 'robot' head
        box(x, y, s, s * 0.8, fc=INK2, r=0.05, z=5)
        for dx in (0.3, 0.7):
            ax.add_patch(Circle((x + dx * s, y + 0.45 * s), 0.09 * s, fc=SURF, ec="none", zorder=6))
        ax.add_patch(Rectangle((x + 0.33 * s, y + 0.16 * s), 0.34 * s, 0.05 * s, fc=SURF, ec="none", zorder=6))
        ax.plot([x + s / 2, x + s / 2], [y + 0.8 * s, y + 0.98 * s], color=INK2, lw=0.9, zorder=5)
        ax.add_patch(Circle((x + s / 2, y + s), 0.06 * s, fc=INK2, ec="none", zorder=5))

    def shield(x, y, s, c):
        pts = [(x, y + 0.78 * s), (x + 0.5 * s, y + s), (x + s, y + 0.78 * s), (x + s, y + 0.42 * s), (x + 0.5 * s, y),
               (x, y + 0.42 * s)]
        ax.add_patch(Polygon(pts, closed=True, fc=c, ec="none", zorder=6))
        ax.plot([x + 0.27 * s, x + 0.45 * s, x + 0.76 * s], [y + 0.52 * s, y + 0.33 * s, y + 0.72 * s], color=SURF, lw=1.1,
                zorder=7, solid_capstyle="round")

    def bubble(x, y, s):
        box(x, y + 0.18 * s, s, 0.62 * s, fc=INK2, r=0.04, z=5)
        ax.add_patch(Polygon([(x + 0.2 * s, y + 0.2 * s), (x + 0.42 * s, y + 0.2 * s), (x + 0.18 * s, y)], fc=INK2, ec="none", zorder=5))
        for dx in (0.3, 0.5, 0.7):
            ax.add_patch(Circle((x + dx * s, y + 0.49 * s), 0.055 * s, fc=SURF, ec="none", zorder=6))

    R = fig.canvas.get_renderer()

    def right(t, gap=0.035):                       # data x just right of a drawn text
        bb = t.get_window_extent(renderer=R)
        return ax.transData.inverted().transform((bb.x1, bb.y0))[0] + gap

    # ---------------------------------------------------------------- stage 1: pairs labelled by humans (real examples)
    X0, CW = 0.03, 2.02
    header(X0, H - 0.12, "1", "Pairs labelled by humans", INK2)
    cards = [(2.03, "Text-to-image", "GenAI-Arena, GenAI-Bench, Pick-a-Pic, ImageRewardDB", "tie: both good"),
             (1.25, "Image editing", "GenAI-Arena", "tie: both bad"),
             (0.47, "Visual question answering", "MLLM-as-a-Judge (pairs, scores)", "tie: same score")]
    for y, task, src, lab in cards:
        box(X0, y, CW, 0.72, fc=GRAY_WASH, r=0.05)
        T(X0 + 0.06, y + 0.645, task, size=6.0, w="bold")
        T(X0 + 0.06, y + 0.545, src, size=4.9, c=MUTED)
        chip(X0 + CW - 0.06, y + 0.645, lab, ha="right")
    y = 2.03
    pic(I(t2i["cand"][0]), X0 + 0.06, y + 0.04, 0.44, "A"); pic(I(t2i["cand"][1]), X0 + 0.54, y + 0.04, 0.44, "B")
    T(X0 + 1.06, y + 0.26, "“A fisheye lens view\nof a turtle sitting\nin a forest.”", size=5.5, c=INK2, st="italic",
      linespacing=1.15)
    y = 1.25
    pic(I(edit["context"][0]), X0 + 0.06, y + 0.04, 0.40, "src")
    arrow(X0 + 0.485, y + 0.24, X0 + 0.555, y + 0.24, ms=4, lw=0.6)
    pic(I(edit["cand"][0]), X0 + 0.58, y + 0.04, 0.40, "A"); pic(I(edit["cand"][1]), X0 + 1.01, y + 0.04, 0.40, "B")
    T(X0 + 1.47, y + 0.24, "“put the zebras\nnext to a river”", size=5.2, c=INK2, st="italic", linespacing=1.15)
    y = 0.47
    pic(I(vqa["context"][0]), X0 + 0.06, y + 0.04, 0.44)
    T(X0 + 0.58, y + 0.42, "Q: How many birds are there?", size=5.4, c=INK, w="bold")
    T(X0 + 0.58, y + 0.285, "A: There are six birds in the image.", size=5.1, c=INK2)
    T(X0 + 0.58, y + 0.175, "B: There are six birds in the image,", size=5.1, c=INK2)
    T(X0 + 0.58, y + 0.075, "     including two bluebirds, two …", size=5.1, c=INK2)
    bubble(X0 + 0.06, 0.07, 0.24)
    T(X0 + 0.36, 0.26, "+ text controls: MT-Bench, Chatbot Arena", size=5.2, c=INK2)
    T(X0 + 0.36, 0.125, "9 benchmarks, 11–44% human ties", size=5.7, w="bold", c=INK)

    # ---------------------------------------------------------------- stage 2: judge signals -> graded claims
    X1 = 2.21
    header(X1, H - 0.12, "2", "Judge signals → graded claims", BLUE)
    arrow(X0 + CW + 0.015, 2.47, X1 + 0.06, 2.47, c=MUTED, lw=1.0, ms=7)
    judge_icon(X1 + 0.07, 2.36, 0.28)
    T(X1 + 0.21, 2.25, "VLM judge", size=5.6, ha="center", w="bold")
    T(X1 + 0.21, 2.15, "6 open models", size=4.8, ha="center", c=MUTED)
    T(X1 + 0.21, 2.075, "7B–38B", size=4.8, ha="center", c=MUTED)
    prompts = [(2.63, "“Which is better, A or B?”", "$p_A$"),
               (2.41, "“A, B, or a tie?”", "$q_A,\\,q_B,\\,q_T$"),
               (2.19, "“Rate this answer, 1–9”", "$s_A,\\,s_B$")]
    for yy, ptxt, sig in prompts:
        box(X1 + 0.47, yy - 0.08, 1.16, 0.16, fc=SURF, ec=AXIS, lw=0.6, r=0.06, z=4)
        T(X1 + 0.52, yy, ptxt, size=5.3)
        arrow(X1 + 0.40, 2.47, X1 + 0.465, yy, c=AXIS, lw=0.6, ms=3.5)
        arrow(X1 + 1.645, yy, X1 + 1.74, yy, c=AXIS, lw=0.6, ms=3.5)
        T(X1 + 1.76, yy, sig, size=5.8)
    T(X1 + 0.47, 2.075, "pairwise prompts: both answer orders", size=4.6, c=MUTED)
    T(X1 + 0.47, 2.005, "pointwise prompt: uncertain pairs only", size=4.6, c=MUTED)
    # calibrator
    box(X1 + 0.05, 1.45, 2.27, 0.50, fc=BLUE_WASH, r=0.05, z=2)
    T(X1 + 0.11, 1.86, "Calibrator", size=5.8, w="bold")
    T(X1 + 0.11, 1.745, "logistic, fit split", size=4.8, c=MUTED)
    T(X1 + 0.11, 1.64, "(an example pair)", size=4.8, c=MUTED)
    for k, (lab, v) in enumerate([("$\\hat p(A)$", .55), ("$\\hat p(T)$", .40), ("$\\hat p(B)$", .05)]):
        yy = 1.855 - 0.14 * k
        T(X1 + 1.12, yy, lab, size=5.6, ha="right")
        ax.add_patch(Rectangle((X1 + 1.16, yy - 0.04), 1.55 * v, 0.08, fc=INK2, ec="none", zorder=4))
        T(X1 + 1.18 + 1.55 * v, yy, f"{v:.2f}".lstrip("0"), size=5.2, c=INK2)
    arrow(X1 + 1.18, 1.445, X1 + 1.18, 1.375, c=INK2, lw=0.8, ms=5)
    T(X1 + 0.06, 1.305, "Issue the first claim that passes (here λ = .2):", size=5.4, c=INK2)
    rows = [(STRICT, "$\\{A\\}$", "A is better", "$1-\\hat p(A)$ = .45 > λ", False),
            (TIE, "$\\{T\\}$", "a tie", "$1-\\hat p(T)$ = .60 > λ", False),
            (WEAK, "$\\{A,T\\}$", "A is not worse", "$\\hat p(B)$ = .05 ≤ λ", True),
            (SURF, "$\\{A,B,T\\}$", "abstain", "otherwise", None)]
    for k, (c, setx, name, test, ok) in enumerate(rows):
        yy = 1.13 - 0.2 * k
        if ok:
            box(X1 + 0.04, yy - 0.088, 2.29, 0.176, fc=SURF, ec=BLUE, lw=0.9, r=0.04, z=2)
        ax.add_patch(Rectangle((X1 + 0.09, yy - 0.055), 0.11, 0.11, fc=c, ec=AXIS if c == SURF else "none", lw=0.5, zorder=4))
        T(X1 + 0.25, yy, setx, size=5.8)
        T(X1 + 0.72, yy, name, size=5.6, w="bold" if ok else "normal")
        T(X1 + 1.42, yy, test, size=5.3, c=INK2)
        if ok is not None:
            T(X1 + 2.26, yy, "✓" if ok else "✗", size=7.0, c=BLUE if ok else MUTED, ha="right", w="bold")
    box(X1 + 0.05, 0.07, 2.27, 0.22, fc=SURF, ec=AXIS, lw=0.6, r=0.05, z=2, ls=(0, (2, 1.5)))
    T(X1 + 0.11, 0.18, "Routing: uncertain pairs can also receive pointwise claims,\n“no clear winner” or “both unacceptable”",
      size=4.9, c=INK2, linespacing=1.12)

    # ---------------------------------------------------------------- stage 3: certify, then audit
    X2 = 4.68
    header(X2, H - 0.12, "3", "Certify, then audit", INK2)
    arrow(X1 + 2.35, 0.93, X2 + 0.02, 1.30, c=MUTED, lw=1.0, ms=7, cs="arc3,rad=0.15")
    box(X2, 1.92, 2.17, 0.78, fc=ORANGE_WASH, r=0.05)
    T(X2 + 0.07, 2.61, "Published certified judges (SCOPE, ToE)", size=5.9, w="bold")
    steps = ["calibrate on\ndecided pairs", "strict claims\n$\\{A\\}$ or $\\{B\\}$", "deploy on all\npairs, ties too"]
    for k, st in enumerate(steps):
        x = X2 + 0.07 + 0.70 * k
        box(x, 2.12, 0.6, 0.40, fc=SURF, ec=AXIS, lw=0.5, r=0.04, z=3)
        T(x + 0.3, 2.37 if k == 0 else 2.32, st, size=4.9, ha="center", linespacing=1.05)
        if k < 2:
            arrow(x + 0.61, 2.32, x + 0.69, 2.32, c=INK2, lw=0.6, ms=4)
    T(X2 + 0.37, 2.195, "✂ ties dropped", size=4.8, c=ORANGE, ha="center", w="bold")
    T(X2 + 0.07, 1.985, f"test risk {f2(min(pub))}–{f2(max(pub))} at α = .2", size=5.6, w="bold")
    T(X2 + 2.12, 1.985, "✗", size=8, c=ORANGE, ha="right", w="bold")
    box(X2, 1.04, 2.17, 0.80, fc=BLUE_WASH, r=0.05)
    T(X2 + 0.07, 1.755, "Ours: ties stay in what counts as an error", size=5.9, w="bold")
    for k, st in enumerate(["one pair per\nsource, ties kept", "Learn-then-Test\non λ"]):
        x = X2 + 0.07 + 0.70 * k
        box(x, 1.29, 0.6, 0.34, fc=SURF, ec=AXIS, lw=0.5, r=0.04, z=3)
        T(x + 0.3, 1.46, st, size=4.9, ha="center", linespacing=1.05)
        arrow(x + 0.61, 1.46, x + 0.69, 1.46, c=INK2, lw=0.6, ms=4)
    shield(X2 + 1.47, 1.31, 0.30, BLUE)
    T(X2 + 1.82, 1.52, "risk ≤ α", size=5.3)
    T(X2 + 1.82, 1.40, "w.p. ≥ 1−δ", size=5.0, c=INK2)
    T(X2 + 0.07, 1.13, f"test risk {f2(ours[0])}–{f2(ours[1])}, coverage up to {f2(cov[1])}", size=5.6, w="bold")
    T(X2 + 2.12, 1.13, "✓", size=8, c=BLUE, ha="right", w="bold")
    box(X2, 0.07, 2.17, 0.89, fc=GRAY_WASH, r=0.05)
    T(X2 + 0.07, 0.875, "Controlled audit", size=5.9, w="bold")
    ctl = [("claim rule", "generic set prediction (APS, LAC)"),
           ("tie signal", "the judge’s p(T) vs. a constant tie prior"),
           ("visual evidence", "image removed, or a wrong image"),
           ("downstream use", "pick one of two; Best-of-N data")]
    for k, (a, b) in enumerate(ctl):
        yy = 0.715 - 0.17 * k
        ax.add_patch(Circle((X2 + 0.11, yy), 0.026, fc=INK2, ec="none", zorder=5))
        t = T(X2 + 0.17, yy, a, size=5.2, w="bold")
        T(right(t, 0.05), yy, b, size=5.0, c=INK2)
    save(fig, out, "fig_overview")


FIGS = {"overview": fig_overview, "problem": fig_problem, "claims": fig_claims, "visual": fig_visual, "scope": fig_scope, "router": fig_router}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--only", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for k in (a.only.split(",") if a.only else FIGS):
        FIGS[k](a.out)
