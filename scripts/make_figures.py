"""Figures for the 'close or can't tell' paper (CVPR format: column 3.25 in, page 6.875 in). Vector PDF + PNG preview.
Re-runnable: every panel uses whatever judges / results exist (local results, results_A, h100/*/merged, paper_tables).
usage (from $EAGLEVLM_DATA): python make_figures.py --out /media/shawn/C/papers/VLM/overleaf/figs
Palette: validated categorical slots (blue #2a78d6 = P(T) / ours, orange #eb6834 = forced-choice U, aqua #1baf7a = third);
text in ink tokens, recessive grid, legend + direct labels."""
import argparse, collections, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import _sm, auroc

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--h100", default="h100")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "text.color": INK, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
J4 = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"]
DN = {"genai_arena": "GenAI-Arena", "mllm_judge": "MLLM-Judge", "editreward": "EditReward", "pp": "Place Pulse",
      "vlrb": "VL-RewardBench", "mmrb": "Multimodal RB", "mjbench": "MJ-Bench", "qbench2": "Q-Bench+", "mmrlhf": "MM-RLHF",
      "mmrb2": "MMRB2", "imagerewarddb": "ImageRewardDB"}
H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))


def save(fig, name):
    fig.savefig(os.path.join(a.out, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(a.out, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig); print("wrote", name)


def load(ds, m):
    R = collections.defaultdict(dict)
    for d in ["results", "results_A"] + sorted(glob.glob(f"{a.h100}/*/merged")):
        f = f"{d}/{ds}__{m}.jsonl"
        if os.path.exists(f):
            for l in open(f):
                x = json.loads(l)
                if "lp" in x:
                    R[x["id"]][(x["cond"], x["order"])] = x["lp"]
    return R


ITEMS = {}


def items(ds):
    if ds not in ITEMS:
        ITEMS[ds] = {json.loads(l)["id"]: json.loads(l) for l in open(f"items/{ds}.jsonl")}
    return ITEMS[ds]


_CH = {}


def channel(ds, m):
    if (ds, m) not in _CH:
        _CH[(ds, m)] = _channel(ds, m)
    return _CH[(ds, m)]


def _channel(ds, m):
    """per pair: U (forced choice), P(T), tie, y, p_bin; None if tri is not complete for this judge."""
    it = items(ds); R = load(ds, m); U, PT, tie, y, pb = [], [], [], [], []
    for i, r in R.items():
        if i not in it or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        p = (_sm(r[("bin", "AB")])["A"] + _sm(r[("bin", "BA")])["B"]) / 2
        U.append(H(p)); pb.append(p); PT.append((_sm(r[("tri", "AB")])["T"] + _sm(r[("tri", "BA")])["T"]) / 2)
        tie.append(it[i]["label"] == "tie"); y.append(it[i]["label"] == 0)
    if len(U) < 0.9 * len(it):
        return None
    return [np.array(v) for v in (U, PT, tie, y, pb)]


# ================================================================ Figure 2: the tie channel
TIE_DS = ["genai_arena", "mllm_judge", "editreward", "pp"]; VER_DS = ["vlrb", "mmrb", "mjbench", "qbench2", "mmrlhf"]
avail = [m for m in J4 if channel("genai_arena", m) is not None]
stats = collections.defaultdict(dict)
for ds in TIE_DS + VER_DS:
    for m in avail:
        c = channel(ds, m)
        if c is None:
            continue
        U, PT, tie, y, pb = c; err = ~tie & ((pb > .5) != y)
        if ds in TIE_DS:
            low = U >= np.quantile(U, .7); sel = low & (tie | err)
            stats[ds][m] = dict(tieU=auroc(U, tie), tieT=auroc(PT, tie), lowU=auroc(U[sel], tie[sel]), lowT=auroc(PT[sel], tie[sel]))
        else:
            stats[ds][m] = dict(errU=auroc(U, err), errT=auroc(PT, err))


def dumbbell(ax, rows, ku, kt, title, xlabel, xlim=(.3, .85)):
    ys = np.arange(len(rows))[::-1]
    ax.axvline(.5, color=MUTED, lw=.6, ls=(0, (3, 2)), zorder=0)
    ax.text(.49, -.62, "chance", color=MUTED, fontsize=5.5, ha="right", va="center")
    ax.set_ylim(-.8, len(rows) - .4)
    for yv, ds in zip(ys, rows):
        S = stats.get(ds, {})
        if not S:
            continue
        u = np.array([S[m][ku] for m in S]); t = np.array([S[m][kt] for m in S])
        ax.plot([u.mean(), t.mean()], [yv, yv], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
        for v, col in ((u, ORANGE), (t, BLUE)):
            if len(v) > 1:
                ax.plot([v.min(), v.max()], [yv, yv], color=col, lw=.8, alpha=.55, zorder=2)
            ax.scatter([v.mean()], [yv], s=26, color=col, edgecolor=SURF, linewidth=1.2, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([DN[d] for d in rows]); ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", color=INK, fontweight="bold"); ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=.5); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)


fig, axs = plt.subplots(1, 4, figsize=(6.875, 1.85), gridspec_kw=dict(width_ratios=[1, 1, 1.1, 1.05], wspace=.75))
dumbbell(axs[0], TIE_DS, "tieU", "tieT", "(a) Human ties", "tie AUROC")
dumbbell(axs[1], TIE_DS, "lowU", "lowT", "(b) Ties vs. errors", "AUROC, least confident 30%")
dumbbell(axs[2], VER_DS, "errU", "errT", "(c) Judge errors", "error AUROC, verified labels")
axs[0].scatter([], [], s=26, color=ORANGE, label="forced-choice uncertainty $U$")
axs[0].scatter([], [], s=26, color=BLUE, label="tie probability $P(T)$")
fig.legend(loc="upper center", bbox_to_anchor=(.42, 1.1), ncol=2, frameon=False, handletextpad=.2, columnspacing=1.5)
# (d) P(T) is tiny but ordered: tie rate by P(T) decile (pooled judges)
ax = axs[3]
for ds, mk, colr in (("genai_arena", "o", BLUE), ("mllm_judge", "s", AQUA)):
    xs, ys_ = [], []
    for m in avail:
        c = channel(ds, m)
        if c is None:
            continue
        U, PT, tie, y, pb = c; q = np.quantile(PT, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, PT, side="right") - 1, 0, 9)
        xs.append([np.median(PT[b == k]) for k in range(10)]); ys_.append([tie[b == k].mean() for k in range(10)])
    if xs:
        x, yy = np.exp(np.mean(np.log(np.array(xs) + 1e-12), 0)), np.mean(ys_, 0)
        ax.plot(x, yy, color=colr, lw=1.6, marker=mk, ms=3.2, mec=SURF, mew=.6)
        ax.text(x[-1] * 1.6, yy[-1], DN[ds], color=INK2, fontsize=6, va="center")
        base = np.mean([channel(ds, m)[2].mean() for m in avail if channel(ds, m) is not None])
        ax.axhline(base, color=colr, lw=.6, ls=(0, (1, 2)), alpha=.8)          # base rate of human ties
ax.set_xscale("log"); ax.set_xlabel("median $P(T)$ in decile (log)"); ax.set_ylabel("human tie rate")
ax.set_title("(d) Tie rate by $P(T)$ decile", loc="left", color=INK, fontweight="bold"); ax.grid(color=GRID, lw=.5); ax.set_axisbelow(True)
ax.set_xlim(right=ax.get_xlim()[1] * 30)
save(fig, "fig_channel")

# ================================================================ Figure 3: confident shared errors
M = json.load(open("paper_tables/mechanism_4j.json")) if os.path.exists("paper_tables/mechanism_4j.json") else {}
X = json.load(open("paper_tables/mechanism_cross.json")) if os.path.exists("paper_tables/mechanism_cross.json") else {}
pp_fresh = os.path.exists("results/pp__qwen25vl7b.jsonl") and os.path.exists("paper_tables/mechanism_4j.json") and \
    os.path.getmtime("paper_tables/mechanism_4j.json") > os.path.getmtime("results/pp__qwen25vl7b.jsonl")
DS3 = [d for d in ["vlrb", "mmrb", "mllm_judge", "genai_arena", "mjbench", "mmrb2", "imagerewarddb", "qbench2", "pp"]
       if d in M and (d != "pp" or pp_fresh)]
fig, axs = plt.subplots(1, 2, figsize=(3.3, 2.05), gridspec_kw=dict(wspace=.12), sharey=True)
ys = np.arange(len(DS3))[::-1]
ax = axs[0]
for yv, d in zip(ys, DS3):
    v = [M[d]["conf_correct"], M[d]["conf_idiosyncratic_err"], M[d]["conf_unanimous_err"]]
    ax.plot([min(v), max(v)], [yv, yv], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    for val, col in zip(v, (BLUE, ORANGE, AQUA)):
        ax.scatter([val], [yv], s=22, color=col, edgecolor=SURF, linewidth=1, zorder=3)
ax.set_yticks(ys); ax.set_yticklabels([DN[d] for d in DS3]); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1); ax.set_xlabel("mean confidence $|2\\bar p-1|$"); ax.grid(axis="x", color=GRID, lw=.5); ax.set_axisbelow(True)
ax.set_title("(a) Confidence", loc="left", color=INK, fontweight="bold")
for lab, col in (("correct verdict", BLUE), ("error of one judge", ORANGE), ("error of all four", AQUA)):
    ax.scatter([], [], s=22, color=col, label=lab)
fig.legend(loc="upper center", bbox_to_anchor=(.42, 1.08), ncol=3, frameon=False, handletextpad=.1, columnspacing=.9)
ax = axs[1]
for yv, d in zip(ys, DS3):
    au, ex = X.get(d, {}).get("auroc_all"), X.get(d, {}).get("auroc_excl_shared")
    if not au:
        continue
    u, e = np.mean(list(au.values())), np.mean(list(ex.values()))
    ax.annotate("", xy=(e, yv), xytext=(u, yv), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=.8, shrinkA=3, shrinkB=3, mutation_scale=6))
    ax.scatter([u], [yv], s=22, color=SURF, edgecolor=INK2, linewidth=.9, zorder=3)
    ax.scatter([e], [yv], s=22, color=INK2, edgecolor=SURF, linewidth=.9, zorder=3)
ax.set_xlim(.55, .9); ax.set_xlabel("AUROC of $U$"); ax.grid(axis="x", color=GRID, lw=.5); ax.set_axisbelow(True)
ax.set_title("(b) Error AUROC", loc="left", color=INK, fontweight="bold")
ax.text(.895, (ys[4] + ys[5]) / 2, "open: all pairs\nfilled: all-four\nerrors removed", color=INK2, fontsize=5.4, ha="right", va="center")
ax.tick_params(axis="y", length=0)
save(fig, "fig_shared")

# ================================================================ Figure 1: three kinds of uncertain or wrong verdicts
EX = [("genai_arena/generation_0143", "genai_arena", "(a) Close: humans tie", "return a tie", BLUE),
      ("mmrb2/t2i_0614", "mmrb2", "(b) Can't tell: a missed detail", "escalate", ORANGE),
      ("mmrb2/t2i_0112", "mmrb2", "(c) Shared, confident error", "audit", AQUA)]
import textwrap
fig = plt.figure(figsize=(6.875, 2.3))
outer = fig.add_gridspec(1, 3, wspace=.16)
for k, (iid, ds, title, route, col) in enumerate(EX):
    it = items(ds)[iid]; r = load(ds, "qwen3vl8b")[iid]
    ab, ba = _sm(r[("tri", "AB")]), _sm(r[("tri", "BA")])
    pA, pB, pT = (ab["A"] + ba["B"]) / 2, (ab["B"] + ba["A"]) / 2, (ab["T"] + ba["T"]) / 2
    pbin = (_sm(r[("bin", "AB")])["A"] + _sm(r[("bin", "BA")])["B"]) / 2
    nwrong = 0
    for m in J4:
        rr = load(ds, m).get(iid, {})
        if ("bin", "AB") in rr and it["label"] != "tie":
            pm = (_sm(rr[("bin", "AB")])["A"] + _sm(rr[("bin", "BA")])["B"]) / 2; nwrong += (pm > .5) != (it["label"] == 0)
    g = outer[0, k].subgridspec(3, 2, height_ratios=[1.0, .24, .46], hspace=.03, wspace=.05)
    for j in (0, 1):
        axi = fig.add_subplot(g[0, j]); im = Image.open(it["cand"][j]).convert("RGB"); im.thumbnail((360, 360))
        w, h = im.size; s = min(w, h); im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        axi.imshow(im); axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_visible(True); sp.set_color(GRID); sp.set_linewidth(.6)
        axi.text(.04, .95, "AB"[j], transform=axi.transAxes, fontsize=7, fontweight="bold", color=INK, va="top",
                 bbox=dict(boxstyle="round,pad=.15", fc=SURF, ec="none", alpha=.85))
        if it["label"] != "tie" and it["label"] == j:
            axi.text(.96, .95, "humans", transform=axi.transAxes, fontsize=6, color=INK, va="top", ha="right",
                     bbox=dict(boxstyle="round,pad=.15", fc=SURF, ec=INK2, lw=.5, alpha=.9))
        if j == 0:
            axi.set_title(title, loc="left", fontsize=7, fontweight="bold", color=INK, x=0)
    axp = fig.add_subplot(g[1, :]); axp.axis("off")
    q = it["question"].strip().rstrip(".")
    wl = textwrap.wrap("\u201c" + q + "\u201d", 44)
    wl = wl[:2] if len(wl) <= 2 else [wl[0], wl[1][:40] + "\u2026\u201d"]
    axp.text(0, .5, "\n".join(wl), fontsize=5.8, style="italic",
             color=INK2, va="center", transform=axp.transAxes, linespacing=1.15)
    axb = fig.add_subplot(g[2, :])
    vals = [pA, pB, pT]; labs = ["A", "B", "T"]; cols = [MUTED, MUTED, BLUE]
    axb.barh([2, 1, 0], [.62 * v for v in vals], color=cols, height=.6, left=.08)
    for yv, v, lab in zip([2, 1, 0], vals, labs):
        axb.text(.06, yv, lab, ha="right", va="center", fontsize=6, color=INK2)
        axb.text(.08 + .62 * v + .015, yv, f"{v:.2f}", va="center", fontsize=5.8, color=INK)
    axb.set_xlim(0, 1.9); axb.set_ylim(-.6, 2.6); axb.axis("off")
    hum = "tie" if it["label"] == "tie" else "prefer " + "AB"[it["label"]]
    info = ("humans: " + hum + "\nforced A/B: " + f"{pbin:.2f}/{1 - pbin:.2f}" + "\n" +
            (f"{nwrong}/4 judges wrong" if it["label"] != "tie" else ""))
    axb.text(1.0, 2.55, info, fontsize=5.6, color=INK2, ha="left", va="top", linespacing=1.3)
    axb.text(1.9, -.55, "\u2192 " + route, fontsize=6.5, color=INK, fontweight="bold", ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=.25", fc=SURF, ec=col, lw=1.1))
save(fig, "fig_teaser")

# ================================================================ Figure 4: routing on one vs two signals (simulation)
def route_acc(U, PT, tie, y, pb, budget, policy, rng, splits=50):
    """accuracy w.r.t. human labels (a tie output is right only on a human tie; an escalated pair is resolved by a human).
    policy 'none': escalate top-budget by U, never tie. 'U': escalate top-budget by U, tie the next band of U (band width
    tuned on a calibration half). 'UT': escalate top-budget by U among pairs not tied; tie if P(T) >= threshold (tuned)."""
    n = len(U); correct_v = (pb > .5) == y; acc = []
    for _ in range(splits):
        perm = rng.permutation(n); cal, te = perm[: n // 2], perm[n // 2:]
        def run(idx, tau):
            k = int(round(budget * len(idx))); u = U[idx]
            if policy == "UT":
                tied = PT[idx] >= tau
                rest = np.where(~tied)[0]; esc = np.zeros(len(idx), bool); esc[rest[np.argsort(-u[rest])[:k]]] = True
                tied &= ~esc
            else:
                order = np.argsort(-u); esc = np.zeros(len(idx), bool); esc[order[:k]] = True
                tied = np.zeros(len(idx), bool)
                if policy == "U":
                    tied[order[k:k + int(round(tau * len(idx)))]] = True
            ok = np.where(esc, True, np.where(tied, tie[idx], ~tie[idx] & correct_v[idx]))
            return ok.mean()
        grid = np.quantile(PT, np.linspace(0, 1, 41)) if policy == "UT" else (np.linspace(0, .6, 31) if policy == "U" else [0])
        tau = max(grid, key=lambda t: run(cal, t)); acc.append(run(te, tau))
    return float(np.mean(acc))


BUD = [0, .05, .1, .2, .3, .4]; ROUT = {}
for ds in ["genai_arena", "mllm_judge"]:
    curves = collections.defaultdict(list)
    for m in avail:
        c = channel(ds, m)
        if c is None:
            continue
        U, PT, tie, y, pb = c; rng = np.random.default_rng(0)
        for pol in ("none", "U", "UT"):
            curves[pol].append([route_acc(U, PT, tie, y, pb, b, pol, rng) for b in BUD])
    ROUT[ds] = {k: np.mean(v, 0) for k, v in curves.items() if v}
fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.7), gridspec_kw=dict(wspace=.32), sharey=False)
for ax, ds in zip(axs, ["genai_arena", "mllm_judge"]):
    R = ROUT.get(ds, {})
    for pol, col, lab, ls in (("none", MUTED, "no tie output", (0, (2, 1.5))), ("U", ORANGE, "one signal ($U$)", "-"),
                              ("UT", BLUE, "two signals ($U$, $P(T)$)", "-")):
        if pol in R:
            ax.plot(np.array(BUD) * 100, R[pol], color=col, lw=1.6, ls=ls, marker="o", ms=2.6, mec=SURF, mew=.5, label=lab)
    ax.set_title(f"({'ab'[['genai_arena', 'mllm_judge'].index(ds)]}) {DN[ds]}", loc="left", color=INK, fontweight="bold")
    ax.set_xlabel("pairs sent to humans (%)"); ax.grid(color=GRID, lw=.5); ax.set_axisbelow(True)
axs[0].set_ylabel("agreement with humans")
axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, 1.42), ncol=3, frameon=False, fontsize=5.6, handlelength=1.6, columnspacing=.8)
save(fig, "fig_routing")
for ds, R in ROUT.items():
    print("ROUTING", ds, {k: [round(x, 3) for x in v] for k, v in R.items()})

# ================================================================ Figure 5: where shared errors live (per subset)
SUBS = []
for ds in ["mllm_judge", "vlrb", "imagerewarddb", "mmrb"]:
    for sub, r in (M.get(ds, {}).get("unanimous_err_by_subset") or {}).items():
        n = sum(1 for x in items(ds).values() if x["subset"] == sub)
        if n >= 60:
            SUBS.append((r, f"{DN[ds]}: {sub}", n))
SUBS.sort(reverse=True); top = SUBS[:7] + SUBS[-4:]
fig, ax = plt.subplots(figsize=(3.3, 2.0))
yv = np.arange(len(top))[::-1]
ax.barh(yv, [t[0] for t in top], color=[AQUA if i < 7 else MUTED for i in range(len(top))], height=.62)
for yy, t in zip(yv, top):
    ax.text(t[0] + .006, yy, f"{t[0]:.2f}", va="center", fontsize=5.6, color=INK)
ax.set_yticks(yv); ax.set_yticklabels([t[1] for t in top], fontsize=5.8); ax.tick_params(axis="y", length=0)
ax.set_xlabel("share of pairs that all four judges get wrong"); ax.grid(axis="x", color=GRID, lw=.5); ax.set_axisbelow(True)
ax.axhline(3.5, color=GRID, lw=.6); ax.text(ax.get_xlim()[1], 3.65, "lowest", ha="right", va="bottom", fontsize=5.4, color=MUTED)
save(fig, "fig_subsets")
print("SUBSETS", [(round(t[0], 3), t[1], t[2]) for t in top])
print("judges with phase-A results:", avail)
