"""Build + execute a timestamped figure notebook for the paper storyline (Fig 1-5).
Data: markdown result tables in paper_figures/data/ (copied from bjserver judgebench/paper_tables).
Output: paper_figures/figures_<YYYYmmdd_HHMM>.ipynb (executed) and paper_figures/figs_<stamp>/fig*.png|pdf.
usage: .figenv/bin/python paper_figures/make_notebook.py   (from the VLM folder)"""
import datetime, os, subprocess, sys
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
nb = nbf.v4.new_notebook()
md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))

md(f"""# EAGLE-VLM paper figures — version {STAMP}
Storyline: **Fig 1** the problem (ties are common; benchmarks and certified judges drop them) →
**Fig 2** published guarantees break, and fixing the calibration alone collapses them (controlled chain) →
**Fig 3** what human ties are made of (human references) →
**Fig 4** two judge phenomena: a suppressed tie channel; pairwise prompts discard absolute quality →
**Fig 5** our graded certification: where it helps, and its per-level limits.
Data = `data/*.md` (snapshot of `judgebench/paper_tables`).""")
code(f"""import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
STAMP = "{STAMP}"; DATA = "{HERE}/data"; OUT = "{HERE}/figs_" + STAMP; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({{"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "legend.frameon": False}})
# Okabe-Ito (colour-blind safe), fixed order
C = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
SETS = ["mtbench_human", "arena55k", "mllm_judge", "genai_arena/generation", "genai_arena/edition", "pickapic_u", "imagerewarddb"]
NICE = {{"mtbench_human": "MT-Bench", "arena55k": "Arena-55k", "mllm_judge": "MLLM-as-a-Judge",
        "genai_arena/generation": "GenAI-Arena gen.", "genai_arena/edition": "GenAI-Arena edit",
        "pickapic_u": "Pick-a-Pic", "imagerewarddb": "ImageRewardDB"}}

def tables(name):
    \"\"\"all markdown tables of data/<name>.md as DataFrames (strings)\"\"\"
    out, cur = [], []
    for line in open(f"{{DATA}}/{{name}}.md"):
        if line.startswith("|"):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif cur:
            out.append(cur); cur = []
    if cur: out.append(cur)
    return [pd.DataFrame([r for r in t[2:] if len(r) == len(t[0])], columns=t[0]) for t in out if len(t) > 2]

def num(s, k=0):
    m = re.findall(r"-?\\d+\\.\\d+|-?\\d+", str(s)); return float(m[k]) if len(m) > k else np.nan

def save(fig, name):
    fig.savefig(f"{{OUT}}/{{name}}.png", bbox_inches="tight"); fig.savefig(f"{{OUT}}/{{name}}.pdf", bbox_inches="tight")""")

md("""## Fig 1 — Ties are common, judges almost never say tie, and dropping ties inflates reported accuracy
(a) human tie share vs the share of pairs on which the judge's tie-option argmax is T (mean over judges, range as error bar).
(b) accuracy as reported (ties dropped) vs the same forced choice on all pairs (tie = error) vs the tie-option 3-way accuracy.""")
code("""rt = tables("repro_tiedrop")[0]
for c in ["human tie", "acc w/o tie", "acc w/ tie", "acc 3-way", "judge tie rate"]: rt[c] = rt[c].map(num)
g = rt.groupby("set"); sets = [s for s in SETS if s in g.groups]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
x = np.arange(len(sets)); w = .38
ht = [g.get_group(s)["human tie"].mean() for s in sets]; jt = g["judge tie rate"]
ax[0].bar(x - w/2, ht, w, color=C[0], label="human tie share")
ax[0].bar(x + w/2, [jt.mean()[s] for s in sets], w, color=C[1], label="judge says T (mean)",
          yerr=[[jt.mean()[s]-jt.min()[s] for s in sets], [jt.max()[s]-jt.mean()[s] for s in sets]], capsize=2)
ax[0].set_xticks(x, [NICE[s] for s in sets], rotation=30, ha="right"); ax[0].set_ylabel("share of pairs"); ax[0].legend(); ax[0].set_title("(a) ties: humans vs judges")
for k, (col, lab) in enumerate([("acc w/o tie", "reported (ties dropped)"), ("acc w/ tie", "all pairs, tie = error"), ("acc 3-way", "tie option, 3-way")]):
    m = g[col].mean(); lo = g[col].min(); hi = g[col].max()
    ax[1].errorbar(x + (k-1)*.22, [m[s] for s in sets], yerr=[[m[s]-lo[s] for s in sets], [hi[s]-m[s] for s in sets]], fmt="o", ms=4, color=C[k], capsize=2, label=lab)
ax[1].set_xticks(x, [NICE[s] for s in sets], rotation=30, ha="right"); ax[1].set_ylabel("accuracy"); ax[1].axhline(.5, color="grey", lw=.6, ls="--")
ax[1].legend(fontsize=7); ax[1].set_title("(b) dropping ties inflates accuracy")
fig.tight_layout(); save(fig, "fig1_problem"); plt.show()""")

md("""## Fig 2 — Controlled chain: published tie-dropping protocols vs the same methods with ties kept vs ours
(a) promised alpha vs realised selective risk of SCOPE and Trust-or-Escalate as published (calibrated without ties, deployed with them), one dot per dataset (mean over judges).
(b) mean prediction-set size at alpha = .2 (1 = decisive or tie claim, 2 = weak claim, 3 = abstain): SCOPE with ties kept (P2), strict claims with all our components (P4), ours graded (G2), ours with every level <= alpha (G3).""")
code("""ch = tables("chain")[0]
ch["step"] = ch["step"].str.split().str[0]
alphas = [c for c in ch.columns if c.startswith("alpha")]
def metric(cell, k):  # acceptance / size / decisive / strict err / risk / violation
    v = re.findall(r"-?\\d+\\.\\d+|-(?=\\s|$)", str(cell).split("(")[0]);
    try: return float(v[k])
    except Exception: return np.nan
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for k, s in enumerate(SETS):
    d = ch[(ch.set == s) & (ch.step == "P1")]
    if len(d):
        ax[0].plot([num(a, 0) for a in alphas], [d[a].map(lambda c: metric(c, 4)).mean() for a in alphas], "o-", color=C[k % 8], ms=3, lw=1, label=NICE[s])
ax[0].plot([0, .35], [0, .35], color="grey", ls="--", lw=.8); ax[0].set_xlabel("promised alpha"); ax[0].set_ylabel("realised risk (ties at test time)")
ax[0].set_title("(a) SCOPE as published"); ax[0].legend(fontsize=6)
steps = [("P2", "SCOPE, ties kept"), ("P4", "strict + all our signals"), ("G2", "graded (ours)"), ("G3", "graded, every level <= alpha")]
x = np.arange(len(SETS)); w = .2; a2 = [a for a in alphas if "0.2" in a][0]
for k, (st, lab) in enumerate(steps):
    sz = [ch[(ch.set == s) & (ch.step == st)][a2].map(lambda c: metric(c, 1)).mean() for s in SETS]
    dc = [ch[(ch.set == s) & (ch.step == st)][a2].map(lambda c: metric(c, 2)).mean() for s in SETS]
    ax[1].bar(x + (k-1.5)*w, sz, w, color=C[k], label=lab); ax[2].bar(x + (k-1.5)*w, dc, w, color=C[k], label=lab)
for a_, t, yl in ((ax[1], "(b) mean set size, alpha .2 (lower = more informative)", (1, 3.05)), (ax[2], "(c) decisive (strict) share, alpha .2", None)):
    a_.set_xticks(x, [NICE[s] for s in SETS], rotation=30, ha="right"); a_.set_title(t)
    if yl: a_.set_ylim(*yl)
fig.legend(*ax[1].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(.62, -.08)); fig.tight_layout(); save(fig, "fig2_chain"); plt.show()""")

md("""## Fig 3 — What human ties are made of (human references, not the judge's view)
(a) ImageRewardDB: human tie rate and judge accuracy on decided pairs vs the human overall-rating gap.
(b) MT-Bench: judge P(T) for consensus ties vs ties that come from expert disagreement.
(c) GenAI-Arena: lower pointwise score of the pair vs judge P(T), per human vote type.""")
code("""ta = tables("tie_anatomy"); a4 = [t for t in ta if "gap" in t.columns][0]; mb = [t for t in ta if "category" in t.columns][0]
gv = [t for t in ta if "vote type" in t.columns][0]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
gaps = ["gap 0", "gap 1", "gap 2", "gap >= 3"]
for k, j in enumerate(a4.judge.unique()):
    d = a4[a4.judge == j].set_index("gap")
    ax[0].plot(range(4), [num(d.loc[g, "judge accuracy (decided)"]) for g in gaps], "o-", color=C[k], label=f"{j} accuracy")
d = a4[a4.judge == a4.judge.unique()[0]].set_index("gap")
ax[0].bar(range(4), [num(d.loc[g, "human tie rate"]) for g in gaps], color="lightgrey", label="human tie rate")
ax[0].set_xticks(range(4), ["0", "1", "2", ">=3"]); ax[0].set_xlabel("human overall-rating gap (1-7 scale)"); ax[0].legend(fontsize=6); ax[0].set_title("(a) ImageRewardDB")
cats = ["consensus tie", "split with tie votes", "opposite (A and B votes)", "unanimous decided"]
for k, j in enumerate(mb.judge.unique()):
    d = mb[mb.judge == j].set_index("category")
    ax[1].bar(np.arange(4) + (k-1)*.25, [num(d.loc[c, "mean P(T)"]) for c in cats], .25, color=C[k], label=j)
ax[1].set_xticks(range(4), ["consensus\\ntie", "split w/\\ntie votes", "A vs B\\nsplit", "unanimous\\ndecided"]); ax[1].set_ylabel("mean judge P(T)"); ax[1].legend(fontsize=6); ax[1].set_title("(b) MT-Bench expert votes")
mk = {"leftvote": "o", "rightvote": "o", "tievote": "s", "bothbad_vote": "X"}
for k, (sub, j) in enumerate([(s, j) for s in ("generation", "edition") for j in gv.judge.unique()]):
    d = gv[(gv.subset == sub) & (gv.judge == j)]
    for _, r in d.iterrows():
        ax[2].scatter(num(r["mean min pointwise E[s]"]), num(r["mean P(T)"]), marker=mk[r["vote type"]], color=C[0] if sub == "generation" else C[5], s=30, alpha=.8)
ax[2].set_xlabel("lower pointwise score of the pair (mean)"); ax[2].set_ylabel("mean judge P(T)")
ax[2].set_title("(c) GenAI-Arena: o decided, s equal, X both bad\\n(blue generation, red edit)")
fig.tight_layout(); save(fig, "fig3_tie_anatomy"); plt.show()""")

md("""## Fig 4 — Two judge phenomena
(a) Suppressed tie channel: human tie rate per bin of the judge's P(T) (pooled over judges). Even P(T) < .001 carries 11-33% human ties.
(b) Pairwise prompts discard absolute quality: CV AUROC of pairwise-only vs pointwise-only signals for targets that differ only in absolute quality.""")
code("""ph = tables("phenomena"); rel = [t for t in ph if "P(T) < .001" in t.columns][0]; p2 = [t for t in ph if "pairwise" in t.columns][0]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
bins = [c for c in rel.columns if c != "set"]
for k, s in enumerate(SETS):
    d = rel[rel.set == s]
    if len(d): ax[0].plot(range(len(bins)), [num(d.iloc[0][b]) for b in bins], "o-", color=C[k % 8], ms=3, label=NICE[s])
ax[0].plot(range(len(bins)), [.0005, .005, .05, .3, .75], color="grey", ls="--", lw=.8, label="calibrated (bin centre)")
ax[0].set_xticks(range(len(bins)), ["<.001", ".001-.01", ".01-.1", ".1-.5", ">=.5"]); ax[0].set_xlabel("judge P(T) bin"); ax[0].set_ylabel("human tie rate")
ax[0].legend(fontsize=6); ax[0].set_title("(a) tie channel is suppressed (reliability)")
d = p2[p2.target.str.contains("AUROC")].copy(); d["pw"] = d["pairwise"].map(num); d["pt"] = d["pointwise"].map(num)
lab = (d.set.str.replace("genai_arena/", "GenAI ") + ": " + d.target.str.replace(" (AUROC)", "").str.replace(r"\\(AUROC.*", "", regex=True))
labels = list(dict.fromkeys(lab)); y = {l: i for i, l in enumerate(labels)}
for k, j in enumerate(d.judge.unique()):
    e = d[d.judge == j]; yy = [y[l] + (k-1)*.15 for l in lab[d.judge == j]]
    ax[1].hlines(yy, e.pw, e.pt, color="lightgrey"); ax[1].scatter(e.pw, yy, color=C[1], s=14, label="pairwise" if k == 0 else None)
    ax[1].scatter(e.pt, yy, color=C[0], s=14, label="pointwise" if k == 0 else None)
ax[1].set_yticks(range(len(labels)), labels, fontsize=7); ax[1].axvline(.5, color="grey", lw=.6, ls="--"); ax[1].set_xlabel("5-fold CV AUROC")
ax[1].legend(fontsize=7); ax[1].set_title("(b) absolute quality: pairwise vs pointwise")
fig.tight_layout(); save(fig, "fig4_phenomena"); plt.show()""")

md("""## Fig 5 — Graded certification: what is accepted, and how reliable each claim level is (alpha = .2, mean over VLM judges)
(a) composition of the outputs: strict / tie / weak / abstain. (b) error rate within each claim level vs alpha; the overall risk is controlled, strict claims are not.""")
code("""gb = tables("graded_breakdown")[0]; gb = gb[gb.alpha == "0.2"]
for c in ["strict share / err", "tie share / err", "weak share / err"]:
    gb[c + "_s"] = gb[c].map(lambda v: num(v, 0)); gb[c + "_e"] = gb[c].map(lambda v: num(v, 1))
gb["abst"] = gb["abstain"].map(num); g = gb.groupby("set")
sets = [s for s in SETS if s in g.groups]; x = np.arange(len(sets))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5)); bot = np.zeros(len(sets))
for k, (c, lab) in enumerate([("strict share / err_s", "strict"), ("tie share / err_s", "tie"), ("weak share / err_s", "weak"), ("abst", "abstain")]):
    v = np.array([g[c].mean()[s] for s in sets]); ax[0].bar(x, v, .6, bottom=bot, color=[C[0], C[2], C[1], "lightgrey"][k], label=lab); bot += v
ax[0].set_xticks(x, [NICE[s] for s in sets], rotation=30, ha="right"); ax[0].set_ylabel("share of test pairs"); ax[0].legend(fontsize=7); ax[0].set_title("(a) output composition")
for k, (c, lab) in enumerate([("strict share / err_e", "strict claims"), ("weak share / err_e", "weak claims")]):
    ax[1].bar(x + (k-.5)*.35, [g[c].mean()[s] for s in sets], .35, color=[C[0], C[1]][k], label=lab)
ax[1].scatter(x, [g["risk"].apply(lambda s: s.map(num).mean())[s] for s in sets], color="black", zorder=3, s=14, label="overall risk")
ax[1].axhline(.2, color=C[5], ls="--", lw=1, label="alpha = .2"); ax[1].set_xticks(x, [NICE[s] for s in sets], rotation=30, ha="right")
ax[1].set_ylabel("error rate"); ax[1].legend(fontsize=7); ax[1].set_title("(b) per-level error (overall LTT)")
fig.tight_layout(); save(fig, "fig5_graded"); plt.show()""")
md("""## Notes for the next version
* Fig 2 add Trust-or-Escalate curves once persona runs on full data finish; add judges R1-Reward / LLaVA-Critic.
* Fig 5: add routing / jury variants (route_exp) and joint-LTT per-level results; MLLM-as-a-Judge per task (14 tasks).
* Fig 6 (planned): scale (8B -> 32B -> 38B/72B) and task taxonomy (blind vs seeing).""")

path = f"{HERE}/figures_{STAMP}.ipynb"; nbf.write(nb, path)
r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", path,
                    "--ExecutePreprocessor.timeout=600"], capture_output=True, text=True)
print(path); print(r.stderr[-3000:] if r.returncode else "executed OK")
