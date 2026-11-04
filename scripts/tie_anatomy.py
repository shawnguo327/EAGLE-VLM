"""Anatomy of human ties against HUMAN quality references (not the judge's own closeness).
A. ImageRewardDB (full 7230 pairs): every image has human ratings overall / alignment / fidelity (1-7) besides the rank;
   a tie = equal rank.  Tie types: good-tie (both overall >= 5), bad-tie (both <= 3), conflict-tie (alignment and fidelity
   favour different images), other.  Questions:
   A1 how do ties differ from decided pairs in the human ratings (rating gap, level, dimension conflict)?
   A2 how predictable are human ties from the human ratings themselves (AUROC; a reference ceiling)?
   A3 Davidson against the human gap: rho(P(T), -|d overall|); and does the judge's own closeness track the human gap?
   A4 by human rating gap: human tie rate, judge accuracy on decided pairs, judge P(T)
   A5 judge pointwise E[s] vs human overall rating (absolute-scale validity)
   A6 judge P(T) AUROC per tie type vs decided pairs
B. MT-Bench (multi-expert votes per pair): consensus tie / split (some tie votes) / opposite (A and B votes, no tie) /
   unanimous decided; judge P(T) and closeness per category.
C. GenAI-Arena (vote types): judge P(T), closeness, pointwise min per type; Davidson rho within each type.
usage (judgebench dir): python tie_anatomy.py --out paper_tables/tie_anatomy.md"""
import argparse, os, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load
from eaglev_stats import auroc
from label_noise_lib import per_judge

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/tie_anatomy.md")
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen3vl32b")
a = ap.parse_args()
JUD = a.judges.split(","); D = np.arange(1, 10)
rho = lambda u, v: spearmanr(u, v).correlation
f3 = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def point(ds, j, ids):
    R = load(ds, j, conds={"score"}); out = {}
    for i in ids:
        r = R.get(i, {})
        if ("score", "A") in r and ("score", "B") in r:
            out[i] = [np.array([r[("score", o)].get(str(d), 0) for d in D]) @ D / sum(r[("score", o)].get(str(d), 0) for d in D) for o in "AB"]
    return out


L = ["# Anatomy of human ties against human quality references", ""]
# ---------------- A. ImageRewardDB
I = items("imagerewarddb"); ids = sorted(I)
ci = [I[i]["meta"]["cand_info"] for i in ids]
ov = np.array([[c[0]["overall"], c[1]["overall"]] for c in ci], float)
al_ = np.array([[c[0]["alignment"], c[1]["alignment"]] for c in ci], float)
fi = np.array([[c[0]["fidelity"], c[1]["fidelity"]] for c in ci], float)
tie = np.array([str(I[i]["label"]) == "tie" for i in ids]); dec = ~tie
y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
dov = np.abs(ov[:, 0] - ov[:, 1]); lvl = ov.mean(1)
conflict = (al_[:, 0] - al_[:, 1]) * (fi[:, 0] - fi[:, 1]) < 0
good, bad = ov.min(1) >= 5, ov.max(1) <= 3
L += ["## A. ImageRewardDB (human ratings per image, 1-7)", "",
      f"pairs {len(ids)}, ties {tie.mean():.3f}", "",
      "### A1. ties vs decided pairs in the human ratings", "",
      "| group | n | mean abs gap overall | share gap = 0 | mean level | both >= 5 | both <= 3 | alignment/fidelity conflict |",
      "|---|---|---|---|---|---|---|---|"]
for name, g in (("tie (equal rank)", tie), ("decided", dec)):
    L.append(f"| {name} | {g.sum()} | {dov[g].mean():.2f} | {(dov[g] == 0).mean():.2f} | {lvl[g].mean():.2f} | {good[g].mean():.2f} | "
             f"{bad[g].mean():.2f} | {conflict[g].mean():.2f} |")
# rank vs rating consistency on decided pairs
rk = np.array([[c[0]["rank"], c[1]["rank"]] for c in ci]); better = np.where(rk[:, 0] < rk[:, 1], 0, 1)
ovb = np.where(ov[:, 0] > ov[:, 1], 0, np.where(ov[:, 0] < ov[:, 1], 1, 2))
L += ["", f"decided pairs whose overall ratings agree with the rank: {(ovb[dec] == better[dec]).mean():.2f}; "
      f"equal overall rating: {(ovb[dec] == 2).mean():.2f}; opposite: {((ovb[dec] != 2) & (ovb[dec] != better[dec])).mean():.2f}", ""]
X = np.c_[dov, np.abs(al_[:, 0] - al_[:, 1]), np.abs(fi[:, 0] - fi[:, 1]), lvl, conflict]
cvp = cross_val_predict(LogisticRegression(max_iter=3000), X, tie.astype(int), cv=5, method="predict_proba")[:, 1]
L += ["### A2. how predictable are human ties from the human ratings themselves", "",
      f"AUROC(-|gap overall|) = {auroc(-dov, tie.astype(float)):.3f};  5-fold logistic on all rating gaps + level + conflict = "
      f"{auroc(cvp, tie.astype(float)):.3f}", "",
      "### A3-A6. judges against the human reference", "",
      "| judge | rho(P(T), -human gap) | rho(P(T), judge closeness) | rho(judge closeness, -human gap) | "
      "rho(pointwise E[s], human overall) | AUROC P(T): good-tie / bad-tie / conflict-tie / all ties |", "|---|---|---|---|---|---|"]
bins = [("gap 0", dov == 0), ("gap 1", dov == 1), ("gap 2", dov == 2), ("gap >= 3", dov >= 3)]
gap_rows = []
for j in JUD:
    P = per_judge("imagerewarddb", j, I); ok = np.array([i in P for i in ids])
    if ok.mean() < .95:
        continue
    p = np.array([P[i][0] if i in P else .5 for i in ids]); pt = np.array([P[i][1][2] if i in P else np.nan for i in ids])
    clo = 1 - 2 * np.abs(p - .5)
    W = point("imagerewarddb", j, ids); hw = np.array([i in W for i in ids])
    rp = "-"
    if hw.mean() > .9:
        sc = np.array([W[i] if i in W else [np.nan, np.nan] for i in ids]); m = hw & ok
        rp = f3(rho(sc[m].ravel(), ov[m].ravel()))
    aus = []
    for g in (good, bad, conflict, np.ones(len(ids), bool)):
        mk = ok & ((tie & g) | dec)
        aus.append(f3(auroc(pt[mk], (tie & g)[mk].astype(float))) if (tie & g & ok).sum() >= 20 else "-")
    L.append(f"| {j} | {f3(rho(pt[ok], -dov[ok]))} | {f3(rho(pt[ok], clo[ok]))} | {f3(rho(clo[ok], -dov[ok]))} | {rp} | {' / '.join(aus)} |")
    for name, g in bins:
        mk = ok & g
        gap_rows.append(f"| {j} | {name} | {mk.sum()} | {tie[mk].mean():.2f} | "
                        f"{((p[mk & dec] > .5) == (y[mk & dec] == 0)).mean():.3f} | {np.nanmean(pt[mk]):.3f} | {np.nanmean(clo[mk]):.3f} |")
L += ["", "(tie types: good = both overall >= 5, bad = both <= 3, conflict = alignment and fidelity favour different images; "
      f"counts among ties: good {(tie & good).sum()}, bad {(tie & bad).sum()}, conflict {(tie & conflict).sum()})", "",
      "### A4. by human overall-rating gap", "", "| judge | gap | n | human tie rate | judge accuracy (decided) | mean P(T) | mean judge closeness |",
      "|---|---|---|---|---|---|---|"] + gap_rows + [""]
# ---------------- B. MT-Bench multi-vote
I = items("mtbench_human"); ids = sorted(I)
vt = [I[i]["meta"]["votes"] for i in ids]; nv = np.array([I[i]["meta"]["n_votes"] for i in ids])
w = np.array([v["win"] for v in vt]); l_ = np.array([v["lose"] for v in vt]); t = np.array([v["tie"] for v in vt])
cat = np.where(nv < 2, "single vote", np.where((t == nv), "consensus tie", np.where(t > 0, "split with tie votes",
               np.where((w > 0) & (l_ > 0), "opposite (A and B votes)", "unanimous decided"))))
L += ["## B. MT-Bench: ties as consensus or as disagreement (pairs with >= 2 expert votes)", "",
      "| judge | category | n | label = tie | mean P(T) | mean closeness 1-2abs(p-.5) |", "|---|---|---|---|---|---|"]
for j in JUD:
    P = per_judge("mtbench_human", j, I)
    for c in ("consensus tie", "split with tie votes", "opposite (A and B votes)", "unanimous decided", "single vote"):
        g = [k for k, i in enumerate(ids) if cat[k] == c and i in P]
        if not g:
            continue
        pt = np.array([P[ids[k]][1][2] for k in g]); cl = np.array([1 - 2 * abs(P[ids[k]][0] - .5) for k in g])
        lt = np.mean([str(I[ids[k]]["label"]) == "tie" for k in g])
        L.append(f"| {j} | {c} | {len(g)} | {lt:.2f} | {pt.mean():.3f} | {cl.mean():.3f} |")
L.append("")
# ---------------- C. GenAI-Arena vote types
I = items("genai_arena")
L += ["## C. GenAI-Arena: judge signals per human vote type", "",
      "| subset | judge | vote type | n | mean P(T) | mean closeness | mean min pointwise E[s] | Davidson rho within type |", "|---|---|---|---|---|---|---|---|"]
for sub in ("generation", "edition"):
    ids = sorted(i for i, x in I.items() if x["subset"] == sub)
    for j in JUD:
        P = per_judge("genai_arena", j, {i: I[i] for i in ids}); W = point("genai_arena", j, ids)
        for v in ("leftvote", "rightvote", "tievote", "bothbad_vote"):
            g = [i for i in ids if I[i]["meta"]["vote_type"] == v and i in P]
            if len(g) < 10:
                continue
            pt = np.array([P[i][1][2] for i in g]); cl = np.array([1 - 2 * abs(P[i][0] - .5) for i in g])
            mn = np.array([min(W[i]) for i in g if i in W])
            L.append(f"| {sub} | {j} | {v} | {len(g)} | {pt.mean():.3f} | {cl.mean():.3f} | {f3(mn.mean() if len(mn) else np.nan)} | {f3(rho(pt, cl))} |")
open(a.out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
