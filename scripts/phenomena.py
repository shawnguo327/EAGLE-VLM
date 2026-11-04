"""Are the two candidate phenomena stable and real?
Phenomenon 1 (suppressed tie channel): the judge's P(T) ranks human ties above decided pairs (AUROC > .5) while its
  level is far below the human tie rate; a single additive bias on log P(T) (fitted on half the data) recovers a large
  part of the tie detection.  Reported per set x judge (+ per tie-prompt paraphrase where those runs exist):
    human tie share | judge tie-argmax share | mean P(T) on ties / on decided | AUROC P(T) |
    3-way accuracy, tie recall, tie precision: raw -> bias-recalibrated (test half)
  plus a reliability table (P(T) bins vs human tie rate).
Phenomenon 2 (pairwise discards absolute quality; pointwise keeps it): with HUMAN absolute references
  ImageRewardDB  pair level = mean human overall rating; 'low pair' = both overall <= 4
  GenAI-Arena    'both bad' vs 'equally good' (two kinds of human tie that differ only in absolute level),
                 'both bad' vs decided
  compare 5-fold CV predictions from pairwise-only signals vs pointwise-only signals vs both.
  Plus: on decided pairs where the pairwise call is closest to 50/50 (top tercile of closeness), accuracy of the
  pointwise-gap direction vs the pairwise direction.
usage (judgebench dir): python phenomena.py --out paper_tables/phenomena.md"""
import argparse, os, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from eaglev_stats import auroc
from label_noise_lib import per_judge, eval_items

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/phenomena.md")
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen35_9b,qwen25vl7b,qwen3vl32b,r1reward,llavacritic7b")
a = ap.parse_args()
JUD = a.judges.split(","); D = np.arange(1, 10)
SETS = ["mtbench_human", "arena55k", "mllm_judge", "genai_arena/generation", "genai_arena/edition", "pickapic_u", "imagerewarddb"]
f3 = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def tri_variant(ds, j, ids, cond):
    R = load(ds, j, conds={cond}); out = {}
    for i in ids:
        r = R.get(i, {})
        if (cond, "AB") in r and (cond, "BA") in r:
            t1, t2 = r[(cond, "AB")], r[(cond, "BA")]
            q = np.array([(t1["A"] + t2["B"]) / 2, (t1["B"] + t2["A"]) / 2, (t1["T"] + t2["T"]) / 2]); out[i] = q / q.sum()
    return out


def recal(Q, y, rng):
    """fit an additive bias b on log P(T) on one half (maximise 3-way accuracy), evaluate on the other half"""
    n = len(y); pm = rng.permutation(n); fh, th = pm[: n // 2], pm[n // 2:]
    L = np.log(Q + 1e-12); bs = np.linspace(0, 12, 121)
    def pred(b, idx):
        Z = L[idx].copy(); Z[:, 2] += b; return Z.argmax(1)
    b = bs[int(np.argmax([(pred(b, fh) == y[fh]).mean() for b in bs]))]
    out = []
    for bb in (0.0, b):
        v = pred(bb, th); t = y[th] == 2; vt = v == 2
        out.append(((v == y[th]).mean(), (vt & t).sum() / max(t.sum(), 1), (vt & t).sum() / vt.sum() if vt.any() else np.nan))
    return b, out


def point(ds, j, ids):
    R = load(ds, j, conds={"score"}); out = {}
    for i in ids:
        r = R.get(i, {})
        if ("score", "A") in r and ("score", "B") in r:
            row = []
            for o in "AB":
                q = np.array([r[("score", o)].get(str(d), 0) for d in D]); q = q / q.sum(); mu = q @ D
                row += [mu, q @ (D - mu) ** 2, q[:3].sum()]
            out[i] = row
    return out


L = ["# Phenomenon 1: suppressed tie channel", "",
     "Per set x judge (tie prompt, order-averaged).  recal = one additive bias on log P(T) fitted on a random half for "
     "3-way accuracy, evaluated on the other half (b = fitted bias in nats).", "",
     "| set | judge | prompt | n | human tie | judge says T | mean P(T) ties / decided | AUROC P(T) | b | 3-way acc raw -> recal | "
     "tie recall raw -> recal | tie precision raw -> recal |", "|" + "---|" * 12]
rel = {}
for spec in SETS:
    ds, sub = (spec.split("/") + [None])[:2]
    I = eval_items(ds, items(ds)); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    for j in JUD:
        P = per_judge(ds, j, I); ids = sorted(P)
        if len(ids) < 200 or len(ids) < .9 * len(I):
            continue
        variants = [("tri", {i: P[i][1] for i in ids})]
        for c in ("tri_p1", "tri_p2", "tri_p3", "tri_sa1"):
            V = tri_variant(ds, j, ids, c)
            if len(V) >= 300:
                variants.append((c, V))
        for cname, V in variants:
            vid = [i for i in ids if i in V]
            y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in vid]); Q = np.array([V[i] for i in vid])
            t = y == 2
            b, ((a0, r0, p0), (a1, r1, p1)) = recal(Q, y, np.random.default_rng(0))
            L.append(f"| {spec} | {j} | {cname}{' (subset)' if len(vid) < .9 * len(ids) else ''} | {len(vid)} | {t.mean():.2f} | "
                     f"{(Q.argmax(1) == 2).mean():.3f} | {Q[t, 2].mean():.3f} / {Q[~t, 2].mean():.3f} | {f3(auroc(Q[:, 2], t.astype(float)))} | "
                     f"{b:.1f} | {a0:.3f} -> {a1:.3f} | {r0:.2f} -> {r1:.2f} | {f3(p0)} -> {f3(p1)} |")
            print(L[-1], flush=True)
            if cname == "tri":
                rel[(spec, j)] = (Q[:, 2], t)
L += ["", "## Reliability of P(T): human tie rate per P(T) bin (pooled over judges)", "",
      "| set | P(T) < .001 | .001-.01 | .01-.1 | .1-.5 | >= .5 |", "|---|---|---|---|---|---|"]
edges = [0, .001, .01, .1, .5, 1.01]
for spec in SETS:
    xs = [rel[k] for k in rel if k[0] == spec]
    if not xs:
        continue
    pt = np.concatenate([x[0] for x in xs]); t = np.concatenate([x[1] for x in xs]); cells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pt >= lo) & (pt < hi); cells.append(f"{t[m].mean():.2f} (n={m.sum()})" if m.sum() else "-")
    L.append(f"| {spec} | " + " | ".join(cells) + " |")

L += ["", "# Phenomenon 2: pairwise signals vs pointwise signals for ABSOLUTE quality", "",
      "5-fold CV.  pairwise = [log P(A), log P(B), log P(T), BPE, |p-.5|, order consistency]; pointwise = [E[s], Var[s], "
      "P(s<=3)] of both candidates (+ min/max); both = union.", "",
      "| set | target | judge | n | pairwise | pointwise | both |", "|---|---|---|---|---|---|---|"]


def cv_auc(X, t):
    return auroc(cross_val_predict(LogisticRegression(max_iter=5000), X, t.astype(int), cv=5, method="predict_proba")[:, 1], t.astype(float))


def cv_rho(X, v):
    return spearmanr(cross_val_predict(Ridge(alpha=1.0), X, v, cv=5), v).correlation


for spec in ("imagerewarddb", "genai_arena/generation", "genai_arena/edition"):
    ds, sub = (spec.split("/") + [None])[:2]
    I = items(ds); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    for j in JUD:
        P = per_judge(ds, j, I); W = point(ds, j, sorted(P)); ids = sorted(i for i in P if i in W)
        if len(ids) < .9 * len(I):
            continue
        p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
        Xp = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), fl]
        Wm = np.array([W[i] for i in ids]); Xw = np.c_[Wm, np.minimum(Wm[:, 0], Wm[:, 3]), np.maximum(Wm[:, 0], Wm[:, 3])]
        Xb = np.c_[Xp, Xw]
        if ds == "imagerewarddb":
            ov = np.array([[c["overall"] for c in I[i]["meta"]["cand_info"]] for i in ids], float)
            lvl = ov.mean(1); low = ov.max(1) <= 4
            L.append(f"| {spec} | pair level (Spearman) | {j} | {len(ids)} | {cv_rho(Xp, lvl):.3f} | {cv_rho(Xw, lvl):.3f} | {cv_rho(Xb, lvl):.3f} |")
            L.append(f"| {spec} | low pair, both <= 4 (AUROC; {low.mean():.2f}) | {j} | {len(ids)} | {cv_auc(Xp, low):.3f} | {cv_auc(Xw, low):.3f} | {cv_auc(Xb, low):.3f} |")
        else:
            vt = np.array([I[i]["meta"]["vote_type"] for i in ids]); bb, eq = vt == "bothbad_vote", vt == "tievote"; dec = ~(bb | eq)
            for name, m, t in (("both bad vs equally good", bb | eq, bb), ("both bad vs decided", bb | dec, bb), ("equally good vs decided", eq | dec, eq)):
                if t[m].sum() >= 20:
                    L.append(f"| {spec} | {name} (AUROC) | {j} | {m.sum()} | {cv_auc(Xp[m], t[m]):.3f} | {cv_auc(Xw[m], t[m]):.3f} | {cv_auc(Xb[m], t[m]):.3f} |")
        # pointwise resolves pairwise-uncertain decided pairs?
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids]); d = y != 2
        clo = 1 - 2 * np.abs(p - .5); hi = clo >= np.quantile(clo[d], 2 / 3)
        m = d & hi; gap = Wm[:, 0] - Wm[:, 3]; nz = m & (gap != 0)
        L.append(f"| {spec} | uncertain decided pairs: pairwise acc / pointwise-gap acc (n) | {j} | {m.sum()} | "
                 f"{((p[m] > .5) == (y[m] == 0)).mean():.3f} | {((gap[nz] > 0) == (y[nz] == 0)).mean():.3f} ({nz.sum()}) | - |")
        print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
