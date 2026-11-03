"""Why can no judge detect Place Pulse ties, and what CAN be certified there?
Data: all Place Pulse 2.0 votes (binary + 'equal'), per category.  Votes on the pairs of our 1860-item eval set are held out.
  BT       L2 Bradley-Terry image scores fitted on the remaining binary votes (human consensus 'oracle judge')
  prop     per-image tie propensity: share of 'equal' among the image's other votes (tie votes / all votes)
  nbr      temporal neighbourhood: share of 'equal' among the 10 votes cast just before/after in the same study (no voter
           ids exist; a high value means ties cluster in time, i.e. behave like a rater/session habit, not pair content)
Part A, held-out 5% of ALL votes (large n): tie-vs-decided AUROC of -|dBT|, prop, nbr, and their CV logistic combination.
Part B, our 1860 items: the same features vs VLM P(T)/BPE, and the oracle as a judge under SCOPE (alpha .2/.3):
  oracle-drop (calibrate without ties, deploy with), oracle-all (tie = error), oracle-3way (tie if |dBT| < tau, tau fit on
  a quarter), and the label ceiling: best possible accuracy against single votes.
usage (judgebench dir): python pp_ties.py --out paper_tables/pp_ties.md"""
import argparse, collections, gzip, json, os, sys
from datetime import datetime
import numpy as np
from scipy import optimize, sparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from eaglev_stats import auroc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ap = argparse.ArgumentParser()
ap.add_argument("--ann", default="/media/shawn/C/papers/VLM/place-pulse-2.0/annotations")
ap.add_argument("--out", default="paper_tables/pp_ties.md")
a = ap.parse_args()
rng = np.random.default_rng(0)

V = []   # (category, left, right, choice 1=left 0=right 2=equal, t, study)
for f, eq in (("pairs_binary.jsonl.gz", False), ("pairs_ties.jsonl.gz", True)):
    for l in gzip.open(f"{a.ann}/{f}", "rt"):
        d = json.loads(l)
        c = 2 if eq else (1 if d["choice"] == "left" else 0)
        V.append((d["category"], d["left_id"], d["right_id"], c, datetime.fromisoformat(d["timestamp"][:19]).timestamp(), d["study_id"]))
print("votes", len(V), flush=True)
I = items("pp")
key = lambda cat, x, y: (cat, min(x, y), max(x, y))
evalpairs = {}
for i, r in I.items():
    ids = [os.path.basename(p).split(".")[0] for p in r["cand"]]
    evalpairs[key(r["subset"], *ids)] = i
held = np.array([key(v[0], v[1], v[2]) in evalpairs for v in V])
testA = (rng.random(len(V)) < 0.05) & ~held
train = ~held & ~testA
print("held (eval pairs)", held.sum(), "test A", testA.sum(), flush=True)

# temporal neighbourhood tie rate (computed on all votes; excludes the vote itself)
nbr = np.zeros(len(V))
by = collections.defaultdict(list)
for k, v in enumerate(V):
    by[v[5]].append(k)
for st, ks in by.items():
    ks = sorted(ks, key=lambda k: V[k][4]); eq = np.array([V[k][3] == 2 for k in ks], float)
    cs = np.r_[0, np.cumsum(eq)]; n = len(ks)
    for j, k in enumerate(ks):
        lo, hi = max(0, j - 10), min(n, j + 11)
        nbr[k] = (cs[hi] - cs[lo] - eq[j]) / max(hi - lo - 1, 1)

# per category BT on training binary votes, and tie propensity on training votes
BT, PROP = {}, {}
for cat in sorted(set(v[0] for v in V)):
    ks = [k for k in np.where(train)[0] if V[k][0] == cat]
    cnt, tie = collections.Counter(), collections.Counter()
    for k in ks:
        for x in (V[k][1], V[k][2]):
            cnt[x] += 1; tie[x] += V[k][3] == 2
    PROP[cat] = {x: (tie[x] + .13) / (cnt[x] + 1) for x in cnt}
    kb = [k for k in ks if V[k][3] != 2]
    ids = {}
    for k in kb:
        ids.setdefault(V[k][1], len(ids)); ids.setdefault(V[k][2], len(ids))
    A = np.array([ids[V[k][1]] for k in kb]); B = np.array([ids[V[k][2]] for k in kb]); y = np.array([V[k][3] for k in kb], float)
    m = len(kb)
    X = sparse.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[A, B])), shape=(m, len(ids)))
    l2 = 1.0

    def f(w):
        z = X @ w[:-1] + w[-1]; p = 1 / (1 + np.exp(-z))
        ll = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum() + l2 * (w[:-1] ** 2).sum()
        g = X.T @ (p - y); return ll, np.r_[g + 2 * l2 * w[:-1], (p - y).sum()]
    w = optimize.minimize(f, np.zeros(len(ids) + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    BT[cat] = ({x: w[j] for x, j in ids.items()}, w[-1])
    print(cat, "BT fitted on", m, flush=True)


def feats(cat, l, r):
    s, _ = BT[cat]; P = PROP[cat]
    d = s.get(l, 0.0) - s.get(r, 0.0)
    return d, P.get(l, .13) + P.get(r, .13)


L = ["# Place Pulse ties: are they predictable at all, and what can be certified?", ""]
# Part A
ka = np.where(testA)[0]
F = np.array([feats(V[k][0], V[k][1], V[k][2]) + (nbr[k],) for k in ka]); yA = np.array([V[k][3] == 2 for k in ka], float)
cvp = lambda X, y: cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
L += ["## A. Held-out 5%% of all votes (n = %d, tie share %.3f): tie vs decided AUROC" % (len(ka), yA.mean()), "",
      "| feature | AUROC |", "|---|---|",
      f"| -abs(dBT) (human consensus closeness) | {auroc(-np.abs(F[:, 0]), yA):.3f} |",
      f"| image tie propensity (sum) | {auroc(F[:, 1], yA):.3f} |",
      f"| temporal neighbour tie rate | {auroc(F[:, 2], yA):.3f} |",
      f"| logistic(closeness, propensity) | {auroc(cvp(np.c_[np.abs(F[:, 0]), F[:, 1]], yA), yA):.3f} |",
      f"| logistic(all three) | {auroc(cvp(np.c_[np.abs(F[:, 0]), F[:, 1], F[:, 2]], yA), yA):.3f} |", ""]
dec = yA == 0; yd = np.array([V[k][3] for k in ka])[dec]
L += [f"Single-vote predictability of decided votes from BT (sign of dBT + left bias): acc {((F[dec, 0] + BT[V[ka[0]][0]][1] > 0) == (yd == 1)).mean():.3f}", ""]
print("\n".join(L), flush=True)

# Part B: eval items
ids_ = list(I)
vote_of = {}
for k in np.where(held)[0]:
    vote_of.setdefault(evalpairs[key(V[k][0], V[k][1], V[k][2])], []).append(k)
rows = []
for i in ids_:
    r = I[i]; cat = r["subset"]; c0, c1 = [os.path.basename(p).split(".")[0] for p in r["cand"]]
    d, pr = feats(cat, c0, c1)
    nb = np.mean([nbr[k] for k in vote_of.get(i, [])]) if vote_of.get(i) else np.nan
    rows.append((i, {"0": 0, "1": 1, "tie": 2}[str(r["label"])], d, pr, nb))
y = np.array([x[1] for x in rows]); d = np.array([x[2] for x in rows]); pr = np.array([x[3] for x in rows]); nb = np.array([x[4] for x in rows])
tie = (y == 2).astype(float)
L += ["## B. Our 1860 eval pairs (tie share %.2f): tie vs decided AUROC" % tie.mean(), "", "| signal | AUROC |", "|---|---|",
      f"| oracle closeness -abs(dBT) | {auroc(-np.abs(d), tie):.3f} |", f"| image tie propensity | {auroc(pr, tie):.3f} |",
      f"| temporal neighbour tie rate (n with vote match {np.isfinite(nb).sum()}) | {auroc(np.nan_to_num(nb, nan=np.nanmean(nb)), tie):.3f} |"]
for j in ("qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"):
    R = load("pp", j, conds={"bin", "tri"}); pt = []; u = []
    for i, *_ in rows:
        rr = R.get(i, {})
        pt.append((rr[("tri", "AB")]["T"] + rr[("tri", "BA")]["T"]) / 2 if ("tri", "BA") in rr and ("tri", "AB") in rr else np.nan)
        if ("bin", "AB") in rr and ("bin", "BA") in rr:
            p = (rr[("bin", "AB")]["A"] / (rr[("bin", "AB")]["A"] + rr[("bin", "AB")]["B"]) + rr[("bin", "BA")]["B"] / (rr[("bin", "BA")]["A"] + rr[("bin", "BA")]["B"])) / 2
            u.append(H(p))
        else:
            u.append(np.nan)
    pt, u = np.array(pt), np.array(u); ok = np.isfinite(pt) & np.isfinite(u)
    L.append(f"| {j}: P(T) / BPE | {auroc(pt[ok], tie[ok]):.3f} / {auroc(u[ok], tie[ok]):.3f} |")

# oracle as judge
s_bt = 1 / (1 + np.exp(-d)); pred = np.where(s_bt > .5, 0, 1); u_bt = H(s_bt)


def thr(s, e, al):
    o = np.argsort(s, kind="stable"); cs = np.cumsum(e[o] - al); ok = np.where(cs <= -1)[0]
    return s[o][ok.max()] if len(ok) else -np.inf


L += ["", "## B2. Human-consensus oracle (BT from all other votes) as the judge, SCOPE, 300 splits", "",
      "| alpha | oracle-drop cov / risk / viol | oracle-all cov / risk | oracle-3way cov / risk / tie% |", "|---|---|---|---|"]
e2 = np.where(y == 2, 1.0, (pred != y).astype(float))
for al in (.2, .3, .4):
    res = collections.defaultdict(list); n = len(y)
    for _ in range(300):
        pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]; cb = pm[: n // 2]
        nt = cb[y[cb] != 2]
        for kk, cset in (("drop", nt), ("all", cb)):
            lam = thr(u_bt[cset], e2[cset], al); acc = u_bt[te] <= lam
            r_ = e2[te][acc].mean() if acc.any() else 0.0; res[kk].append((acc.mean(), r_, r_ > al))
        taus = np.quantile(np.abs(d[fit]), np.linspace(0, .6, 13))
        best = max(taus, key=lambda t: (np.where(np.abs(d[fit]) < t, 2, pred[fit]) == y[fit]).mean())
        o3 = np.where(np.abs(d) < best, 2, pred); e3 = (o3 != y).astype(float)
        s3 = np.where(o3 == 2, np.abs(d) / (best + 1e-9), 1 / (np.abs(d) - best + 1e-3))     # far from tau = confident
        lam = thr(s3[cal], e3[cal], al); acc = s3[te] <= lam
        res["3"].append((acc.mean(), e3[te][acc].mean() if acc.any() else 0.0, (o3[te][acc] == 2).mean() if acc.any() else 0.0))
    f = lambda k: np.mean(res[k], 0)
    L.append(f"| {al} | {f('drop')[0]:.2f} / {f('drop')[1]:.3f} / {f('drop')[2]:.2f} | {f('all')[0]:.2f} / {f('all')[1]:.3f} | "
             f"{f('3')[0]:.2f} / {f('3')[1]:.3f} / {f('3')[2]:.2f} |")
L += ["", f"Oracle accuracy on decided eval pairs: {((pred == y)[y != 2]).mean():.3f};  on all pairs (tie = wrong): {(pred == y).mean():.3f}"]
open(a.out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
