"""Place Pulse: certify the POPULATION preference instead of a single vote.
Single votes on Place Pulse are not certifiable even for a human-consensus oracle (pp_ties.py): ties and decided votes
are driven by the rater, not the pair.  The population-level question 'which place do people in general find safer /
livelier / ...' has a well-defined answer: the sign of the Bradley-Terry score difference fitted on all OTHER votes.
Target:  y_c = 0 if BT(cand0) > BT(cand1) else 1, from BT fitted on all votes except those on the 1860 eval pairs
         (so the eval pair's own vote never enters its target).
Target reliability: split the training votes in two random halves, fit BT on each; the share of eval pairs on which the
         two halves agree in sign; |dBT| strata.
Judges: VLM order-averaged p_bar (bin) + P(T) (tri); a 2-class logistic head on the judge features is fit on 25%, the
         strict claim 'k preferred by the population' is calibrated on 25% (SCOPE and LTT, delta .1) and tested on 50%.
Compared with the same judges against the single-vote label (tie = error), the setting of the main table.
usage (judgebench dir): python pp_consensus.py --out paper_tables/pp_consensus.md"""
import argparse, collections, gzip, json, os, sys
import numpy as np
from scipy import optimize, sparse
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, H
from label_noise_lib import per_judge, thr
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--ann", default="/media/shawn/C/papers/VLM/place-pulse-2.0/annotations")
ap.add_argument("--out", default="paper_tables/pp_consensus.md")
ap.add_argument("--splits", type=int, default=200)
ap.add_argument("--delta", type=float, default=0.1)
a = ap.parse_args()
AL = [0.1, 0.2, 0.3]
JUD = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"]
rng0 = np.random.default_rng(0)

V = []   # (category, left, right, choice 1=left 0=right 2=equal)
for f, eq in (("pairs_binary.jsonl.gz", False), ("pairs_ties.jsonl.gz", True)):
    for l in gzip.open(f"{a.ann}/{f}", "rt"):
        d = json.loads(l)
        V.append((d["category"], d["left_id"], d["right_id"], 2 if eq else (1 if d["choice"] == "left" else 0)))
I = items("pp")
key = lambda cat, x, y: (cat, min(x, y), max(x, y))
ev = {}
for i, r in I.items():
    c0, c1 = [os.path.basename(p).split(".")[0] for p in r["cand"]]
    ev[i] = (r["subset"], c0, c1)
held = set(key(*v) for v in ev.values())
train = [k for k, v in enumerate(V) if key(v[0], v[1], v[2]) not in held]
half = rng0.random(len(V)) < .5
print("votes", len(V), "train", len(train), flush=True)


def fit_bt(ks):
    """per category L2 BT on the binary votes among ks; ties are ignored (they carry no direction)"""
    out = {}
    for cat in sorted(set(V[k][0] for k in ks)):
        kb = [k for k in ks if V[k][0] == cat and V[k][3] != 2]
        ids = {}
        for k in kb:
            ids.setdefault(V[k][1], len(ids)); ids.setdefault(V[k][2], len(ids))
        A = np.array([ids[V[k][1]] for k in kb]); B = np.array([ids[V[k][2]] for k in kb]); y = np.array([V[k][3] for k in kb], float)
        m = len(kb)
        X = sparse.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[A, B])), shape=(m, len(ids)))

        def f(w):
            z = X @ w[:-1] + w[-1]; p = 1 / (1 + np.exp(-z))
            ll = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum() + (w[:-1] ** 2).sum()
            g = X.T @ (p - y); return ll, np.r_[g + 2 * w[:-1], (p - y).sum()]
        w = optimize.minimize(f, np.zeros(len(ids) + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
        out[cat] = {x: w[j] for x, j in ids.items()}
    return out


def dbt(B, i):
    cat, c0, c1 = ev[i]; s = B.get(cat, {})
    return s.get(c0, np.nan) - s.get(c1, np.nan)


BT = fit_bt(train)
B1 = fit_bt([k for k in train if half[k]]); B2 = fit_bt([k for k in train if not half[k]])
ids_all = sorted(i for i in I if not np.isnan(dbt(BT, i)))
d = np.array([dbt(BT, i) for i in ids_all]); d1 = np.array([dbt(B1, i) for i in ids_all]); d2 = np.array([dbt(B2, i) for i in ids_all])
ok = ~np.isnan(d1) & ~np.isnan(d2)
L = ["# Place Pulse: certifying the population preference (BT on all other votes) instead of a single vote", "",
     f"eval pairs with a BT target: {len(ids_all)} / {len(I)}.", "",
     "## Target reliability (split-half BT sign agreement)", "", "| |dBT| stratum | n | sign agreement of the two halves |", "|---|---|---|"]
qs = np.quantile(np.abs(d), [0, .25, .5, .75, 1])
for lo, hi in zip(qs[:-1], qs[1:]):
    m = ok & (np.abs(d) >= lo) & (np.abs(d) <= hi)
    L.append(f"| {lo:.2f}-{hi:.2f} | {m.sum()} | {(np.sign(d1[m]) == np.sign(d2[m])).mean():.3f} |")
L.append(f"| all | {ok.sum()} | {(np.sign(d1[ok]) == np.sign(d2[ok])).mean():.3f} |")
# single-vote agreement with the consensus (how noisy single votes are w.r.t. the population target)
lab = {i: str(I[i]["label"]) for i in ids_all}
dec = np.array([lab[i] != "tie" for i in ids_all])
yv = np.array([0 if lab[i] == "0" else 1 for i in ids_all])
L += ["", f"single decided vote agrees with the consensus sign: {((d[dec] < 0).astype(int) == yv[dec]).mean():.3f} "
      f"(n={dec.sum()}); single-vote tie share {1 - dec.mean():.3f}", ""]


def ltt_strict(s_cal, e_cal, s_fit, e_fit, al, delta):
    G = np.linspace(0, 1, 201)[1:]
    pv = lambda s, e, lam: binom.cdf(int(e[s <= lam].sum()), int((s <= lam).sum()), al) if (s <= lam).any() else 1.0
    st = int(np.argmin([pv(s_fit, e_fit, g) for g in G])); best = -np.inf
    for g in G[st:]:
        if pv(s_cal, e_cal, g) <= delta:
            best = g
        else:
            break
    return best


L += ["## Judges: single-vote target vs population target (strict claim 'k preferred')", "",
      "Cells = coverage / selective risk / violation rate at alpha.  acc = accuracy of the order-averaged forced choice "
      "against the target (single vote: decided votes only).", "",
      "| judge | target | n | acc | " + " | ".join(f"SCOPE @{x} | LTT @{x}" for x in AL) + " |", "|" + "---|" * (4 + 2 * len(AL))]
for j in JUD:
    P = per_judge("pp", j, I)
    ids = [i for i in ids_all if i in P]
    if len(ids) < 200:
        continue
    p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
    F = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), fl, p]
    dd = np.array([dbt(BT, i) for i in ids])
    ysv = np.array([{"0": 0, "1": 1, "tie": 2}[lab[i]] for i in ids]); yc = (dd < 0).astype(int)
    for tname, y in (("single vote (tie = error)", ysv), ("population (BT sign)", yc)):
        n = len(y); rng = np.random.default_rng(0); R = collections.defaultdict(list)
        for _ in range(a.splits):
            pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
            m = LogisticRegression(max_iter=3000).fit(F[fit], y[fit])
            Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(F)
            k = Pr[:, :2].argmax(1); s = 1 - Pr[np.arange(n), k]          # estimated error of 'k preferred' (tie = error)
            e = (y != k).astype(float)
            for al in AL:
                for key_, lam in (("SCOPE", thr(s[cal], e[cal], al)), ("LTT", ltt_strict(s[cal], e[cal], s[fit], e[fit], al, a.delta))):
                    acc = s[te] <= lam; r = e[te][acc].mean() if acc.any() else 0.0
                    R[(key_, al)].append((acc.mean(), r, float(r > al)))
        nt = y != 2
        accu = ((p[nt] > .5) == (y[nt] == 0)).mean()
        L.append(f"| {j} | {tname} | {n} | {accu:.3f} | " + " | ".join(
            "{:.2f} / {:.3f} / {:.2f}".format(*np.mean(R[(kk, al)], 0)) for al in AL for kk in ("SCOPE", "LTT")) + " |")
        print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n"); print("\n".join(L[:14]))
