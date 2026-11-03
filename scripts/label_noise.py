"""Is the text-vs-image certification gap a label-noise effect?  MT-Bench human turn 1 has 1-7 votes per pair (majority
label); GenAI-Arena has exactly 1 vote per pair.  Re-label MT-Bench with ONE random vote per pair (20 draws) and compare
certified coverage with the majority label.  Also: pairs with >= 3 votes only (cleanest labels), majority vs single vote.
Rules (same judge outputs, 25% fit / 25% cal / 50% test, SCOPE threshold, as tie_cert.py):
  strict   'k better' via the 3-class head, tie = error
  graded   strict > tie > weak, one threshold (exploratory version)
  bin-acc  forced-choice accuracy on pairs whose label is not a tie
usage (judgebench dir): python label_noise.py --out paper_tables/label_noise.md"""
import argparse, os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/label_noise.md")
ap.add_argument("--splits", type=int, default=60)
ap.add_argument("--draws", type=int, default=20)
ap.add_argument("--alphas", default="0.1,0.2,0.3")
a = ap.parse_args()
AL = [float(x) for x in a.alphas.split(",")]
JUD = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3_8b", "qwen25_7b"]
VOTES = "/media/shawn/C/papers/VLM/judgebench_local/items/mtbench_votes_turn1.jsonl"


def per_judge(ds, j, I):
    R = load(ds, j, conds={"bin", "tri"}); out = {}
    for i, r in R.items():
        if i not in I or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
        f1, f2 = b1["A"] / (b1["A"] + b1["B"]), b2["A"] / (b2["A"] + b2["B"])
        q = np.array([(t1["A"] + t2["B"]) / 2, (t1["B"] + t2["A"]) / 2, (t1["T"] + t2["T"]) / 2])
        out[i] = ((f1 + 1 - f2) / 2, q / q.sum(), float((f1 > .5) == (f2 > .5)))
    return out


def thr(s, e, al):
    o = np.argsort(s, kind="stable"); cs = np.cumsum(e[o] - al); ok = np.where(cs <= -1)[0]
    return s[o][ok.max()] if len(ok) else -np.inf


def _claims(Ss, lam):
    ok = Ss <= lam; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def hier_lam(Ss, Es, al):
    best = -np.inf
    for lam in np.unique(Ss.ravel()):
        anyok, ch = _claims(Ss, lam)
        if anyok.any() and (Es[np.arange(len(Es)), np.maximum(ch, 0)][anyok] - al).sum() <= -1:
            best = lam
    return best


def evaluate(F, p, y, splits, seed):
    """mean test coverage of strict and graded rules at each alpha, plus forced-choice accuracy on non-tie labels"""
    n = len(y); rng = np.random.default_rng(seed); res = collections.defaultdict(list)
    for _ in range(splits):
        pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
        m = LogisticRegression(max_iter=3000).fit(F[fit], y[fit])
        Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(F)
        k = Pr[:, :2].argmax(1); other = 1 - k; pk = Pr[np.arange(n), k]; po = Pr[np.arange(n), other]
        es = (y != k).astype(float)
        Ss = np.c_[1 - pk, 1 - Pr[:, 2], po]; Es = np.c_[y != k, y != 2, y == other].astype(float)
        for al in AL:
            lam = thr((1 - pk)[cal], es[cal], al); res[("strict", al)].append(((1 - pk)[te] <= lam).mean())
            acc, _ = _claims(Ss[te], hier_lam(Ss[cal], Es[cal], al)); res[("graded", al)].append(acc.mean())
    nt = y != 2
    return {k: np.mean(v) for k, v in res.items()} | {"acc": ((p[nt] > .5) == (y[nt] == 0)).mean(), "tie": (y == 2).mean()}


I = items("mtbench_human")
V = collections.defaultdict(list)
for r in map(json.loads, open(VOTES)):
    V[(r["question_id"], frozenset((r["model_a"], r["model_b"])))].append(r)
ids, votes = [], []
for i, x in I.items():
    key = (x["meta"]["question_id"], frozenset(x["meta"]["models"]))
    if key not in V:
        continue
    m0 = x["meta"]["models"][0]
    lab = [2 if v["winner"].startswith("tie") else (0 if (v["model_a"] if v["winner"] == "model_a" else v["model_b"]) == m0 else 1)
           for v in V[key]]
    ids.append(i); votes.append(lab)
nv = np.array([len(v) for v in votes])
print(f"matched {len(ids)} / {len(I)} pairs; votes per pair: mean {nv.mean():.2f}")
L = ["# Label noise vs certification (MT-Bench human, turn 1)", "",
     f"Majority label (as in the paper tables) vs ONE random vote per pair ({a.draws} draws).  Cells = mean test coverage "
     f"({a.splits} splits); acc = forced-choice accuracy on non-tie labels.  GenAI-Arena has 1 vote per pair.", "",
     "| judge | pairs | label | tie share | acc | " + " | ".join(f"strict @{x} | graded @{x}" for x in AL) + " |",
     "|" + "---|" * (5 + 2 * len(AL))]
for j in JUD:
    P = per_judge("mtbench_human", j, I)
    for sub, mask in (("all pairs", np.ones(len(ids), bool)), (">=3 votes", nv >= 3)):
        sel = [k for k, i in enumerate(ids) if mask[k] and i in P]
        if len(sel) < 100:
            continue
        p = np.array([P[ids[k]][0] for k in sel]); Q = np.array([P[ids[k]][1] for k in sel]); fl = np.array([P[ids[k]][2] for k in sel])
        F = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), fl]
        y_maj = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[ids[k]]["label"])] for k in sel])
        rows = [("majority", evaluate(F, p, y_maj, a.splits, 0))]
        rng = np.random.default_rng(1); D = []
        for d in range(a.draws):
            y1 = np.array([votes[k][rng.integers(len(votes[k]))] for k in sel])
            D.append(evaluate(F, p, y1, max(10, a.splits // 4), d))
        rows.append(("1 random vote", {key: np.mean([x[key] for x in D]) for key in D[0]}))
        for lab, r in rows:
            L.append(f"| {j} | {sub} (n={len(sel)}) | {lab} | {r['tie']:.2f} | {r['acc']:.3f} | "
                     + " | ".join(f"{r[('strict', x)]:.2f} | {r[('graded', x)]:.2f}" for x in AL) + " |")
            print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
