"""Downstream value of certified graded claims: recovering a model leaderboard from judge claims + a human-label budget.
Ground truth: Bradley-Terry strengths fitted on the FULL source data (tie = half a win): arena55k -> all 57k battles of
lmarena arena-human-preference-55k; genai_arena -> all 4175 GenAI-Arena-Bench pairs of the subset; mtbench_human -> the
910 pairs themselves (6 models, saturates; reported for completeness).
Human routing: tiered, random within a tier.  strict+H: abstained pairs first.  graded+H: abstained, then weak claims, then tie
claims.  judge-all and human-only: random pairs.  (Least-certain-first routing was tried and hurts ranking.)
Per split (fit 25% / calibrate 25% / test 50%, as tie_cert.py), on the test half, with a budget of b * n_test human labels:
  human        b * n_test random test pairs get their human label; nothing else
  judge-all    uncertified: the judge's order-averaged p_bar on every pair, humans overwrite b * n_test random pairs
  strict+H     certified strict claims (alpha) -> hard win; humans label abstained pairs (random order) up to the budget
  graded+H     certified graded claims (strict > tie > weak, one threshold, exploratory SCOPE version as tie_cert.py):
               strict -> hard win, tie -> .5, weak 'k >= other' -> the calibration-set mean human score of k among accepted
               weak claims; humans label abstained pairs up to the budget
Metric: Kendall tau between the BT strengths of the method and the ground truth, over models with >= MINAPP appearances in the
(sub)set; mean over splits.  Also: human labels each method needs to reach the tau that human-only reaches with 50% of the
test half.
usage (judgebench dir): python rank_recover.py --out paper_tables/rank_recover.md"""
import argparse, os, sys, collections, json, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from sklearn.linear_model import LogisticRegression
from scipy.stats import kendalltau
from label_noise_lib import ltt

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/rank_recover.md")
ap.add_argument("--splits", type=int, default=40)
ap.add_argument("--alpha", type=float, default=0.2)
ap.add_argument("--minapp", type=int, default=20)
ap.add_argument("--ltt", action="store_true", help="Learn-then-Test (delta .1) instead of SCOPE for strict and graded")
a = ap.parse_args()
BUD = [0, .05, .1, .2, .3, .5, .75, 1.0]
SETS = [("text", "mtbench_human", None), ("text", "arena55k", None),
        ("img", "genai_arena", "edition"), ("img", "genai_arena", "generation")]
VLM = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"]; LLM = ["qwen3_8b", "qwen25_7b"]


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
    """SCOPE linearised threshold, tie-aware: the condition is checked only at the last position of each distinct score,
    because deployment accepts every item with s <= lambda"""
    o = np.argsort(s, kind="stable"); ss = s[o]; cs = np.cumsum(e[o] - al)
    last = np.r_[ss[1:] != ss[:-1], True]; ok = np.where((cs <= -1) & last)[0]
    return ss[ok.max()] if len(ok) else -np.inf


def _claims(Ss, lam):
    ok = Ss <= lam; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def hier_lam(Ss, Es, al):
    best = -np.inf
    for lam in np.unique(Ss.ravel()):
        anyok, ch = _claims(Ss, lam)
        if anyok.any() and (Es[np.arange(len(Es)), np.maximum(ch, 0)][anyok] - al).sum() <= -1:
            best = lam
    return best


def bt(pairs, s, nm):
    """BT strengths from soft outcomes s = score of the first model in each pair (weighted logistic regression)"""
    if len(pairs) == 0:
        return np.zeros(nm)
    X = np.zeros((len(pairs), nm)); X[np.arange(len(pairs)), pairs[:, 0]] = 1; X[np.arange(len(pairs)), pairs[:, 1]] -= 1
    XX = np.r_[X, X]; yy = np.r_[np.ones(len(X)), np.zeros(len(X))]; w = np.r_[s, 1 - s] + 1e-9
    if w[:len(X)].sum() < 1e-6 or w[len(X):].sum() < 1e-6:
        return X.T @ (s - .5)
    return LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(XX, yy, sample_weight=w).coef_[0]


FULL = "/media/shawn/C/papers/VLM/judgebench_full"


def gt_pairs(ds, sub, I):
    """(model_a, model_b, score_a) over the full source data"""
    if ds == "arena55k":
        out = []
        for r in csv.DictReader(open(f"{FULL}/raw/lmarena-ai__arena-human-preference-55k/train.csv")):
            out.append((r["model_a"], r["model_b"], 1. if r["winner_model_a"] == "1" else 0. if r["winner_model_b"] == "1" else .5))
        return out
    if ds == "genai_arena":
        out = []
        for r in map(json.loads, open(f"{FULL}/items/genai_arena.jsonl")):
            if r["subset"] == sub:
                lab = str(r["label"]); out.append((*r["meta"]["models"], 1. if lab == "0" else 0. if lab == "1" else .5))
        return out
    return [(*I[i]["meta"]["models"], 1. if str(I[i]["label"]) == "0" else 0. if str(I[i]["label"]) == "1" else .5) for i in I]


L = ["# Leaderboard recovery from certified judge claims + human labels", "",
     f"alpha = {a.alpha}; {a.splits} splits (25/25/50); ground truth = BT on all human labels of the (sub)set (tie = .5).  "
     "Cells = mean Kendall tau vs ground truth at a human budget of b x n_test.  'need' = human labels (share of n_test) needed "
     "to reach human-only tau at b = .5 (>1 = never).", ""]
hdr = "| dataset | judge | n | models | tie | method | coverage | " + " | ".join(f"b={b}" for b in BUD) + " | need |"
L += [hdr, "|" + "---|" * (8 + len(BUD))]
for kind, ds, sub in SETS:
    I0 = items(ds); I = {i: x for i, x in I0.items() if sub is None or x["subset"] == sub}
    js = VLM + (LLM if kind == "text" else [])
    for j in js:
        P = per_judge(ds, j, I)
        ids = sorted(P)
        if len(ids) < 200:
            continue
        cnt = collections.Counter(m for i in ids for m in I[i]["meta"]["models"])
        keep = sorted(m for m, c in cnt.items() if c >= a.minapp); mi = {m: k for k, m in enumerate(keep)}
        ids = [i for i in ids if all(m in mi for m in I[i]["meta"]["models"])]
        n = len(ids); nm = len(keep)
        pairs = np.array([[mi[m] for m in I[i]["meta"]["models"]] for i in ids])
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids]); hs = np.where(y == 0, 1., np.where(y == 1, 0., .5))
        G = [g for g in gt_pairs(ds, sub, I) if g[0] in mi and g[1] in mi]
        gt = bt(np.array([[mi[g[0]], mi[g[1]]] for g in G]), np.array([g[2] for g in G]), nm)
        p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); flip = np.array([P[i][2] for i in ids])
        u = H(p); F = np.c_[np.log(Q + 1e-9), u, np.abs(p - .5), flip]
        rng = np.random.default_rng(0); res = collections.defaultdict(list); cov = collections.defaultdict(list)
        tau = lambda est: kendalltau(est, gt).statistic
        for _ in range(a.splits):
            pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]; nt = len(te)
            m = LogisticRegression(max_iter=3000).fit(F[fit], y[fit])
            Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(F)
            k = Pr[:, :2].argmax(1); other = 1 - k; pk = Pr[np.arange(n), k]; po = Pr[np.arange(n), other]
            # strict
            es = (y != k).astype(float)
            lam = ltt((1 - pk)[cal, None], es[cal, None], a.alpha, .1, (1 - pk)[fit, None], es[fit, None]) if a.ltt else thr((1 - pk)[cal], es[cal], a.alpha)
            acc_s = (1 - pk) <= lam
            # graded
            Ss = np.c_[1 - pk, 1 - Pr[:, 2], po]; Es = np.c_[y != k, y != 2, y == other].astype(float)
            lamh = ltt(Ss[cal], Es[cal], a.alpha, .1, Ss[fit], Es[fit]) if a.ltt else hier_lam(Ss[cal], Es[cal], a.alpha)
            acc_h, ch = _claims(Ss, lamh)
            wk = cal[acc_h[cal] & (ch[cal] == 2)]
            wval = np.mean(np.where(k[wk] == 0, hs[wk], 1 - hs[wk])) if len(wk) else .75   # mean human score of the claimed side
            sk = lambda v: np.where(k == 0, v, 1 - v)                                     # score of cand0 given score v of side k
            s_h = np.where(ch == 0, sk(1.), np.where(ch == 1, .5, sk(wval)))
            cov["strict"].append(acc_s[te].mean()); cov["graded+H"].append(acc_h[te].mean())
            hum_order = rng.permutation(te)
            # tiered routing, random within a tier (pure least-certain-first hurts ranking: the hardest pairs are near-equal
            # and carry little ranking information).  strict: abstained first.  graded: abstained, then weak, then tie claims.
            rk = rng.permutation(n).astype(float)
            r_s = te[np.lexsort((rk[te], acc_s[te]))]
            tier = np.where(~acc_h, 0, np.where(ch == 2, 1, np.where(ch == 1, 2, 3)))
            r_h = te[np.lexsort((rk[te], tier[te]))]
            r_j = hum_order                                    # judge-all: humans overwrite random pairs
            for b in BUD:
                nb = int(round(b * nt))
                hl = hum_order[:nb]
                res[("human", b)].append(tau(bt(pairs[hl], hs[hl], nm)))
                sj = p.copy(); sj[r_j[:nb]] = hs[r_j[:nb]]
                res[("judge-all", b)].append(tau(bt(pairs[te], sj[te], nm)))
                for key, acc, sval, order in (("strict+H", acc_s, sk(1.), r_s), ("graded+H", acc_h, s_h, r_h)):
                    hum = order[:nb]; sv = sval.copy(); sv[hum] = hs[hum]
                    idx = te[acc[te] | np.isin(te, hum)]
                    res[(key, b)].append(tau(bt(pairs[idx], sv[idx], nm)))
        ref = np.mean(res[("human", .5)])
        for key in ("human", "judge-all", "strict+H", "graded+H"):
            curve = [np.mean(res[(key, b)]) for b in BUD]
            need = next((b for b, t in zip(BUD, curve) if t >= ref - 1e-9), None)
            c = {"human": 0., "judge-all": 1.}.get(key, np.mean(cov.get(key if key != "strict+H" else "strict", [0])))
            L.append(f"| {ds}{'/' + sub if sub else ''} | {j}{' (LLM)' if j in LLM else ''} | {n} | {nm} | {(y == 2).mean():.2f} | {key} | "
                     f"{c:.2f} | " + " | ".join(f"{t:.3f}" for t in curve) + f" | {need if need is not None else '>1'} |")
            print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
