"""Graded claims, level by level.  Same head / splits / LTT as cert_main (25% fit, 25% cal, 50% test, delta .1).
For every set x judge x alpha, over test pairs (mean of S random splits):
  share of each output: strict 'k better' | tie | weak 'k not worse' | abstain
  error rate WITHIN each claim level (strict: y != k; tie: y != tie; weak: y == other) and overall selective risk
  trivial weak error = (1 - tie)/2 (a weak claim in a random direction) and the weak error of the head's direction on
  ALL pairs (what the weak claim would cost without selection)
  mean set size (strict/tie 1, weak 2, abstain 3)
usage (judgebench dir): python graded_breakdown.py --out paper_tables/graded_breakdown.md"""
import argparse, collections, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, H
from label_noise_lib import per_judge, eval_items, ltt, _claims

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/graded_breakdown.md")
ap.add_argument("--splits", type=int, default=100)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--perlevel", action="store_true", help="per-level guarantee: strict, tie, weak each certified at alpha with "
                "delta/3, sequentially on three disjoint calibration folds (tie / weak only on pairs left by the earlier levels)")
a = ap.parse_args()
AL = [0.1, 0.2, 0.3]
SETS = [("text", "mtbench_human", None), ("text", "arena55k", None), ("resp", "mllm_judge", None),
        ("img", "genai_arena", "generation"), ("img", "genai_arena", "edition"), ("img", "pickapic_u", None),
        ("img", "imagerewarddb", None)]
VLM = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b"]; LLM = ["qwen3_8b", "qwen25_7b"]
f2 = lambda v: "-" if v is None or np.isnan(v) else f"{v:.2f}"
f3 = lambda v: "-" if v is None or np.isnan(v) else f"{v:.3f}"

def perlevel(S3, E3, fit, cal, te, al):
    """sequential per-level certification; each level's threshold uses its own fold, so each level's selective risk is
    controlled at alpha w.p. >= 1 - delta/3 (union bound: all three simultaneously w.p. >= 1 - delta)"""
    folds = np.array_split(cal, 3); lam = [-np.inf] * 3
    for c in range(3):
        f = folds[c]
        left = np.ones(len(f), bool)
        for c0 in range(c):                                   # pairs already claimed by a stronger level
            left &= ~(S3[f, c0] <= lam[c0])
        ff = fit                                              # start point of the fixed sequence from the fit split
        lf = np.ones(len(ff), bool)
        for c0 in range(c):
            lf &= ~(S3[ff, c0] <= lam[c0])
        if left.sum() == 0:
            continue
        lam[c] = ltt(S3[f][left][:, [c]], E3[f][left][:, [c]], al, a.delta / 3, S3[ff][lf][:, [c]], E3[ff][lf][:, [c]])
    n = len(te); ch = np.full(n, -1)
    for c in (2, 1, 0):                                       # strongest claim wins
        ch[S3[te, c] <= lam[c]] = c
    return ch >= 0, ch


L = ["# Graded claims level by level" + (" (per-level guarantee)" if a.perlevel else ""), "",
     f"{a.splits} splits; shares over test pairs; err = error rate among test pairs that received that claim (pooled over "
     "splits); risk = overall selective risk; triv = (1 - tie)/2; weak@all = error of the weak claim in the head's "
     "direction if issued on every pair.", "",
     "| set | judge | n | tie | alpha | strict share / err | tie share / err | weak share / err | abstain | risk | set size | triv | weak@all |",
     "|" + "---|" * 13]
for kind, ds, sub in SETS:
    I = eval_items(ds, items(ds)); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    for j in VLM + (LLM if kind == "text" else []):
        P = per_judge(ds, j, I); ids = sorted(P)
        if len(ids) < 200 or len(ids) < .95 * len(I):
            continue
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
        F = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), fl]; n = len(y)
        acc_ = collections.defaultdict(lambda: np.zeros(2)); share = collections.defaultdict(list); risk = collections.defaultdict(list)
        size = collections.defaultdict(list); wall = []
        rng = np.random.default_rng(0)
        for _ in range(a.splits):
            pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
            m = LogisticRegression(max_iter=3000).fit(F[fit], y[fit])
            Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(F)
            k = Pr[:, :2].argmax(1); o = 1 - k
            S3 = np.c_[1 - Pr[np.arange(n), k], 1 - Pr[:, 2], Pr[np.arange(n), o]]
            E3 = np.c_[y != k, y != 2, y == o].astype(float)
            wall.append(E3[te, 2].mean())
            for al in AL:
                if a.perlevel:
                    ok, ch = perlevel(S3, E3, fit, cal, te, al)
                else:
                    lam = ltt(S3[cal], E3[cal], al, a.delta, S3[fit], E3[fit]); ok, ch = _claims(S3[te], lam)
                e = E3[te][np.arange(len(te)), np.maximum(ch, 0)]
                for c, name in ((0, "strict"), (1, "tie"), (2, "weak")):
                    mk = ok & (ch == c); share[(al, name)].append(mk.mean()); acc_[(al, name)] += [e[mk].sum(), mk.sum()]
                share[(al, "abstain")].append((~ok).mean())
                risk[al].append(e[ok].mean() if ok.any() else np.nan)
                size[al].append(np.where(~ok, 3, np.where(ch == 2, 2, 1)).mean())
        tie = (y == 2).mean()
        for al in AL:
            cells = []
            for name in ("strict", "tie", "weak"):
                er, cnt = acc_[(al, name)]
                cells.append(f"{np.mean(share[(al, name)]):.2f} / {f3(er / cnt if cnt else np.nan)}")
            L.append(f"| {ds}{'/' + sub if sub else ''} | {j}{' (LLM)' if j in LLM else ''} | {n} | {tie:.2f} | {al} | "
                     + " | ".join(cells) + f" | {np.mean(share[(al, 'abstain')]):.2f} | {f3(np.nanmean(risk[al]))} | "
                     f"{np.mean(size[al]):.2f} | {(1 - tie) / 2:.3f} | {np.mean(wall):.3f} |")
            print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
