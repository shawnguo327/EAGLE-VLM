"""Tie-aware certified judging on real data with human ties.  Labels y in {0: cand0 better, 1: cand1 better, 2: tie}.
Splits: fit 25% (3-class head) / calibrate 25% / test 50%, 200 random splits; SCOPE linearised threshold for selective rules,
split-conformal (LAC) for prediction sets.  All rules use the same judge outputs; nothing is tuned on the test half.
Features for the 3-class head (per judge): log P0, log P1, log PT (A/B/T prompt, both orders aligned), BPE and |p_bar-.5|
(A/B prompt), order flip (A/B verdict changes with the order), and optionally the PANEL: mean P(T) and mean p_bar over the
4 VLM judges (or the 2 LLM + 4 VLM judges on text sets).
Claims and their error:
  strict   'k better'            error unless y == k                  (the published claim; tie = error)
  three    'k better' | 'tie'    error unless y == claim
  weak     'k at least as good'  error only if y == other             (a tie is compatible)
  mixed    strict when the head is confident enough, otherwise weak: the claim with the highest coverage at alpha;
           implemented as one score over the union of candidate claims (strict claims scored by 1-P(y=k), weak claims by
           P(y=other)); each item picks its lowest-risk claim, SCOPE threshold on that score
Also: bin-drop (published: BPE, calibrated on non-ties, deployed on all), predicted vs realised risk of bin-drop
(r + pi(1-r)), and LAC prediction sets at 1-alpha: marginal coverage, mean size, share {k}, {k,tie}, other.
usage (judgebench dir): python tie_cert.py --out paper_tables/tie_cert.md"""
import argparse, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/tie_cert.md")
ap.add_argument("--splits", type=int, default=200)
ap.add_argument("--alphas", default="0.1,0.2,0.3")
a = ap.parse_args()
AL = [float(x) for x in a.alphas.split(",")]
SETS = {"text": ["mtbench_human", "arena55k", "arena_tt"], "resp": ["mllm_judge"], "img": ["genai_arena", "editreward", "pp"]}
VLM = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"]; LLM = ["qwen3_8b", "qwen25_7b"]


def per_judge(ds, j, I):
    R = load(ds, j, conds={"bin", "tri"}); out = {}
    for i, r in R.items():
        if i not in I or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
        f1, f2 = b1["A"] / (b1["A"] + b1["B"]), b2["A"] / (b2["A"] + b2["B"])
        p = (f1 + 1 - f2) / 2
        q = np.array([(t1["A"] + t2["B"]) / 2, (t1["B"] + t2["A"]) / 2, (t1["T"] + t2["T"]) / 2]); q = q / q.sum()
        out[i] = (p, q, float((f1 > .5) == (f2 > .5)))
    return out


def _claims(Ss, lam):
    """per item: the MOST INFORMATIVE claim (strict > tie > weak) whose estimated error <= lam; -1 = abstain"""
    ok = Ss <= lam; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def hier_lam(Ss, Es, al):
    """one threshold lam on the head's estimated error; each item makes the most informative claim within lam.
    lam = largest grid value with sum_cal(E - alpha) <= -1 (SCOPE linearised condition; exploratory: the claim can switch
    with lam, so the loss path is not monotone -> to be replaced by fixed-sequence Learn-then-Test for the paper)."""
    best = -np.inf
    for lam in np.unique(Ss.ravel()):
        anyok, ch = _claims(Ss, lam)
        if anyok.any() and (Es[np.arange(len(Es)), np.maximum(ch, 0)][anyok] - al).sum() <= -1:
            best = lam
    return best


def hier_apply(Ss, Es, lam):
    anyok, ch = _claims(Ss, lam)
    return anyok, ch, np.where(anyok, Es[np.arange(len(Es)), np.maximum(ch, 0)], 0.0)


def thr(s, e, al):
    o = np.argsort(s, kind="stable"); cs = np.cumsum(e[o] - al); ok = np.where(cs <= -1)[0]
    return s[o][ok.max()] if len(ok) else -np.inf


L = ["# Tie-aware certified judging (real data with human ties)", "",
     f"{a.splits} splits: 25% fit / 25% calibrate / 50% test.  Cells = coverage (share of test pairs that receive a certified claim) "
     "/ realised risk.  Panel = the 3-class head also sees the panel's mean P(T) and mean p_bar.", ""]
T1 = ["## 1. Coverage of certified claims", "",
      "| kind | dataset | judge | n | tie | alpha | bin-drop (published) | strict | three-way | weak | mixed | mixed+panel | mixed claims: strict/tie/weak | HIER (+panel) | HIER share of test pairs: strict/tie/weak | trivial weak-claim error (1-tie)/2 |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
T2 = ["", "## 2. Why the published protocol fails: predicted risk r + pi(1-r) vs realised (alpha = .2)", "",
      "| dataset | judge | r (non-tie error, accepted) | pi (ties among accepted) | tie share overall | predicted | realised |", "|---|---|---|---|---|---|---|"]
T3 = ["", "## 3. Conformal prediction sets over {cand0, cand1, tie} (LAC, head with panel)", "",
      "| dataset | judge | 1-alpha | coverage | mean size | {k} | {k, tie} | {tie} | other sets |", "|---|---|---|---|---|---|---|---|---|"]
agg = collections.defaultdict(list)
for kind, dss in SETS.items():
    for ds in dss:
        I = items(ds)
        js = VLM + (LLM if kind == "text" else [])
        P = {j: per_judge(ds, j, I) for j in js}
        P = {j: v for j, v in P.items() if len(v) > 100}
        if not P:
            continue
        common = sorted(set.intersection(*[set(v) for v in P.values()]))
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in common]); n = len(y)
        panel_pt = np.mean([[P[j][i][1][2] for i in common] for j in P], 0)
        panel_p = np.mean([[P[j][i][0] for i in common] for j in P], 0)
        for j in P:
            p = np.array([P[j][i][0] for i in common]); Q = np.array([P[j][i][1] for i in common]); flip = np.array([P[j][i][2] for i in common])
            u = H(p); yb = np.where(p > .5, 0, 1); eb = (yb != y).astype(float)
            F0 = np.c_[np.log(Q + 1e-9), u, np.abs(p - .5), flip]
            F1 = np.c_[F0, panel_pt, np.abs(panel_p - .5), H(panel_p)]
            rng = np.random.default_rng(0)
            res = collections.defaultdict(list)
            for _ in range(a.splits):
                pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
                cb = pm[: n // 2]; nt = cb[y[cb] != 2]
                for al in AL:
                    lam = thr(u[nt], eb[nt], al); acc = u[te] <= lam; e2 = np.where(y == 2, 1.0, eb)
                    res[("drop", al)].append((acc.mean(), e2[te][acc].mean() if acc.any() else 0.0))
                    if al == .2:
                        r_ = eb[te][acc & (y[te] != 2)].mean() if (acc & (y[te] != 2)).any() else 0.0
                        pi = (y[te][acc] == 2).mean() if acc.any() else 0.0
                        res["pred"].append((r_, pi, r_ + pi * (1 - r_), e2[te][acc].mean() if acc.any() else 0.0))
                for tag, F in (("", F0), ("+panel", F1)):
                    m = LogisticRegression(max_iter=3000, C=1.0).fit(F[fit], y[fit])
                    Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(F)
                    k = Pr[:, :2].argmax(1); other = 1 - k
                    pk = Pr[np.arange(n), k]; po = Pr[np.arange(n), other]
                    # candidate claims: (score = estimated error prob, error indicator, type)
                    s_strict, e_strict = 1 - pk, (y != k).astype(float)
                    c3 = Pr.argmax(1); s_three, e_three = 1 - Pr.max(1), (y != c3).astype(float)
                    s_weak, e_weak = po, (y == other).astype(float)
                    # mixed: per item the claim with the lowest estimated error among strict(k), tie, weak(k); ties in
                    # estimated error are broken towards the more informative claim (strict > tie > weak)
                    S = np.c_[1 - pk, 1 - Pr[:, 2], po + 1e-6]; ch = S.argmin(1)
                    s_mix = S[np.arange(n), ch]
                    e_mix = np.where(ch == 0, y != k, np.where(ch == 1, y != 2, y == other)).astype(float)
                    if tag:
                        Ss = np.c_[1 - pk, 1 - Pr[:, 2], po]; Es = np.c_[y != k, y != 2, y == other].astype(float)
                        for al in AL:
                            lam = hier_lam(Ss[cal], Es[cal], al); acc, ch_, e_ = hier_apply(Ss[te], Es[te], lam)
                            res[("hier", al)].append((acc.mean(), e_[acc].mean() if acc.any() else 0.0))
                            res[("hierch", al)].append([(ch_ == c).mean() for c in (0, 1, 2)])
                    for al in AL:
                        for key, s, e in (("strict", s_strict, e_strict), ("three", s_three, e_three), ("weak", s_weak, e_weak), ("mixed", s_mix, e_mix)):
                            if tag and key != "mixed":
                                continue
                            lam = thr(s[cal], e[cal], al); acc = s[te] <= lam
                            res[(key + tag, al)].append((acc.mean(), e[te][acc].mean() if acc.any() else 0.0))
                            if key == "mixed" and not tag:
                                res[("mixch", al)].append([(ch[te][acc] == c).mean() if acc.any() else 0 for c in (0, 1, 2)])
                    if tag:   # LAC prediction sets
                        for al in AL:
                            sc = 1 - Pr[cal, y[cal]]; qh = np.quantile(sc, min(1, np.ceil((len(cal) + 1) * (1 - al)) / len(cal)), method="higher")
                            Sset = (1 - Pr[te]) <= qh
                            cov = Sset[np.arange(len(te)), y[te]].mean(); size = Sset.sum(1)
                            single_k = ((size == 1) & ~Sset[:, 2]).mean(); kt = ((size == 2) & Sset[:, 2]).mean(); tonly = ((size == 1) & Sset[:, 2]).mean()
                            res[("set", al)].append((cov, size.mean(), single_k, kt, tonly, 1 - single_k - kt - tonly))
            for al in AL:
                f = lambda key: "{:.2f} / {:.3f}".format(*np.mean(res[(key, al)], 0))
                mc = np.mean(res[("mixch", al)], 0)
                T1.append(f"| {kind} | {ds} | {j}{' (LLM)' if j in LLM else ''} | {n} | {(y == 2).mean():.2f} | {al} | {f('drop')} | {f('strict')} | "
                          f"{f('three')} | {f('weak')} | {f('mixed')} | {f('mixed+panel')} | {mc[0]:.2f}/{mc[1]:.2f}/{mc[2]:.2f} | {f('hier')} | "
                          + "/".join(f"{x:.2f}" for x in np.mean(res[("hierch", al)], 0)) + f" | {(1 - (y == 2).mean()) / 2:.3f} |")
                for key in ("drop", "strict", "three", "weak", "mixed", "mixed+panel", "hier"):
                    agg[(kind, key, al)].append(np.mean(res[(key, al)], 0)[0])
                sv = np.mean(res[("set", al)], 0)
                T3.append(f"| {ds} | {j} | {1 - al:.1f} | {sv[0]:.3f} | {sv[1]:.2f} | {sv[2]:.2f} | {sv[3]:.2f} | {sv[4]:.2f} | {sv[5]:.2f} |")
            pr = np.mean(res["pred"], 0)
            T2.append(f"| {ds} | {j} | {pr[0]:.3f} | {pr[1]:.3f} | {(y == 2).mean():.2f} | {pr[2]:.3f} | {pr[3]:.3f} |")
            print(ds, j, flush=True)
S = ["", "## Mean coverage by kind", "", "| kind | alpha | bin-drop | strict | three-way | weak | mixed | mixed+panel | hier |", "|---|---|---|---|---|---|---|---|---|"]
for kind in SETS:
    for al in AL:
        if (kind, "strict", al) in agg:
            S.append(f"| {kind} | {al} | " + " | ".join(f"{np.mean(agg[(kind, k, al)]):.2f}" for k in ("drop", "strict", "three", "weak", "mixed", "mixed+panel", "hier")) + " |")
txt = "\n".join(L + T1 + S + T2 + T3) + "\n"
open(a.out, "w").write(txt); print("\n".join(S + T2))
