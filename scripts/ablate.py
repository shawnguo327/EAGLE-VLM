"""Component ablation of the certified graded judge.  One component changed at a time; reference configuration =
signals 'base' + logistic head + full graded output + overall LTT.  Same 25/25/50 splits for every variant.
  signals      bin       [BPE, |p-.5|, logit p]                      (forced-choice prompt only)
               tri       [log P0, log P1, log PT]                     (tie-option prompt only)
               bin+tri   both, no order-consistency flag
               base      bin+tri + order-consistency                  (reference)
               +pt-mean  base + pointwise E[sA], E[sB], min, max, diff       (image / response sets with 'score' runs)
               +pt-unc   +pt-mean + pointwise Var, entropy, P(s<=3) of each candidate
               +pt-trustjudge  base + TrustJudge distribution-sensitive pointwise score (logit/T softmax, T=.2) features
  baseline     TrustJudge likelihood-aware pairwise aggregation (no head), delta-margin tie, LTT on the margin
  calibrator   raw       no learning: (P0, P1, PT) = order-averaged tie-prompt probabilities
               temp      one temperature on the tie-prompt log-probs, fitted by NLL on the fit split
               logistic  multinomial logistic regression on the signals (reference)
               gbm       gradient-boosted trees (sklearn HistGradientBoosting), a non-linear head
  output       strict | strict+tie | strict+weak | full (reference)
  certify      LTT (overall, reference) | per-level LTT (each level at alpha, delta/3, disjoint folds) |
               joint LTT (intersection-union p-value: max over levels; all levels <= alpha simultaneously)
Metrics on test pairs: mean set size (strict/tie 1, weak 2, abstain 3), decisive (strict) share, tie-claim share,
error of strict claims, overall selective risk.  alpha = .2 (and .3), delta = .1.
usage (judgebench dir): python ablate.py --out paper_tables/ablate.md"""
import argparse, collections, os, sys
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from label_noise_lib import per_judge, eval_items, ltt, _claims

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/ablate.md")
ap.add_argument("--splits", type=int, default=50)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--alphas", default="0.2,0.3")
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen3vl32b")
ap.add_argument("--sets", default="mllm_judge,mtbench_human,arena55k,genai_arena/generation,genai_arena/edition,pickapic_u,imagerewarddb")
a = ap.parse_args()
AL = [float(x) for x in a.alphas.split(",")]
D = np.arange(1, 10)


def point(ds, j, ids):
    R = load(ds, j, conds={"score"}); out = {}
    for i in ids:
        r = R.get(i, {})
        if ("score", "A") in r and ("score", "B") in r:
            row = []
            for o in "AB":
                q = np.array([r[("score", o)].get(str(d), 0) for d in D]); q = q / q.sum(); mu = q @ D
                lg = np.log((q + 1e-8) / (1 - q + 1e-8)) / 0.2; w = np.exp(lg - lg.max()); w /= w.sum()
                row.append((mu, q @ (D - mu) ** 2, -(q * np.log(q + 1e-12)).sum(), q[:3].sum(), w @ D))
            out[i] = row
    return out


from scipy.stats import binom
GRID = np.linspace(0, 1, 201)[1:]


def _pjoint(S, E, lam, al):
    """intersection-union p-value for H0: some claim level has selective risk > alpha (levels issuing nothing are vacuous)"""
    ok, ch = _claims(S, lam); ps = [0.0]
    for c in range(3):
        mk = ok & (ch == c)
        if mk.any():
            ps.append(binom.cdf(int(E[mk, c].sum()), int(mk.sum()), al))
    return max(ps) if ok.any() else 1.0


def joint(S3, E3, fit, cal, te, al):
    """fixed-sequence LTT on the joint p-value: every claim level controlled at alpha simultaneously, w.p. >= 1 - delta"""
    s0 = int(np.argmin([_pjoint(S3[fit], E3[fit], g, al) for g in GRID])); best = -np.inf
    for g in GRID[s0:]:
        if _pjoint(S3[cal], E3[cal], g, al) <= a.delta:
            best = g
        else:
            break
    return _claims(S3[te], best)


def perlevel(S3, E3, fit, cal, te, al):
    folds = np.array_split(cal, 3); lam = [-np.inf] * 3
    for c in range(3):
        f = folds[c]; left = np.ones(len(f), bool); lf = np.ones(len(fit), bool)
        for c0 in range(c):
            left &= ~(S3[f, c0] <= lam[c0]); lf &= ~(S3[fit, c0] <= lam[c0])
        if left.sum() and np.isfinite(S3[f][left][:, c]).any():
            lam[c] = ltt(S3[f][left][:, [c]], E3[f][left][:, [c]], al, a.delta / 3, S3[fit][lf][:, [c]], E3[fit][lf][:, [c]])
    ch = np.full(len(te), -1)
    for c in (2, 1, 0):
        ch[S3[te, c] <= lam[c]] = c
    return ch >= 0, ch


def probs(cal_name, F, Q, y, fit):
    n = len(y)
    if cal_name == "raw":
        return Q.copy()
    if cal_name == "temp":
        L = np.log(Q + 1e-9)
        def nll(t):
            Z = L / t; Z = Z - Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
            return -np.log(P[fit, y[fit]] + 1e-12).mean()
        t = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
        Z = L / t; Z = Z - Z.max(1, keepdims=True); P = np.exp(Z); return P / P.sum(1, keepdims=True)
    m = (LogisticRegression(max_iter=5000) if cal_name == "logistic" else
         HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0))
    m.fit(F[fit], y[fit]); Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(F); return Pr


def evaluate(F, Q, y, cal_name="logistic", output="full", certify="ltt"):
    n = len(y); rng = np.random.default_rng(0); R = collections.defaultdict(list)
    for _ in range(a.splits):
        pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
        Pr = probs(cal_name, F, Q, y, fit)
        k = Pr[:, :2].argmax(1); o = 1 - k
        S3 = np.c_[1 - Pr[np.arange(n), k], 1 - Pr[:, 2], Pr[np.arange(n), o]]
        E3 = np.c_[y != k, y != 2, y == o].astype(float)
        drop = {"strict": [1, 2], "strict+tie": [2], "strict+weak": [1], "full": []}[output]
        S3[:, drop] = np.inf
        for al in AL:
            ok, ch = perlevel(S3, E3, fit, cal, te, al) if certify == "perlevel" else joint(S3, E3, fit, cal, te, al) \
                if certify == "joint" else _claims(
                S3[te], ltt(S3[cal], E3[cal], al, a.delta, S3[fit], E3[fit]))
            e = E3[te][np.arange(len(te)), np.maximum(ch, 0)]
            st = ok & (ch == 0)
            R[al].append((np.where(~ok, 3, np.where(ch == 2, 2, 1)).mean(), st.mean(), (ok & (ch == 1)).mean(),
                          e[st].sum(), st.sum(), e[ok].mean() if ok.any() else np.nan))
    out = {}
    for al in AL:
        Z = np.array(R[al], float)
        out[al] = (Z[:, 0].mean(), Z[:, 1].mean(), Z[:, 2].mean(), Z[:, 3].sum() / Z[:, 4].sum() if Z[:, 4].sum() else np.nan,
                   np.nanmean(Z[:, 5]) if (~np.isnan(Z[:, 5])).any() else np.nan)
    return out


def trustjudge_pair(Q, y):
    """TrustJudge Option B: m = p_order1 + swap(p_order2) (= 2 x our order-averaged tie-prompt probabilities); verdict =
    argmax m, tie if (top1 - top2)/2 < d with d tuned on the fit split for 3-way accuracy; confidence = top1 - top2.
    Then the same LTT (strict and tie claims; no weak claims) on 1 - margin."""
    n = len(y); rng = np.random.default_rng(0); R = collections.defaultdict(list)
    srt = np.sort(Q, 1); margin = srt[:, 2] - srt[:, 1]; am = Q.argmax(1)
    for _ in range(a.splits):
        pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
        ds_ = np.linspace(0, .5, 51)
        acc = [np.mean(np.where(margin[fit] / 2 < d, 2, am[fit]) == y[fit]) for d in ds_]
        d = ds_[int(np.argmax(acc))]; v = np.where(margin / 2 < d, 2, am)
        S1 = (1 - margin)[:, None]; E1 = (v != y).astype(float)[:, None]
        for al in AL:
            lam = ltt(S1[cal], E1[cal], al, a.delta, S1[fit], E1[fit]); ok = S1[te, 0] <= lam
            st = ok & (v[te] != 2); e = E1[te, 0]
            R[al].append((np.where(ok, 1, 3).mean(), st.mean(), (ok & (v[te] == 2)).mean(), e[st].sum(), st.sum(),
                          e[ok].mean() if ok.any() else np.nan))
    out = {}
    for al in AL:
        Z = np.array(R[al], float)
        out[al] = (Z[:, 0].mean(), Z[:, 1].mean(), Z[:, 2].mean(), Z[:, 3].sum() / Z[:, 4].sum() if Z[:, 4].sum() else np.nan,
                   np.nanmean(Z[:, 5]) if (~np.isnan(Z[:, 5])).any() else np.nan)
    return out


fmt = lambda r: "{:.2f} / {:.2f} / {:.2f} / {} / {}".format(r[0], r[1], r[2], "-" if np.isnan(r[3]) else f"{r[3]:.3f}",
                                                         "-" if np.isnan(r[4]) else f"{r[4]:.3f}")
L = ["# Component ablation (one change at a time from the reference: base signals + logistic + full graded + LTT)", "",
     f"{a.splits} splits.  Cells = set size / decisive share / tie-claim share / strict-claim error / overall risk.", "",
     "| set | judge | n | variant | " + " | ".join(f"alpha {x}" for x in AL) + " |", "|" + "---|" * (4 + len(AL))]
for spec in a.sets.split(","):
    ds, sub = (spec.split("/") + [None])[:2]
    I = eval_items(ds, items(ds)); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    for j in a.judges.split(","):
        P = per_judge(ds, j, I); ids = sorted(P)
        if len(ids) < 200 or len(ids) < .95 * len(I):
            continue
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
        lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        sig = {"bin": np.c_[H(p), np.abs(p - .5), lp], "tri": np.log(Q + 1e-9),
               "bin+tri": np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5)], "base": np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), fl]}
        W = point(ds, j, ids)
        if len(W) >= .95 * len(ids):
            keep = np.array([i in W for i in ids]); ids_k = [i for i in ids if i in W]
            y, p, Q, fl = y[keep], p[keep], Q[keep], fl[keep]; sig = {k: v[keep] for k, v in sig.items()}
            M = np.array([[W[i][0][0], W[i][1][0]] for i in ids_k]); U = np.array([[*W[i][0][1:4], *W[i][1][1:4]] for i in ids_k])
            sig["+pt-mean"] = np.c_[sig["base"], M, M.min(1), M.max(1), M[:, 0] - M[:, 1]]
            sig["+pt-unc"] = np.c_[sig["+pt-mean"], U]
            TJ = np.array([[W[i][0][4], W[i][1][4]] for i in ids_k])          # TrustJudge distribution-sensitive score
            sig["+pt-trustjudge"] = np.c_[sig["base"], TJ, TJ.min(1), TJ.max(1), TJ[:, 0] - TJ[:, 1]]
        runs = [(f"signals: {k}", dict(F=v)) for k, v in sig.items()]
        runs += [(f"calibrator: {c}", dict(F=sig["base"], cal_name=c)) for c in ("raw", "temp", "gbm")]
        runs += [(f"output: {o}", dict(F=sig["base"], output=o)) for o in ("strict", "strict+tie", "strict+weak")]
        runs += [("certify: per-level LTT (3 folds)", dict(F=sig["base"], certify="perlevel")),
                 ("certify: joint LTT (every level <= alpha)", dict(F=sig["base"], certify="joint"))]
        if "+pt-unc" in sig:
            runs += [("+pt-unc, joint LTT", dict(F=sig["+pt-unc"], certify="joint"))]
        r = trustjudge_pair(Q, y)
        L.append(f"| {spec} | {j} | {len(y)} | baseline: TrustJudge likelihood-aware (no head) + LTT | " + " | ".join(fmt(r[al]) for al in AL) + " |")
        print(L[-1], flush=True)
        for name, kw in runs:
            r = evaluate(Q=Q, y=y, **kw)
            L.append(f"| {spec} | {j} | {len(y)} | {name} | " + " | ".join(fmt(r[al]) for al in AL) + " |")
            print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
