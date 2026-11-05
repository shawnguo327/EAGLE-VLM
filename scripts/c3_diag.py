"""C3 diagnostics for the graded-claim certification (same features, head and claims as chain.py G2 / G3):
 (1) Is 'violation > delta' a test-sample artefact?  LTT bounds the POPULATION risk of the chosen threshold
     (P(risk > alpha) <= delta); chain.py counts a split as violated when the finite TEST risk exceeds alpha, which also
     happens when the population risk is just below alpha.  Per split we additionally report
       viol_sig  test risk significantly above alpha (one-sided binomial test at .05 on the accepted test pairs)
     and the across-split mean / sd of the test risk.
 (2) Is the every-level guarantee (G3) power-limited?  Same splits with a larger calibration share:
       split 25/25/50 (default) vs 25/50/25 (fit / cal / test).
 (3) Strict-claim error: pooled error of 'k better' claims under G2 and G3.
Cell = acceptance / decisive (strict) / strict err / mean risk (sd) / violation / viol_sig.
usage (judgebench dir): python c3_diag.py --out paper_tables/c3_diag.md"""
import argparse, collections, os, sys
import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from label_noise_lib import eval_items, ltt, _claims

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/c3_diag.md")
ap.add_argument("--splits", type=int, default=100)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--alphas", default="0.1,0.2,0.3")
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen35_9b,qwen25vl7b,qwen3vl32b,internvl35_38b")
ap.add_argument("--splitsets", default="25/25/50,25/50/25")
ap.add_argument("--methods", default="G2,G3")
ap.add_argument("--sets", default="mllm_judge_sp,mllm_judge,genaibench,mtbench_human,arena55k,genai_arena/generation,genai_arena/edition")
a = ap.parse_args()
AL = [float(x) for x in a.alphas.split(",")]
GRID = np.linspace(0, 1, 201)[1:]
SPLITS = {"25/25/50": (.25, .50), "25/50/25": (.25, .75)}      # (end of fit, end of cal) as fractions of n


def signals(ds, j, I):
    R = load(ds, j, conds={"bin", "tri"}); out = {}
    for i, r in R.items():
        if i not in I or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
        f1 = b1["A"] / (b1["A"] + b1["B"]); f2 = b2["B"] / (b2["A"] + b2["B"])
        q1 = np.array([t1["A"], t1["B"], t1["T"]]); q2 = np.array([t2["B"], t2["A"], t2["T"]])
        out[i] = (f1, f2, q1 / q1.sum(), q2 / q2.sum())
    return out


def pjoint(S, E, lam, al):
    ok, ch = _claims(S, lam); ps = [0.0]
    for c in range(3):
        mk = ok & (ch == c)
        if mk.any():
            ps.append(binom.cdf(int(E[mk, c].sum()), int(mk.sum()), al))
    return max(ps) if ok.any() else 1.0


def pstrict(S, E, lam, al):
    """G4: overall risk <= alpha AND strict-claim risk <= alpha (intersection-union over the two)"""
    ok, ch = _claims(S, lam)
    if not ok.any():
        return 1.0
    e = E[np.arange(len(E)), np.maximum(ch, 0)]
    ps = [binom.cdf(int(e[ok].sum()), int(ok.sum()), al)]
    mk = ok & (ch == 0)
    if mk.any():
        ps.append(binom.cdf(int(E[mk, 0].sum()), int(mk.sum()), al))
    return max(ps)


def ltt_joint(Sc, Ec, Sf, Ef, al, pf=None):
    pf = pf or pjoint
    s0 = int(np.argmin([pf(Sf, Ef, g, al) for g in GRID])); best = -np.inf
    for g in GRID[s0:]:
        if pf(Sc, Ec, g, al) <= a.delta:
            best = g
        else:
            break
    return best


LS = np.linspace(.04, .40, 10)          # fixed candidate strict thresholds (chosen before seeing data; Bonferroni over them)


def claims2(S, ls, lw):
    ok = np.c_[S[:, 0] <= ls, S[:, 1] <= lw, S[:, 2] <= lw]; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def p2(S, E, ls, lw, al):
    ok, ch = claims2(S, ls, lw)
    if not ok.any():
        return 1.0
    e = E[np.arange(len(E)), np.maximum(ch, 0)]
    ps = [binom.cdf(int(e[ok].sum()), int(ok.sum()), al)]
    mk = S[:, 0] <= ls
    if mk.any():
        ps.append(binom.cdf(int(E[mk, 0].sum()), int(mk.sum()), al))
    return max(ps)


def ltt_two(Sc, Ec, Sf, Ef, al):
    """G5: separate thresholds for strict claims (ls) and for tie / weak claims (lw).  Hypothesis for (ls, lw):
    overall risk > alpha OR strict-claim risk > alpha (p = max).  Bonferroni over the 10 fixed ls values (delta / 10 each),
    fixed-sequence over lw within each (start chosen on the fit split).  Returns the certified pair with the largest
    calibration acceptance."""
    best, bacc = (-np.inf, -np.inf), -1.0
    for ls in LS:
        s0 = int(np.argmin([p2(Sf, Ef, ls, g, al) for g in GRID])); lw = None
        for g in GRID[s0:]:
            if p2(Sc, Ec, ls, g, al) <= a.delta / len(LS):
                lw = g
            else:
                break
        if lw is not None:
            acc = claims2(Sc, ls, lw)[0].mean()
            if acc > bacc:
                best, bacc = (ls, lw), acc
    return best


R = collections.defaultdict(list)


def add(key, al, ok, ch, err):
    st = ok & (ch == 0); k = int(ok.sum()); e = int(err[ok].sum()) if k else 0
    r = e / k if k else np.nan
    sig = float(binom.sf(e - 1, k, al) < .05) if k else np.nan          # P(X >= e | k, alpha) < .05
    R[key + (al,)].append((ok.mean(), st.mean(), err[st].sum(), st.sum(), r, float(r > al) if k else np.nan, sig))


def cell(key, al):
    Z = np.array(R.get(key + (al,), []), float)
    if not len(Z):
        return "-"
    es = Z[:, 2].sum() / Z[:, 3].sum() if Z[:, 3].sum() else np.nan
    g = lambda v: "-" if v != v else f"{v:.2f}"
    rr = Z[:, 4][~np.isnan(Z[:, 4])]
    return (f"{Z[:, 0].mean():.2f} / {Z[:, 1].mean():.2f} / {g(es)} / "
            + (f"{rr.mean():.3f} ({rr.std():.3f})" if len(rr) else "-")
            + f" / {g(np.nanmean(Z[:, 5]) if len(rr) else np.nan)} / {g(np.nanmean(Z[:, 6]) if len(rr) else np.nan)}")


rows = []
for spec in a.sets.split(","):
    ds, sub = (spec.split("/") + [None])[:2]
    I = eval_items(ds, items(ds, rest=ds not in ("mllm_judge_sp", "genaibench")))
    I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    for j in a.judges.split(","):
        Sg = signals(ds, j, I); ids = sorted(Sg)
        if len(ids) < 200 or len(ids) < .95 * len(I):
            continue
        n = len(ids); y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        f1 = np.array([Sg[i][0] for i in ids]); f2 = np.array([Sg[i][1] for i in ids])
        Q = (np.array([Sg[i][2] for i in ids]) + np.array([Sg[i][3] for i in ids])) / 2; p = (f1 + f2) / 2
        F = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), ((f1 > .5) == (f2 > .5)).astype(float)]
        rng = np.random.default_rng(0)
        for _ in range(a.splits):
            pm = rng.permutation(n)
            for sname, (e1, e2) in [(k, v) for k, v in SPLITS.items() if k in a.splitsets.split(",")]:
                fit, cal, te = pm[: int(e1 * n)], pm[int(e1 * n): int(e2 * n)], pm[int(e2 * n):]
                m = LogisticRegression(max_iter=5000).fit(F[fit], y[fit]); Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(F)
                k = Pr[:, :2].argmax(1); o = 1 - k
                S = np.c_[1 - Pr[np.arange(n), k], 1 - Pr[:, 2], Pr[np.arange(n), o]]; E = np.c_[y != k, y != 2, y == o].astype(float)
                for al in AL:
                    for meth in a.methods.split(","):
                        if meth == "G5":
                            ls, lw = ltt_two(S[cal], E[cal], S[fit], E[fit], al); ok, ch = claims2(S[te], ls, lw)
                            add((spec, j, sname, meth), al, ok, ch, E[te][np.arange(len(te)), np.maximum(ch, 0)]); continue
                        lam = (ltt(S[cal], E[cal], al, a.delta, S[fit], E[fit]) if meth == "G2" else
                               ltt_joint(S[cal], E[cal], S[fit], E[fit], al) if meth == "G3" else
                               ltt_joint(S[cal], E[cal], S[fit], E[fit], al, pstrict))
                        ok, ch = _claims(S[te], lam)
                        add((spec, j, sname, meth), al, ok, ch, E[te][np.arange(len(te)), np.maximum(ch, 0)])
        for sname in [k for k in SPLITS if k in a.splitsets.split(",")]:
            for meth in a.methods.split(","):
                rows.append(f"| {spec} | {j} | {n} | {(y == 2).mean():.2f} | {sname} | {meth} | "
                            + " | ".join(cell((spec, j, sname, meth), al) for al in AL) + " |")
                print(rows[-1], flush=True)
L = ["# C3 diagnostics: violation vs test noise, calibration size for every-level control, strict error", "",
     f"{a.splits} splits, delta {a.delta}.  Cell = acceptance / decisive / strict err / mean test risk (sd) / violation "
     "(test risk > alpha) / viol_sig (test risk significantly > alpha, binomial .05).  G2 = overall risk, G3 = every level, G4 = overall AND strict-claim risk <= alpha (one threshold), G5 = same constraints with separate strict / tie-weak thresholds.", "",
     "| set | judge | n | tie | split | method | " + " | ".join(f"alpha {x}" for x in AL) + " |", "|" + "---|" * (6 + len(AL))] + rows
open(a.out, "w").write("\n".join(L) + "\n")
