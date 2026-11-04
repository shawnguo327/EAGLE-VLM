"""Routing / ensemble experiments aimed at a stronger main effect (same splits / metrics as chain.py):
  base   G2 / G3 with one judge's pairwise head (reference)
  E1     jury: the pairwise head features of ALL available VLM judges concatenated (5 judges)
  E1+pt  jury + pointwise features of the judges that have 'score' runs
  E2     both-bad routing (GenAI only, where 'both bad' is a human label): the tie claim is split into
         'both unacceptable' (true iff human vote = both bad) and 'equal' (true iff tie vote); a 4-class head on
         pairwise + pointwise features; claims strict / both-bad / equal / weak; graded LTT and joint LTT
  E3     recalibrated tie channel: separate 2-class tie detector (tie vs not) on [log P(T) + pointwise], its score
         replaces 1 - P_head(tie) for the tie claim
Metrics: acceptance / set size / decisive / tie-type claims / strict error / risk / violation, alpha .1 .2 .3.
usage (judgebench dir): python route_exp.py --out paper_tables/route_exp.md"""
import argparse, collections, os, sys
import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from label_noise_lib import per_judge, eval_items, ltt

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/route_exp.md")
ap.add_argument("--splits", type=int, default=100)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--sets", default="mllm_judge,mtbench_human,arena55k,genai_arena/generation,genai_arena/edition,pickapic_u,imagerewarddb")
a = ap.parse_args()
AL = [0.1, 0.2, 0.3]; GRID = np.linspace(0, 1, 201)[1:]; D = np.arange(1, 10)
VLM = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b"]


def claims(S, lam):
    ok = S <= lam; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def pj(S, E, lam, al):
    ok, ch = claims(S, lam); ps = [0.0]
    for c in range(S.shape[1]):
        mk = ok & (ch == c)
        if mk.any():
            ps.append(binom.cdf(int(E[mk, c].sum()), int(mk.sum()), al))
    return max(ps) if ok.any() else 1.0


def p1(S, E, lam, al):
    ok, ch = claims(S, lam); n = int(ok.sum())
    return 1.0 if n == 0 else binom.cdf(int(E[np.arange(len(E)), np.maximum(ch, 0)][ok].sum()), n, al)


def fst(Sc, Ec, Sf, Ef, al, joint):
    f = pj if joint else p1
    s0 = int(np.argmin([f(Sf, Ef, g, al) for g in GRID])); best = -np.inf
    for g in GRID[s0:]:
        if f(Sc, Ec, g, al) <= a.delta:
            best = g
        else:
            break
    return best


def feats(ds, j, ids, I):
    P = per_judge(ds, j, I)
    if not all(i in P for i in ids):
        return None
    p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
    return np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), fl]


def pfeats(ds, j, ids):
    R = load(ds, j, conds={"score"}); rows = []
    for i in ids:
        r = R.get(i, {})
        if not (("score", "A") in r and ("score", "B") in r):
            return None
        row = []
        for o in "AB":
            q = np.array([r[("score", o)].get(str(d), 0) for d in D]); q = q / q.sum(); mu = q @ D
            row += [mu, q @ (D - mu) ** 2, q[:3].sum()]
        rows.append(row + [min(row[0], row[3]), max(row[0], row[3])])
    return np.array(rows)


def graded(Pr, lab):
    """Pr columns: P(cand0), P(cand1), then one column per tie-type claim; lab: 0, 1, or 2+ (tie types)"""
    n = len(lab); k = Pr[:, :2].argmax(1); o = 1 - k
    cols = [1 - Pr[np.arange(n), k]] + [1 - Pr[:, c] for c in range(2, Pr.shape[1])] + [Pr[np.arange(n), o]]
    errs = [lab != k] + [lab != c for c in range(2, Pr.shape[1])] + [lab == o]
    return np.c_[tuple(cols)], np.c_[tuple(errs)].astype(float)


def evaluate(F, lab, nclass, tie_detector=None):
    n = len(lab); rng = np.random.default_rng(0); R = collections.defaultdict(list)
    for _ in range(a.splits):
        pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
        m = LogisticRegression(max_iter=5000).fit(F[fit], lab[fit]); Pr = np.zeros((n, nclass)); Pr[:, m.classes_] = m.predict_proba(F)
        if tie_detector is not None:
            t = LogisticRegression(max_iter=5000).fit(tie_detector[fit], (lab[fit] >= 2).astype(int))
            Pr[:, 2] = t.predict_proba(tie_detector)[:, 1]
        S, E = graded(Pr, lab); W = S.shape[1] - 1                                   # last column = weak claim
        for al in AL:
            for key, joint in (("graded", False), ("joint", True)):
                ok, ch = claims(S[te], fst(S[cal], E[cal], S[fit], E[fit], al, joint))
                e = E[te][np.arange(len(te)), np.maximum(ch, 0)]
                size = np.where(~ok, 3, np.where(ch == W, 2, 1)); st = ok & (ch == 0)
                r = e[ok].mean() if ok.any() else np.nan
                R[(key, al)].append((ok.mean(), size.mean(), st.mean(), (ok & (ch > 0) & (ch < W)).mean(), e[st].sum(), st.sum(),
                                     r, float(r > al) if ok.any() else np.nan))
    out = {}
    for k, Z in R.items():
        Z = np.array(Z, float)
        out[k] = (Z[:, 0].mean(), Z[:, 1].mean(), Z[:, 2].mean(), Z[:, 3].mean(), Z[:, 4].sum() / Z[:, 5].sum() if Z[:, 5].sum() else np.nan,
                  np.nanmean(Z[:, 6]) if (~np.isnan(Z[:, 6])).any() else np.nan, np.nanmean(Z[:, 7]) if (~np.isnan(Z[:, 7])).any() else np.nan)
    return out


f = lambda v: "none" if np.isnan(v) else f"{v:.2f}"
cell = lambda r: f"{r[0]:.2f} / {r[1]:.2f} / {r[2]:.2f} / {r[3]:.2f} / {f(r[4])} / {f(r[5])} / {f(r[6])}"
L = ["# Routing / ensemble experiments", "", f"{a.splits} splits.  Cell = acceptance / set size / decisive / tie-type claims / "
     "strict error / risk / violation.  graded = overall LTT; joint = every claim level <= alpha.", "",
     "| set | variant | rule | " + " | ".join(f"alpha {x}" for x in AL) + " |", "|" + "---|" * (3 + len(AL))]
for spec in a.sets.split(","):
    ds, sub = (spec.split("/") + [None])[:2]
    I = eval_items(ds, items(ds)); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    P0 = per_judge(ds, "qwen3vl8b", I); ids = sorted(i for i in P0 if all(True for _ in [0]))
    FJ = {j: feats(ds, j, ids, I) for j in VLM}; FJ = {j: v for j, v in FJ.items() if v is not None}
    PW = {j: pfeats(ds, j, ids) for j in FJ}; PW = {j: v for j, v in PW.items() if v is not None}
    y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
    runs = []
    for j in ("qwen3vl8b", "qwen3vl32b"):
        if j in FJ:
            runs.append((f"base: {j}", FJ[j], y, 3, None))
    if len(FJ) >= 3:
        Fj = np.concatenate([FJ[j] for j in sorted(FJ)], 1); runs.append((f"E1 jury ({len(FJ)} judges)", Fj, y, 3, None))
        if PW:
            Fp = np.concatenate([Fj] + [PW[j] for j in sorted(PW)], 1)
            runs.append((f"E1+pt jury + pointwise ({len(PW)})", Fp, y, 3, None))
            runs.append(("E3 jury + separate tie detector [log P(T) all judges + pointwise]", Fj, y, 3,
                         np.concatenate([np.c_[tuple(FJ[j][:, 2] for j in sorted(FJ))]] + [PW[j] for j in sorted(PW)], 1)))
    if ds == "genai_arena" and PW:
        vt = np.array([I[i]["meta"]["vote_type"] for i in ids]); y4 = np.where(vt == "bothbad_vote", 2, np.where(vt == "tievote", 3, y))
        if (y4 == 3).sum() >= 20:
            Fp = np.concatenate([np.concatenate([FJ[j] for j in sorted(FJ)], 1)] + [PW[j] for j in sorted(PW)], 1)
            runs.append(("E2 both-bad routing: strict / both-bad / equal / weak (jury + pointwise)", Fp, y4, 4, None))
            j = "qwen3vl8b"
            runs.append((f"E2 single judge {j}: strict / both-bad / equal / weak", np.c_[FJ[j], PW[j]] if j in PW else FJ[j], y4, 4, None))
    for name, F, lab, nc, td in runs:
        r = evaluate(F, lab, nc, td)
        for rule in ("graded", "joint"):
            L.append(f"| {spec} | {name} | {rule} | " + " | ".join(cell(r[(rule, al)]) for al in AL) + " |"); print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
