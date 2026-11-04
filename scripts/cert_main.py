"""Main table: certified judging on real data WITH human ties (full GenAI-Arena, natural-rate Arena 55k).
Splits: 25% fit (3-class head) / 25% calibrate / 50% test; S random splits.  Labels y in {0, 1, 2 = tie}.
Rules (all on the same judge outputs):
  drop          published protocol: BPE score, SCOPE threshold calibrated on NON-TIE pairs, deployed on all pairs
                (a tie counts as an error for the 'k better' verdict)
  strict-SCOPE  'k better' from the 3-class head, calibrated on all pairs (tie = error), SCOPE linearised threshold
  strict-LTT    same, Learn-then-Test
  graded-SCOPE  strict > tie > weak claims, one threshold on the head's estimated error (exploratory)
  graded-LTT    same claims, Learn-then-Test
Learn-then-Test (Angelopoulos et al.; fixed-sequence testing): lambda grid ordered from conservative to liberal; for each
lambda on a fixed grid, H0: selective risk > alpha, p-value = BinomCDF(errors among accepted cal items; n_accepted,
alpha).  The sequence starts at the grid value with the smallest p-value on the FIT split (independent of calibration)
and walks to more liberal thresholds while p <= delta, stopping at the first non-rejection.  Valid for ANY fixed order, so the non-monotone graded path is
fine: P(selective risk of the chosen lambda <= alpha) >= 1 - delta.
Cells: mean test coverage / mean test selective risk / violation rate (share of splits with test risk > alpha);
claim mix for graded-LTT; trivial weak-claim error (1 - tie) / 2 (a weak claim in a random direction).
usage (judgebench dir): python cert_main.py --out paper_tables/cert_main.md"""
import argparse, os, sys, collections
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, H
from label_noise_lib import per_judge, thr, _claims, hier_lam, eval_items
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/cert_main.md")
ap.add_argument("--splits", type=int, default=100)
ap.add_argument("--alphas", default="0.1,0.2,0.3")
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--grid", type=int, default=200)
ap.add_argument("--sets", default="")
ap.add_argument("--min-share", type=float, default=0.95, help="skip a judge that covers less of the set (a partial run "
                "can be a biased sample, e.g. a tie-free cap)")
ap.add_argument("--baselines", type=int, default=1)
ap.add_argument("--fit", type=float, default=0.25, help="share of items for the 3-class head; calibration gets the rest of the first half")
a = ap.parse_args()
AL = [float(x) for x in a.alphas.split(",")]
SETS = [("text", "mtbench_human", None), ("text", "arena55k", None), ("resp", "mllm_judge", None),
        ("img", "genai_arena", None), ("img", "genai_arena", "generation"), ("img", "genai_arena", "edition"),
        ("img", "pickapic_u", None), ("img", "imagerewarddb", None),
        ("appx", "pickapic", None), ("appx", "editreward", None), ("appx", "pp", None)]
# pickapic_u = PickScore's standard splits (test_unique + validation_unique); pickapic = 3 random test shards (appendix)       # appx: few ties (EditReward) / street-view (Place Pulse)
if a.sets:
    SETS = [s for s in SETS if s[1] in a.sets.split(",")]
VLM = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b"]; LLM = ["qwen3_8b", "qwen25_7b"]


GRID = None


def _pval(Ss, Es, lam, al):
    acc, ch = _claims(Ss, lam); n = int(acc.sum())
    if n == 0:
        return 1.0
    k = int(Es[np.arange(len(Es)), np.maximum(ch, 0)][acc].sum())
    return binom.cdf(k, n, al)


def ltt(Ss, Es, al, delta, Sf=None, Ef=None):
    """fixed-sequence LTT.  Grid: fixed (data-independent) thresholds on the estimated error in [0, 1].  The sequence
    starts at lam0 = the grid value with the smallest binomial p-value on the FIT split (independent of the calibration
    split, so the order is fixed w.r.t. calibration data) and walks towards more liberal thresholds; stop at the first
    non-rejection.  Claim = most informative candidate claim with estimated error <= lam."""
    G = GRID
    s0 = int(np.argmin([_pval(Sf, Ef, lam, al) for lam in G])) if Sf is not None else 0
    best = -np.inf
    for lam in G[s0:]:
        if _pval(Ss, Es, lam, al) <= delta:
            best = lam
        else:
            break
    return best


def run(F, u, p, y, splits, Q=None):
    n = len(y); rng = np.random.default_rng(0); R = collections.defaultdict(list)
    yb = np.where(p > .5, 0, 1); eb = (yb != y).astype(float)
    for _ in range(splits):
        nf = int(round(a.fit * n)); pm = rng.permutation(n); fit, cal, te = pm[:nf], pm[nf: n // 2], pm[n // 2:]
        cb = pm[: n // 2]; nt = cb[y[cb] != 2]
        m = LogisticRegression(max_iter=3000).fit(F[fit], y[fit])
        Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(F)
        k = Pr[:, :2].argmax(1); other = 1 - k; pk = Pr[np.arange(n), k]; po = Pr[np.arange(n), other]
        S1 = (1 - pk)[:, None]; E1 = (y != k).astype(float)[:, None]
        S3 = np.c_[1 - pk, 1 - Pr[:, 2], po]; E3 = np.c_[y != k, y != 2, y == other].astype(float)
        if a.baselines:
            # B-tieerr: the minimal fix of the published protocol - same BPE score and forced-choice verdict, but
            #           calibrated on all pairs with a tie counted as an error
            # B-tri:    ask the judge with a tie option and take its ternary verdict (argmax of the order-averaged A/B/T
            #           probabilities), score = 1 - max prob; a tie verdict is correct only on a human tie
            # B-head3:  ternary verdict of our 3-class head (argmax incl. tie): isolates the value of weak claims
            # B-graded-raw: strict / tie / weak claims from the raw A/B/T probabilities without the head
            vt = Q.argmax(1); vh = Pr.argmax(1)
            B = [("B-tieerr", u[:, None], eb[:, None]),
                 ("B-tri", (1 - Q.max(1))[:, None], (vt != y).astype(float)[:, None]),
                 ("B-head3", (1 - Pr.max(1))[:, None], (vh != y).astype(float)[:, None])]
            kr = Q[:, :2].argmax(1); orr = 1 - kr
            SR = np.c_[1 - Q[np.arange(n), kr], 1 - Q[:, 2], Q[np.arange(n), orr]]
            ER = np.c_[y != kr, y != 2, y == orr].astype(float)

        def rec(key, al, acc, err, ch=None):
            r = err[acc].mean() if acc.any() else np.nan          # empty acceptance: risk undefined, not 0
            R[(key, al)].append((acc.mean(), r, float(r > al) if acc.any() else np.nan, float(not acc.any())))
            if ch is not None:
                R[(key + "mix", al)].append([(ch[acc] == c).mean() if acc.any() else 0 for c in (0, 1, 2)])

        for al in AL:
            lam = thr(u[nt], eb[nt], al); acc = u[te] <= lam
            rec("drop", al, acc, np.where(y[te] == 2, 1.0, eb[te]))
            lam = thr(S1[cal, 0], E1[cal, 0], al); rec("strict-SCOPE", al, S1[te, 0] <= lam, E1[te, 0])
            lam = ltt(S1[cal], E1[cal], al, a.delta, S1[fit], E1[fit]); rec("strict-LTT", al, S1[te, 0] <= lam, E1[te, 0])
            for key, lam in (("graded-SCOPE", hier_lam(S3[cal], E3[cal], al)), ("graded-LTT", ltt(S3[cal], E3[cal], al, a.delta, S3[fit], E3[fit]))):
                acc, ch = _claims(S3[te], lam)
                rec(key, al, acc, E3[te][np.arange(len(te)), np.maximum(ch, 0)], ch)
            if not a.baselines:
                continue
            # ---- baselines, all calibrated with Learn-then-Test on the same fit / cal / test split ----
            for key, S, E in B:
                lam = ltt(S[cal], E[cal], al, a.delta, S[fit], E[fit]); rec(key, al, S[te, 0] <= lam, E[te, 0])
            lam = ltt(SR[cal], ER[cal], al, a.delta, SR[fit], ER[fit]); acc, ch = _claims(SR[te], lam)
            rec("B-graded-raw", al, acc, ER[te][np.arange(len(te)), np.maximum(ch, 0)])
    return R


GRID = np.linspace(0, 1, a.grid + 1)[1:]
KEYS = ["drop", "strict-SCOPE", "strict-LTT", "graded-SCOPE", "graded-LTT"] + \
    (["B-tieerr", "B-tri", "B-head3", "B-graded-raw"] if a.baselines else [])
L = ["# Certified judging on real data with human ties (main table)", "",
     f"{a.splits} splits ({a.fit:.0%} fit / {.5 - a.fit:.0%} cal / 50% test); LTT delta = {a.delta}.  Cells = coverage / selective risk / "
     "violation rate (share of splits with test risk > alpha), risk and violation over splits with a non-empty accepted "
     "set; '(x% empty)' = share of splits accepting nothing.  LTT bounds the POPULATION selective risk w.p. 1-delta; the "
     "empirical violation rate on a finite test half can exceed delta when few items are accepted.  drop = published protocol (calibrated without ties, "
     "deployed with them).  mix = graded-LTT accepted claims strict / tie / weak.  triv = error of a weak claim in a random "
     "direction, (1 - tie)/2: graded claims are only non-trivial for alpha < triv.", "",
     "| kind | set | judge | n | tie | alpha | " + " | ".join(KEYS) + " | graded-LTT mix | triv |", "|" + "---|" * (8 + len(KEYS))]
agg = collections.defaultdict(list)
for kind, ds, sub in SETS:
    I = eval_items(ds, items(ds)); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    for j in VLM + (LLM if kind == "text" else []):
        P = per_judge(ds, j, I)
        ids = sorted(P)
        if len(ids) < 200 or len(ids) < a.min_share * len(I):
            if ids:
                print(f"[skip] {ds}/{sub} {j}: {len(ids)}/{len(I)} items judged", file=sys.stderr)
            continue
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
        u = H(p); F = np.c_[np.log(Q + 1e-9), u, np.abs(p - .5), fl]
        R = run(F, u, p, y, a.splits, Q); tie = (y == 2).mean()
        name = ds + (f"/{sub}" if sub else "")
        for al in AL:
            cells = []
            for key in KEYS:
                Z = np.array(R[(key, al)]); c = Z[:, 0].mean(); r, v = np.nanmean(Z[:, 1]) if (Z[:, 3] < 1).any() else np.nan, \
                    np.nanmean(Z[:, 2]) if (Z[:, 3] < 1).any() else np.nan
                cells.append(f"{c:.2f} / {r:.3f} / {v:.2f} ({Z[:, 3].mean():.0%} empty)" if Z[:, 3].mean() > 0 else f"{c:.2f} / {r:.3f} / {v:.2f}")
                if sub is None:                                   # genai_arena counted once (overall), subsets reported only
                    agg[(kind, "LLM" if j in LLM else ("VLM-32B" if j == "qwen3vl32b" else "VLM-7-9B"), key, al)].append(c)
            mx = np.mean(R[("graded-LTTmix", al)], 0)
            L.append(f"| {kind} | {name} | {j}{' (LLM)' if j in LLM else ''} | {len(ids)} | {tie:.2f} | {al} | " + " | ".join(cells)
                     + f" | {mx[0]:.2f}/{mx[1]:.2f}/{mx[2]:.2f} | {(1 - tie) / 2:.3f} |")
            print(L[-1], flush=True)
L += ["", "## Mean coverage by kind and judge type (genai_arena counted once, overall set only)", "",
      "| kind | judges | alpha | " + " | ".join(KEYS) + " |", "|---|---|---|" + "---|" * len(KEYS)]
for kind in ("text", "resp", "img", "appx"):
    for jt in ("VLM-7-9B", "VLM-32B", "LLM"):
        for al in AL:
            if (kind, jt, KEYS[0], al) in agg:
                L.append(f"| {kind} | {jt} | {al} | " + " | ".join(f"{np.mean(agg[(kind, jt, k, al)]):.2f}" for k in KEYS) + " |")
open(a.out, "w").write("\n".join(L) + "\n")
