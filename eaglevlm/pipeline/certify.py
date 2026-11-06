"""Certification experiment ('certify'): published tie-dropping protocols -> our graded claims, on the same splits.
Ported unchanged from scripts/chain.py (P1..G3) and scripts/c3_diag.py (G4, G5, significance-corrected violation).
Every method is deployed on ALL test pairs (ties included); a tie is an error for a 'k better' claim, a non-tie is an
error for a 'tie' claim, and only the opposite verdict is an error for a weak 'k not worse' claim.
Methods:
  P1  SCOPE as published (BPE of order-averaged forced choice; calibration WITHOUT ties; SCOPE linearised threshold)
  P1s same, single order          P2  SCOPE, ties kept in calibration       P3  strict + LTT (BPE), ties kept
  P4  strict + LTT with all our signal components and the 3-class calibrator
  TC  tie calibration (Deutsch et al., EMNLP 2023, "Ties Matter"): 'tie' when |p - .5| <= eps, otherwise the forced-choice
      winner; eps maximises three-way accuracy on the calibration split; a claim for every pair, no guarantee
  T1  Trust-or-Escalate as published (4 simulated annotators; calibration without ties)   T2  ToE, ties kept
  G0  graded, no calibrator       G1  graded + calibrator, single-order features      G2  graded + calibrator (ours)
  G3  graded, every claim level <= alpha (intersection-union)
  G4  graded, overall AND strict-claim risk <= alpha, one threshold
  G5  graded, overall AND strict-claim risk <= alpha, separate strict / tie-weak thresholds (Bonferroni over 11 strict
      values, one of which switches strict claims off)
  G3 / G4 / G5 (and router 'every level'): a claim level that can be issued but is absent from calibration is uncertified
  G2pw / G5pw  as G2 / G5 with the judge's pointwise (single-candidate 1-9 score) features added to the calibrator
  P4pw  compute-matched control: P4 (strict claims only) with the same pointwise-augmented calibrator
  CPL / CPA  split-conformal prediction sets over {A, B, T} from the same 3-class calibrator as ours: LAC (Sadinle et al.,
      JASA 2019; score 1 - p(y)) and APS (Romano et al., NeurIPS 2020; cumulative mass, non-randomised).  They guarantee
      marginal miscoverage <= alpha over ALL pairs (a full set counts as covered); here a full set is an abstention and
      the risk is measured among the other sets, as for every method (an empty set is an error)
  PWT  pointwise grading (single-answer scores, as in MT-Bench's single-answer mode): the candidate with the higher
      expected 1-9 score wins, tie if |difference| <= eps tuned on the calibration split; a claim for every pair, no guarantee
  PWL  the same pointwise difference as a confidence score for strict claims, certified with LTT (ties count as errors)
  Same-target baselines (same LTT selective-risk certificate as G2; review W8):
  G2s  singleton claims only ({A}, {B}, {T}; no weak level) from the same calibrator - isolates the weak claims
  G2b  graded claims from a calibrator that sees only the forced-choice (binary) signals, no tie-prompt p(T)
  SL / SA  generic set predictors over {A, B, T} certified with the same LTT: LAC sets {y: p(y) >= 1 - q} and APS sets
       (labels by decreasing probability until the mass reaches q); a full set is an abstention, an empty set too
  T3   simulated-annotator signal keeping the tie: the persona prompts' A/B/T probabilities (averaged over personas and
       orders) turned into graded claims and certified with the same LTT (on benchmarks where the personas were run)
  judge "jury:<m1>+<m2>+...": the calibrator sees the features of all member judges (pairs judged by every member);
  P-type baselines use the members' averaged probabilities; T methods are not defined for a jury
Optional cfg 'unit: source1' (review W5, main source-level variant): split by source as below, and calibrate on ONE pair
drawn at random from every calibration source, so the calibration pairs are independent and the ordinary binomial LTT is
exact for the risk of a pair drawn uniformly from a uniformly drawn new source (every method, incl. G3-G5, unchanged).
Optional cfg 'unit: source' (review W5): pairs that share a source (same question / prompt and the same context images)
form a group; fit / calibration / test are split by source, and every LTT p-value is computed from per-source losses
(Hoeffding-Bentkus on L_g = e_g - alpha a_g + alpha in [0, 1], with a_g / e_g the accepted / accepted-and-wrong fractions of
source g), which stays valid under any dependence within a source.  Certified object: E[e_g] / E[a_g] <= alpha, the risk of
a pair drawn uniformly from a uniformly drawn new source (summary column risk_source); the pair-level unit is the default.
Optional cfg 'group: meta.<key>' (e.g. meta.question_index for Best-of-N candidate pairs): the fit / calibration / test
split is drawn over groups, so all pairs of a group fall in the same part; without it the split is over pairs, as in the
original scripts.
Summary columns risk / guarantee_success / significant_violation are over the splits with at least one answer (as
before); splits_answered, pooled_risk, guarantee_success_all and significant_all make the denominators explicit.
Per split and alpha one CSV row: acceptance, mean set size, decisive share, strict errors / strict claims, accepted,
errors, risk, violated (risk > alpha), significant (one-sided binomial p < .05 that risk > alpha), empty, and the human ties
among the accepted pairs and among the strict claims (summary: tie_share_accepted / tie_share_strict, pooled over splits;
compare with human_tie, the tie rate of all pairs - Proposition 1: a strict claim on a tie is always an error)."""
import collections
import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression

GRID = np.linspace(0, 1, 201)[1:]
LS = np.r_[-np.inf, np.linspace(.04, .40, 10)]       # strict-claim thresholds of G5; -inf = strict claims switched off
SA = ["tri_sa1", "tri_sa2", "tri_sa3", "tri_sa4"]
H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))


# ---------------- thresholds (scripts/label_noise_lib.py)
def thr(s, e, al):
    """SCOPE linearised threshold, tie-aware (checked at the last position of each distinct score)"""
    o = np.argsort(s, kind="stable"); ss = s[o]; cs = np.cumsum(e[o] - al)
    last = np.r_[ss[1:] != ss[:-1], True]; ok = np.where((cs <= -1) & last)[0]
    return ss[ok.max()] if len(ok) else -np.inf


def claims(S, lam):
    ok = S <= lam; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def hb_pvalue(mu, m, R):
    """Hoeffding-Bentkus p-value for H0: E[L] >= R from m independent losses in [0, 1] with mean mu (Bates et al., 2021)"""
    if m == 0 or mu >= R:
        return 1.0
    h1 = (mu * np.log(mu / R) if mu > 0 else 0.0) + (1 - mu) * np.log((1 - mu) / (1 - R))
    return float(min(1.0, np.exp(-m * h1), np.e * binom.cdf(np.ceil(m * mu), m, R)))


def group_pvalue(acc, err, g, al):
    """p-value of 'source-level selective risk > alpha' from per-source accepted / error fractions (any within-source
    dependence allowed): L_g = e_g - alpha a_g + alpha in [0, 1], H0: E[L] >= alpha"""
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    ag = np.bincount(inv, acc.astype(float)) / cnt; eg = np.bincount(inv, err * acc) / cnt
    if ag.sum() == 0:
        return 1.0
    return hb_pvalue(float((eg - al * ag + al).mean()), len(cnt), al)


def _pval(S, E, lam, al, g=None):
    acc, ch = claims(S, lam); n = int(acc.sum())
    if n == 0:
        return 1.0
    e = E[np.arange(len(E)), np.maximum(ch, 0)]
    if g is not None:
        return group_pvalue(acc, e, g, al)
    return binom.cdf(int(e[acc].sum()), n, al)


def ltt(S, E, al, delta, Sf=None, Ef=None, grid=GRID, gc=None, gf=None):
    """fixed-sequence Learn-then-Test; start = grid value with the smallest p-value on the FIT split.  gc / gf: source ids
    of the calibration / fit rows for the source-level unit (None = pairs)"""
    s0 = int(np.argmin([_pval(Sf, Ef, lam, al, gf) for lam in grid])) if Sf is not None else 0
    best = -np.inf
    for lam in grid[s0:]:
        if _pval(S, E, lam, al, gc) <= delta:
            best = lam
        else:
            break
    return best


def lvl_p(E, mk, c, al):
    """p-value of 'the risk of claim level c exceeds alpha'.  A level the policy can issue but that does not occur in the
    calibration sample is NOT certified (p = 1): an unobserved level is no evidence of a safe one.  Only a level the
    policy switches off by construction may be left out (review W4; theory notes, binomial_pvalue)."""
    n = int(mk.sum())
    return float(binom.cdf(int(E[mk, c].sum()), n, al)) if n else 1.0


def pjoint(S, E, lam, al):
    ok, ch = claims(S, lam)
    if not ok.any():
        return 1.0
    return max(lvl_p(E, ok & (ch == c), c, al) for c in range(S.shape[1]))


def pstrict(S, E, lam, al):
    ok, ch = claims(S, lam)
    if not ok.any():
        return 1.0
    e = E[np.arange(len(E)), np.maximum(ch, 0)]
    return max(float(binom.cdf(int(e[ok].sum()), int(ok.sum()), al)), lvl_p(E, ok & (ch == 0), 0, al))


def ltt_p(Sc, Ec, Sf, Ef, al, delta, pf):
    s0 = int(np.argmin([pf(Sf, Ef, g, al) for g in GRID])); best = -np.inf
    for g in GRID[s0:]:
        if pf(Sc, Ec, g, al) <= delta:
            best = g
        else:
            break
    return best


def claims2(S, ls, lw):
    ok = np.c_[S[:, 0] <= ls, S[:, 1] <= lw, S[:, 2] <= lw]; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def p2(S, E, ls, lw, al):
    ok, ch = claims2(S, ls, lw)
    if not ok.any():
        return 1.0
    e = E[np.arange(len(E)), np.maximum(ch, 0)]
    ps = [float(binom.cdf(int(e[ok].sum()), int(ok.sum()), al))]
    if ls > -np.inf:                   # strict claims are possible under this policy -> they must be certified too
        ps.append(lvl_p(E, S[:, 0] <= ls, 0, al))
    return max(ps)


def ltt_two(Sc, Ec, Sf, Ef, al, delta):
    best, bacc = (-np.inf, -np.inf), -1.0
    for ls in LS:
        s0 = int(np.argmin([p2(Sf, Ef, ls, g, al) for g in GRID])); lw = None
        for g in GRID[s0:]:
            if p2(Sc, Ec, ls, g, al) <= delta / len(LS):
                lw = g
            else:
                break
        if lw is not None:
            acc = claims2(Sc, ls, lw)[0].mean()
            if acc > bacc:
                best, bacc = (ls, lw), acc
    return best


# ---------------- signals
def signals(store, ds, j, I):
    """per id: forced-choice P(cand 0 better) per order, tie-prompt probs per order (aligned to cand 0 / cand 1 / T)"""
    R = store.load(ds, j, conds={"bin", "tri"}); out = {}
    for i, r in R.items():
        if i not in I or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
        f1 = b1["A"] / (b1["A"] + b1["B"]); f2 = b2["B"] / (b2["A"] + b2["B"])
        q1 = np.array([t1["A"], t1["B"], t1["T"]]); q2 = np.array([t2["B"], t2["A"], t2["T"]])
        out[i] = (f1, f2, q1 / q1.sum(), q2 / q2.sum())
    return out


def persona(store, ds, j, ids):
    R = store.load(ds, j, conds=set(SA)); out = {}
    for i in ids:
        r = R.get(i, {})
        if all((c, o) in r for c in SA for o in ("AB", "BA")):
            ps = []
            for c in SA:
                t1, t2 = r[(c, "AB")], r[(c, "BA")]
                pa = (t1["A"] + t2["B"]) / 2; pb = (t1["B"] + t2["A"]) / 2; ps.append(pa / (pa + pb))
            out[i] = float(np.mean(ps))
    return out


def pointwise_feats(store, ds, j, ids):
    """[E_A, E_B, Var_A, Var_B, P(s<=3)_A, P(s<=3)_B, min E, max E, |E_A - E_B|] from the 1-9 score prompt, or None;
    NaN where a score is missing (imputed inside each split from the fit rows)"""
    from eaglevlm.pipeline.router import pointwise
    M = pointwise(store, ds, j, ids)
    if M is None:
        return None
    return np.c_[M[:, 0], M[:, 3], M[:, 1], M[:, 4], M[:, 2], M[:, 5], np.minimum(M[:, 0], M[:, 3]), np.maximum(M[:, 0], M[:, 3]),
                 np.abs(M[:, 0] - M[:, 3])]


def head(F, y, fit):
    m = LogisticRegression(max_iter=5000).fit(F[fit], y[fit]); Pr = np.zeros((len(y), 3)); Pr[:, m.classes_] = m.predict_proba(F)
    return Pr


def graded_scores(Pr, y):
    n = len(y); k = Pr[:, :2].argmax(1); o = 1 - k
    S = np.c_[1 - Pr[np.arange(n), k], 1 - Pr[:, 2], Pr[np.arange(n), o]]; E = np.c_[y != k, y != 2, y == o].astype(float)
    return S, E


def row(ok, ch, err, al, tie=None, g=None):
    """claim types ch: 0 strict {k}; 1 tie {T}; 2 weak {k,T}; 3 'not a tie' {A,B} (only from generic set predictors).
    g: source ids of the test rows (source-level unit) -> source-weighted test risk sum_g e_g / sum_g a_g"""
    size = np.where(~ok, 3, np.where(ch >= 2, 2, 1)); st = ok & (ch == 0); k = int(ok.sum()); e = int(err[ok].sum()) if k else 0
    risk = e / k if k else np.nan
    d = dict(acceptance=float(ok.mean()), set_size=float(size.mean()), decisive=float(st.mean()), strict_errors=int(err[st].sum()),
             strict_claims=int(st.sum()), accepted=k, errors=e, risk=risk, violated=(float(risk > al) if k else np.nan),
             significant=(float(binom.sf(e - 1, k, al) < .05) if k else np.nan), empty=float(not ok.any()),
             tie_claims=float((ok & (ch == 1)).mean()), weak_claims=float((ok & (ch == 2)).mean()),
             nondirectional=float((ok & (ch == 3)).mean()))
    if g is not None:
        _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
        ag = np.bincount(inv, ok.astype(float)) / cnt; eg = np.bincount(inv, (err * ok).astype(float)) / cnt
        rs = float(eg.sum() / ag.sum()) if ag.sum() else np.nan
        d.update(risk_source=rs, sources=int(len(cnt)), violated_source=(float(rs > al) if ag.sum() else np.nan))
    if tie is not None:                     # human ties among the accepted pairs / among the strict claims (Proposition 1)
        d.update(accepted_ties=int((ok & tie).sum()), strict_on_ties=int((st & tie).sum()))
    return d


def source_key(item):
    """pairs sharing a source: the same question / prompt and the same context images (review W5)"""
    return item["question"] + "\x1f" + "\x1f".join(item["context"])


def make_sets(Pr, q, kind):
    """generic label sets over {0: A, 1: B, 2: T}: LAC {y: p(y) >= 1 - q}; APS = labels by decreasing p until mass >= q"""
    if kind == "lac":
        return Pr >= 1 - q
    o = np.argsort(-Pr, axis=1, kind="stable"); cs = np.take_along_axis(Pr, o, 1).cumsum(1)
    k = (cs < q).sum(1) + 1; S = np.zeros_like(Pr, bool)
    for j in range(3):
        S[np.arange(len(Pr)), o[:, j]] = j < k
    return S


def set_claims(Pr, y, q, kind):
    """(accepted, claim type, error) for row(): a set of size 1 or 2 is a claim, a full or empty set an abstention;
    q = -inf (nothing certified) abstains everywhere"""
    S = make_sets(Pr, q, kind) if q > -np.inf else np.ones_like(Pr, bool); size = S.sum(1); ok = (size > 0) & (size < 3)
    ch = np.where(size == 2, np.where(S[:, 2], 2, 3), np.where(S[:, 2] & (size == 1), 1, 0))
    return ok, ch, (~S[np.arange(len(y)), y]).astype(float)


def ltt_sets(Pr, y, cal, fit, al, delta, kind, grid=GRID, gi=None):
    """the same fixed-sequence LTT as ltt() over the set parameter q, walking from conservative to informative policies:
    here a larger q means larger sets (more abstention), so the sequence runs from the fit-chosen start towards smaller q"""
    def pv(idx, q):
        ok, _, e = set_claims(Pr[idx], y[idx], q, kind); n = int(ok.sum())
        if gi is not None:
            return group_pvalue(ok, e, gi[idx], al) if n else 1.0
        return float(binom.cdf(int(e[ok].sum()), n, al)) if n else 1.0
    grid = grid[::-1]
    s0 = int(np.argmin([pv(fit, q) for q in grid])); best = -np.inf
    for q in grid[s0:]:
        if pv(cal, q) <= delta:
            best = q
        else:
            break
    return best


def persona3(store, ds, j, ids):
    """persona prompts' A/B/T probabilities, aligned to cand 0 / cand 1 / tie, averaged over personas and both orders"""
    R = store.load(ds, j, conds=set(SA)); out = {}
    for i in ids:
        r = R.get(i, {})
        if all((c, o) in r for c in SA for o in ("AB", "BA")):
            q = np.mean([[(r[(c, "AB")]["A"] + r[(c, "BA")]["B"]) / 2, (r[(c, "AB")]["B"] + r[(c, "BA")]["A"]) / 2,
                          (r[(c, "AB")]["T"] + r[(c, "BA")]["T"]) / 2] for c in SA], 0)
            out[i] = q / q.sum()
    return out


def conformal_sets(Pr, y, cal, te, al, score):
    """split-conformal sets over the classes {0: A, 1: B, 2: T} (LAC or non-randomised APS) -> (accepted, claim type,
    error) on the test split in the format of row(): accepted = not the full set; error = label not in the set"""
    n = len(cal); q_level = min(1.0, np.ceil((n + 1) * (1 - al)) / n)
    if score == "lac":
        qhat = np.quantile(1 - Pr[cal, y[cal]], q_level, method="higher")
        S = (1 - Pr[te]) <= qhat
    else:
        o = np.argsort(-Pr, axis=1, kind="stable"); cs = np.take_along_axis(Pr, o, 1).cumsum(1)
        rank = np.argsort(o, axis=1)                                  # position of each class in the sorted order
        sc = cs[np.arange(len(Pr)), rank[np.arange(len(Pr)), y]]       # mass up to and including the true class
        qhat = np.quantile(sc[cal], q_level, method="higher")
        k = (cs[te] < qhat).sum(1) + 1                                 # smallest prefix whose mass reaches qhat
        S = np.zeros((len(te), 3), bool)
        for j in range(3):
            S[np.arange(len(te)), o[te, j]] = j < k
    size = S.sum(1); ok = size < 3
    ch = np.where(size == 2, np.where(S[:, 2], 2, 3), np.where(S[:, 2] & (size == 1), 1, 0))
    err = (~S[np.arange(len(te)), y[te]]).astype(float)
    return ok, ch, err


def run(cfg, store, log):
    methods = cfg["methods"]; AL = cfg["alphas"]; delta = cfg["delta"]; fit_f, cal_f = cfg["split"]["fit"], cfg["split"]["cal"]
    out = []
    for spec in cfg["datasets"]:
        ds, I = __import__("eaglevlm.pipeline.io", fromlist=["dataset_items"]).dataset_items(store, spec)
        for j in cfg["judges"]:
            members = j.split(":", 1)[1].split("+") if j.startswith("jury:") else [j]
            SG = {m: signals(store, ds, m, I) for m in members}
            ids = sorted(set.intersection(*[set(v) for v in SG.values()]))
            if len(ids) < 200 or len(ids) < cfg.get("min_coverage", .95) * len(I):
                log(f"skip {spec} {j}: {len(ids)} / {len(I)} items with both prompts in both orders"); continue
            n = len(ids); y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids]); Ty = y == 2
            grp = cfg.get("group"); unit_src = cfg.get("unit", "pair") == "source"; unit_one = cfg.get("unit", "pair") == "source1"
            if grp or unit_src or unit_one:
                keys = [str(I[i]["meta"][grp.split(".", 1)[1]]) if grp else source_key(I[i]) for i in ids]
                ug, gi = np.unique(np.array(keys), return_inverse=True)
            else:
                gi = None
            parts_full, parts_one, parts_bin, P_, Q_, F1_, Q1_, C_ = [], [], [], [], [], [], [], []
            for m in members:
                Sg = SG[m]
                f1 = np.array([Sg[i][0] for i in ids]); f2 = np.array([Sg[i][1] for i in ids])
                q1 = np.array([Sg[i][2] for i in ids]); q2 = np.array([Sg[i][3] for i in ids])
                p = (f1 + f2) / 2; Q = (q1 + q2) / 2; cons = ((f1 > .5) == (f2 > .5)).astype(float)
                parts_full.append(np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), cons])
                parts_one.append(np.c_[np.log(q1 + 1e-9), H(f1), np.abs(f1 - .5)])
                parts_bin.append(np.c_[H(p), np.abs(p - .5), cons, p])
                P_.append(p); Q_.append(Q); F1_.append(f1); Q1_.append(q1); C_.append(cons)
            if len(members) == 1:
                F_full, F_one, F_bin = parts_full[0], parts_one[0], parts_bin[0]
            else:
                F_full, F_one, F_bin = np.concatenate(parts_full, 1), np.concatenate(parts_one, 1), np.concatenate(parts_bin, 1)
                p, Q, f1, q1 = np.mean(P_, 0), np.mean(Q_, 0), np.mean(F1_, 0), np.mean(Q1_, 0)
            PWF = None
            if {"G2pw", "G5pw", "P4pw", "PWT", "PWL"} & set(methods):
                pws = [pointwise_feats(store, ds, m, ids) for m in members]; pws = [x for x in pws if x is not None]
                PWF = np.nanmean(pws, 0) if pws else None     # may hold NaN (missing score): imputed per split from fit rows
            fc = np.where(p > .5, 0, 1); eb = (fc != y).astype(float); et = np.where(y == 2, 1.0, eb)
            fc1 = np.where(f1 > .5, 0, 1); et1 = np.where(y == 2, 1.0, (fc1 != y).astype(float))
            nsa = 0; QSA = None
            if "T3" in methods and len(members) == 1:
                QS = persona3(store, ds, j, ids)
                if sum(i in QS for i in ids) >= .95 * n:
                    QSA = np.array([QS.get(i, np.full(3, 1 / 3)) for i in ids])
            if {"T1", "T2"} & set(methods) and len(members) == 1:
                PS = persona(store, ds, j, ids); has = np.array([i in PS for i in ids]); nsa = int(has.sum())
                if nsa >= 200:
                    ps = np.array([PS.get(i, .5) for i in ids]); sa_s = 1 - np.maximum(ps, 1 - ps)
                    sa_fc = np.where(ps > .5, 0, 1); sa_et = np.where(y == 2, 1.0, (sa_fc != y).astype(float))
            rng = np.random.default_rng(cfg.get("seed", 0))
            for sp in range(cfg["n_splits"]):
                if grp or unit_src or unit_one:
                    rk = np.empty(len(ug), int); rk[rng.permutation(len(ug))] = np.arange(len(ug)); r = rk[gi]
                    g1, g2 = int(round(fit_f * len(ug))), int(round((fit_f + cal_f) * len(ug)))
                    fit, cal, te = np.where(r < g1)[0], np.where((r >= g1) & (r < g2))[0], np.where(r >= g2)[0]
                    if unit_one:                # one random pair per calibration source -> independent calibration units
                        pc = rng.permutation(cal); _, first = np.unique(gi[pc], return_index=True); cal = np.sort(pc[first])
                else:
                    pm = rng.permutation(n)
                    a1, a2 = int(round(fit_f * n)), int(round((fit_f + cal_f) * n))
                    if fit_f == .25 and cal_f == .25:
                        a1, a2 = n // 4, n // 2                     # identical to the original scripts
                    fit, cal, te = pm[:a1], pm[a1:a2], pm[a2:]
                calnt = cal[y[cal] != 2]
                GC, GF, TG = (gi[cal], gi[fit], gi[te]) if unit_src else (None, None, gi[te] if unit_one else None)
                Prf = head(F_full, y, fit); S3, E3 = graded_scores(Prf, y)
                need1 = "G1" in methods
                if need1:
                    Pr1 = head(F_one, y, fit); S31, E31 = graded_scores(Pr1, y)
                S0, E0 = graded_scores(Q, y)
                if "G2b" in methods:
                    Prb = head(F_bin, y, fit); S3b, E3b = graded_scores(Prb, y)
                if QSA is not None:
                    S3t, E3t = graded_scores(QSA, y)
                if PWF is not None:
                    PWFs = PWF.copy(); bad = np.isnan(PWFs)
                    if bad.any():                   # review W6: imputation fixed on the fit rows only
                        PWFs[bad] = np.nanmean(PWF[fit], 0)[np.where(bad)[1]]
                    Prp = head(np.c_[F_full, PWFs], y, fit); S3p, E3p = graded_scores(Prp, y)
                u, u1 = H(p), H(f1); zeros = np.zeros(len(te), int)
                for al in AL:
                    res = {}
                    if "P1" in methods:
                        res["P1"] = row(u[te] <= thr(u[calnt], eb[calnt], al), zeros, et[te], al, Ty[te], TG)
                    if "P1s" in methods:
                        res["P1s"] = row(u1[te] <= thr(u1[calnt], (fc1 != y)[calnt].astype(float), al), zeros, et1[te], al, Ty[te], TG)
                    if "P2" in methods:
                        res["P2"] = row(u[te] <= thr(u[cal], et[cal], al), zeros, et[te], al, Ty[te], TG)
                    if "P3" in methods:
                        lam = ltt(u[cal][:, None], et[cal][:, None], al, delta, u[fit][:, None], et[fit][:, None], gc=GC, gf=GF)
                        res["P3"] = row(u[te] <= lam, zeros, et[te], al, Ty[te], TG)
                    if "P4" in methods:
                        lam = ltt(S3[cal][:, [0]], E3[cal][:, [0]], al, delta, S3[fit][:, [0]], E3[fit][:, [0]], gc=GC, gf=GF)
                        res["P4"] = row(S3[te, 0] <= lam, zeros, E3[te, 0], al, Ty[te], TG)
                    for key, score in (("CPL", "lac"), ("CPA", "aps")):
                        if key in methods:
                            res[key] = row(*conformal_sets(Prf, y, cal, te, al, score), al, Ty[te], TG)
                    if PWF is not None and {"PWT", "PWL"} & set(methods):
                        dE = PWFs[:, 0] - PWFs[:, 1]; pw_fc = np.where(dE > 0, 0, 1)    # ties of the means -> B (rare)
                        if "PWT" in methods:
                            if al == AL[0]:
                                mg2 = np.abs(dE); oc2 = cal[np.argsort(mg2[cal], kind="stable")]; mc2 = mg2[oc2]
                                ht2 = np.cumsum(y[oc2] == 2); hs2 = np.cumsum(pw_fc[oc2] == y[oc2]); tot2 = hs2[-1]
                                last2 = np.r_[mc2[1:] != mc2[:-1], True]; acc2 = (ht2 + tot2 - hs2)[last2]
                                eps2 = mc2[last2][int(np.argmax(acc2))] if acc2.max() > tot2 else -1.0
                                pwt_ch = np.where(mg2[te] <= eps2, 1, 0)
                                pwt_err = np.where(pwt_ch == 1, y[te] != 2, pw_fc[te] != y[te]).astype(float)
                            res["PWT"] = row(np.ones(len(te), bool), pwt_ch, pwt_err, al, Ty[te], TG)
                        if "PWL" in methods:
                            sp = 1 - np.minimum(np.abs(dE), 8) / 8; ep = np.where(y == 2, 1.0, (pw_fc != y).astype(float))
                            lam = ltt(sp[cal][:, None], ep[cal][:, None], al, delta, sp[fit][:, None], ep[fit][:, None], gc=GC, gf=GF)
                            res["PWL"] = row(sp[te] <= lam, zeros, ep[te], al, Ty[te], TG)
                    if "TC" in methods:     # tie calibration (Deutsch et al., EMNLP 2023): 'tie' if |p - .5| <= eps, else the
                        if al == AL[0]:     # forced-choice winner; eps maximises 3-way accuracy on the calibration split.
                            mg = np.abs(p - .5); oc = cal[np.argsort(mg[cal], kind="stable")]; mc = mg[oc]   # every pair gets a
                            ht = np.cumsum(y[oc] == 2); hs = np.cumsum(fc[oc] == y[oc]); tot = hs[-1]        # claim; no guarantee
                            last = np.r_[mc[1:] != mc[:-1], True]; acc = (ht + tot - hs)[last]
                            eps = mc[last][int(np.argmax(acc))] if acc.max() > tot else -1.0
                            tc_ch = np.where(mg[te] <= eps, 1, 0)
                            tc_err = np.where(tc_ch == 1, y[te] != 2, fc[te] != y[te]).astype(float)
                        res["TC"] = row(np.ones(len(te), bool), tc_ch, tc_err, al, Ty[te], TG)
                    if nsa >= 200:
                        ct, tt, ft = cal[has[cal]], te[has[te]], fit[has[fit]]
                        ctn = ct[y[ct] != 2]
                        for key, cc in (("T1", ctn), ("T2", ct)):
                            if key not in methods:
                                continue
                            ec = np.where(y[cc] == 2, 1.0, (sa_fc[cc] != y[cc]).astype(float))
                            ef = np.where(y[ft] == 2, 1.0, (sa_fc[ft] != y[ft]).astype(float)) if key == "T2" else (sa_fc[ft] != y[ft]).astype(float)
                            lam = ltt(sa_s[cc][:, None], ec[:, None], al, delta, sa_s[ft][:, None], ef[:, None], gc=(gi[cc] if unit_src else None), gf=(gi[ft] if unit_src else None))
                            res[key] = row(sa_s[tt] <= lam, np.zeros(len(tt), int), sa_et[tt], al, Ty[tt], (gi[tt] if unit_src else None))
                    for key, S, E in (("G0", S0, E0), ("G1", S31 if need1 else None, E31 if need1 else None), ("G2", S3, E3)):
                        if key in methods:
                            ok, ch = claims(S[te], ltt(S[cal], E[cal], al, delta, S[fit], E[fit], gc=GC, gf=GF))
                            res[key] = row(ok, ch, E[te][np.arange(len(te)), np.maximum(ch, 0)], al, Ty[te], TG)
                    for key, S, E in (("G2b", S3b if "G2b" in methods else None, E3b if "G2b" in methods else None),
                                      ("T3", S3t if QSA is not None else None, E3t if QSA is not None else None)):
                        if key in methods and S is not None:
                            ok, ch = claims(S[te], ltt(S[cal], E[cal], al, delta, S[fit], E[fit], gc=GC, gf=GF))
                            res[key] = row(ok, ch, E[te][np.arange(len(te)), np.maximum(ch, 0)], al, Ty[te], TG)
                    if "G2s" in methods:            # singletons only: drop the weak level
                        S2, E2 = S3[:, :2], E3[:, :2]
                        ok, ch = claims(S2[te], ltt(S2[cal], E2[cal], al, delta, S2[fit], E2[fit], gc=GC, gf=GF))
                        res["G2s"] = row(ok, ch, E2[te][np.arange(len(te)), np.maximum(ch, 0)], al, Ty[te], TG)
                    for key, kind in (("SL", "lac"), ("SA", "aps")):
                        if key in methods:
                            q = ltt_sets(Prf, y, cal, fit, al, delta, kind, gi=(gi if unit_src else None))
                            res[key] = row(*set_claims(Prf[te], y[te], q, kind), al, Ty[te], TG)
                    if "G3" in methods:
                        ok, ch = claims(S3[te], ltt_p(S3[cal], E3[cal], S3[fit], E3[fit], al, delta, pjoint))
                        res["G3"] = row(ok, ch, E3[te][np.arange(len(te)), np.maximum(ch, 0)], al, Ty[te], TG)
                    if "G4" in methods:
                        ok, ch = claims(S3[te], ltt_p(S3[cal], E3[cal], S3[fit], E3[fit], al, delta, pstrict))
                        res["G4"] = row(ok, ch, E3[te][np.arange(len(te)), np.maximum(ch, 0)], al, Ty[te], TG)
                    if "G5" in methods:
                        ls, lw = ltt_two(S3[cal], E3[cal], S3[fit], E3[fit], al, delta); ok, ch = claims2(S3[te], ls, lw)
                        res["G5"] = row(ok, ch, E3[te][np.arange(len(te)), np.maximum(ch, 0)], al, Ty[te], TG)
                    if PWF is not None and "P4pw" in methods:      # compute-matched control for G2pw / G5pw: the same
                        lam = ltt(S3p[cal][:, [0]], E3p[cal][:, [0]], al, delta, S3p[fit][:, [0]], E3p[fit][:, [0]], gc=GC, gf=GF)   # features,
                        res["P4pw"] = row(S3p[te, 0] <= lam, zeros, E3p[te, 0], al, Ty[te], TG)                       # strict claims only
                    if PWF is not None and "G2pw" in methods:
                        ok, ch = claims(S3p[te], ltt(S3p[cal], E3p[cal], al, delta, S3p[fit], E3p[fit], gc=GC, gf=GF))
                        res["G2pw"] = row(ok, ch, E3p[te][np.arange(len(te)), np.maximum(ch, 0)], al, Ty[te], TG)
                    if PWF is not None and "G5pw" in methods:
                        ls, lw = ltt_two(S3p[cal], E3p[cal], S3p[fit], E3p[fit], al, delta); ok, ch = claims2(S3p[te], ls, lw)
                        res["G5pw"] = row(ok, ch, E3p[te][np.arange(len(te)), np.maximum(ch, 0)], al, Ty[te], TG)
                    for mth, r in res.items():
                        out.append(dict(dataset=spec, judge=j, n=n, human_tie=float((y == 2).mean()),
                                        n_persona=(nsa if mth in ("T1", "T2") else None), method=mth, alpha=al, split=sp, **r))
            log(f"done {spec} {j} (n={n})")
    return out


def summarize(rows):
    """per dataset x judge x method x alpha: means over splits; strict-claim error pooled over splits (as the original)"""
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["judge"], r["method"], r["alpha"])].append(r)
    out = []
    for (ds, j, m, al), R in g.items():
        nn = lambda k: [r[k] for r in R if r[k] == r[k] and r[k] is not None]
        sc = sum(r["strict_claims"] for r in R); risk = nn("risk")
        out.append(dict(dataset=ds, judge=j, method=m, alpha=al, n=R[0]["n"], human_tie=R[0]["human_tie"], n_persona=R[0]["n_persona"],
                        splits=len(R), acceptance=np.mean([r["acceptance"] for r in R]), decisive=np.mean([r["decisive"] for r in R]),
                        set_size=np.mean([r["set_size"] for r in R]),
                        strict_error=(sum(r["strict_errors"] for r in R) / sc if sc else np.nan),
                        risk=(np.mean(risk) if risk else np.nan), risk_sd=(np.std(risk) if risk else np.nan),
                        human_agreement=(1 - np.mean(risk) if risk else np.nan),
                        violation=(np.mean(nn("violated")) if risk else np.nan),
                        guarantee_success=(1 - np.mean(nn("violated")) if risk else np.nan),
                        significant_violation=(np.mean(nn("significant")) if risk else np.nan),
                        empty_splits=np.mean([r["empty"] for r in R]),
                        # review W2/W3: answered splits, pooled ratio (SCOPE's ratio-of-expectations object) and event
                        # rates over ALL splits, where a split without any answer counts as no violation
                        tie_claim_share=np.mean([r.get("tie_claims", np.nan) for r in R]),
                        weak_share=np.mean([r.get("weak_claims", np.nan) for r in R]),
                        nondirectional_share=np.mean([r.get("nondirectional", np.nan) for r in R]),
                        risk_source=(np.nanmean([r.get("risk_source", np.nan) for r in R]) if any(r.get("risk_source", np.nan) == r.get("risk_source", np.nan) for r in R) else np.nan),
                        guarantee_success_source=(1 - float(np.mean([r.get("violated_source") == 1.0 for r in R])) if any("risk_source" in r for r in R) else np.nan),
                        splits_answered=sum(r["accepted"] > 0 for r in R),
                        accepted_median=float(np.median([r["accepted"] for r in R])),
                        pooled_risk=(sum(r["errors"] for r in R) / sum(r["accepted"] for r in R) if sum(r["accepted"] for r in R) else np.nan),
                        violation_all=float(np.mean([r["violated"] == 1.0 for r in R])),
                        guarantee_success_all=float(1 - np.mean([r["violated"] == 1.0 for r in R])),
                        significant_all=float(np.mean([r["significant"] == 1.0 for r in R])),
                        tie_share_accepted=(sum(r["accepted_ties"] for r in R) / max(sum(r["accepted"] for r in R), 1)
                                            if all(r.get("accepted_ties") is not None for r in R) else np.nan),
                        tie_share_strict=(sum(r["strict_on_ties"] for r in R) / sc
                                          if sc and all(r.get("strict_on_ties") is not None for r in R) else np.nan)))
    return out
