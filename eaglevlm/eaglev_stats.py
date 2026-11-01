"""EAGLE-V statistics: confidence scores, risk-controlled selective judging, escalation ladder, and
confidence-routed prediction-powered Bradley-Terry ranking.

Inputs are per-item records built from judge_bench.py outputs (see `load_records`):
  pa      : order-averaged P(cand0 wins)                      (EAGLE p~, SCOPE's p-bar)
  lo      : order-averaged log-odds that cand0 wins           (non-saturating version of the same signal)
  ed      : pointwise expected-score difference (cand0 - cand1), if the score condition was run
  y       : 1 if cand0 truly better, 0 if cand1, None for ties
Everything here is CPU-only and runs on saved outputs.
"""
import json, math, collections
import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------- loading

def _sm(lp):
    m = max(lp.values()); e = {k: math.exp(v - m) for k, v in lp.items()}; z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def load_records(items_path, res_path):
    items = {json.loads(l)["id"]: json.loads(l) for l in open(items_path)}
    R = collections.defaultdict(dict)
    for l in open(res_path):
        d = json.loads(l)
        if "lp" in d:
            R[d["id"]][(d["cond"], d["order"])] = d["lp"]
    out = []
    for i, it in items.items():
        r = R.get(i, {})
        if ("bin", "AB") not in r or ("bin", "BA") not in r:
            continue
        ab, ba = r[("bin", "AB")], r[("bin", "BA")]
        pab, pba = _sm(ab), _sm(ba)
        pa = (pab["A"] + pba["B"]) / 2
        # log-odds of cand0 in each order, from raw log-probs (no renormalisation needed for a difference)
        lo = ((ab["A"] - ab["B"]) + (ba["B"] - ba["A"])) / 2
        rec = dict(id=i, subset=it.get("subset", ""), pa=pa, lo=lo,
                   y=None if it["label"] == "tie" else int(it["label"] == 0), tie=it["label"] == "tie",
                   cons=(pab["A"] > .5) == (pba["B"] > .5))
        if ("score", "A") in r and ("score", "B") in r:
            ea = sum(int(k) * v for k, v in _sm(r[("score", "A")]).items())
            eb = sum(int(k) * v for k, v in _sm(r[("score", "B")]).items())
            rec["ed"] = ea - eb
        out.append(rec)
    return out

# ----------------------------------------------------------------------------- confidence scores

def conf_scores(recs):
    """Return dict name -> array of confidence (higher = more confident) for the pairwise verdict."""
    pa = np.array([r["pa"] for r in recs]); lo = np.array([r["lo"] for r in recs])
    S = {"dp": np.abs(2 * pa - 1),                 # EAGLE Δp  (rank-equivalent to SCOPE's BPE)
         "abs_logodds": np.abs(lo)}                 # same signal in logit space: does not saturate at 1
    if all("ed" in r for r in recs):
        ed = np.array([r["ed"] for r in recs])
        # pointwise agreement: positive if pointwise margin agrees with pairwise direction
        S["dp_x_pw_agree"] = S["abs_logodds"] * np.where(np.sign(ed) == np.sign(lo), 1.0, 0.25)
    return S


def auroc(score, correct):
    score = np.asarray(score, float); correct = np.asarray(correct, int)
    if correct.min() == correct.max():
        return float("nan")
    return float(stats.mannwhitneyu(score[correct == 1], score[correct == 0]).statistic /
                 ((correct == 1).sum() * (correct == 0).sum()))


def saturation(score, q=0.95):
    """Share of items tied at the top of the score's range (fraction within 1e-3 of the max)."""
    s = np.asarray(score, float)
    return float((s >= s.max() - 1e-3 * max(1.0, abs(s.max()))).mean())

# ----------------------------------------------------------------------------- risk control

def cp_upper(k, n, delta):
    """One-sided Clopper-Pearson upper bound on a binomial rate."""
    if n == 0:
        return 1.0
    return 1.0 if k >= n else float(stats.beta.ppf(1 - delta, k + 1, n - k))


def start_cov(n, alpha, delta):
    """First coverage on the fixed path: enough accepted items that zero errors could certify alpha
    (n_min = log(delta)/log(1-alpha)), doubled for slack; depends only on n, alpha, delta (data-independent)."""
    n_min = math.ceil(math.log(delta) / math.log(1 - alpha))
    return min(1.0, max(0.05, 2 * n_min / max(n, 1)))


def calibrate_threshold(conf, err, alpha, delta=0.1, grid=None):
    """SCOPE / Learn-then-Test style: fixed-sequence test from the strictest threshold downward.
    Returns the smallest threshold t such that, for every threshold from the top down to t,
    the CP upper bound of the error among accepted items (conf >= t) is <= alpha.
    Fixed-sequence testing makes the family-wise error <= delta, so with prob >= 1-delta the selective
    risk at the returned threshold is <= alpha (exchangeable calibration/test data)."""
    conf = np.asarray(conf, float); err = np.asarray(err, int)
    if grid is None:  # fixed, data-independent order: coverage 5%, 7.5%, ... 100% (strict -> lenient)
        grid = np.quantile(conf, 1 - np.linspace(start_cov(len(conf), alpha, delta), 1.0, 39))
    best = np.inf
    for t in grid:  # descending
        m = conf >= t
        if m.sum() == 0:
            continue
        if cp_upper(int(err[m].sum()), int(m.sum()), delta) <= alpha:
            best = t
        else:
            break
    return best


def selective_eval(conf, err, alpha, delta=0.1, seed=0, n_splits=50, cal_frac=0.5):
    """Repeated random cal/test splits. Reports mean coverage and realised risk on the test half,
    plus the fraction of splits where test risk exceeded alpha (should be about <= delta)."""
    rng = np.random.default_rng(seed); n = len(conf); cov, risk, viol = [], [], []
    for _ in range(n_splits):
        p = rng.permutation(n); c, t = p[:int(cal_frac * n)], p[int(cal_frac * n):]
        tau = calibrate_threshold(conf[c], err[c], alpha, delta)
        m = conf[t] >= tau
        cov.append(m.mean())
        r = err[t][m].mean() if m.any() else 0.0
        risk.append(r); viol.append(r > alpha)
    return dict(coverage=float(np.mean(cov)), risk=float(np.mean(risk)), violation_rate=float(np.mean(viol)))


def ladder_eval(recs, alpha, delta=0.1, seed=0, n_splits=50, conf_name="abs_logodds"):
    """Escalation ladder with a joint guarantee:
       rung 1 pairwise (accept if conf >= t1) -> rung 2 pointwise (accept if |ed| >= t2) -> rung 3 human.
    Error on human-labelled items is 0 by definition (target = agreement with humans).
    The pair (t1, t2) is searched on a 1-D path (equal-quantile pairs, strict -> lenient) with fixed-sequence
    testing on the CP bound of the *overall* error rate among machine-decided items. Objective: max automation."""
    B = [r for r in recs if r["y"] is not None and "ed" in r]
    if not B:
        return None
    S = conf_scores(B); c1 = S[conf_name]
    lo = np.array([r["lo"] for r in B]); ed = np.array([r["ed"] for r in B]); y = np.array([r["y"] for r in B])
    e1 = ((lo > 0).astype(int) != y).astype(int)
    e2 = np.where(ed == 0, 1, ((ed > 0).astype(int) != y).astype(int))
    c2 = np.abs(ed)
    rng = np.random.default_rng(seed); n = len(B)
    qs = 1 - np.linspace(start_cov(n // 2, alpha, delta), 1.0, 39)
    res = collections.defaultdict(list)
    for _ in range(n_splits):
        p = rng.permutation(n); c, t = p[: n // 2], p[n // 2:]
        chosen = None
        for q in qs:  # path: both thresholds at the same upper quantile, strict -> lenient
            t1 = np.quantile(c1[c], q)
            t2 = np.quantile(c2[c], q)
            a1 = c1[c] >= t1; a2 = (~a1) & (c2[c] >= t2)
            k = int(e1[c][a1].sum() + e2[c][a2].sum()); m = int(a1.sum() + a2.sum())
            if m > 0 and cp_upper(k, m, delta) <= alpha:
                chosen = (t1, t2)
            else:
                break
        if chosen is None:
            chosen = (np.inf, np.inf)
        t1, t2 = chosen
        a1 = c1[t] >= t1; a2 = (~a1) & (c2[t] >= t2); hum = ~(a1 | a2)
        m = a1.sum() + a2.sum()
        risk = (e1[t][a1].sum() + e2[t][a2].sum()) / m if m else 0.0
        res["auto"].append(1 - hum.mean()); res["pair_share"].append(a1.mean()); res["point_share"].append(a2.mean())
        res["risk"].append(risk); res["viol"].append(risk > alpha)
    return {k: float(np.mean(v)) for k, v in res.items()}

# ----------------------------------------------------------------------------- PPI ranking

def bt_fit(pairs, wins, w, n_items, l2=0.1, iters=200):
    """Weighted Bradley-Terry M-estimation. pairs: (m,2) int; wins: P(first beats second) in [0,1] (soft ok);
    w: per-row weights, may be negative (PPI rectifier rows). Damped Newton with a PSD Hessian built from |w| and
    backtracking on the weighted loss, so negative weights cannot make the step diverge. Returns centred strengths."""
    pairs = np.asarray(pairs); wins = np.asarray(wins, float); w = np.asarray(w, float)
    th = np.zeros(n_items)

    def loss(t):
        d = t[pairs[:, 0]] - t[pairs[:, 1]]
        return float(np.sum(w * (np.logaddexp(0, d) - wins * d)) + l2 * t @ t)
    cur = loss(th)
    for _ in range(iters):
        d = th[pairs[:, 0]] - th[pairs[:, 1]]; p = 1 / (1 + np.exp(-d))
        g = np.zeros(n_items); r = w * (wins - p)
        np.add.at(g, pairs[:, 0], r); np.add.at(g, pairs[:, 1], -r); g -= 2 * l2 * th
        H = np.zeros((n_items, n_items)); h = np.abs(w) * p * (1 - p)
        np.add.at(H, (pairs[:, 0], pairs[:, 0]), h); np.add.at(H, (pairs[:, 1], pairs[:, 1]), h)
        np.add.at(H, (pairs[:, 0], pairs[:, 1]), -h); np.add.at(H, (pairs[:, 1], pairs[:, 0]), -h)
        H += (2 * l2 + 1e-6) * np.eye(n_items)
        step = np.linalg.solve(H, g)
        a = 1.0
        while a > 1e-4:
            new = loss(th + a * step)
            if new <= cur:
                break
            a /= 2
        if a <= 1e-4:
            break
        th = th + a * step; delta = cur - new; cur = new
        if np.abs(a * step).max() < 1e-7 or delta < 1e-10:
            break
    return th - th.mean()


def ppi_bt(pairs, yhat, y, labelled, pi, n_items, lam=1.0):
    """Active PPI (Zrnic & Candes 2024) for BT: unlabelled-judge term + inverse-propensity rectifier.
    L(θ) = Σ_i ℓ(θ; ŷ_i)·λ + Σ_i (ξ_i/π_i)[ℓ(θ; y_i) − λ ℓ(θ; ŷ_i)].  Implemented as a weighted BT fit
    with duplicated rows (judge rows weight λ(1 − ξ/π), human rows weight ξ/π)."""
    xi = labelled.astype(float)
    P = np.vstack([pairs, pairs[labelled]])
    W = np.r_[lam * (1 - xi / pi), (1 / pi)[labelled]]
    Y = np.r_[yhat, y[labelled]]
    return bt_fit(P, Y, W, n_items)


def routed_propensity(phat, budget_frac, floor=0.02):
    """Active-inference propensities: π_i ∝ sqrt(p̂(1−p̂)) (the judge's own predicted error std), mean = budget_frac."""
    phat = np.asarray(phat, float)
    u = np.sqrt(np.clip(phat * (1 - phat), 1e-4, None))
    p = u / u.mean() * budget_frac
    p = np.clip(p, floor, 1.0)
    for _ in range(50):  # renormalise mean after clipping
        p = np.clip(p * budget_frac / p.mean(), floor, 1.0)
    return p


def ppi_bt_tuned(pairs, yhat, y, labelled, pi, n_items, lams=(0, .25, .5, .75, 1.0), B=20, seed=0):
    """PPI++-style power tuning: pick λ minimising bootstrap variance of θ over labelled rows (λ=0 is the
    human-only IPW estimator, so tuning can never be much worse than ignoring the judge)."""
    rng = np.random.default_rng(seed); idx = np.flatnonzero(labelled); best = None
    for lam in lams:
        ths = []
        for _ in range(B):
            lab = np.zeros_like(labelled); lab[rng.choice(idx, len(idx))] = True
            ths.append(ppi_bt(pairs, yhat, y, lab, pi, n_items, lam))
        v = np.var(np.array(ths), 0).sum()
        if best is None or v < best[0]:
            best = (v, lam)
    return ppi_bt(pairs, yhat, y, labelled, pi, n_items, best[1]), best[1]


def _ladder_apply(C, E, idx, t):
    """Route items idx through rungs with thresholds t; return (errors, decided, undecided mask, per-rung accept masks)."""
    undecided = np.ones(len(idx), bool); k = m = 0; accs = []
    for r in range(len(C)):
        acc = undecided & (C[r][idx] >= t[r]); k += int(E[r][idx][acc].sum()); m += int(acc.sum())
        undecided &= ~acc; accs.append(acc)
    return k, m, undecided, accs


def ladder_multi(rungs, alpha, delta=0.1, seed=0, n_splits=50, costs=None, rhos=(0.0, 0.5, 1.0, 2.0)):
    """General escalation ladder. rungs: list of (conf, err) arrays aligned over the same items, in escalation order;
    a final human rung (error 0) is implicit. A *path family* is indexed by a mixing ratio rho_r >= 0 for every rung
    after the first: at step q of the path, rung 0 accepts its top (1-q) quantile and rung r its top min(1, rho_r(1-q))
    quantile (rho_r = 0 switches rung r off). Every path is walked by fixed-sequence testing on the Clopper-Pearson
    bound of the error among machine-decided items at level delta/|family| (Bonferroni over paths, so the guarantee
    holds for whichever path is then picked); the path with the lowest calibration cost (or highest automation) wins.
    costs[r] = cost of *reaching* rung r; human cost is costs[-1] if len(costs) == len(rungs) + 1.
    Returns mean automation, per-rung shares, risk, violation rate, cost, and how often each rung was switched on."""
    import itertools
    R = len(rungs); n = len(rungs[0][0]); rng = np.random.default_rng(seed)
    C = [np.asarray(c, float) for c, _ in rungs]; E = [np.asarray(e, int) for _, e in rungs]
    family = [(1.0,) + combo for combo in itertools.product(rhos, repeat=R - 1)]
    d_path = delta / len(family)
    hc = costs[R] if costs is not None and len(costs) > R else 0.0

    def mean_cost(undecided, accs):
        if costs is None:
            return -(1 - undecided.mean())
        reach = np.ones(len(undecided), bool); c = np.zeros(len(undecided))
        for r in range(R):
            c += reach * costs[r]; reach &= ~accs[r]
        return float((c + undecided * hc).mean())

    res = collections.defaultdict(list)
    for _ in range(n_splits):
        p = rng.permutation(n); cal, te = p[: n // 2], p[n // 2:]
        qs = 1 - np.linspace(start_cov(len(cal), alpha, d_path), 1.0, 39)
        best, best_obj = [np.inf] * R, np.inf
        for rho in family:
            chosen = None
            for q in qs:
                t = [np.quantile(C[r][cal], max(0.0, 1 - rho[r] * (1 - q))) if rho[r] > 0 else np.inf for r in range(R)]
                k, m, _, _ = _ladder_apply(C, E, cal, t)
                if m > 0 and cp_upper(k, m, d_path) <= alpha:
                    chosen = t
                else:
                    break
            if chosen is None:
                continue
            _, _, und, accs = _ladder_apply(C, E, cal, chosen)
            obj = mean_cost(und, accs)
            if obj < best_obj:
                best, best_obj = chosen, obj
        k, m, undecided, accs = _ladder_apply(C, E, te, best)
        risk = k / m if m else 0.0
        res["auto"].append(1 - undecided.mean()); res["risk"].append(risk); res["viol"].append(risk > alpha)
        cst = mean_cost(undecided, accs)
        res["cost"].append(cst if costs is not None else float(R + hc))
        for r in range(R):
            res[f"share_r{r}"].append(accs[r].mean()); res[f"on_r{r}"].append(float(np.isfinite(best[r])))
    return {k: float(np.mean(v)) for k, v in res.items()}
