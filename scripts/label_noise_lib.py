"""Shared helpers for the certification analyses (tie_cert protocol): judge signals and strict / graded coverage."""
import collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from tie_mech import load


def per_judge(ds, j, I, idmap=lambda i: i):
    """{base id: (p_bar, [P0, P1, PT] order-aligned, order-consistent 0/1)} over items of I with bin + tri in both orders"""
    R = load(ds, j, conds={"bin", "tri"}); out = {}
    for i, r in R.items():
        if i not in I or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
        f1, f2 = b1["A"] / (b1["A"] + b1["B"]), b2["A"] / (b2["A"] + b2["B"])
        q = np.array([(t1["A"] + t2["B"]) / 2, (t1["B"] + t2["A"]) / 2, (t1["T"] + t2["T"]) / 2])
        out[idmap(i)] = ((f1 + 1 - f2) / 2, q / q.sum(), float((f1 > .5) != (f2 > .5)))
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


def evaluate_cov(F, y, AL, splits, seed):
    """mean test coverage of the strict and graded (exploratory one-threshold) rules"""
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
    return {k: np.mean(v) for k, v in res.items()}


def _pval(Ss, Es, lam, al):
    from scipy.stats import binom
    acc, ch = _claims(Ss, lam); n = int(acc.sum())
    if n == 0:
        return 1.0
    return binom.cdf(int(Es[np.arange(len(Es)), np.maximum(ch, 0)][acc].sum()), n, al)


GRID = np.linspace(0, 1, 201)[1:]


def ltt(Ss, Es, al, delta, Sf=None, Ef=None, grid=GRID):
    """fixed-sequence Learn-then-Test (same as cert_main.ltt): fixed grid on the estimated error; start at the grid value
    with the smallest binomial p-value on the FIT split; walk to more liberal thresholds while p <= delta."""
    s0 = int(np.argmin([_pval(Sf, Ef, lam, al) for lam in grid])) if Sf is not None else 0
    best = -np.inf
    for lam in grid[s0:]:
        if _pval(Ss, Es, lam, al) <= delta:
            best = lam
        else:
            break
    return best


# ---- analysis-time sample corrections (review 2026-11-03 evening) ----
FULL = "/media/shawn/C/papers/VLM/judgebench_local/items/{}_full.jsonl"


def _strat(x):
    return (x["subset"], str(x["label"]) == "tie")


def proportional(ds, I, seed=0):
    """Largest subsample of I whose (subset x tie) composition equals that of the full benchmark <ds>_full.jsonl.
    The capped base files were built round-robin over subset x label, so base + random rest is NOT a random sample of
    the benchmark; this restores exact proportions (random within each stratum, fixed seed)."""
    import json, random
    full = collections.Counter(_strat(json.loads(l)) for l in open(FULL.format(ds)))
    nf = sum(full.values()); have = collections.defaultdict(list)
    for i, x in I.items():
        have[_strat(x)].append(i)
    N = min(len(have[s]) * nf // c for s, c in full.items() if c)
    rng = random.Random(seed); keep = {}
    for s, c in full.items():
        q = round(N * c / nf)
        for i in rng.sample(sorted(have[s]), min(q, len(have[s]))):
            keep[i] = I[i]
    return keep


def dedup_pairs(I):
    """one item per unordered candidate pair (repeat votes on the identical pair would sit in fit, cal and test at once):
    keep the first id of a pair if all its labels agree after orientation, otherwise drop the pair."""
    g = collections.defaultdict(list)
    for i in sorted(I):
        g[frozenset(I[i]["cand"])].append(i)
    keep = {}
    for ids in g.values():
        labs = {str(I[i]["label"]) if str(I[i]["label"]) == "tie" else I[i]["cand"][int(I[i]["label"])] for i in ids}
        if len(labs) == 1:
            keep[ids[0]] = I[ids[0]]
    return keep


def eval_items(ds, I):
    """the item set a main analysis should use for <ds>"""
    if ds in ("imagerewarddb", "mllm_judge"):
        import json
        if len(I) == sum(1 for _ in open(FULL.format(ds))):      # complete benchmark judged (make_rest3.py)
            return I
        return proportional(ds, I)
    if ds == "pickapic":
        return dedup_pairs(I)
    return I
