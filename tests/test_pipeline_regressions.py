"""Regression tests for the certification pipeline on synthetic inputs (the pipeline functions are imported unchanged).

    python tests/test_pipeline_regressions.py        (or: pytest tests/test_pipeline_regressions.py)

Adapted from the read-only audit script of the 11-07 re-review (reaudit_checks.py), with the expectations updated to
the fixed code:
  1. zero-count claim levels (review W4): a level the fitted policy can issue but that never occurs in calibration gets
     p = 1, so the old rare-claim counterexample is certified by none of G3 / G4 / G5 / the router's every-level rule;
  2. the G5 budget of eleven strict thresholds at delta / 11: 21 zero-error calibration pairs (p = .8^21 > delta / 11)
     certify nothing, 22 pairs (p = .8^22 < delta / 11) certify the strict-off chain;
  3. fit-only router features: changing one calibration row changes no fit or other row's features;
  4. source1 splits: no source in two parts, exactly one calibration pair per calibration source;
  5. coverage estimands: pair acceptance and source-uniform coverage differ (two sources with 100 and 1 pairs);
  6. splits without any answer in the summary, and the matched pooled source ratio;
  7. the overall router is not a nested extension of the graded policy (why the paper drops that mapping);
  8. PWL no longer overwrites the split index (review 11-07, N1): every row's split is an int and the
     (method, alpha, split) keys are unique.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eaglevlm.pipeline import certify as c, router as r  # noqa: E402

ALPHA, DELTA = .2, .1
# a common item (estimated .25 / .25 / .50) and a rare one (.99 / .005 / .005), both with human label T
S, E = c.graded_scores(np.array([[.25, .25, .5], [.99, .005, .005]]), np.array([2, 2]))


def test_rare_claim_counterexample():
    Sf, Ef = np.repeat(S[:1], 100, 0), np.repeat(E[:1], 100, 0)      # calibration holds only the common item
    for pf in (c.pjoint, c.pstrict):                                  # G3 every level, G4 overall + strict
        assert c.ltt_p(Sf, Ef, Sf, Ef, ALPHA, DELTA, pf) == -np.inf
    ls, lw = c.ltt_two(Sf, Ef, Sf, Ef, ALPHA, DELTA)                  # G5: strict claims must stay switched off
    ok, ch = c.claims2(S, ls, lw)
    assert ls == -np.inf and not any(E[np.arange(2), np.maximum(ch, 0)][ok])
    assert r.fst(Sf, Ef, Sf, Ef, ALPHA, True, DELTA, np.zeros(3)) == -np.inf


def test_g5_delta_over_11_boundary():
    assert len(c.LS) == 11
    th = {}
    for n in (21, 22):
        ss, ee = np.repeat(S[:1], n, 0), np.repeat(E[:1], n, 0)
        th[n] = c.ltt_two(ss, ee, ss, ee, ALPHA, DELTA)
    assert .8 ** 21 > DELTA / 11 > .8 ** 22
    assert th[21][1] == -np.inf and th[22][1] > -np.inf and th[22][0] == -np.inf


def test_fit_only_router_features():
    M = np.array([[2., .2, .3, 3., .1, .2], [4., .3, .1, 5., .2, .1], [np.nan, .1, .1, 6., .2, .1], [7., .1, .2, 8., .1, .1]])
    M2 = M.copy(); M2[3] = [1., 99., .9, 1.5, 99., .9]
    F, F2 = r.lvl_feats(M, np.zeros(4), np.array([0, 1])), r.lvl_feats(M2, np.zeros(4), np.array([0, 1]))
    assert np.array_equal(F[:3], F2[:3], equal_nan=True)


def test_source1_splits():
    groups = np.repeat(np.arange(80), np.arange(1, 81) % 5 + 1); rng = np.random.default_rng(0)
    for _ in range(100):
        fit, cal, te = r.splits(len(groups), groups, {"unit": "source1", "split": {"fit": .25, "cal": .5}}, rng)
        fs, cs, ts = (set(groups[x]) for x in (fit, cal, te))
        assert not (fs & cs or fs & ts or cs & ts)
        assert len(cal) == len(cs) == 40 and len(fs) == len(ts) == 20


def test_coverage_estimands():
    ok = np.array([True] * 100 + [False])
    d = c.row(ok, np.zeros(101, int), np.zeros(101), ALPHA, np.zeros(101, bool), np.array([0] * 100 + [1]))
    assert np.isclose(d["acceptance"], 100 / 101) and np.isclose(d["source_coverage"], .5)
    assert d["accepted_sources"] == 1 and d["src_acc_mass"] == 1.0 and d["risk_source"] == 0.0


def test_empty_split_summary():
    rows = []
    for sp, (acc, er) in enumerate([([False, False], [0., 0.]), ([True, True], [1., 0.]), ([True, True], [0., 0.])]):
        d = c.row(np.array(acc), np.zeros(2, int), np.array(er), ALPHA, np.zeros(2, bool), np.array([0, 1]))
        rows.append(dict(dataset="mock", judge="mock", method="G2", alpha=ALPHA, n=2, human_tie=0., n_persona=None, split=sp, **d))
    m = c.summarize(rows)[0]
    assert m["splits_answered"] == 2 and np.isclose(m["guarantee_success_all"], 2 / 3)
    assert m["guarantee_success"] == .5 and m["pooled_risk"] == .25
    assert np.isclose(m["pooled_risk_source"], .25) and np.isclose(m["source_coverage"], 2 / 3)
    assert np.isclose(m["guarantee_success_source"], 2 / 3)


def test_overall_router_not_nested():
    fitG = np.tile([.9, .9, .1], (100, 1)); fitE = np.zeros_like(fitG)
    calG = np.vstack([np.tile([.9, .9, .1], (50, 1)), np.tile([.9, .9, .9], (50, 1))])
    calE = np.vstack([np.zeros((50, 3)), np.ones((50, 3))])
    fitR, fitRE = np.c_[fitG, np.full((100, 2), .1)], np.c_[fitE, np.ones((100, 2))]
    calR, calRE = np.c_[calG, np.full((100, 2), .1)], np.c_[calE, np.ones((100, 2))]
    base = r.fst(calG, calE, fitG, fitE, ALPHA, False, DELTA, np.zeros(3))
    routed = r.fst(calR, calRE, fitR, fitRE, ALPHA, False, DELTA, np.zeros(5))
    # the graded policy certifies half the pairs; adding an uncertifiable absolute branch loses all of it
    assert r.claims(calG, base)[0].mean() == .5 and routed == -np.inf


class _Store:
    """240 synthetic rows from 80 sources, every prompt condition present"""
    items_dirs = []

    def __init__(self):
        self.I, self.R = {}, {}
        for i in range(240):
            k = f"i{i:04d}"
            self.I[k] = {"id": k, "question": f"q{i // 3}", "context": [], "label": ["0", "1", "tie"][i % 3], "subset": "all"}
            p = [.8, .2, .5][i % 3]; q = np.array([.1, .1, .1]); q[i % 3] = .8
            self.R[k] = {("bin", "AB"): {"A": p, "B": 1 - p}, ("bin", "BA"): {"A": 1 - p, "B": p},
                         ("tri", "AB"): dict(zip(["A", "B", "T"], q)), ("tri", "BA"): dict(zip(["B", "A", "T"], q)),
                         ("score", "A"): {str(d): float(d == (i % 8 + 1)) for d in range(1, 10)},
                         ("score", "B"): {str(d): float(d == ((i + 3) % 8 + 1)) for d in range(1, 10)}}

    def items(self, ds, rest=None):
        return self.I

    def load(self, ds, model, conds=None):
        return {i: {k: v for k, v in rs.items() if conds is None or k[0] in conds} for i, rs in self.R.items()}


def test_pwl_keeps_split_index():
    cfg = {"datasets": ["synthetic"], "judges": ["synthetic"], "methods": ["PWL", "G2"], "alphas": [.2], "delta": .1,
           "split": {"fit": .25, "cal": .5}, "unit": "source1", "n_splits": 3}
    rows = c.run(cfg, _Store(), lambda _: None)
    assert rows and all(isinstance(x["split"], (int, np.integer)) for x in rows)
    keys = [(x["method"], x["alpha"], int(x["split"])) for x in rows]
    assert len(keys) == len(set(keys)) and {k[2] for k in keys} == {0, 1, 2}
    assert {k[0] for k in keys} == {"PWL", "G2"}


def test_thr_exact_boundary():                 # review round 6, R6-N1 (the reviewer's checks)
    from fractions import Fraction
    s = np.arange(1, 11) / 10; e = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    assert c.thr(s, e, .2) == 1.0                  # the last prefix is exactly -1 and passes
    assert c.thr(s, e, .2, "float") == 0.6          # the earlier float cumulative sum missed it
    assert c.thr(np.full(5, .5), np.array([0, 0, 0, 0, 1]), .25) == -np.inf   # a score group passes whole or not at all
    assert c.thr(np.full(4, .5), np.zeros(4), .25) == .5
    assert c.thr(np.array([]), np.array([]), .2) == -np.inf
    assert c.thr(s, np.ones(10), .2) == -np.inf
    assert c.thr(s, e, Fraction(1, 5)) == c.thr(s, e, .2)


def test_thr_literal_sup():                     # R6-N2: the supremum of the passing thresholds need not pass
    s = np.array([.1, .2, .3, .4, .5]); e = np.array([0, 0, 0, 0, 1])
    assert c.thr(s, e, .25) == .4 and c.thr(s, e, .25, "sup") == .5
    assert c.thr(s, np.zeros(5), .25, "sup") == np.inf and c.thr(s, np.zeros(5), .25) == .5
    assert c.thr(s, np.ones(5), .25, "sup") == -np.inf


def test_scope_exact_counterexamples():         # supplement: exact i.i.d. counterexamples (alpha = 1/4, n = 4)
    from fractions import Fraction as F
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    import scope_exact_counterexample as X
    out = X.main()
    assert out[("A", "observed")][0] == F(1, 2) and out[("A", "sup")][0] == F(1, 2)
    assert out[("A", "observed")][2] == F(1, 64) and out[("A", "observed")][3] == F(-3, 320)
    assert out[("B", "observed")][0] == F(1827, 4904) and out[("B", "sup")][0] == F(3, 8)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
