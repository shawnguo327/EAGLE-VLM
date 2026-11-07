"""Exact finite-sample counterexamples to SCOPE's ratio-of-expectations guarantee (supplement, theory), computed with
rational arithmetic through the production threshold rules of eaglevlm.pipeline.certify.thr (observed-score rule, and the
published supremum read literally).  i.i.d. calibration and test points; alpha = 1/4, n = 4.

  A  constant score, errors Bernoulli(1/2) (no safe threshold): Q = 1/2 under both rules; the two sides of SCOPE's
     Eq. (13) are E[L(Z_5, lam)] = 1/64 and E[(1/5) sum_i L(Z_i, lam)] = -3/320 (opposite signs).
  B  a safe threshold exists: Low (score .1, never wrong, prob. 1/4); High (score .5) correct w.p. 3/8, wrong w.p. 3/8.
     Accepting Low only has coverage 1/4 and risk 0.  Q = 1827/4904 (observed rule) and 3/8 (literal supremum).
Constructions from the round-6 internal review, re-derived and checked here independently.
    python scripts/scope_exact_counterexample.py
"""
import itertools, os, sys
from fractions import Fraction as F
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eaglevlm.pipeline.certify import thr

ALPHA, N = F(1, 4), 4


def ratio(states, rule):
    """states: list of (score, error, probability); returns (E[ES], E[S], E[L test], E[mean L over n+1]) exactly"""
    ES = Ss = Ltest = Lavg = F(0)
    for seq in itertools.product(states, repeat=N):
        pr = F(1)
        for _, _, p in seq:
            pr *= p
        lam = thr(np.array([x[0] for x in seq]), np.array([x[1] for x in seq]), ALPHA, rule)
        acc = [(sc <= lam, er, p) for sc, er, p in states]                 # a new point of each state
        s_new = sum(p for ok, _, p in acc if ok); es_new = sum(p * er for ok, er, p in acc if ok)
        ES += pr * es_new; Ss += pr * s_new
        l_new = sum(p * (er - ALPHA) for ok, er, p in acc if ok)          # E[L(Z_{n+1}, lam) | calibration]
        l_cal = sum((er - ALPHA) for sc, er, _ in seq if sc <= lam)         # sum over the n calibration points
        Ltest += pr * l_new; Lavg += pr * (l_cal + l_new) / (N + 1)
    return ES, Ss, Ltest, Lavg


def main():
    A = [(0.5, 0, F(1, 2)), (0.5, 1, F(1, 2))]
    B = [(0.1, 0, F(1, 4)), (0.5, 0, F(3, 8)), (0.5, 1, F(3, 8))]
    out = {}
    for name, st in (("A", A), ("B", B)):
        for rule in ("observed", "sup"):
            ES, Ss, Lt, La = ratio(st, rule)
            out[(name, rule)] = (ES / Ss, Ss, Lt, La)
            print(f"{name} {rule:8s}  E[S]={Ss}  Q=E[ES]/E[S]={ES / Ss} ({float(ES / Ss):.6f})  "
                  f"Eq.13: E[L(Z_5)]={Lt}  E[mean L]={La}")
    return out


if __name__ == "__main__":
    main()
