"""Monte Carlo companion to scripts/scope_exact_counterexample.py (supplement, theory): SCOPE's calibration condition
sum_i 1{s_i <= lam}(E_i - alpha) <= -1 on exchangeable synthetic data, under two readings of its threshold - the largest
observed calibration score that passes ('observed', our reproduction) and the published supremum read literally ('sup') -
both evaluated exactly (eaglevlm.pipeline.certify.thr; review round 6), plus the earlier float evaluation for reference.
Per setting and rule: share of draws with at least one test answer, mean test coverage E[S], and the ratio of
expectations E[E*S]/E[S] pooled over independent draws with a delta-method standard error.
    python scripts/scope_counterexample.py > results/scope_counterexample/output_<date>_exact.txt
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eaglevlm.pipeline.certify import thr
alpha = .2
RULES = ("observed", "sup", "float")


def run(name, perr, n=500, ntest=4000, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    num = {r: np.zeros(reps) for r in RULES}; den = {r: np.zeros(reps) for r in RULES}; anyacc = {r: 0 for r in RULES}
    for k in range(reps):
        s = rng.random(n + ntest); e = (rng.random(n + ntest) < perr(s)).astype(float)
        for r in RULES:
            acc = s[n:] <= thr(s[:n], e[:n], alpha, r)
            anyacc[r] += bool(acc.any()); num[r][k] = (e[n:] * acc).mean(); den[r][k] = acc.mean()
    for r in RULES:
        d = den[r].sum(); q = num[r].sum() / d if d else float("nan")
        se = np.sqrt(((num[r] - q * den[r]) ** 2).sum()) / d if d else float("nan")
        print(f"{name:58s} {r:8s} draws with a test answer {anyacc[r] / reps:.3f}  mean coverage {den[r].mean():.4f}  "
              f"ratio of expectations {q:.4f} (s.e. {se:.4f})", flush=True)


run("infeasible: P(err)=.30 for every score", lambda s: np.full_like(s, .30))
run("infeasible: P(err)=.25 for every score", lambda s: np.full_like(s, .25))
run("tie floor: P(err)=.25+.5*s (min risk .25 > alpha)", lambda s: .25 + .5 * s)
run("feasible, monotone: P(err)=.4*s (risk below alpha at low s)", lambda s: .4 * s)
run("feasible, monotone: P(err)=.05+.6*s", lambda s: .05 + .6 * s)
run("feasible, small n=100: P(err)=.4*s", lambda s: .4 * s, n=100)
run("feasible, monotone: P(err)=.05+.6*s, 100,000 draws", lambda s: .05 + .6 * s, reps=100000, seed=1)
