"""Counterexample for a ratio-of-expectations guarantee (supplement, theory).  SCOPE threshold (production certify.thr = Eq. 6 of SCOPE: largest lam with sum_i S(x_i,lam)(E_i - alpha) <= -1) on
exchangeable synthetic data, independent repetitions; ratio of expectations E[E*S]/E[S] estimated by pooling, with a
delta-method standard error over the independent draws."""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from eaglevlm.pipeline.certify import thr
alpha = .2
def run(name, perr, n=500, ntest=4000, reps=20000, seed=0):
    rng = np.random.default_rng(seed); nums, dens = np.zeros(reps), np.zeros(reps); nacc = 0
    for k in range(reps):
        s = rng.random(n + ntest); e = (rng.random(n + ntest) < perr(s)).astype(float)
        lam = thr(s[:n], e[:n], alpha); acc = s[n:] <= lam
        if acc.any():
            nacc += 1
        nums[k] = (e[n:] * acc).mean(); dens[k] = acc.mean()
    r = nums.sum() / dens.sum() if dens.sum() else float("nan")
    se = np.sqrt(((nums - r * dens) ** 2).sum()) / dens.sum() if dens.sum() else float("nan")   # delta method over draws
    print(f"{name:58s} runs accepting {nacc / reps:.3f}  ratio of expectations {r:.4f} (s.e. {se:.4f})")
run("infeasible: P(err)=.30 for every score", lambda s: np.full_like(s, .30))
run("infeasible: P(err)=.25 for every score", lambda s: np.full_like(s, .25))
run("tie floor: P(err)=.25+.5*s (min risk .25 > alpha)", lambda s: .25 + .5 * s)
run("feasible, monotone: P(err)=.4*s (risk below alpha at low s)", lambda s: .4 * s)
run("feasible, monotone: P(err)=.05+.6*s", lambda s: .05 + .6 * s)
run("feasible, small n=100: P(err)=.4*s", lambda s: .4 * s, n=100)
run("feasible, monotone: P(err)=.05+.6*s, 100,000 draws", lambda s: .05 + .6 * s, reps=100000, seed=1)
