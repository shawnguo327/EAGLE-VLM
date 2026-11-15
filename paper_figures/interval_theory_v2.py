"""Window model, corrected (2026-11-15, after an external review).  The first version (interval_theory.py) started the
fixed sequence at the first threshold that PASSED on the calibration sample itself; that start depends on the calibration
data and the procedure is then not a valid fixed-sequence test (it accepts unsafe rules more often than delta).  Here the
start is chosen as in the paper's pipeline (eaglevlm/pipeline/certify.py, ltt): the threshold with the smallest p-value on
an independent FIT sample (n_fit = n_cal / 2, the 25 / 50 split), then the sequence walks laxer on the calibration sample
until the first failure; if the start fails, nothing is certified.
Model as before: a pair is 'both failed' with probability pi; evidence x ~ N(d, 1) on such pairs, N(0, 1) otherwise; the
policy rejects when x >= t.  Output: correct certified rejections per 100 pairs.
usage: python interval_theory_v2.py <shared_failure_curve.csv> <out.json>      |      python interval_theory_v2.py audit"""
import json, sys
import numpy as np
from scipy.stats import norm, binom
N_CAL, N_FIT, DELTA, REP = 90, 45, .1, 1500
GRID_T = np.linspace(4.5, -4.5, 181)                       # strict -> lax
def k_star(n, alpha, delta=DELTA):
    k = np.arange(n + 1); ok = binom.cdf(k, n, alpha) <= delta; return int(k[ok].max()) if ok.any() else -1
def pvals(x, isF, alpha):
    ans = x[:, :, None] >= GRID_T[None, None, :]; na = ans.sum(1); ne = (ans & ~isF[:, :, None]).sum(1)
    return np.where(na > 0, binom.cdf(ne, np.maximum(na, 1), alpha), 1.0)
def certified_index(pi, d, alpha, rng, rep=REP, old=False):
    """index of the certified threshold per repetition (-1: nothing certified)"""
    isF = rng.random((rep, N_CAL)) < pi; x = rng.standard_normal((rep, N_CAL)) + d * isF; passed = pvals(x, isF, alpha) <= DELTA
    if old: start = np.where(passed.any(1), passed.argmax(1), -1)                                   # first version: data-dependent start
    else:
        isFf = rng.random((rep, N_FIT)) < pi; xf = rng.standard_normal((rep, N_FIT)) + d * isFf; start = pvals(xf, isFf, alpha).argmin(1)
    out = np.full(rep, -1)
    for r in range(rep):
        j = start[r]
        if j < 0 or not passed[r, j]: continue
        while j + 1 < len(GRID_T) and passed[r, j + 1]: j += 1
        out[r] = j
    return out
def correct_per_100(pi, d, alpha, rng, rep=REP):
    tpr = 1 - norm.cdf(GRID_T - d); j = certified_index(pi, d, alpha, rng, rep); return 100 * pi * np.where(j >= 0, tpr[np.maximum(j, 0)], 0).mean()
def audit(rep=20000):
    """uninformative evidence (d = 0): every threshold has true risk 1 - pi.  Share of repetitions that certify anything."""
    rng = np.random.default_rng(1); res = {}
    for al in (.1, .2, .3):
        for excess in (.0, .01, .05):
            pi = 1 - al - excess                 # true risk of every rule = al + excess (at excess 0 the risk equals the target: acceptance may reach delta)
            new = np.mean(np.concatenate([certified_index(pi, 0., al, rng, 2000) for _ in range(rep // 2000)]) >= 0)
            old = np.mean(np.concatenate([certified_index(pi, 0., al, rng, 2000, old=True) for _ in range(rep // 2000)]) >= 0)
            res[f"alpha{al}_risk{al + excess:.2f}"] = dict(corrected=float(new), first_version=float(old)); print(f"alpha {al} true risk {al + excess:.2f}: accepts an unsafe rule  corrected {new:.3f}   first version {old:.3f}   (delta = {DELTA})")
    for al in (.1, .2, .3):
        ks = k_star(N_CAL, al); ph = 1 - ks / N_CAL; res[f"reject_all_alpha{al}"] = {f"pi={p:.3f}": float(binom.cdf(ks, N_CAL, 1 - p)) for p in (ph - .05, ph, ph + .03, ph + .06, min(ph + .1, .999))}
        print(f"alpha {al}: k* = {ks}, pi_hi = {ph:.3f}; P[reject-all passes] at pi =", res[f"reject_all_alpha{al}"])
    return res
if __name__ == "__main__":
    if sys.argv[1] == "audit":
        json.dump(audit(), open(sys.argv[2], "w"), indent=1); sys.exit()
    import csv
    D = list(csv.DictReader(open(sys.argv[1]))); res = {"version": "v2: start chosen on an independent fit sample"}
    res["pi_hi"] = {str(a): 1 - k_star(N_CAL, a) / N_CAL for a in (.1, .2, .3)}
    res["n_min_zero_error"] = {str(a): int(np.ceil(np.log(DELTA) / np.log(1 - a))) for a in (.1, .2, .3)}
    PI = np.round(np.linspace(.05, .99, 48), 3); DS = np.round(np.arange(1.0, 2.001, .02), 2)          # finer grid: the two fitted separations differ by less than .1
    TAB = {(a, d): np.array([correct_per_100(p, d, a, np.random.default_rng(1000 * k + int(100 * a))) for k, p in enumerate(PI)]) for a in (.1, .2, .3) for d in DS}   # common random numbers across d: the gain is a smooth difference
    f = lambda a, d, pis: np.interp(pis, PI, TAB[(a, d)])
    for scope in ("open", "gpt4o"):
        M0 = sorted([r for r in D if r["scope"] == scope and r["tau"] == "mid"], key=lambda r: float(r["pi"]))
        M = {c: np.array([float(r[c]) for r in M0]) for c in ("pi", "correct_PW", "correct_B+T")}; pis = M["pi"]; fit = {}
        for name, col in (("base", "correct_PW"), ("tie", "correct_B+T")):
            err = {d: float(np.mean((f(.2, d, pis) - M[col]) ** 2)) for d in DS}; d = min(err, key=err.get)
            fit[name] = dict(d=float(d), auroc=float(norm.cdf(d / np.sqrt(2))), rmse=float(np.sqrt(err[d])))
        res[scope] = dict(fit=fit, pi=pis.tolist(), observed_base=M["correct_PW"].tolist(), observed_tie=M["correct_B+T"].tolist(), model_base=f(.2, fit["base"]["d"], pis).tolist(), model_tie=f(.2, fit["tie"]["d"], pis).tolist(),
                          curves={str(a): dict(pi=PI.tolist(), base=TAB[(a, fit["base"]["d"])].tolist(), tie=TAB[(a, fit["tie"]["d"])].tolist()) for a in (.1, .2, .3)})
        g = {}
        for a in (.1, .2, .3):
            gain = np.convolve(TAB[(a, fit["tie"]["d"])] - TAB[(a, fit["base"]["d"])], np.ones(5) / 5, mode="same"); k = int(np.argmax(gain[2:-2])) + 2; pos = PI[gain >= .5 * gain.max()]
            g[str(a)] = dict(peak_pi=float(PI[k]), peak_gain=float(gain[k]), half_max_window=[float(pos.min()), float(pos.max())])
        res[scope]["predicted_gain"] = g; print(scope, "fit", fit); print("   predicted gain (5-point smoothed): peak and half-maximum window:", g)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
