"""Do vision-specific probes carry error information beyond order-symmetric confidence?
Signals per strict pair (judge m):  lam = |order-averaged log-odds| (base);  lowres_shift = |pbar - pbar_lowres|,
lowres_flip = verdict changes at 112px;  blind_shift = |pbar - pbar_blind|, blind_same = verdict unchanged without
the question image (resp only), blind_conf = |2 pbar_blind - 1|.
Reports AUROC of each signal, error rate by flip status, cross-fitted logistic combination (AUROC and certified
coverage vs lam alone).  usage (from $EAGLEVLM_DATA): python probe_analysis.py qwen3vl8b [--alphas .2,.3]"""
import argparse, collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import load_records, auroc, selective_eval, calibrate_threshold, _sm

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--alphas", default=".2,.3")
ap.add_argument("--ds", default="vlrb,mmrb,mllm_judge,mmrlhf,genai_arena,mjbench,mmrb2,imagerewarddb,qbench2,editreward,pp")
a = ap.parse_args()


def probe_pbar(path):
    R = collections.defaultdict(dict)
    if not os.path.exists(path):
        return {}
    for l in open(path):
        d = json.loads(l)
        if "lp" in d:
            R[(d["id"], d["cond"])][d["order"]] = d["lp"]
    out = collections.defaultdict(dict)
    for (i, c), v in R.items():
        if "AB" in v and "BA" in v:
            out[i][c] = (_sm(v["AB"])["A"] + _sm(v["BA"])["B"]) / 2
    return out


def logit_fit(X, y, l2=1.0, iters=50):
    X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H, g)
    return w


def crossfit_score(X, correct, k=5, seed=0):
    """out-of-fold predicted P(correct); standardise features on the training folds."""
    n = len(X); idx = np.random.default_rng(seed).permutation(n); s = np.zeros(n)
    for f in range(k):
        te = idx[f::k]; tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = logit_fit((X[tr] - mu) / sd, correct[tr].astype(float))
        s[te] = 1 / (1 + np.exp(-np.c_[np.ones(len(te)), (X[te] - mu) / sd] @ w))
    return s


def selective_fit_eval(X, err, alpha, delta=0.1, n_splits=50, fit_frac=0.5, seed=0):
    """Combined score without leakage: per split, cal/test 50/50; the logistic combiner is fitted on fit_frac of the
    calibration half and the threshold certified on the rest, so the guarantee is untouched."""
    rng = np.random.default_rng(seed); n = len(err); cov, risk = [], []
    for _ in range(n_splits):
        p = rng.permutation(n); cal, te = p[: n // 2], p[n // 2:]
        k = int(fit_frac * len(cal)); fit, cer = cal[:k], cal[k:]
        mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-9
        w = logit_fit((X[fit] - mu) / sd, 1.0 - err[fit])
        sc = lambda idx: np.c_[np.ones(len(idx)), (X[idx] - mu) / sd] @ w
        tau = calibrate_threshold(sc(cer), err[cer], alpha, delta)
        m = sc(te) >= tau; cov.append(m.mean()); risk.append(err[te][m].mean() if m.any() else 0.0)
    return float(np.mean(cov)), float(np.mean(risk))


def veto_eval(conf, veto, err, alpha, delta=0.1, n_splits=50, seed=0):
    """Vision veto: pairs flagged by the probe are always escalated; lam is thresholded on the rest.
    No fitting, full calibration half used. Coverage is over all test pairs."""
    rng = np.random.default_rng(seed); n = len(err); cov, risk = [], []
    c2 = np.where(veto, conf.min() - 1.0, conf)
    for _ in range(n_splits):
        p = rng.permutation(n); cal, te = p[: n // 2], p[n // 2:]
        tau = calibrate_threshold(c2[cal], err[cal], alpha, delta)
        m = c2[te] >= tau; cov.append(m.mean()); risk.append(err[te][m].mean() if m.any() else 0.0)
    return float(np.mean(cov)), float(np.mean(risk))


def family_eval(conf, vetoes, err, alpha, delta=0.1, n_splits=50, seed=0):
    """Task-adaptive veto: candidate rules = lam with each veto mask (one mask is all-False = no veto). Each rule is
    certified at delta/K on the calibration half (Bonferroni), the one with the largest calibration coverage is
    deployed. Returns test coverage, risk and how often each rule was picked."""
    rng = np.random.default_rng(seed); n = len(err); K = len(vetoes); cov, risk = [], []; pick = np.zeros(K)
    C = [np.where(v, conf.min() - 1.0, conf) for v in vetoes]
    for _ in range(n_splits):
        p = rng.permutation(n); cal, te = p[: n // 2], p[n // 2:]
        best, bc, bt = 0, -1.0, np.inf
        for k in range(K):
            tau = calibrate_threshold(C[k][cal], err[cal], alpha, delta / K); c = (C[k][cal] >= tau).mean()
            if c > bc:
                best, bc, bt = k, c, tau
        pick[best] += 1; m = C[best][te] >= bt
        cov.append(m.mean()); risk.append(err[te][m].mean() if m.any() else 0.0)
    return float(np.mean(cov)), float(np.mean(risk)), (pick / n_splits).round(2).tolist()


summary = {}
for ds in a.ds.split(","):
    base = f"results/{ds}__{a.model}.jsonl"
    if not os.path.exists(base):
        continue
    recs = {r["id"]: r for r in load_records(f"items/{ds}.jsonl", base) if r["y"] is not None}
    P = probe_pbar(f"results_probe/{ds}__{a.model}.jsonl")
    ids = [i for i in recs if "bin_lowres" in P.get(i, {})]
    if len(ids) < 50:
        continue
    has_blind = all("bin_blind" in P[i] for i in ids)
    pa = np.array([recs[i]["pa"] for i in ids]); y = np.array([recs[i]["y"] for i in ids])
    correct = (pa > .5) == (y == 1)
    lam = np.abs([recs[i]["lo"] for i in ids])
    plo = np.array([P[i]["bin_lowres"] for i in ids])
    feats = {"lam": lam, "lowres_shift": -np.abs(pa - plo), "lowres_flip": -((plo > .5) != (pa > .5)).astype(float),
             "lowres_conf": np.abs(2 * plo - 1)}
    if has_blind:
        pb = np.array([P[i]["bin_blind"] for i in ids])
        feats.update({"blind_shift": np.abs(pa - pb), "blind_same": -((pb > .5) == (pa > .5)).astype(float),
                      "blind_conf": -np.abs(2 * pb - 1)})
    # signs chosen so that larger = more likely correct under the hypothesis; AUROC < .5 means the opposite
    res = {"n": len(ids), "acc": float(correct.mean()), "auroc": {k: float(auroc(v, correct)) for k, v in feats.items()}}
    flip = (plo > .5) != (pa > .5)
    res["lowres_flip_rate"] = float(flip.mean())
    res["acc_lowres_flip"] = float(correct[flip].mean()) if flip.any() else None
    res["acc_lowres_stable"] = float(correct[~flip].mean())
    if has_blind:
        same = (pb > .5) == (pa > .5)
        res["blind_same_rate"] = float(same.mean())
        res["acc_blind_same"] = float(correct[same].mean()); res["acc_blind_diff"] = float(correct[~same].mean()) if (~same).any() else None
    X_base = np.c_[lam]
    X_all = np.c_[[feats[k] for k in feats]].T
    s_base, s_all = crossfit_score(X_base, correct), crossfit_score(X_all, correct)
    res["auroc_combined"] = float(auroc(s_all, correct)); res["auroc_lam"] = float(auroc(lam, correct))
    err = (~correct).astype(int)
    for al in map(float, a.alphas.split(",")):
        res[f"cov_lam@{al}"] = selective_eval(lam, err, al)["coverage"]
        res[f"cov_comb@{al}"] = selective_fit_eval(X_all, err, al)[0]
        res[f"cov_veto_lowres@{al}"] = veto_eval(lam, flip, err, al)[0]
        if has_blind:
            res[f"cov_veto_blind@{al}"] = veto_eval(lam, same & (np.abs(2 * pb - 1) > .9), err, al)[0]
        V = [np.zeros(len(ids), bool), flip] + ([same & (np.abs(2 * pb - 1) > .9), flip | (same & (np.abs(2 * pb - 1) > .9))] if has_blind else [])
        c, r, pk = family_eval(lam, V, err, al)
        res[f"cov_adapt@{al}"] = c; res[f"risk_adapt@{al}"] = r; res[f"pick_adapt@{al}"] = pk
    summary[ds] = res
    print(f"{ds:13s} n={len(ids)} acc={res['acc']:.3f} AUROC lam={res['auroc_lam']:.3f} comb={res['auroc_combined']:.3f} | "
          + " ".join(f"{k}={v:.3f}" for k, v in res["auroc"].items() if k != "lam")
          + f" | lowres flip={res['lowres_flip_rate']:.2f} acc(flip/stable)={res['acc_lowres_flip'] or 0:.2f}/{res['acc_lowres_stable']:.2f}"
          + (f" | blind same={res['blind_same_rate']:.2f} acc(same/diff)={res['acc_blind_same']:.2f}/{res['acc_blind_diff'] or 0:.2f}" if has_blind else "")
          + " | " + " ".join(f"cov@{al}: lam {res[f'cov_lam@{float(al)}']:.2f} comb {res[f'cov_comb@{float(al)}']:.2f} vetoLR {res[f'cov_veto_lowres@{float(al)}']:.2f}" + (f" vetoBL {res[f'cov_veto_blind@{float(al)}']:.2f}" if has_blind else "") + f" ADAPT {res[f'cov_adapt@{float(al)}']:.2f} risk {res[f'risk_adapt@{float(al)}']:.2f} pick {res[f'pick_adapt@{float(al)}']}" for al in a.alphas.split(",")))
os.makedirs("paper_tables", exist_ok=True)
json.dump(summary, open(f"paper_tables/probe_{a.model}.json", "w"), indent=1)
