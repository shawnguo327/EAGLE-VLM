"""Decision utility of absolute (pointwise) information on pairs without a clear winner (2026-11-07).
Review 11-07 (tie_reframing_discussion, report 02, experiment 1): with a pre-declared acceptability threshold tau on the
human ratings, a system that sees two candidates for one source ships A, ships B, rejects both, or escalates. Does
pointwise information change these decisions at a certified risk, and does a gate on the judge's tie probability pick the
pairs that need it better than an entropy gate or a random gate at the same call budget?

Human ground truth (rating benchmarks only; arena votes do not say whether a winner is acceptable):
  acceptable_k = mean human rating of candidate k >= tau, with tau fixed in advance as the midpoint of the rating scale
  ('mid'; 1-5: 3, [0,1]: .5, 1-7: 4) and, as a pre-declared sensitivity, the top quarter ('high'; 1-5: 4, [0,1]: .75, 1-7: 5.5)
Loss (absolute ship-or-reject): 'ship k' is wrong iff k is not acceptable; 'reject both' is wrong iff A or B is acceptable.
Relative error (reported, not certified): shipping the candidate with the lower mean rating of a decided pair.
Policies (one 4-class acceptability head per feature set, multinomial logistic on standardised features, fitted on the fit
split; action = the most probable of ship A = P(A ok), ship B = P(B ok), reject = P(neither ok); answered when that
probability >= 1 - lambda; lambda by the same fixed-sequence Learn-then-Test on one pair per calibration source):
  pair_forced    pairwise features, ship actions only (no reject)                       4 calls / pair
  pair_reject    pairwise features, ship or reject                                       4
  pair_T_esc     as pair_reject, but escalate every pair whose A/B/T prompt answers T    4
  gate_entropy   pairwise; the most uncertain half (forced-choice entropy above the fit median) also scored pointwise and
                 decided by the pairwise+pointwise head                                  5 on average
  gate_tie       the same, routing the half with the highest tie probability             5
  gate_random    the same, routing a random half                                         5
  pointwise_only pointwise features only                                                 2
  both_all       pairwise + pointwise for every pair                                     6
  oracle_tie     routes the human ties (not deployable; an upper reference)              4 + 2 x tie rate
Primary endpoint: shipped acceptable candidates per 100 pairs (source-weighted) at certified risk <= alpha.
usage (repo root): python scripts/decision_utility.py --out <dir> [--splits 100] [--workers 16]"""
import argparse, ast, collections, csv, os, sys
from multiprocessing import Pool
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eaglevlm.pipeline.run import load_cfg                                       # noqa: E402
from eaglevlm.pipeline.io import Store, dataset_items                            # noqa: E402
from eaglevlm.pipeline.certify import signals, source_key, pointwise_feats, ltt, H   # noqa: E402
from eaglevlm.pipeline.router import splits                                      # noqa: E402

SPECS = ["genaibench", "evalmuse", "imagenhub/t2i", "imagenhub/edit", "imagerewarddb", "mllm_judge_sp"]
SCALE = {"genaibench": (1, 5), "evalmuse": (1, 5), "imagenhub": (0, 1), "imagerewarddb": (1, 7), "mllm_judge_sp": (1, 5)}
JUDGES = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b", "internvl35_38b"]
AL, DELTA = [.1, .2, .3], .1
POL = ["pair_forced", "pair_reject", "pair_T_esc", "gate_entropy", "gate_tie", "gate_random", "pointwise_only", "both_all", "oracle_tie"]


def lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def ratings(spec, x):
    ds = spec.split("/")[0]; m = x.get("meta", {}) or {}
    if ds == "imagerewarddb":
        return [float(c["overall"]) for c in lit(m["cand_info"])]
    return [float(s) for s in lit(m["scores"])]


def head(F, y, fit):
    sc = StandardScaler().fit(F[fit]); m = LogisticRegression(max_iter=5000).fit(sc.transform(F[fit]), y[fit])
    P = np.zeros((len(y), 4)); P[:, m.classes_] = m.predict_proba(sc.transform(F)); return P


def act(P, allow_reject=True):
    c = np.c_[P[:, 0] + P[:, 1], P[:, 0] + P[:, 2], P[:, 3] if allow_reject else np.zeros(len(P))]
    return c.argmax(1), 1 - c.max(1)            # 0 ship A, 1 ship B, 2 reject


def cell(args):
    spec, j, n_splits, P_ = args
    st = Store(P_["items"], P_["outputs"], P_.get("items_override"), P_.get("snapshot"))
    ds, I = dataset_items(st, spec)
    S = signals(st, ds, j, I); ids = sorted(S)
    if len(ids) < 200 or len(ids) < .95 * len(I):
        return [], f"skip {spec} {j}: signals {len(ids)} / {len(I)}"
    PWF = pointwise_feats(st, ds, j, ids)
    if PWF is None or np.isnan(PWF).any(1).mean() > .05:
        return [], f"skip {spec} {j}: pointwise"
    f1 = np.array([S[i][0] for i in ids]); f2 = np.array([S[i][1] for i in ids])
    Q = (np.array([S[i][2] for i in ids]) + np.array([S[i][3] for i in ids])) / 2; p = (f1 + f2) / 2
    cons = ((f1 > .5) == (f2 > .5)).astype(float)
    Fpair = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), cons]
    R = np.array([ratings(spec, I[i]) for i in ids]); lo, hi = SCALE[spec.split("/")[0]]
    tie = np.array([str(I[i]["label"]) == "tie" for i in ids]); saysT = Q.argmax(1) == 2
    gi = np.unique(np.array([source_key(I[i]) for i in ids]), return_inverse=True)[1]
    out = []
    for tname, frac in (("mid", .5), ("high", .75)):
        tau = lo + frac * (hi - lo); acc = R >= tau - 1e-9
        y = np.where(acc[:, 0] & acc[:, 1], 0, np.where(acc[:, 0], 1, np.where(acc[:, 1], 2, 3)))
        worse = np.where(R[:, 0] > R[:, 1], 1, np.where(R[:, 1] > R[:, 0], 0, -1))   # index of the worse candidate, -1 if equal
        rng = np.random.default_rng(0)
        for sp in range(n_splits):
            fit, cal, te = splits(len(ids), gi, {"unit": "source1", "split": {"fit": .25, "cal": .5}}, rng)
            pw = PWF.copy(); bad = np.isnan(pw)
            if bad.any():
                pw[bad] = np.nanmean(PWF[fit], 0)[np.where(bad)[1]]
            Fb = np.c_[Fpair, pw]
            Pp, Pw, Pb = head(Fpair, y, fit), head(pw, y, fit), head(Fb, y, fit)
            hmed, tmed = np.median(H(p)[fit]), np.median(Q[fit, 2]); rnd = np.random.default_rng(1000 + sp).random(len(ids)) < .5
            routes = {"gate_entropy": H(p) > hmed, "gate_tie": Q[:, 2] > tmed, "gate_random": rnd, "oracle_tie": tie}
            pol = {}
            a, s = act(Pp, False); pol["pair_forced"] = (a, s, 4.0)
            a, s = act(Pp); pol["pair_reject"] = (a, s, 4.0)
            pol["pair_T_esc"] = (a, np.where(saysT, 2.0, s), 4.0)
            for k, r in routes.items():
                ab, sb = act(Pb); a2 = np.where(r, ab, a); s2 = np.where(r, sb, s); pol[k] = (a2, s2, 4.0 + 2 * r[te].mean())
            a, s = act(Pw); pol["pointwise_only"] = (a, s, 2.0)
            a, s = act(Pb); pol["both_all"] = (a, s, 6.0)
            for name, (a_, s_, calls) in pol.items():
                ship = a_ < 2
                err = np.where(ship, ~acc[np.arange(len(ids)), np.minimum(a_, 1)], acc[:, 0] | acc[:, 1]).astype(float)
                good_ship = (ship & acc[np.arange(len(ids)), np.minimum(a_, 1)]).astype(float)
                good_rej = ((~ship) & ~(acc[:, 0] | acc[:, 1])).astype(float)
                rel = (ship & (worse == np.minimum(a_, 1))).astype(float)
                for al in AL:
                    lam = ltt(s_[cal][:, None], err[cal][:, None], al, DELTA, s_[fit][:, None], err[fit][:, None])
                    ok = s_[te] <= lam
                    _, inv, cnt = np.unique(gi[te], return_inverse=True, return_counts=True)
                    sw = lambda v: float((np.bincount(inv, v * ok) / cnt).mean())
                    out.append(dict(dataset=spec, judge=j, tau=tname, policy=name, alpha=al, split=sp, calls=calls,
                                    automated=sw(np.ones(len(te))), shipped_ok=sw(good_ship[te]), rejected_ok=sw(good_rej[te]),
                                    acc_mass=float((np.bincount(inv, ok.astype(float)) / cnt).sum()),
                                    err_mass=float((np.bincount(inv, err[te] * ok) / cnt).sum()),
                                    rel_err=float(rel[te][ok & ship[te]].mean()) if (ok & ship[te]).any() else np.nan,
                                    both_ok=float((y == 0).mean()), neither_ok=float((y == 3).mean()), n=len(ids)))
    return out, f"done {spec} {j}: {len(ids)} pairs"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--only", default="")
    a = ap.parse_args()
    paths = load_cfg("configs/exp/certify_source.yaml")["paths"]
    specs = [s for s in SPECS if not a.only or s in a.only.split(",")]
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        res = pool.map(cell, [(s, j, a.splits, paths) for s in specs for j in JUDGES], chunksize=1)
    rows = [r for rr, _ in res for r in rr]
    with open(f"{a.out}/decision_utility_splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    G = collections.defaultdict(list)
    for r in rows:
        G[(r["dataset"], r["tau"], r["alpha"], r["policy"], r["judge"])].append(r)
    S = collections.defaultdict(list)
    for (ds, t, al, pol, j), rr in G.items():
        am = sum(r["acc_mass"] for r in rr); em = sum(r["err_mass"] for r in rr)
        S[(ds, t, al, pol)].append(dict(shipped_ok=np.mean([r["shipped_ok"] for r in rr]), rejected_ok=np.mean([r["rejected_ok"] for r in rr]),
                                        automated=np.mean([r["automated"] for r in rr]), risk=em / am if am else np.nan,
                                        rel=np.nanmean([r["rel_err"] for r in rr]) if any(r["rel_err"] == r["rel_err"] for r in rr) else np.nan,
                                        calls=np.mean([r["calls"] for r in rr]), both_ok=rr[0]["both_ok"], neither_ok=rr[0]["neither_ok"]))
    L = ["# Decision utility of absolute information (shipped acceptable per 100 pairs at certified risk)", "",
         __doc__.split("usage")[0].strip(), "", "Mean over judges; risk pooled over splits per judge, then averaged (worst judge in brackets).", ""]
    for ds in specs:
        for t in ("mid", "high"):
            v0 = [x for k, x in S.items() if k[0] == ds and k[1] == t]
            if not v0:
                continue
            L += [f"## {ds}, tau = {t} (both acceptable {v0[0][0]['both_ok']:.2f}, neither {v0[0][0]['neither_ok']:.2f})", "",
                  "| alpha | policy | J | calls | shipped ok /100 | correct rejects /100 | automated | risk [worst] | relative err. |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for al in AL:
                for pol in POL:
                    v = S.get((ds, t, al, pol))
                    if not v:
                        continue
                    g = lambda k: np.array([x[k] for x in v], float)
                    rk = g("risk")
                    L.append(f"| {al} | {pol} | {len(v)} | {g('calls').mean():.1f} | {100 * g('shipped_ok').mean():.1f} | {100 * g('rejected_ok').mean():.1f} | "
                             f"{g('automated').mean():.3f} | {np.nanmean(rk):.3f} [{np.nanmax(rk) if (~np.isnan(rk)).any() else float('nan'):.3f}] | {np.nanmean(g('rel')):.3f} |")
            L.append("")
    L += ["## Log", ""] + [f"- {m}" for _, m in res]
    open(f"{a.out}/decision_utility.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))
