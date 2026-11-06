"""Router experiment ('router'): pairwise graded claims, with the pairwise-uncertain half routed to pointwise
absolute-level claims ('both unacceptable', 'no clear winner').  2026-11-06 redesign after review (W6/W7): the graded
claims (incl. the tie claim) are kept for every pair and absolute-level claims are added only where no graded claim
passes; all feature normalisations are fit-only; optional unit: source1 (split by source, one pair per calibration
source).  The pre-review version (scripts/router_v2.py port) is in git history (commit 442345a and earlier).
Level labels from human absolute references:
  mllm_judge_sp / genaibench   human 1-5 (mean of raters): no clear winner = |gap| <= 1, both unacceptable = both <= 2
  imagerewarddb                human overall 1-7: |gap| <= 1, both <= 3
  datasets with meta.scale     a quarter of the score range (polaris 0-1, agiqa3k 0-5)
  genai_arena                  votes: no clear winner = tie or both-bad vote, both unacceptable = both-bad vote
  others (no absolute scores)  no clear winner = human tie; 'both unacceptable' is never claimed
Variants: G2 (no routing), control (routed, level claims with the routed base rate as estimated error - no pointwise),
cells (fixed 3x3x2 map), logistic (per level claim), jury (pointwise ranks averaged over all judges with pointwise runs).
Rules: overall risk <= alpha, or every claim level <= alpha (intersection-union).  One CSV row per split."""
import collections
import numpy as np
from scipy.stats import binom, rankdata
from sklearn.linear_model import LogisticRegression

GRID = np.linspace(0, 1, 201)[1:]; D = np.arange(1, 10)
H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))
VARIANTS = ["pairwise graded (G2)", "control: routed, no pointwise", "router: cells", "router: logistic", "router: jury pointwise"]


def claims(S, lam):
    ok = S <= lam; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def pj(S, E, lam, al, joint):
    ok, ch = claims(S, lam)
    if not ok.any():
        return 1.0
    if not joint:
        return binom.cdf(int(E[np.arange(len(E)), np.maximum(ch, 0)][ok].sum()), int(ok.sum()), al)
    # every level the policy can issue (finite score somewhere) must be certified; an unobserved one gets p = 1 (review W4)
    ps = [0.0]
    for c in range(S.shape[1]):
        if not np.isfinite(S[:, c]).any():
            continue                    # level switched off by construction (e.g. no routed pairs / degenerate estimate)
        mk = ok & (ch == c); n = int(mk.sum())
        ps.append(float(binom.cdf(int(E[mk, c].sum()), n, al)) if n else 1.0)
    return max(ps)


def fst(Sc, Ec, Sf, Ef, al, joint, delta):
    s0 = int(np.argmin([pj(Sf, Ef, g, al, joint) for g in GRID])); best = -np.inf
    for g in GRID[s0:]:
        if pj(Sc, Ec, g, al, joint) <= delta:
            best = g
        else:
            break
    return best


def per_judge(store, ds, j, I):
    R = store.load(ds, j, conds={"bin", "tri"}); out = {}
    for i, r in R.items():
        if i not in I or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
        f1, f2 = b1["A"] / (b1["A"] + b1["B"]), b2["A"] / (b2["A"] + b2["B"])
        q = np.array([(t1["A"] + t2["B"]) / 2, (t1["B"] + t2["A"]) / 2, (t1["T"] + t2["T"]) / 2])
        out[i] = ((f1 + 1 - f2) / 2, q / q.sum(), float((f1 > .5) != (f2 > .5)))
    return out


def level_labels(ds, I, ids):
    if "scale" in I[ids[0]].get("meta", {}):
        lo, hi = I[ids[0]]["meta"]["scale"]; q = (hi - lo) / 4
        s = np.array([I[i]["meta"]["scores"] for i in ids], float); return np.abs(s[:, 0] - s[:, 1]) <= q, s.max(1) <= lo + q
    if ds in ("mllm_judge_sp", "genaibench"):
        s = np.array([I[i]["meta"]["scores"] for i in ids], float); return np.abs(s[:, 0] - s[:, 1]) <= 1, s.max(1) <= 2
    if ds == "imagerewarddb":
        s = np.array([[c["overall"] for c in I[i]["meta"]["cand_info"]] for i in ids], float)
        return np.abs(s[:, 0] - s[:, 1]) <= 1, s.max(1) <= 3
    if "vote_type" in I[ids[0]].get("meta", {}):
        v = np.array([I[i]["meta"]["vote_type"] for i in ids]); return np.isin(v, ["tievote", "bothbad_vote"]), v == "bothbad_vote"
    # no absolute reference: 'no clear winner' = human tie; 'both unacceptable' is never claimed
    t = np.array([str(I[i]["label"]) == "tie" for i in ids]); return t, np.zeros(len(ids), bool)


def pointwise(store, ds, j, ids):
    """per pair [E_A, Var_A, P(<=3)_A, E_B, Var_B, P(<=3)_B] from the 1-9 score prompt; NaN where missing (imputed inside
    each split from the fit rows only, review W6); None if more than 5% of the pairs miss a score"""
    R = store.load(ds, j, conds={"score"}); M = np.full((len(ids), 6), np.nan)
    for k, i in enumerate(ids):
        r = R.get(i, {})
        if ("score", "A") in r and ("score", "B") in r:
            row = []
            for o in "AB":
                q = np.array([r[("score", o)].get(str(d), 0) for d in D]); q = q / q.sum(); mu = q @ D
                row += [mu, q @ (D - mu) ** 2, q[:3].sum()]
            M[k] = row
    if np.isnan(M).any(1).mean() > .05:
        return None
    return M


def lvl_feats(M, u, fit):
    """absolute-level features with every normalisation fixed on the FIT rows (review W6): missing scores imputed with
    fit means, ranks = empirical CDF of the fit candidates' expected scores, so a pair's features depend only on its
    own pointwise scores and the fit data (deployable with pointwise calls for the routed pairs only)"""
    M = M.copy(); mu = np.nanmean(M[fit], 0); bad = np.isnan(M); M[bad] = mu[np.where(bad)[1]]
    pool = np.sort(np.r_[M[fit, 0], M[fit, 3]])
    rk = np.c_[np.searchsorted(pool, M[:, 0], side="right"), np.searchsorted(pool, M[:, 3], side="right")] / len(pool)
    lo, hi = rk.min(1), rk.max(1)
    return np.c_[rk, lo, hi, np.abs(rk[:, 0] - rk[:, 1]), M[:, [1, 4, 2, 5]], u]


def splits(n, groups, cfg, rng):
    """pair-level split (n//4, n//4, rest; as before) or, for unit: source1, a split by source with one random pair per
    calibration source (review W5; same design as certify)"""
    if cfg.get("unit", "pair") != "source1":
        pm = rng.permutation(n); return pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
    fit_f, cal_f = cfg.get("split", {}).get("fit", .25), cfg.get("split", {}).get("cal", .25)
    ug, gi = np.unique(groups, return_inverse=True)
    rk = np.empty(len(ug), int); rk[rng.permutation(len(ug))] = np.arange(len(ug)); r = rk[gi]
    g1, g2 = int(round(fit_f * len(ug))), int(round((fit_f + cal_f) * len(ug)))
    fit, cal, te = np.where(r < g1)[0], np.where((r >= g1) & (r < g2))[0], np.where(r >= g2)[0]
    pc = rng.permutation(cal); _, first = np.unique(gi[pc], return_index=True)
    return fit, np.sort(pc[first]), te


def run(cfg, store, log):
    """Router (review W6/W7 design): every variant keeps the graded claims (strict, tie, weak) of G2 and adds the two
    absolute-level claims only where no graded claim passes, on the routed (pairwise-uncertain) half; claim columns
    [strict, tie, weak, both unacceptable, no clear winner]; one threshold, certified with the same fixed-sequence LTT.
    All normalisations are fit-only; per-claim shares are reported, and the coverage gain over G2 on the same split."""
    from eaglevlm.pipeline.io import NO_REST
    from eaglevlm.pipeline.certify import source_key
    AL = cfg["alphas"]; delta = cfg["delta"]; out = []
    for spec in cfg["datasets"]:
        ds, sub = (spec.split("/") + [None])[:2]
        I = store.items(ds, rest=ds not in NO_REST); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
        ids_all = sorted(I)
        PJ = {j: per_judge(store, ds, j, I) for j in cfg["judges"]}
        PW = {j: pointwise(store, ds, j, ids_all) for j in PJ if len(PJ[j]) >= .95 * len(I)}
        PW = {j: v for j, v in PW.items() if v is not None}
        if not PW:
            log(f"skip {spec}: no judge with pointwise scores"); continue
        ncw_all, bun_all = level_labels(ds, I, ids_all)
        for j in [j for j in PJ if j in PW]:
            P = PJ[j]; keep = np.array([i in P for i in ids_all]); ids = [i for i in ids_all if i in P]; pos = np.where(keep)[0]
            n = len(ids); y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
            p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
            u = H(p); Fp = np.c_[np.log(Q + 1e-9), u, np.abs(p - .5), fl]
            ncw, bun = ncw_all[keep], bun_all[keep]
            groups = np.array([source_key(I[i]) for i in ids])
            rng = np.random.default_rng(cfg.get("seed", 0))
            for sp in range(cfg["n_splits"]):
                fit, cal, te = splits(n, groups, cfg, rng)
                m = LogisticRegression(max_iter=5000).fit(Fp[fit], y[fit]); Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(Fp)
                k = Pr[:, :2].argmax(1); o = 1 - k
                s_str, s_tie, s_weak = 1 - Pr[np.arange(n), k], 1 - Pr[:, 2], Pr[np.arange(n), o]
                e_str, e_tie, e_weak = (y != k), (y != 2), (y == o)
                routed = u >= np.quantile(u[fit], .5); rf = fit[routed[fit]]
                Xl = lvl_feats(PW[j][keep], u, fit)
                fit_all = pos[fit]                                   # jury: each judge's features fixed on the same fit pairs
                jury = np.mean([lvl_feats(PW[jj], np.zeros(len(ids_all)), fit_all)[:, :8] for jj in PW], 0)[keep]
                Xj = np.c_[jury, u]; inf = np.full(n, np.inf)
                G = np.c_[s_str, s_tie, s_weak]; GE = np.c_[e_str, e_tie, e_weak].astype(float)
                variants = {"pairwise graded (G2)": (G, GE)}
                est0 = [np.full(n, 1 - lab[rf].mean()) for lab in (bun, ncw)]
                LE = np.c_[GE, ~bun, ~ncw].astype(float)
                variants["control: routed, no pointwise"] = (np.c_[G, np.where(routed, est0[0], inf), np.where(routed, est0[1], inf)], LE)
                for vname, X in (("router: cells", None), ("router: logistic", Xl), ("router: jury pointwise", Xj)):
                    est = []
                    for lab in (bun, ncw):
                        if vname == "router: cells":
                            cell = np.array([f"{int(r[2] * 3)}|{int(r[3] * 3)}|{int(r[4] < .15)}" for r in Xl])
                            hit = {c: (lab[rf][cell[rf] == c].mean() if (cell[rf] == c).sum() >= 20 else 0.0) for c in set(cell)}
                            est.append(np.array([1 - hit[c] for c in cell]))
                        else:
                            if lab[rf].all() or not lab[rf].any():
                                est.append(np.full(n, np.inf)); continue
                            c = LogisticRegression(max_iter=5000).fit(X[rf], lab[rf]); est.append(1 - c.predict_proba(X)[:, 1])
                    variants[vname] = (np.c_[G, np.where(routed, est[0], inf), np.where(routed, est[1], inf)], LE)
                base = {}
                for vname, (S, E) in variants.items():
                    for al in AL:
                        for rule, joint in (("overall", False), ("every level", True)):
                            ok, ch = claims(S[te], fst(S[cal], E[cal], S[fit], E[fit], al, joint, delta))
                            e = E[te][np.arange(len(te)), np.maximum(ch, 0)]
                            st, ti, wk, lv = ok & (ch == 0), ok & (ch == 1), ok & (ch == 2), ok & (ch >= 3)
                            kk = int(ok.sum()); risk = float(e[ok].mean()) if kk else np.nan
                            _, inv, cnt = np.unique(groups[te], return_inverse=True, return_counts=True)   # source-weighted test risk
                            ag = np.bincount(inv, ok.astype(float)) / cnt; eg = np.bincount(inv, (e * ok).astype(float)) / cnt
                            rsrc = float(eg.sum() / ag.sum()) if ag.sum() else np.nan
                            if vname == "pairwise graded (G2)":
                                base[(al, rule)] = float(ok.mean())
                            out.append(dict(dataset=spec, judge=j, n=n, human_tie=float((y == 2).mean()), no_clear_winner=float(ncw.mean()),
                                            both_unacceptable=float(bun.mean()), variant=vname, rule=rule, alpha=al, split=sp,
                                            acceptance=float(ok.mean()), decisive=float(st.mean()), tie_claims=float(ti.mean()),
                                            weak=float(wk.mean()), level=float(lv.mean()), level_correct=float((1 - e[lv]).sum()),
                                            level_claims=int(lv.sum()), accepted=kk, errors=int(e[ok].sum()) if kk else 0, risk=risk,
                                            violated=(float(risk > al) if kk else np.nan),
                                            significant=(float(binom.sf(int(e[ok].sum()) - 1, kk, al) < .05) if kk else np.nan),
                                            gain_vs_g2=float(ok.mean()) - base.get((al, rule), np.nan),
                                            risk_source=rsrc, violated_source=(float(rsrc > al) if kk else np.nan),
                                            routed=(float(routed[te].mean()) if vname != "pairwise graded (G2)" else 0.0)))
            log(f"done {spec} {j} (n={n})")
    return out


def summarize(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["judge"], r["variant"], r["rule"], r["alpha"])].append(r)
    out = []
    for (ds, j, v, rule, al), R in g.items():
        risk = [r["risk"] for r in R if r["risk"] == r["risk"]]; vi = [r["violated"] for r in R if r["violated"] == r["violated"]]
        sg = [r["significant"] for r in R if r["significant"] == r["significant"]]; lc = sum(r["level_claims"] for r in R)
        dec = np.mean([r["decisive"] for r in R]); lev = np.mean([r["level"] for r in R])
        ka = sum(r["accepted"] for r in R)
        out.append(dict(dataset=ds, judge=j, variant=v, rule=rule, alpha=al, n=R[0]["n"], human_tie=R[0]["human_tie"],
                        tie_claims=np.mean([r.get("tie_claims", np.nan) for r in R]),
                        gain_vs_g2=np.mean([r.get("gain_vs_g2", np.nan) for r in R]),
                        pooled_risk=(sum(r["errors"] for r in R) / ka if ka else np.nan),
                        splits_answered=sum(r["accepted"] > 0 for r in R),
                        guarantee_success_all=float(1 - np.mean([r["violated"] == 1.0 for r in R])),
                        risk_source=(np.nanmean([r.get("risk_source", np.nan) for r in R]) if any(r.get("risk_source", np.nan) == r.get("risk_source", np.nan) for r in R) else np.nan),
                        guarantee_success_source=float(1 - np.mean([r.get("violated_source") == 1.0 for r in R])),
                        no_clear_winner=R[0]["no_clear_winner"], both_unacceptable=R[0]["both_unacceptable"], splits=len(R),
                        acceptance=np.mean([r["acceptance"] for r in R]), informative=dec + lev, decisive=dec, level=lev,  # informative = strict + absolute-level (tie claims reported separately)
                        level_precision=(sum(r["level_correct"] for r in R) / lc if lc else np.nan), weak=np.mean([r["weak"] for r in R]),
                        risk=(np.mean(risk) if risk else np.nan), human_agreement=(1 - np.mean(risk) if risk else np.nan),
                        violation=(np.mean(vi) if vi else np.nan), guarantee_success=(1 - np.mean(vi) if vi else np.nan),
                        significant_violation=(np.mean(sg) if sg else np.nan), routed=np.mean([r["routed"] for r in R])))
    return out
