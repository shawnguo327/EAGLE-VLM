"""Router experiment ('router'): pairwise graded claims, with the pairwise-uncertain half routed to pointwise
absolute-level claims ('both unacceptable', 'no clear winner').  Ported unchanged from scripts/router_v2.py.
Level labels from human absolute references:
  mllm_judge_sp / genaibench   human 1-5 (mean of raters): no clear winner = |gap| <= 1, both unacceptable = both <= 2
  imagerewarddb                human overall 1-7: |gap| <= 1, both <= 3
  datasets with meta.scale     a quarter of the score range (polaris 0-1, agiqa3k 0-5)
  genai_arena                  votes: no clear winner = tie or both-bad vote, both unacceptable = both-bad vote
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
    ps = [0.0]
    for c in range(S.shape[1]):
        mk = ok & (ch == c)
        if mk.any():
            ps.append(binom.cdf(int(E[mk, c].sum()), int(mk.sum()), al))
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
    v = np.array([I[i]["meta"]["vote_type"] for i in ids]); return np.isin(v, ["tievote", "bothbad_vote"]), v == "bothbad_vote"


def pointwise(store, ds, j, ids):
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
    M[np.isnan(M).any(1)] = np.nanmean(M, 0)
    return M


def lvl_feats(M, u):
    rk = (rankdata(np.r_[M[:, 0], M[:, 3]]) / (2 * len(M))).reshape(2, -1).T
    lo, hi = rk.min(1), rk.max(1)
    return np.c_[rk, lo, hi, np.abs(rk[:, 0] - rk[:, 1]), M[:, [1, 4, 2, 5]], u]


def run(cfg, store, log):
    from eaglevlm.pipeline.io import NO_REST
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
        jury = np.mean([lvl_feats(PW[j], np.zeros(len(ids_all)))[:, :8] for j in PW], 0)
        for j in [j for j in PJ if j in PW]:
            P = PJ[j]; keep = np.array([i in P for i in ids_all]); ids = [i for i in ids_all if i in P]
            n = len(ids); y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
            p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
            u = H(p); Fp = np.c_[np.log(Q + 1e-9), u, np.abs(p - .5), fl]
            ncw, bun = ncw_all[keep], bun_all[keep]
            Xl = lvl_feats(PW[j][keep], u); Xj = np.c_[jury[keep], u]
            rng = np.random.default_rng(cfg.get("seed", 0))
            for sp in range(cfg["n_splits"]):
                pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
                m = LogisticRegression(max_iter=5000).fit(Fp[fit], y[fit]); Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(Fp)
                k = Pr[:, :2].argmax(1); o = 1 - k
                s_str, s_tie, s_weak = 1 - Pr[np.arange(n), k], 1 - Pr[:, 2], Pr[np.arange(n), o]
                e_str, e_tie, e_weak = (y != k), (y != 2), (y == o)
                routed = u >= np.quantile(u[fit], .5); rf = fit[routed[fit]]
                inf = np.full(n, np.inf)
                variants = {"pairwise graded (G2)": (np.c_[s_str, s_tie, s_weak], np.c_[e_str, e_tie, e_weak])}
                est0 = [np.full(n, 1 - lab[rf].mean()) for lab in (bun, ncw)]
                variants["control: routed, no pointwise"] = (np.c_[s_str, np.where(routed, est0[0], inf), np.where(routed, est0[1], inf), s_weak],
                                                             np.c_[e_str, ~bun, ~ncw, e_weak].astype(float))
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
                    variants[vname] = (np.c_[s_str, np.where(routed, est[0], inf), np.where(routed, est[1], inf), s_weak],
                                       np.c_[e_str, ~bun, ~ncw, e_weak].astype(float))
                for vname, (S, E) in variants.items():
                    for al in AL:
                        for rule, joint in (("overall", False), ("every level", True)):
                            ok, ch = claims(S[te], fst(S[cal], E[cal], S[fit], E[fit], al, joint, delta))
                            e = E[te][np.arange(len(te)), np.maximum(ch, 0)]; W = S.shape[1] - 1
                            lv = ok & (ch > 0) & (ch < W); st = ok & (ch == 0); kk = int(ok.sum())
                            risk = float(e[ok].mean()) if kk else np.nan
                            out.append(dict(dataset=spec, judge=j, n=n, human_tie=float((y == 2).mean()), no_clear_winner=float(ncw.mean()),
                                            both_unacceptable=float(bun.mean()), variant=vname, rule=rule, alpha=al, split=sp,
                                            acceptance=float(ok.mean()), decisive=float(st.mean()), level=float(lv.mean()),
                                            level_correct=float((1 - e[lv]).sum()), level_claims=int(lv.sum()), weak=float((ok & (ch == W)).mean()),
                                            accepted=kk, errors=int(e[ok].sum()) if kk else 0, risk=risk,
                                            violated=(float(risk > al) if kk else np.nan),
                                            significant=(float(binom.sf(int(e[ok].sum()) - 1, kk, al) < .05) if kk else np.nan),
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
        out.append(dict(dataset=ds, judge=j, variant=v, rule=rule, alpha=al, n=R[0]["n"], human_tie=R[0]["human_tie"],
                        no_clear_winner=R[0]["no_clear_winner"], both_unacceptable=R[0]["both_unacceptable"], splits=len(R),
                        acceptance=np.mean([r["acceptance"] for r in R]), informative=dec + lev, decisive=dec, level=lev,
                        level_precision=(sum(r["level_correct"] for r in R) / lc if lc else np.nan), weak=np.mean([r["weak"] for r in R]),
                        risk=(np.mean(risk) if risk else np.nan), human_agreement=(1 - np.mean(risk) if risk else np.nan),
                        violation=(np.mean(vi) if vi else np.nan), guarantee_success=(1 - np.mean(vi) if vi else np.nan),
                        significant_violation=(np.mean(sg) if sg else np.nan), routed=np.mean([r["routed"] for r in R])))
    return out
