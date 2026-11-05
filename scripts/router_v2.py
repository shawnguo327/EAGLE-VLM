"""Router v2: pairwise judging, handed to pointwise ABSOLUTE-LEVEL claims where the pairwise judge is uncertain.
Claims (strongest first): strict 'k better' | 'both unacceptable' | 'no clear winner' | weak 'k not worse' | abstain
  level labels from human absolute references:
    mllm_judge_sp  human 1-5 per answer: no clear winner = |gap| <= 1, both unacceptable = both <= 2
    genaibench     mean of 3 human 1-5 ratings per image: same thresholds
    imagerewarddb  human overall 1-7 per image: no clear winner = |gap| <= 1, both unacceptable = both <= 3
    genai_arena    votes: no clear winner = tie or both-bad vote, both unacceptable = both-bad vote
Pairwise part: 3-class head on the judge's pairwise features (same as G2) -> strict / weak claims.
Routing: pairs whose forced-choice entropy is in the top half of the FIT split are routed (pointwise is queried only for them).
Level-claim estimators (fit split, routed pairs only), features = pointwise ranks (within judge) of both answers, min, max,
|gap|, variances, P(s<=3), and the pairwise entropy:
  cells     fixed 3x3x2 map (level x gap x uncertainty), estimated error = 1 - cell hit rate
  logistic  logistic regression per level claim
  jury      logistic on the AVERAGE pointwise rank features over all judges with pointwise runs
Certification: fixed-sequence LTT, overall risk and every-level (intersection-union), delta .1, 25/25/50 splits.
Control: 'routed, no pointwise' issues the same level claims on the routed pairs with the routed base rate as the
estimated error (isolates what pointwise adds).
Metrics: acceptance, decisive (strict) share, level-claim share and precision, weak share, informative share
(strict + level), risk, violation, routed share.
usage (judgebench dir): python router_v2.py --out paper_tables/router_v2.md   (also writes router_v2.jsonl)"""
import argparse, collections, json, os, sys
import numpy as np
from scipy.stats import binom, rankdata
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from label_noise_lib import per_judge

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/router_v2.md")
ap.add_argument("--splits", type=int, default=100)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--sets", default="mllm_judge_sp,imagerewarddb,genai_arena/generation,genai_arena/edition")
ap.add_argument("--judges", default="qwen3vl32b,qwen3vl8b,internvl35_8b,qwen35_9b,qwen25vl7b,internvl35_38b")
a = ap.parse_args()
AL = [0.1, 0.2, 0.3]; GRID = np.linspace(0, 1, 201)[1:]; D = np.arange(1, 10)


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


def fst(Sc, Ec, Sf, Ef, al, joint):
    s0 = int(np.argmin([pj(Sf, Ef, g, al, joint) for g in GRID])); best = -np.inf
    for g in GRID[s0:]:
        if pj(Sc, Ec, g, al, joint) <= a.delta:
            best = g
        else:
            break
    return best


def level_labels(ds, I, ids):
    if "scale" in I[ids[0]].get("meta", {}):                          # agiqa3k / polaris: a quarter of the score range
        lo, hi = I[ids[0]]["meta"]["scale"]; q = (hi - lo) / 4
        s = np.array([I[i]["meta"]["scores"] for i in ids], float); return np.abs(s[:, 0] - s[:, 1]) <= q, s.max(1) <= lo + q
    if ds in ("mllm_judge_sp", "genaibench"):                       # human 1-5 per answer / mean of 3 raters per image
        s = np.array([I[i]["meta"]["scores"] for i in ids], float); return np.abs(s[:, 0] - s[:, 1]) <= 1, s.max(1) <= 2
    if ds == "imagerewarddb":
        s = np.array([[c["overall"] for c in I[i]["meta"]["cand_info"]] for i in ids], float)
        return np.abs(s[:, 0] - s[:, 1]) <= 1, s.max(1) <= 3
    v = np.array([I[i]["meta"]["vote_type"] for i in ids]); return np.isin(v, ["tievote", "bothbad_vote"]), v == "bothbad_vote"


def pointwise(ds, j, ids):
    R = load(ds, j, conds={"score"}); M = np.full((len(ids), 6), np.nan)
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
    rk = (rankdata(np.r_[M[:, 0], M[:, 3]]) / (2 * len(M))).reshape(2, -1).T     # rank of E[s] within the judge
    lo, hi = rk.min(1), rk.max(1)
    return np.c_[rk, lo, hi, np.abs(rk[:, 0] - rk[:, 1]), M[:, [1, 4, 2, 5]], u]


rec = []; L = ["# Router v2: pairwise -> pointwise absolute-level claims", "",
               f"{a.splits} splits, delta {a.delta}.  Cell = acceptance / decisive / level claims (precision) / weak / risk / violation; "
               "informative = strict + level claims.  routed = share of pairs that query pointwise.", ""]
for spec in a.sets.split(","):
    ds, sub = (spec.split("/") + [None])[:2]
    I = items(ds, rest=(ds not in ("mllm_judge_sp", "genaibench", "agiqa3k", "polaris"))); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    ids_all = sorted(I)
    PJ = {j: per_judge(ds, j, I) for j in a.judges.split(",")}
    PW = {j: pointwise(ds, j, ids_all) for j in PJ if len(PJ[j]) >= .95 * len(I)}
    PW = {j: v for j, v in PW.items() if v is not None}
    if not PW:
        continue
    ncw_all, bun_all = level_labels(ds, I, ids_all)
    jury = np.mean([lvl_feats(PW[j], np.zeros(len(ids_all)))[:, :8] for j in PW], 0)
    for j in [j for j in PJ if j in PW]:
        P = PJ[j]; keep = np.array([i in P for i in ids_all]); ids = [i for i in ids_all if i in P]
        n = len(ids); y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
        u = H(p); Fp = np.c_[np.log(Q + 1e-9), u, np.abs(p - .5), fl]
        ncw, bun = ncw_all[keep], bun_all[keep]
        Xl = lvl_feats(PW[j][keep], u); Xj = np.c_[jury[keep], u]
        R = collections.defaultdict(list); rng = np.random.default_rng(0)
        for _ in range(a.splits):
            pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
            m = LogisticRegression(max_iter=5000).fit(Fp[fit], y[fit]); Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(Fp)
            k = Pr[:, :2].argmax(1); o = 1 - k
            s_str, s_tie, s_weak = 1 - Pr[np.arange(n), k], 1 - Pr[:, 2], Pr[np.arange(n), o]
            e_str, e_tie, e_weak = (y != k), (y != 2), (y == o)
            routed = u >= np.quantile(u[fit], .5); rf = fit[routed[fit]]
            inf = np.full(n, np.inf)
            variants = {"pairwise graded (G2)": (np.c_[s_str, s_tie, s_weak], np.c_[e_str, e_tie, e_weak])}
            # control: same routing and level claims, NO pointwise - estimated error = base rate among routed fit pairs
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
                S = np.c_[s_str, np.where(routed, est[0], inf), np.where(routed, est[1], inf), s_weak]
                E = np.c_[e_str, ~bun, ~ncw, e_weak].astype(float)
                variants[vname] = (S, E)
            for vname, (S, E) in variants.items():
                for al in AL:
                    for rule, joint in (("overall", False), ("every level", True)):
                        ok, ch = claims(S[te], fst(S[cal], E[cal], S[fit], E[fit], al, joint))
                        e = E[te][np.arange(len(te)), np.maximum(ch, 0)]; W = S.shape[1] - 1
                        lv = ok & (ch > 0) & (ch < W); st = ok & (ch == 0)
                        R[(vname, rule, al)].append((ok.mean(), st.mean(), lv.mean(), (1 - e[lv]).sum(), lv.sum(), (ok & (ch == W)).mean(),
                                                     e[ok].mean() if ok.any() else np.nan, float(e[ok].mean() > al) if ok.any() else np.nan,
                                                     routed[te].mean() if vname != "pairwise graded (G2)" else 0.0))
        L += [f"## {spec} / {j}: n = {n}, human tie {np.mean(y == 2):.2f}, no-clear-winner {ncw.mean():.2f}, both-unacceptable {bun.mean():.2f}", "",
              "| variant | rule | " + " | ".join(f"alpha {x}" for x in AL) + " | routed |", "|---|---|" + "---|" * len(AL) + "---|"]
        for (vname, rule, al), Z in sorted(R.items(), key=lambda kv: (list(variants).index(kv[0][0]), kv[0][1], kv[0][2])):
            pass
        for vname in variants:
            for rule in ("overall", "every level"):
                cells = []
                for al in AL:
                    Z = np.array(R[(vname, rule, al)], float)
                    prec = Z[:, 3].sum() / Z[:, 4].sum() if Z[:, 4].sum() else np.nan
                    d = dict(set=spec, judge=j, variant=vname, rule=rule, alpha=al, accept=Z[:, 0].mean(), decisive=Z[:, 1].mean(),
                             level=Z[:, 2].mean(), level_prec=prec, weak=Z[:, 5].mean(), informative=Z[:, 1].mean() + Z[:, 2].mean(),
                             risk=float(np.nanmean(Z[:, 6])), violation=float(np.nanmean(Z[:, 7])), routed=Z[:, 8].mean())
                    rec.append(d)
                    cells.append(f"{d['accept']:.2f} / {d['decisive']:.2f} / {d['level']:.2f} ({'none' if np.isnan(prec) else f'{prec:.2f}'}) / "
                                 f"{d['weak']:.2f} / {d['risk']:.3f} / {d['violation']:.2f}")
                L.append(f"| {vname} | {rule} | " + " | ".join(cells) + f" | {d['routed']:.2f} |")
        L.append(""); print(spec, j, flush=True)
        open(a.out, "w").write("\n".join(L) + "\n")
        with open(a.out.replace(".md", ".jsonl"), "w") as f:
            for d in rec:
                f.write(json.dumps(d) + "\n")
