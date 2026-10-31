import pickle, json, itertools, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from common import *

R = np.random.default_rng(0)
P = pickle.load(open(f"{WS}/out/prep.pkl", "rb")); L = P["ladder"]; T = P["targets"]
G = {}
for l in open(f"{WS}/out/gens.jsonl"):
    r = json.loads(l); G[r["i"]] = [[tuple(x) for x in recs] for recs in r["recs"]]
gidx = sorted(i for i in G if G[i] and len(G[i][0]) >= 2)
S = {"n_targets": len(T), "n_targets_with_gen": len(gidx)}


def boot(x, B=1000, f=np.mean):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return [float("nan")] * 3
    bs = [f(x[R.integers(0, len(x), len(x))]) for _ in range(B)]
    return [float(f(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def fmt(c, p=3):
    return f"{c[0]:.{p}f} [{c[1]:.{p}f}, {c[2]:.{p}f}]"

md = []
# ---------------- generation quality ----------------
rows = []
for i in gidx:
    t = T[i]
    ms = {f"gen{j}": all_metrics(g, t["real"]) for j, g in enumerate(G[i]) if g}
    for b in ["copy7", "copy1", "markov", "stayhome", "other_day"]:
        ms[b] = all_metrics(t[b], t["real"])
    for k, v in ms.items():
        rows.append({"i": i, "method": k, **v})
gm = pd.DataFrame(rows); gm.to_csv(f"{WS}/out/gen_metrics.csv", index=False)
gm["kind"] = gm.method
gmean = gm[gm.method.str.startswith("gen")].groupby("i")[["dtw", "geobleu", "slot_km"]].mean().reset_index()
gmean["method"] = gmean["kind"] = "llm_gen(mean of 6 samples)"
gm = pd.concat([gm[~gm.method.str.startswith("gen") | (gm.method == "gen0")].assign(kind=lambda x: x.method.replace({"gen0": "llm_gen(sample 0)"})), gmean, gm[gm.method.str.startswith("gen") & (gm.method != "gen0")]], ignore_index=True)
gm_all = gm.copy()
S["gen_quality"] = {}
md.append("## Generation quality vs actual next day (mean [95% bootstrap CI]; DTW and slot error in km, lower better; GEO-BLEU-like higher better)\n")
md.append("| method | DTW km/step | median DTW | GEO-BLEU-like | slot error km | median slot err |\n|---|---|---|---|---|---|")
for k, g in gm[gm.kind.isin(["llm_gen(sample 0)", "llm_gen(mean of 6 samples)", "copy7", "copy1", "markov", "stayhome", "other_day"])].groupby("kind"):
    d = {m: boot(g[m]) for m in ["dtw", "geobleu", "slot_km"]}
    d["dtw_median"] = float(g.dtw.median()); d["slot_km_median"] = float(g.slot_km.median())
    S["gen_quality"][k] = d
    md.append(f"| {k} | {fmt(d['dtw'],2)} | {d['dtw_median']:.2f} | {fmt(d['geobleu'])} | {fmt(d['slot_km'],2)} | {d['slot_km_median']:.2f} |")
pv = gm.pivot_table(index="i", columns="kind", values="dtw")
S["paired_dtw_llmmean_minus_copy7"] = boot(pv["llm_gen(mean of 6 samples)"] - pv["copy7"])
S["paired_dtw_llmmean_minus_copy1"] = boot(pv["llm_gen(mean of 6 samples)"] - pv["copy1"])
md.append(f"\nPaired DTW difference (LLM mean-of-6 minus copy Day -7): {fmt(S['paired_dtw_llmmean_minus_copy7'],2)} km; minus copy Day -1: {fmt(S['paired_dtw_llmmean_minus_copy1'],2)} km.")
# diversity / copying
div, cp = [], []
for i in gidx:
    seqs = [tuple(seq_of(g)) for g in G[i] if g]
    div.append(len(set(seqs)))
    c7 = tuple(seq_of(T[i]["copy7"]))
    cp.append(np.mean([dtw(list(s), list(c7)) if s else np.nan for s in seqs]))
S["gen_distinct_of_6"] = boot(div); S["gen_dtw_to_day-7_km"] = boot(cp)
S["gen_parse_fail_rate"] = float(np.mean([len(g) < 2 for i in G for g in G[i]]))
md.append(f"\nDistinct location sequences among the 6 samples per user-day: {fmt(S['gen_distinct_of_6'],2)}; "
          f"mean DTW of LLM samples to the Day -7 record: {fmt(S['gen_dtw_to_day-7_km'],2)} km; parse-fail (<2 records) rate {S['gen_parse_fail_rate']:.3f}.\n")

# ---------------- LR feature baseline ----------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
S["lr_auc"] = {}
Xr = np.array([feats(r) for r in L["real"][:300]])
for rung in ["gauss1", "gauss2", "gauss4", "gauss8", "tshuffle", "markov", "uniform", "llmgen"]:
    fk = [G[i][0] for i in gidx] if rung == "llmgen" else L[rung][300:600]
    Xf = np.array([feats(r) for r in fk])
    X = np.vstack([Xr, Xf]); y = np.r_[np.ones(len(Xr)), np.zeros(len(Xf))]
    pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=5, method="predict_proba")[:, 1]
    S["lr_auc"][rung] = float(roc_auc_score(y, pr))

# ---------------- context-free judge ----------------
order = ["real", "gauss1", "gauss2", "gauss4", "gauss8", "uniform"]
if os.path.exists(f"{WS}/out/judge_cf.jsonl"):
    cf = pd.read_json(f"{WS}/out/judge_cf.jsonl", lines=True)
    S["cf_calls"] = 2 * len(cf)
    md.append("## Q1: context-free judge, real vs rung (pairs from different source user-days; order-averaged)\n")
    md.append("| rung | n | judge acc (p_real>0.5) | mean p_real | mean Δp | order-consistent (same pick both orders) | LR-feature AUC (5-fold) |\n|---|---|---|---|---|---|---|")
    S["cf_real_vs"] = {}
    for rung, g in cf[cf.a == "real"].groupby("b"):
        acc = boot((g.p_a > 0.5).astype(float)); pm = boot(g.p_a)
        cons = float(np.mean((g.pA_ab > 0.5) != (g.pA_ba > 0.5)))
        S["cf_real_vs"][rung] = {"n": len(g), "acc": acc, "p_real": pm, "dp": float(g.dp.mean()), "consistency": cons}
        md.append(f"| {rung} | {len(g)} | {fmt(acc)} | {fmt(pm)} | {g.dp.mean():.3f} | {cons:.2f} | {S['lr_auc'].get(rung, float('nan')):.3f} |")
    # Bradley-Terry over all rungs from soft outcomes
    rungs = sorted(set(cf.a) | set(cf.b) - {"real2"})
    idx = {r: j for j, r in enumerate(rungs)}
    cf2 = cf[cf.b != "real2"]

    def bt(df):
        th = np.zeros(len(rungs))
        A = df.a.map(idx).values; B = df.b.map(idx).values; w = df.p_a.values
        for _ in range(3000):
            p = 1 / (1 + np.exp(-(th[A] - th[B])))
            gth = np.zeros(len(rungs)); np.add.at(gth, A, w - p); np.add.at(gth, B, -(w - p))
            th += 0.5 * gth / len(df) * len(rungs); th -= th.mean()
        return th
    th = bt(cf2)
    bs = []
    for _ in range(200):
        bs.append(bt(cf2.iloc[R.integers(0, len(cf2), len(cf2))]))
    bs = np.array(bs)
    S["bt"] = {r: [float(th[idx[r]]), float(np.percentile(bs[:, idx[r]], 2.5)), float(np.percentile(bs[:, idx[r]], 97.5))] for r in rungs}
    tau = kendalltau([-j for j in range(len(order))], [th[idx[r]] for r in order])[0]
    taus = [kendalltau([-j for j in range(len(order))], [b[idx[r]] for r in order])[0] for b in bs]
    S["ladder_kendall_tau"] = [float(tau), float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))]
    md.append("\n## Q1: Bradley-Terry scores over all rungs (soft outcomes, 200 bootstrap) \n")
    md.append("| rung | BT score [95% CI] |\n|---|---|")
    for r in sorted(rungs, key=lambda r: -th[idx[r]]):
        md.append(f"| {r} | {fmt(S['bt'][r],2)} |")
    md.append(f"\nKendall τ of BT scores vs true ladder order {order}: {fmt(S['ladder_kendall_tau'],2)}\n")
    # adjacent / ladder pair ordering accuracy
    md.append("| ladder pair (better vs worse) | n | P(better preferred) | mean p_better |\n|---|---|---|---|")
    S["ladder_pairs"] = {}
    for x, y in itertools.combinations(order, 2):
        g = cf[(cf.a == x) & (cf.b == y)]
        if len(g) == 0: continue
        a = boot((g.p_a > 0.5).astype(float))
        S["ladder_pairs"][f"{x}>{y}"] = a
        md.append(f"| {x} > {y} | {len(g)} | {fmt(a)} | {g.p_a.mean():.3f} |")
    # position bias
    S["position"] = {"P(A)_overall": float(np.r_[cf.pA_ab, cf.pA_ba].mean()),
                     "frac_both_orders_pick_A": float(np.mean((cf.pA_ab > 0.5) & (cf.pA_ba > 0.5))),
                     "frac_both_orders_pick_B": float(np.mean((cf.pA_ab < 0.5) & (cf.pA_ba < 0.5))),
                     "first_token_mass_mean": float(np.r_[cf.mass1, cf.mass2].mean())}
    null = cf[cf.b == "real2"]
    S["null_real_vs_real"] = {"n": len(null), "mean_dp": float(null.dp.mean()), "dp_q90": float(null.dp.quantile(.9))}

# ---------------- calibration ----------------
def calib(df, label, bins=(0, .1, .2, .4, .6, .8, 1.0001)):
    out = []
    md.append(f"\n### Δp calibration: {label}\n\n| Δp bin | n | acc | mean conf (=(1+Δp)/2) |\n|---|---|---|---|")
    conf = (1 + df.dp) / 2
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (df.dp >= lo) & (df.dp < hi)
        if m.sum() == 0: continue
        acc = float(df.correct[m].mean())
        out.append({"lo": lo, "hi": hi, "n": int(m.sum()), "acc": acc, "conf": float(conf[m].mean())})
        md.append(f"| [{lo:.1f},{min(hi,1):.1f}) | {int(m.sum())} | {acc:.3f} | {conf[m].mean():.3f} |")
    ece = sum(o["n"] * abs(o["acc"] - o["conf"]) for o in out) / len(df)
    from sklearn.metrics import roc_auc_score as auc
    a = float(auc(df.correct, df.dp)) if df.correct.nunique() > 1 else float("nan")
    md.append(f"\nECE={ece:.3f}; AUROC(Δp → correct)={a:.3f}; overall acc={df.correct.mean():.3f}, n={len(df)}\n")
    for tau_ in [0.1, 0.2, 0.4, 0.6]:
        m = df.dp >= tau_
        md.append(f"- Δp≥{tau_}: coverage {m.mean():.2f}, acc {df.correct[m].mean() if m.any() else float('nan'):.3f}")
    return {"bins": out, "ece": ece, "auroc": a, "acc": float(df.correct.mean()), "n": len(df)}

if os.path.exists(f"{WS}/out/judge_cf.jsonl"):
    md.append("\n## Q3: Δp calibration")
    known = cf[cf.b != "real2"].copy()
    rank = {r: j for j, r in enumerate(order)}
    kk = known[known.a.isin(order) & known.b.isin(order)].copy()
    kk["correct"] = (kk.p_a > 0.5) == (kk.a.map(rank) < kk.b.map(rank))
    S["calib_cf_ladder"] = calib(kk, "context-free, all ladder pairs (real, gauss1-8, uniform)")
    rv = known[known.a == "real"].copy(); rv["correct"] = rv.p_a > 0.5
    S["calib_cf_real_vs_any"] = calib(rv, "context-free, real vs every non-real rung")

if os.path.exists(f"{WS}/out/judge_cc.jsonl"):
    cc = pd.read_json(f"{WS}/out/judge_cc.jsonl", lines=True)
    S["cc_calls"] = 2 * len(cc)
    md.append("\n## Q2: context-conditioned judge — actual next day vs candidate (given past 7 days)\n")
    md.append("| candidate | n | judge acc (picks actual) | mean p_actual | mean Δp | order-consistent |\n|---|---|---|---|---|---|")
    S["cc"] = {}
    for b, g in cc.groupby("b"):
        acc = boot((g.p_a > 0.5).astype(float))
        S["cc"][b] = {"n": len(g), "acc": acc, "p_actual": float(g.p_a.mean()), "dp": float(g.dp.mean())}
        md.append(f"| {b} | {len(g)} | {fmt(acc)} | {g.p_a.mean():.3f} | {g.dp.mean():.3f} | {np.mean((g.pA_ab > 0.5) != (g.pA_ba > 0.5)):.2f} |")
    cands_of = lambda i, b: {"llmgen": G[i][0], "other_day": T[i]["other_day"], "markov": T[i]["markov"], "copy7": T[i]["copy7"],
                             "other_user": T[gidx[(gidx.index(i) + 1) % len(gidx)]]["real"], "stayhome": T[i]["stayhome"]}[b]
    def ctxdist(i, recs):
        sq = seq_of(recs)
        return min(dtw(sq, seq_of(r)) for k, r in T[i]["ctx"] if r) if sq else np.nan
    cs = []
    for _, r in cc.iterrows():
        i = int(r.i); dr = ctxdist(i, T[i]["real"]); dc = ctxdist(i, cands_of(i, r.b))
        cs.append((r.b, r.p_a, dr, dc))
    cs = pd.DataFrame(cs, columns=["b", "p_a", "d_real", "d_cand"])
    cs = cs[cs.d_real != cs.d_cand]
    agree_ctx = ((cs.p_a > 0.5) == (cs.d_real < cs.d_cand))
    S["cc_pick_closer_to_context"] = boot(agree_ctx.astype(float))
    S["cc_pick_closer_to_context_by_cand"] = {b: float(g.mean()) for b, g in agree_ctx.groupby(cs.b)}
    S["cc_real_closer_to_context_than_cand"] = {b: float((g.d_real < g.d_cand).mean()) for b, g in cs.groupby("b")}
    md.append(f"\nTypicality check: fraction of context-conditioned decisions where the judge picks the candidate closer (min DTW) to one of the 7 context days: {fmt(S['cc_pick_closer_to_context'])}; by candidate: "
              + ", ".join(f"{b} {v:.2f}" for b, v in S["cc_pick_closer_to_context_by_cand"].items())
              + ". Fraction where the actual day is closer to context than the candidate: " + ", ".join(f"{b} {v:.2f}" for b, v in S["cc_real_closer_to_context_than_cand"].items()) + "\n")
    cc["correct"] = cc.p_a > 0.5
    S["calib_cc"] = calib(cc, "context-conditioned, actual vs all candidates")
    # does the judge's preference for a candidate over truth track candidate quality?
    q = gm.set_index(["i", "method"])
    xs = []
    for _, r in cc.iterrows():
        meth = "gen0" if r.b == "llmgen" else r.b
        if (r.i, meth) in q.index:
            xs.append((r.b, 1 - r.p_a, q.loc[(r.i, meth)].dtw))
    xs = pd.DataFrame(xs, columns=["b", "p_cand", "dtw"])
    from scipy.stats import spearmanr
    S["cc_pcand_vs_dtw_spearman"] = float(spearmanr(xs.p_cand, xs.dtw, nan_policy="omit")[0])

if os.path.exists(f"{WS}/out/judge_bon.jsonl"):
    bon = pd.read_json(f"{WS}/out/judge_bon.jsonl", lines=True)
    S["bon_calls"] = 2 * len(bon)
    res = {k: [] for k in ["random", "judge", "oracle", "worst", "medoid", "gated0.2", "gated0.4", "gated0.6"]}
    agree = []; close_pick = []
    for i, g in bon.groupby("i"):
        n = len(G[i]); sc = np.zeros(n); cnt = np.zeros(n); conf = np.zeros(n)
        for _, r in g.iterrows():
            ra_, rb_ = int(r.a), int(r.b)
            sc[ra_] += r.p_a; sc[rb_] += 1 - r.p_a; cnt[ra_] += 1; cnt[rb_] += 1
            conf[ra_] += r.dp; conf[rb_] += r.dp
        valid = cnt > 0
        sc = np.where(valid, sc / np.maximum(cnt, 1), -1)
        ms = gm[(gm.i == i) & gm.method.str.startswith("gen")].set_index("method")
        d = np.array([ms.loc[f"gen{j}"].dtw if f"gen{j}" in ms.index else np.nan for j in range(n)])
        gb = np.array([ms.loc[f"gen{j}"].geobleu if f"gen{j}" in ms.index else np.nan for j in range(n)])
        sk = np.array([ms.loc[f"gen{j}"].slot_km if f"gen{j}" in ms.index else np.nan for j in range(n)])
        vj = [j for j in range(n) if valid[j] and not np.isnan(d[j])]
        if len(vj) < 2: continue
        pick = max(vj, key=lambda j: sc[j])
        seqs = {j: seq_of(G[i][j]) for j in vj}
        med = min(vj, key=lambda j: np.mean([dtw(seqs[j], seqs[k]) for k in vj if k != j]))
        # gate: margin of top pick = its mean soft win-rate minus 0.5, scaled to [0,1]
        margin = 2 * sc[pick] - 1
        triple = lambda j: (d[j], gb[j], sk[j])
        res["random"].append((np.nanmean(d[vj]), np.nanmean(gb[vj]), np.nanmean(sk[vj])))
        res["judge"].append(triple(pick)); res["medoid"].append(triple(med))
        res["oracle"].append(triple(min(vj, key=lambda j: d[j]))); res["worst"].append(triple(max(vj, key=lambda j: d[j])))
        for tg in [0.2, 0.4, 0.6]:
            res[f"gated{tg}"].append(triple(pick if margin >= tg else med))
        agree.append(margin)
        cd = {j: min(dtw(seqs[j], seq_of(r)) for k, r in T[i]["ctx"] if r) for j in vj}
        close_pick.append(cd[pick] <= min(cd.values()) + 1e-9)
    S["bon"] = {}
    md.append("\n## Q2: best-of-n selection among the 6 LLM samples (context-conditioned judge, round-robin, soft win-rate)\n")
    md.append("| selector | DTW km/step | GEO-BLEU-like | slot error km |\n|---|---|---|---|")
    for k, v in res.items():
        v = np.array(v)
        S["bon"][k] = {"dtw": boot(v[:, 0]), "geobleu": boot(v[:, 1]), "slot_km": boot(v[:, 2]), "n": len(v)}
        md.append(f"| {k} | {fmt(S['bon'][k]['dtw'],2)} | {fmt(S['bon'][k]['geobleu'])} | {fmt(S['bon'][k]['slot_km'],2)} |")
    # paired difference judge - random
    jd = np.array(res["judge"])[:, 0] - np.array(res["random"])[:, 0]
    S["bon_judge_minus_random_dtw"] = boot(jd)
    md.append(f"\nPaired DTW(judge pick) − DTW(random): {fmt(S['bon_judge_minus_random_dtw'],3)} km (n={len(jd)}); "
              f"gate margins: mean {np.mean(agree):.2f}; fraction ≥0.2: {np.mean(np.array(agree)>=0.2):.2f}, ≥0.4: {np.mean(np.array(agree)>=0.4):.2f}, ≥0.6: {np.mean(np.array(agree)>=0.6):.2f}")
    S["bon_pick_is_closest_to_context"] = float(np.mean(close_pick))
    md.append(f"Judge pick is the sample closest to a context day in {np.mean(close_pick):.2f} of cases.")
    md.append("(gatedτ: use judge pick if its margin 2·winrate−1 ≥ τ, else medoid of samples)")

for job, title in [("cfm", "Q1b: context-free judge, MATCHED pairs (real day vs its own perturbation)"),
                   ("cfh", "Q1d: context-free judge with an explicit hint describing jitter artefacts (unmatched pairs)"),
                   ("ccd", "Q2b: context-conditioned judge with a de-biasing instruction ('the real day is often not a copy of history')"),
                   ("cft", "Q1c: thinking-mode judge (Qwen3-8B, enable_thinking, hard vote per order), unmatched pairs")]:
    if os.path.exists(f"{WS}/out/judge_{job}.jsonl"):
        dj = pd.read_json(f"{WS}/out/judge_{job}.jsonl", lines=True)
        S[f"{job}_calls"] = 2 * len(dj); S[job] = {}
        md.append(f"\n## {title}\n\n| rung | n | acc (p_real>0.5; ties count 0.5) | mean p_real | order-consistent | parse ok |\n|---|---|---|---|---|---|")
        for b, g in dj.groupby("b"):
            acc = boot(np.where(g.p_a > 0.5, 1.0, np.where(g.p_a == 0.5, 0.5, 0.0)))
            S[job][b] = {"n": len(g), "acc": acc, "p_real": float(g.p_a.mean())}
            md.append(f"| {b} | {len(g)} | {fmt(acc)} | {g.p_a.mean():.3f} | {np.mean((g.pA_ab > 0.5) != (g.pA_ba > 0.5)):.2f} | {np.r_[g.mass1, g.mass2].mean():.2f} |")
        if job == "cfm":
            from scipy.stats import spearmanr
            dd = [dtw(seq_of(L["real"][int(r.k)]), seq_of(L[r.b][int(r.k)])) for _, r in dj.iterrows()]
            dj["distort"] = dd
            g = dj[dj.b.str.startswith("gauss")]
            S["cfm_spearman_preal_vs_dtw_distortion"] = float(spearmanr(g.p_a, g.distort)[0])
            md.append(f"\nSpearman(p_real, DTW distortion of the perturbed copy) over gauss rungs: {S['cfm_spearman_preal_vs_dtw_distortion']:.3f} (n={len(g)})")
            dj["correct"] = dj.p_a > 0.5
            S["calib_cfm"] = calib(dj, "matched real vs perturbation")

if os.path.exists(f"{WS}/out/judge_cfB.jsonl"):
    PB = pickle.load(open(f"{WS}/out/prep_B.pkl", "rb")); LB = PB["ladder"]
    dj = pd.read_json(f"{WS}/out/judge_cfB.jsonl", lines=True)
    S["cfB_calls"] = 2 * len(dj); S["cfB"] = {}; S["lr_auc_B"] = {}
    XrB = np.array([feats(r) for r in LB["real"][:300]])
    for rung in ["gauss1", "gauss2", "gauss4", "gauss8", "tshuffle", "markov", "uniform"]:
        Xf = np.array([feats(r) for r in LB[rung][300:600]])
        X = np.vstack([XrB, Xf]); y = np.r_[np.ones(len(XrB)), np.zeros(len(Xf))]
        pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=5, method="predict_proba")[:, 1]
        S["lr_auc_B"][rung] = float(roc_auc_score(y, pr))
    md.append("\n## Replication on City B (6000-user subsample; same pipeline, context-free judge)\n\n| pair (first should win) | n | acc | mean p_first | order-consistent | LR AUC |\n|---|---|---|---|---|---|")
    for (x, yb), g in dj.groupby(["a", "b"], sort=False):
        acc = boot((g.p_a > 0.5).astype(float)); S["cfB"][f"{x}>{yb}"] = {"n": len(g), "acc": acc, "p": float(g.p_a.mean())}
        md.append(f"| {x} > {yb} | {len(g)} | {fmt(acc)} | {g.p_a.mean():.3f} | {np.mean((g.pA_ab > 0.5) != (g.pA_ba > 0.5)):.2f} | {S['lr_auc_B'].get(yb, float('nan')) if x == 'real' else float('nan'):.3f} |")

# position bias per job
S["position_by_job"] = {}
for job in ["cf", "cfm", "cc", "bon", "cfh", "ccd", "cfB"]:
    fp = f"{WS}/out/judge_{job}.jsonl"
    if os.path.exists(fp):
        dj = pd.read_json(fp, lines=True)
        S["position_by_job"][job] = {"P(A)": float(np.r_[dj.pA_ab, dj.pA_ba].mean()),
                                     "order_consistent": float(np.mean((dj.pA_ab > 0.5) != (dj.pA_ba > 0.5))),
                                     "both_A": float(np.mean((dj.pA_ab > 0.5) & (dj.pA_ba > 0.5))),
                                     "both_B": float(np.mean((dj.pA_ab < 0.5) & (dj.pA_ba < 0.5)))}
md.append("\n## Position bias by job\n\n| job | mean P(A) | same item both orders | A both orders | B both orders |\n|---|---|---|---|---|")
for k, v in S["position_by_job"].items():
    md.append(f"| {k} | {v['P(A)']:.3f} | {v['order_consistent']:.2f} | {v['both_A']:.2f} | {v['both_B']:.2f} |")

calls = {}
for f in ["gen", "cf", "cc", "bon", "cfm", "cft", "cfh", "ccd", "cfB"]:
    if os.path.exists(f"{WS}/out/usage_{f}.json"):
        calls[f] = json.load(open(f"{WS}/out/usage_{f}.json"))
S["usage"] = calls
json.dump(S, open(f"{WS}/out/summary.json", "w"), indent=1)
open(f"{WS}/out/results_tables.md", "w").write("\n".join(md))
print("\n".join(md))
print(json.dumps({k: S[k] for k in ["lr_auc", "usage"] + [k for k in ["position", "null_real_vs_real", "cc_pcand_vs_dtw_spearman"] if k in S]}, indent=1))
