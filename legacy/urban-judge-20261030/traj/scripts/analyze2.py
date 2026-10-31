"""Follow-up round analysis: evidence-augmented judge, hybrid LR/judge gate, Qwen3-14B scale, best-of-n with evidence."""
import pickle, json, itertools, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from common import *

R = np.random.default_rng(0)
P = pickle.load(open(f"{WS}/out/prep.pkl", "rb")); L = P["ladder"]; T = P["targets"]
G = {}
for l in open(f"{WS}/out/gens.jsonl"):
    r = json.loads(l); G[r["i"]] = [[tuple(x) for x in recs] for recs in r["recs"]]
gidx = sorted(i for i in G if G[i] and len(G[i][0]) >= 2)
F = {}
md = ["\n\n# Follow-up round (evidence-augmented judge, hybrid gate, Qwen3-14B)\n"]


def boot(x, B=1000, f=np.mean):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return [float("nan")] * 3
    bs = [f(x[R.integers(0, len(x), len(x))]) for _ in range(B)]
    return [float(f(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def fmt(c, p=3):
    return f"{c[0]:.{p}f} [{c[1]:.{p}f}, {c[2]:.{p}f}]"


def load(job):
    fp = f"{WS}/out/judge_{job}.jsonl"
    return pd.read_json(fp, lines=True) if os.path.exists(fp) and os.path.getsize(fp) > 0 else None


def calib_stats(df):
    conf = (1 + df.dp) / 2; bins = [0, .1, .2, .4, .6, .8, 1.0001]; e = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (df.dp >= lo) & (df.dp < hi)
        if m.any(): e += m.sum() * abs(df.correct[m].mean() - conf[m].mean())
    au = float(roc_auc_score(df.correct, df.dp)) if df.correct.nunique() > 1 else float("nan")
    hi = df.dp >= 0.6
    return {"n": len(df), "acc": float(df.correct.mean()), "ece": float(e / len(df)), "auroc_dp": au,
            "acc_dp>=0.6": float(df.correct[hi].mean()) if hi.any() else float("nan"), "cov_dp>=0.6": float(hi.mean())}


def posbias(df):
    return {"P(A)": float(np.r_[df.pA_ab, df.pA_ba].mean()), "order_consistent": float(np.mean((df.pA_ab > 0.5) != (df.pA_ba > 0.5)))}

# ---------------- LR scores for exactly the judged items ----------------
FAKE = ["gauss1", "gauss2", "gauss4", "gauss8", "tshuffle", "markov", "uniform", "llmgen"]
Xr = np.array([feats(r) for r in L["real"][:300]])
fake_items = {r: ([G[i][0] for i in gidx] if r == "llmgen" else L[r][300:600]) for r in FAKE}
lr_real, lr_fake, lr_auc = {}, {}, {}
pipe = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
for rung in FAKE:
    Xf = np.array([feats(r) for r in fake_items[rung]])
    X = np.vstack([Xr, Xf]); y = np.r_[np.ones(len(Xr)), np.zeros(len(Xf))]
    pr = cross_val_predict(pipe(), X, y, cv=5, method="predict_proba")[:, 1]
    lr_real[rung], lr_fake[rung] = pr[:len(Xr)], pr[len(Xr):]; lr_auc[rung] = float(roc_auc_score(y, pr))
# pooled LR: real vs all fake rungs together (does not know which degradation)
Xall = [Xr] + [np.array([feats(r) for r in fake_items[rg]]) for rg in FAKE]
X = np.vstack(Xall); y = np.r_[np.ones(len(Xr)), np.zeros(len(X) - len(Xr))]
pr = cross_val_predict(pipe(), X, y, cv=5, method="predict_proba")[:, 1]
pool_real = pr[:len(Xr)]; pool_fake = {}; o = len(Xr)
for rg, xa in zip(FAKE, Xall[1:]):
    pool_fake[rg] = pr[o:o + len(xa)]; o += len(xa)
F["lr_auc_per_rung"] = lr_auc
F["lr_auc_pooled"] = {rg: float(roc_auc_score(np.r_[np.ones(300), np.zeros(len(pool_fake[rg]))], np.r_[pool_real, pool_fake[rg]])) for rg in FAKE}


def fake_pos(rung, k):
    return (k + 7) % len(gidx) if rung == "llmgen" else k


def lr_pair_p(rung, k, pooled=False):
    sr = (pool_real if pooled else lr_real[rung])[k]; sf = (pool_fake[rung] if pooled else lr_fake[rung])[fake_pos(rung, k)]
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(lg(sr) - lg(sf))))

# ---------------- (A)/(C) context-free comparisons ----------------
CFJOBS = [("cf", "8B plain"), ("cfe", "8B +stats"), ("cfe2", "8B +stats+real-day primer"), ("cfs", "8B stats-only+primer"), ("cf_14b", "14B plain"), ("cfe_14b", "14B +stats"), ("cfs_14b", "14B stats-only+primer")]
rungs7 = ["gauss1", "gauss2", "gauss4", "gauss8", "tshuffle", "markov", "llmgen"]
D = {j: load(j) for j, _ in CFJOBS}
F["cf_acc"] = {}; F["cf_calib"] = {}; F["cf_null_dp"] = {}; F["cf_position"] = {}
hdr = "| rung | " + " | ".join(n for j, n in CFJOBS if D[j] is not None) + " | LR per-rung (same pairs) | LR pooled (same pairs) | LR AUC per-rung |"
md.append("## A/C. Context-free judge: accuracy of picking the real day (unmatched pairs, n=150 per rung, order-averaged)\n")
md.append(hdr); md.append("|" + "---|" * (hdr.count("|") - 1))
for rung in rungs7 + ["uniform"]:
    cells = []
    for j, n in CFJOBS:
        d = D[j]
        if d is None: continue
        g = d[(d.a == "real") & (d.b == rung)]
        if len(g) == 0: cells.append("—"); continue
        a = boot((g.p_a > 0.5).astype(float)); F["cf_acc"].setdefault(j, {})[rung] = a
        cells.append(fmt(a, 2))
    lrp = np.mean([lr_pair_p(rung, k) > 0.5 for k in range(150)]); lrq = np.mean([lr_pair_p(rung, k, True) > 0.5 for k in range(150)])
    F.setdefault("lr_pair_acc", {})[rung] = float(lrp); F.setdefault("lr_pooled_pair_acc", {})[rung] = float(lrq)
    md.append(f"| {rung} | " + " | ".join(cells) + f" | {lrp:.2f} | {lrq:.2f} | {lr_auc[rung]:.3f} |")
md.append("\n**Calibration / position bias (real vs the 7 non-real rungs above; null = real vs real)**\n")
md.append("| judge | n | acc | ECE | AUROC(Δp→correct) | acc@Δp≥0.6 (cov) | null mean Δp | P(A) | same pick both orders |\n|---|---|---|---|---|---|---|---|---|")
for j, n in CFJOBS:
    d = D[j]
    if d is None: continue
    k = d[(d.a == "real") & d.b.isin(rungs7)].copy(); k["correct"] = k.p_a > 0.5
    if j.endswith("_14b") and j != "cf_14b":
        k = k[k.b.isin(["gauss1", "gauss2", "gauss4", "gauss8", "llmgen"])]
    c = calib_stats(k); F["cf_calib"][j] = c
    nl = d[d.b == "real2"]; F["cf_null_dp"][j] = float(nl.dp.mean()) if len(nl) else float("nan")
    pb = posbias(d); F["cf_position"][j] = pb
    md.append(f"| {n} | {c['n']} | {c['acc']:.3f} | {c['ece']:.3f} | {c['auroc_dp']:.3f} | {c['acc_dp>=0.6']:.3f} ({c['cov_dp>=0.6']:.2f}) | {F['cf_null_dp'][j]:.2f} | {pb['P(A)']:.2f} | {pb['order_consistent']:.2f} |")
md.append("\n**Same-subset comparison (gauss1/2/4/8 + llmgen only, n≤750):**\n\n| judge | acc | ECE | AUROC(Δp) | acc@Δp≥0.6 (cov) |\n|---|---|---|---|---|")
for j, n in CFJOBS:
    d = D[j]
    if d is None: continue
    k = d[(d.a == "real") & d.b.isin(["gauss1", "gauss2", "gauss4", "gauss8", "llmgen"])].copy(); k["correct"] = k.p_a > 0.5
    c = calib_stats(k); F.setdefault("cf_calib_5rung", {})[j] = c
    md.append(f"| {n} | {c['acc']:.3f} | {c['ece']:.3f} | {c['auroc_dp']:.3f} | {c['acc_dp>=0.6']:.3f} ({c['cov_dp>=0.6']:.2f}) |")
md.append("\n(14B +stats was run only on gauss1/2/4/8, llmgen and the null pairs; its row covers those 5 rungs. Plain-8B row covers all 7.)")

# Spearman of p_real with LR margin: does the stats judge just copy the LR features?
for j, n in CFJOBS:
    d = D[j]
    if d is None: continue
    g = d[(d.a == "real") & d.b.isin(rungs7)]
    lp = [lr_pair_p(r.b, int(r.k), True) for _, r in g.iterrows()]
    F.setdefault("spearman_judge_vs_pooledLR", {})[j] = float(spearmanr(g.p_a, lp)[0])
md.append("\nSpearman(judge p_real, pooled-LR pair probability) on the same pairs: " + ", ".join(f"{n}: {F['spearman_judge_vs_pooledLR'][j]:.2f}" for j, n in CFJOBS if j in F.get("spearman_judge_vs_pooledLR", {})))

# BT ladder for 14B
d14 = D.get("cf_14b")
if d14 is not None and set(["gauss1", "uniform"]).issubset(set(d14.b)):
    order = ["real", "gauss1", "gauss2", "gauss4", "gauss8", "uniform"]
    cf2 = d14[d14.b != "real2"]; rungs = sorted(set(cf2.a) | set(cf2.b)); idx = {r: j for j, r in enumerate(rungs)}

    def bt(df):
        th = np.zeros(len(rungs)); A = df.a.map(idx).values; B = df.b.map(idx).values; w = df.p_a.values
        for _ in range(3000):
            p = 1 / (1 + np.exp(-(th[A] - th[B]))); gth = np.zeros(len(rungs))
            np.add.at(gth, A, w - p); np.add.at(gth, B, -(w - p)); th += 0.5 * gth / len(df) * len(rungs); th -= th.mean()
        return th
    th = bt(cf2); bs = np.array([bt(cf2.iloc[R.integers(0, len(cf2), len(cf2))]) for _ in range(200)])
    F["bt_14b"] = {r: [float(th[idx[r]]), float(np.percentile(bs[:, idx[r]], 2.5)), float(np.percentile(bs[:, idx[r]], 97.5))] for r in rungs}
    taus = [kendalltau(range(6, 0, -1), [b[idx[r]] for r in order])[0] for b in bs]
    F["ladder_tau_14b"] = [float(kendalltau(range(6, 0, -1), [th[idx[r]] for r in order])[0]), float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))]
    md.append("\n**Qwen3-14B Bradley–Terry over 9 rungs:** " + " > ".join(f"{r} {F['bt_14b'][r][0]:.2f}" for r in sorted(rungs, key=lambda r: -th[idx[r]]))
              + f"; Kendall τ on noise ladder = {fmt(F['ladder_tau_14b'],2)}")
    adj = []
    for x, y in [("gauss1", "gauss2"), ("gauss2", "gauss4"), ("gauss4", "gauss8"), ("gauss8", "uniform")]:
        g = d14[(d14.a == x) & (d14.b == y)]; adj.append(f"{x}>{y} {np.mean(g.p_a > 0.5):.2f}")
    md.append("14B adjacent-rung pair accuracy: " + ", ".join(adj))

# ---------------- (B) hybrid gate ----------------
md.append("\n## B. Hybrid Δp-gated LR/judge (τ chosen on one random half of pairs, evaluated on the other, both directions averaged; pooled LR = one model for all degradations)\n")
md.append("| judge | rung | judge | LR pooled | judge→LR (judge if Δp≥τ) | LR→judge (LR if Δ_LR≥τ) | avg-prob ensemble | LR per-rung | J→LRrung | LRrung→J |\n|---|---|---|---|---|---|---|---|---|---|")
taus = np.round(np.arange(0, 1.0001, 0.05), 2)
F["hybrid"] = {}


def gate_eval(df, mode, col="plr"):
    df = df.assign(plr=df[col])
    rng = np.random.default_rng(1); perm = rng.permutation(len(df)); h = [perm[:len(df) // 2], perm[len(df) // 2:]]
    accs = []
    for tr, te in [(h[0], h[1]), (h[1], h[0])]:
        def acc(ix, t):
            s = df.iloc[ix]
            if mode == "J>LR": pick = np.where(s.dp >= t, s.p_a > 0.5, s.plr > 0.5)
            else: pick = np.where(np.abs(2 * s.plr - 1) >= t, s.plr > 0.5, s.p_a > 0.5)
            return pick.mean()
        best = max(taus, key=lambda t: acc(tr, t)); accs.append(acc(te, best))
    return float(np.mean(accs))

for j, n in CFJOBS:
    d = D[j]
    if d is None: continue
    g = d[(d.a == "real") & d.b.isin(rungs7)].copy()
    if len(g) == 0: continue
    g["plr"] = [lr_pair_p(r.b, int(r.k), True) for _, r in g.iterrows()]
    g["plr_r"] = [lr_pair_p(r.b, int(r.k), False) for _, r in g.iterrows()]
    rows = [("ALL", g)] + [(rg, g[g.b == rg]) for rg in rungs7 if (g.b == rg).any()]
    for rg, s in rows:
        s = s.reset_index(drop=True)
        res = {"judge": float((s.p_a > 0.5).mean()), "lr": float((s.plr > 0.5).mean()), "J>LR": gate_eval(s, "J>LR"), "LR>J": gate_eval(s, "LR>J"),
               "avg": float(((s.p_a + s.plr) / 2 > 0.5).mean()), "n": len(s),
               "lr_rung": float((s.plr_r > 0.5).mean()), "J>LRr": gate_eval(s, "J>LR", "plr_r"), "LRr>J": gate_eval(s, "LR>J", "plr_r")}
        F["hybrid"].setdefault(j, {})[rg] = res
        if rg in ("ALL", "llmgen", "tshuffle", "gauss1", "markov"):
            md.append(f"| {n} | {rg} (n={len(s)}) | {res['judge']:.3f} | {res['lr']:.3f} | {res['J>LR']:.3f} | {res['LR>J']:.3f} | {res['avg']:.3f} | {res['lr_rung']:.3f} | {res['J>LRr']:.3f} | {res['LRr>J']:.3f} |")

# ---------------- context-conditioned ----------------
CC = [("cc", "8B plain"), ("cce", "8B +stats"), ("cc_14b", "14B plain"), ("cce_14b", "14B +stats")]
C = {j: load(j) for j, _ in CC}
md.append("\n## A/C. Context-conditioned judge: P(actual next day chosen)\n")
md.append("| candidate | " + " | ".join(n for j, n in CC if C[j] is not None) + " |\n|---|" + "---|" * sum(C[j] is not None for j, _ in CC))
F["cc_acc"] = {}
for b in ["llmgen", "copy7", "other_day", "markov", "other_user", "stayhome"]:
    cells = []
    for j, n in CC:
        d = C[j]
        if d is None: continue
        g = d[d.b == b]
        if len(g) == 0: cells.append("—"); continue
        a = boot((g.p_a > 0.5).astype(float)); F["cc_acc"].setdefault(j, {})[b] = a; cells.append(fmt(a, 2))
    md.append(f"| {b} | " + " | ".join(cells) + " |")


def ctxdist(i, recs):
    sq = seq_of(recs)
    return min(dtw(sq, seq_of(r)) for k, r in T[i]["ctx"] if r) if sq else np.nan

cand = lambda i, b: {"llmgen": G[i][0], "other_day": T[i]["other_day"], "markov": T[i]["markov"], "copy7": T[i]["copy7"],
                     "other_user": T[gidx[(gidx.index(i) + 1) % len(gidx)]]["real"], "stayhome": T[i]["stayhome"]}[b]
dcache = {}
md.append("\n| judge | typicality (picks candidate closer to history; llmgen/copy7/other_day pairs) | ECE | AUROC(Δp) | acc | P(A) | same pick both orders |\n|---|---|---|---|---|---|---|")
for j, n in CC:
    d = C[j]
    if d is None: continue
    s = d[d.b.isin(["llmgen", "copy7", "other_day"])].copy()
    ty = []
    for _, r in s.iterrows():
        i = int(r.i)
        if (i, "real") not in dcache: dcache[(i, "real")] = ctxdist(i, T[i]["real"])
        if (i, r.b) not in dcache: dcache[(i, r.b)] = ctxdist(i, cand(i, r.b))
        dr, dc = dcache[(i, "real")], dcache[(i, r.b)]
        if dr != dc: ty.append(float((r.p_a > 0.5) == (dr < dc)))
    s["correct"] = s.p_a > 0.5
    c = calib_stats(s); pb = posbias(d)
    F.setdefault("cc_typicality", {})[j] = boot(ty); F.setdefault("cc_calib", {})[j] = c; F.setdefault("cc_position", {})[j] = pb
    md.append(f"| {n} | {fmt(F['cc_typicality'][j],2)} | {c['ece']:.3f} | {c['auroc_dp']:.3f} | {c['acc']:.3f} | {pb['P(A)']:.2f} | {pb['order_consistent']:.2f} |")

# ---------------- best-of-n with evidence ----------------
gm = pd.read_csv(f"{WS}/out/gen_metrics.csv")
for job, name in [("bon", "8B plain"), ("bone", "8B +stats"), ("bone_14b", "14B +stats")]:
    b = load(job)
    if b is None: continue
    out = {"random": [], "judge": [], "oracle": [], "medoid": [], "gb_random": [], "gb_judge": [], "sk_random": [], "sk_judge": []}
    for i, g in b.groupby("i"):
        n = len(G[i]); sc = np.zeros(n); cnt = np.zeros(n)
        for _, r in g.iterrows():
            sc[int(r.a)] += r.p_a; sc[int(r.b)] += 1 - r.p_a; cnt[int(r.a)] += 1; cnt[int(r.b)] += 1
        ms = gm[(gm.i == i) & gm.method.str.startswith("gen")].set_index("method")
        d = np.array([ms.loc[f"gen{j}"].dtw if f"gen{j}" in ms.index else np.nan for j in range(n)])
        vj = [j for j in range(n) if cnt[j] > 0 and not np.isnan(d[j])]
        if len(vj) < 2: continue
        sc = sc / np.maximum(cnt, 1); pick = max(vj, key=lambda j: sc[j])
        seqs = {j: seq_of(G[i][j]) for j in vj}
        med = min(vj, key=lambda j: np.mean([dtw(seqs[j], seqs[k]) for k in vj if k != j]))
        gb = np.array([ms.loc[f"gen{j}"].geobleu if f"gen{j}" in ms.index else np.nan for j in range(n)])
        sk = np.array([ms.loc[f"gen{j}"].slot_km if f"gen{j}" in ms.index else np.nan for j in range(n)])
        out["gb_random"].append(np.mean(gb[vj])); out["gb_judge"].append(gb[pick]); out["sk_random"].append(np.mean(sk[vj])); out["sk_judge"].append(sk[pick])
        out["random"].append(np.mean(d[vj])); out["judge"].append(d[pick]); out["oracle"].append(min(d[vj])); out["medoid"].append(d[med])
    F.setdefault("bon", {})[job] = {k: boot(v) for k, v in out.items()}
    F["bon"][job]["judge_minus_random"] = boot(np.array(out["judge"]) - np.array(out["random"]))
    F["bon"][job]["position"] = posbias(b)
    F["bon"][job]["gb_judge_minus_random"] = boot(np.array(out["gb_judge"]) - np.array(out["gb_random"]))
    F["bon"][job]["sk_judge_minus_random"] = boot(np.array(out["sk_judge"]) - np.array(out["sk_random"]))
md.append("\n## Best-of-6 (DTW km/step vs actual next day; lower is better)\n\n| judge | random | judge pick | medoid | oracle | judge − random (paired) | GEO-BLEU-like judge − random | slot-err judge − random | P(A) |\n|---|---|---|---|---|---|---|---|---|")
for job, v in F.get("bon", {}).items():
    md.append(f"| {job} | {v['random'][0]:.2f} | {v['judge'][0]:.2f} | {v['medoid'][0]:.2f} | {v['oracle'][0]:.2f} | {fmt(v['judge_minus_random'],2)} | {fmt(v['gb_judge_minus_random'],3)} | {fmt(v['sk_judge_minus_random'],2)} | {v['position']['P(A)']:.2f} |")

calls = {}
for f in os.listdir(f"{WS}/out"):
    if f.startswith("usage_"):
        calls[f[6:-5]] = json.load(open(f"{WS}/out/{f}"))
F["usage_all_jobs"] = calls
F["usage_total"] = {k: sum(v[k] for v in calls.values()) for k in ["calls", "prompt_tokens", "completion_tokens"]}
S = json.load(open(f"{WS}/out/summary.json")); S["followup"] = F
json.dump(S, open(f"{WS}/out/summary.json", "w"), indent=1)
open(f"{WS}/out/results_followup.md", "w").write("\n".join(md))
print("\n".join(md)); print(F["usage_total"])
