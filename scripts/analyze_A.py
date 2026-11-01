"""Phase A / scale analysis: do explicit "close" channels (tri: P(T); str7: P(4), P(3..5)) separate
"the two candidates are close" from "the judge cannot tell", where the forced A/B readout cannot?
Per dataset x judge (order-averaged; str7 BA is mirrored k -> 8-k):
  U_bin    entropy of the order-averaged A/B probability (baseline)
  pT       P(T) under tri;  tri_U = entropy of A vs B within tri
  p4, pmid P(4) and P(3..5) under str7; s7_abs = |E[str7] - 4|
Metrics: tie AUROC (human ties), low-confidence tie-vs-error AUROC, MMRB2 Spearman with 3-annotator strength and
split-vs-unanimous AUROC, error AUROC on strict pairs, tri/str7 accuracy and T-rate, cross-fitted logistic
[U_bin, pT, p4] for tie detection.
usage (from $EAGLEVLM_DATA): python analyze_A.py [--judges ...] [--dirs results,results_A] [--tag A]"""
import argparse, ast, collections, json, math, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import _sm, auroc

ap = argparse.ArgumentParser()
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen35_9b,qwen25vl7b")
ap.add_argument("--dirs", default="results,results_A")
ap.add_argument("--ds", default="genai_arena,mllm_judge,pp,editreward,mmrb2,vlrb,mmrb,mjbench,imagerewarddb,qbench2,mmrlhf")
ap.add_argument("--tag", default="A")
a = ap.parse_args()
H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))


def load(ds, m):
    R = collections.defaultdict(dict)
    for d in a.dirs.split(","):
        f = f"{d}/{ds}__{m}.jsonl"
        if os.path.exists(f):
            for l in open(f):
                x = json.loads(l)
                if "lp" in x:
                    R[x["id"]][(x["cond"], x["order"])] = x["lp"]
    return R


def feats(r):
    f = {}
    if ("bin", "AB") in r and ("bin", "BA") in r:
        p = (_sm(r[("bin", "AB")])["A"] + _sm(r[("bin", "BA")])["B"]) / 2
        f["p_bin"] = p; f["U_bin"] = H(p)
    if ("tri", "AB") in r and ("tri", "BA") in r:
        ab, ba = _sm(r[("tri", "AB")]), _sm(r[("tri", "BA")])
        f["pT"] = (ab["T"] + ba["T"]) / 2
        pa = (ab["A"] / (ab["A"] + ab["B"]) + ba["B"] / (ba["A"] + ba["B"])) / 2
        f["p_tri"] = pa; f["tri_U"] = H(pa)
        f["tri_argT"] = float((ab["T"] > max(ab["A"], ab["B"])) + (ba["T"] > max(ba["A"], ba["B"]))) / 2
    if ("str7", "AB") in r and ("str7", "BA") in r:
        ab, ba = _sm(r[("str7", "AB")]), _sm(r[("str7", "BA")])
        d = np.array([(ab[str(k)] + ba[str(8 - k)]) / 2 for k in range(1, 8)])     # in cand0 = A orientation
        f["p4"] = d[3]; f["pmid"] = d[2:5].sum()
        e = (d * np.arange(1, 8)).sum(); f["s7_abs"] = abs(e - 4); f["p_s7"] = d[:3].sum() / max(1e-9, d[:3].sum() + d[4:].sum())
        f["s7_order_gap"] = abs(sum(k * ab[str(k)] for k in range(1, 8)) - sum((8 - k) * ba[str(k)] for k in range(1, 8)))
    return f


def logit_cv(X, y, k=5):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9); X = np.c_[np.ones(len(X)), X]; s = np.zeros(len(y))
    idx = np.random.default_rng(0).permutation(len(y))
    for f in range(k):
        te = idx[f::k]; tr = np.setdiff1d(idx, te); w = np.zeros(X.shape[1])
        for _ in range(30):
            p = 1 / (1 + np.exp(-X[tr] @ w)); g = X[tr].T @ (p - y[tr]) + np.r_[0, w[1:]]
            Hs = (X[tr] * (p * (1 - p))[:, None]).T @ X[tr] + np.diag(np.r_[0, np.ones(len(w) - 1)])
            w -= np.linalg.solve(Hs, g)
        s[te] = X[te] @ w
    return s


out = {}
for ds in a.ds.split(","):
    items = {json.loads(l)["id"]: json.loads(l) for l in open(f"items/{ds}.jsonl")}
    for m in a.judges.split(","):
        R = load(ds, m)
        F = {i: feats(R[i]) for i in R if i in items}
        ids = [i for i, f in F.items() if {"U_bin", "pT", "p4"} <= set(f)]
        if len(ids) < 30:
            continue
        g = lambda k: np.array([F[i][k] for i in ids])
        tie = np.array([items[i]["label"] == "tie" for i in ids]); y = np.array([items[i]["label"] == 0 for i in ids])
        strict = ~tie; err = strict & ((g("p_bin") > .5) != y)
        res = {"n": len(ids), "n_tie": int(tie.sum()),
               "acc_bin": float(((g("p_bin") > .5) == y)[strict].mean()),
               "acc_tri": float(((g("p_tri") > .5) == y)[strict].mean()),
               "acc_s7": float(((g("p_s7") > .5) == y)[strict].mean()),
               "T_rate_argmax": float(g("tri_argT").mean()), "mean_pT": float(g("pT").mean())}
        # "closeness" signals: higher = more likely close/tie
        close = {"U_bin": g("U_bin"), "pT": g("pT"), "p4": g("p4"), "pmid": g("pmid"), "-s7_abs": -g("s7_abs"),
                 "tri_U": g("tri_U")}
        if tie.sum() >= 15:
            res["tie_auroc"] = {k: round(auroc(v, tie), 3) for k, v in close.items()}
            X = np.c_[g("U_bin"), g("pT"), g("p4"), g("pmid")]
            res["tie_auroc"]["logit[U,pT,p4,pmid]"] = round(auroc(logit_cv(X, tie.astype(float)), tie), 3)
            res["tie_auroc"]["logit[U]"] = round(auroc(logit_cv(X[:, :1], tie.astype(float)), tie), 3)
            low = g("U_bin") >= np.quantile(g("U_bin"), .7); sel = low & (tie | err)
            res["lowconf_tie_vs_err"] = {k: round(auroc(v[sel], tie[sel]), 3) for k, v in close.items()}
            res["mean_pT_tie_vs_strict"] = [round(float(g("pT")[tie].mean()), 3), round(float(g("pT")[strict].mean()), 3)]
        if ds == "mmrb2":
            ha = []
            for i in ids:
                mt = items[i]["meta"]; mt = ast.literal_eval(mt) if isinstance(mt, str) else mt
                h = mt["human_annotations"]; ha.append(ast.literal_eval(h) if isinstance(h, str) else h)
            ha = np.array(ha, float); st = np.abs(ha.mean(1) - 3.5); unan = ((ha <= 3).all(1)) | ((ha >= 4).all(1))
            res["mmrb2_rho_strength"] = {k: round(float(stats.spearmanr(-v, st)[0]), 3) for k, v in close.items()}
            res["mmrb2_auroc_split"] = {k: round(auroc(v, ~unan), 3) for k, v in close.items()}
        # error detection on strict pairs: higher = more likely wrong
        res["err_auroc"] = {k: round(auroc(v[strict], err[strict]), 3) for k, v in
                            {**close, "s7_order_gap": g("s7_order_gap")}.items()}
        out[f"{ds}|{m}"] = res
        print(f"{ds:13s} {m:14s} n={res['n']:5d} acc bin/tri/s7 {res['acc_bin']:.3f}/{res['acc_tri']:.3f}/{res['acc_s7']:.3f} "
              f"T-rate {res['T_rate_argmax']:.2f} pT {res['mean_pT']:.2f}"
              + (f" | tieAUROC {res['tie_auroc']}" if "tie_auroc" in res else "")
              + (f" | lowconf tie-vs-err {res['lowconf_tie_vs_err']}" if "lowconf_tie_vs_err" in res else "")
              + (f" | rho {res['mmrb2_rho_strength']} split {res['mmrb2_auroc_split']}" if ds == "mmrb2" else "")
              + f" | errAUROC {res['err_auroc']}", flush=True)
os.makedirs("paper_tables", exist_ok=True)
json.dump(out, open(f"paper_tables/analyze_{a.tag}.json", "w"), indent=1)
