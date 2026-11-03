"""Claim-level verification on real response pairs (make_claim_diff.py + C1_claims).
Per pair: points k with p_k = P(response-1 claim is accurate | image) (order-averaged), q_k the same without the image.
  n      number of differing factual points (0 = the LLM found no factual disagreement)
  M      evidence margin  sum_k (p_k - .5)   (>0 favours response 1)
  cert   mean |p_k - .5|  (how decisively the judge verifies the differing facts)
  use    mean |p_k - q_k| (how much the image moves the claim check)
Questions:
 (1) human ties (MLLM-Judge): do claim features add to P(T)/U?  CV logistic AUROC.
 (2) decided pairs with verified labels (VL-RewardBench, MM RewardBench, MLLM-Judge non-ties): accuracy of sign(M) vs the
     judge's verdict, and of 'judge unless low confidence, then claims'.
 (3) a three-way split of the judge's low-confidence pairs (top 30% U or P(T)):  no disagreement / decisive verification /
     undecided verification, with the human-tie rate and judge error rate in each part.
usage (from $EAGLEVLM_DATA): python claims.py --judges qwen3vl8b,... --out paper_tables/claims.md"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, signals
from eaglev_stats import auroc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen25vl7b,llavaov7b")
ap.add_argument("--pairs", default="/media/shawn/C/papers/VLM/judgebench_local/items/claim_check_pairs.jsonl")
ap.add_argument("--out", default="paper_tables/claims.md")
a = ap.parse_args()
PAIRS = {json.loads(l)["id"]: json.loads(l) for l in open(a.pairs)}
IC = items("claim_check")
DS = {"mllm_judge": "mllm_judge", "vlrb": "vlrb_s32", "mmrb": "mmrb_s32"}


def p0(r, c):
    if (c, "AB") not in r or (c, "BA") not in r:
        return None
    ab, ba = r[(c, "AB")], r[(c, "BA")]
    return (ab["A"] / (ab["A"] + ab["B"]) + ba["B"] / (ba["A"] + ba["B"])) / 2


def cv(X, y):
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[tr], y[tr]); out[te] = m.predict_proba(X[te])[:, 1]
    return auroc(out, y)


L = ["# Claim-level verification on real response pairs", ""]
for j in a.judges.split(","):
    RC = load("claim_check", j)
    if not RC:
        continue
    pts = {}
    for i, r in RC.items():
        it = IC.get(i)
        if not it:
            continue
        p, q = p0(r, "bin"), p0(r, "bin_blind")
        if p is None:
            continue
        s1 = p if it["label"] == 0 else 1 - p                       # P(response-1 claim accurate)
        s1b = None if q is None else (q if it["label"] == 0 else 1 - q)
        pts.setdefault(it["meta"]["pair"], []).append((s1, s1b))
    L += [f"## {j}", ""]
    for dsn, dsf in DS.items():
        I = items(dsf); R = load(dsf, j) or load(dsn, j)          # general judges ran the full set (same ids)
        rows = []
        for pid, pr in PAIRS.items():
            if pr["ds"] != dsf.split("_s32")[0] and pr["ds"] != dsf:
                continue
            if pid not in I or pid not in R or not pr["parsed"]:
                continue
            s = signals(R[pid])
            if not s:
                continue
            P = pts.get(pid, [])
            if len(P) != pr["n_points"]:
                continue                                              # claim checks missing
            ps = np.array([x[0] for x in P]) if P else np.zeros(0)
            qs = np.array([x[1] if x[1] is not None else .5 for x in P]) if P else np.zeros(0)
            M = float((ps - .5).sum()); cert = float(np.abs(ps - .5).mean()) if P else 0.0
            use = float(np.abs(ps - qs).mean()) if P else 0.0
            it = I[pid]
            rows.append(dict(tie=it["label"] == "tie", lab=it["label"], n=len(P), M=M, cert=cert, use=use,
                             pt=s[0], u=s[1], p=s[2]))
        if len(rows) < 50:
            continue
        L += [f"### {dsn} (n = {len(rows)}; pairs with no differing point: {np.mean([x['n'] == 0 for x in rows]):.2f})", ""]
        pt = np.array([x["pt"] for x in rows]); u = np.array([x["u"] for x in rows])
        F = np.array([[x["n"] == 0, x["n"], abs(x["M"]), x["cert"], x["use"]] for x in rows], float)
        tie = np.array([x["tie"] for x in rows]).astype(int)
        if tie.sum() >= 20:
            base = np.c_[np.log(pt + 1e-4), np.log(u + 1e-4)]
            L.append(f"- human tie AUROC: P(T) {auroc(pt, tie):.3f}, U {auroc(u, tie):.3f}, no-disagreement {auroc(F[:, 0], tie):.3f}, "
                     f"-|M| {auroc(-F[:, 2], tie):.3f}, -cert {auroc(-F[:, 3], tie):.3f};  CV [P(T),U] {cv(base, tie):.3f} -> "
                     f"+claims {cv(np.c_[base, F], tie):.3f};  claims alone {cv(F, tie):.3f}")
        dec = [x for x in rows if not x["tie"]]
        if dec:
            y0 = np.array([x["lab"] == 0 for x in dec])
            jv = np.array([x["p"] > .5 for x in dec]); cvd = np.array([x["M"] > 0 if x["M"] != 0 else x["p"] > .5 for x in dec])
            uu = np.array([x["u"] for x in dec]); thr = np.quantile(uu, .7)
            hyb = np.where(uu >= thr, cvd, jv)
            L.append(f"- decided pairs (n = {len(dec)}): judge acc {np.mean(jv == y0):.3f}; claim-verdict acc {np.mean(cvd == y0):.3f} "
                     f"(on pairs with >=1 point: {np.mean((cvd == y0)[[x['n'] > 0 for x in dec]]):.3f}); "
                     f"judge, low-confidence 30% -> claims: {np.mean(hyb == y0):.3f}")
        # three-way split of the low-confidence pairs
        conf = np.maximum(pt, u / np.log(2))
        lo = conf >= np.quantile(conf, .7)
        grp = np.where(F[:, 0] == 1, "no disagreement", np.where(F[:, 3] >= .25, "decisive check", "undecided check"))
        err = np.array([np.nan if x["tie"] else float((x["p"] > .5) != (x["lab"] == 0)) for x in rows])
        L.append("- low-confidence pairs (top 30% of max(P(T), U/ln2)) split by claim check:")
        for gname in ("no disagreement", "decisive check", "undecided check"):
            m = lo & (grp == gname)
            if m.sum():
                L.append(f"  - {gname}: n={m.sum()}, human-tie rate {tie[m].mean():.2f}, judge error rate (non-tie) "
                         f"{np.nanmean(err[m]) if np.isfinite(err[m]).any() else float('nan'):.2f}")
        L.append(f"  - (all pairs: human-tie rate {tie.mean():.2f}, judge error rate {np.nanmean(err):.2f})")
        L.append("")
open(a.out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
