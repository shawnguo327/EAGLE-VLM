"""Does low judge confidence mean the two candidates are close (aleatoric) or the judge cannot tell (epistemic)?
Per pair, from saved outputs only:
  single judge, two orders: U = H(pbar), A = mean_o H(p_o) (each order individually unsure), E = U - A (order disagreement = JS)
  pointwise: pw_gap = |E[s0]-E[s1]|, pw_eq = P(s0 == s1) under the two score distributions
  ensemble of J judges x 2 orders: U_ens, A_ens, E_ens (= mutual information, BALD)
Targets: human tie (GenAI-Arena tievote / bothbad, MLLM-Judge, Place Pulse, EditReward), MMRB2 3-annotator strength
|mean-3.5| and unanimity, and judge error on strict pairs.
usage (from $EAGLEVLM_DATA): python tie_vs_unsure.py [--judges ...]"""
import argparse, ast, collections, json, math, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import _sm, auroc

ap = argparse.ArgumentParser(); ap.add_argument("--judges", default="qwen3vl8b,qwen35_9b,qwen25vl7b,internvl35_8b")
ap.add_argument("--ds", default="genai_arena,mllm_judge,pp,editreward,mmrb2,vlrb,mmrb,mjbench,imagerewarddb,qbench2,mmrlhf")
a = ap.parse_args(); J = a.judges.split(",")
H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))


def calls(ds, m):
    R = collections.defaultdict(dict)
    for l in open(f"results/{ds}__{m}.jsonl"):
        d = json.loads(l)
        if "lp" in d:
            R[d["id"]][(d["cond"], d["order"])] = d["lp"]
    out = {}
    for i, r in R.items():
        if ("bin", "AB") in r and ("bin", "BA") in r:
            o = {"p": [_sm(r[("bin", "AB")])["A"], _sm(r[("bin", "BA")])["B"]]}   # P(cand0 wins) in each order
            if ("score", "A") in r and ("score", "B") in r:
                sa, sb = _sm(r[("score", "A")]), _sm(r[("score", "B")])
                o["ed"] = sum(int(k) * v for k, v in sa.items()) - sum(int(k) * v for k, v in sb.items())
                o["eq"] = sum(sa[k] * sb.get(k, 0) for k in sa)
            out[i] = o
    return out


def signals(ps):
    p = np.clip(np.array(ps), 1e-6, 1 - 1e-6); U = H(p.mean()); A = H(p).mean(); return U, A, U - A


res = {}
for ds in a.ds.split(","):
    items = {json.loads(l)["id"]: json.loads(l) for l in open(f"items/{ds}.jsonl")}
    C = {m: calls(ds, m) for m in J if os.path.exists(f"results/{ds}__{m}.jsonl")}
    ids = sorted(set.intersection(*[set(v) for v in C.values()]))
    meta = {i: (ast.literal_eval(items[i]["meta"]) if isinstance(items[i].get("meta"), str) else items[i].get("meta", {})) for i in ids}
    tie = np.array([items[i]["label"] == "tie" for i in ids]); y = np.array([items[i]["label"] == 0 for i in ids])
    out = {"n": len(ids), "n_tie": int(tie.sum())}
    sig = {}
    for m in C:
        S = np.array([signals(C[m][i]["p"]) for i in ids])
        sig[f"{m}:U"], sig[f"{m}:A"], sig[f"{m}:E"] = S[:, 0], S[:, 1], S[:, 2]
        if all("ed" in C[m][i] for i in ids):
            sig[f"{m}:-pwgap"] = -np.abs([C[m][i]["ed"] for i in ids]); sig[f"{m}:pweq"] = np.array([C[m][i]["eq"] for i in ids])
    S = np.array([signals(sum((C[m][i]["p"] for m in C), [])) for i in ids])
    sig["ens:U"], sig["ens:A"], sig["ens:E"] = S[:, 0], S[:, 1], S[:, 2]
    # cross-judge only (order-averaged per judge, then disagreement across judges)
    S2 = np.array([signals([np.mean(C[m][i]["p"]) for m in C]) for i in ids]); sig["xjudge:E"] = S2[:, 2]
    agg = lambda key: {k.split(":")[0] if k.startswith(tuple(J)) else k: v for k, v in key.items()}
    def mean_over_judges(fn):
        d = collections.defaultdict(list)
        for k, v in sig.items():
            mm, s = k.split(":"); d[s if mm in J else k].append(fn(v))
        return {k: round(float(np.nanmean(v)), 3) for k, v in d.items()}
    strict = ~tie
    pred = {m: np.array([np.mean(C[m][i]["p"]) > .5 for i in ids]) for m in C}
    if tie.sum() >= 15:
        out["tie_auroc"] = mean_over_judges(lambda v: auroc(v, tie))   # higher signal -> tie
        # within the most uncertain 30% (by each judge U), can components separate ties from judge errors?
        def lowconf(v, k=None):
            return v
        d = collections.defaultdict(list)
        for m in C:
            U = sig[f"{m}:U"]; low = U >= np.quantile(U, .7); err = strict & (pred[m] != y)
            sel = low & (tie | err)
            for s in ["A", "E", "-pwgap", "pweq"]:
                if f"{m}:{s}" in sig and tie[sel].any() and err[sel].any():
                    d[s].append(auroc(sig[f"{m}:{s}"][sel], tie[sel]))
            d["tie_share_lowconf"].append(tie[low].mean()); d["err_share_lowconf"].append(err[low].mean())
        out["lowconf_tie_vs_err"] = {k: round(float(np.mean(v)), 3) for k, v in d.items()}
        if ds == "genai_arena":
            vt = np.array([meta[i]["vote_type"] for i in ids])
            out["tie_auroc_tievote"] = mean_over_judges(lambda v: auroc(v[(vt == "tievote") | strict], vt[(vt == "tievote") | strict] == "tievote"))
            out["tie_auroc_bothbad"] = mean_over_judges(lambda v: auroc(v[(vt == "bothbad_vote") | strict], vt[(vt == "bothbad_vote") | strict] == "bothbad_vote"))
    # error detection on strict pairs (higher signal -> error)
    d = collections.defaultdict(list)
    for m in C:
        err = (pred[m] != y)[strict]
        for s in ["U", "A", "E", "-pwgap", "pweq"]:
            if f"{m}:{s}" in sig:
                d[s].append(auroc(sig[f"{m}:{s}"][strict], err))
        for s in ["ens:U", "ens:A", "ens:E", "xjudge:E"]:
            d[s].append(auroc(sig[s][strict], err))
    out["err_auroc"] = {k: round(float(np.mean(v)), 3) for k, v in d.items()}
    if ds == "mmrb2":
        ha = np.array([ast.literal_eval(meta[i]["human_annotations"]) for i in ids], float)
        strength = np.abs(ha.mean(1) - 3.5); unan = ((ha <= 3).all(1)) | ((ha >= 4).all(1))
        out["mmrb2_spearman_with_human_strength"] = mean_over_judges(lambda v: stats.spearmanr(-v, strength)[0])
        out["mmrb2_auroc_split_vs_unanimous"] = mean_over_judges(lambda v: auroc(v, ~unan))
        out["mmrb2_split_share"] = float((~unan).mean())
        acc = np.mean([(pred[m] == y)[unan].mean() for m in C]), np.mean([(pred[m] == y)[~unan].mean() for m in C])
        out["mmrb2_acc_unanimous_vs_split"] = [round(float(x), 3) for x in acc]
    res[ds] = out; print(ds, json.dumps(out))
os.makedirs("paper_tables", exist_ok=True); json.dump(res, open("paper_tables/tie_vs_unsure.json", "w"), indent=1)
