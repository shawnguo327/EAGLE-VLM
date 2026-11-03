"""Selective judging with guarantees (Trust-or-Escalate / SCOPE style), VLM vs LLM judges, same protocol, existing outputs.
Per (dataset, judge), from the A/B forced-choice prompt in both orders (cond 'bin') and the A/B/T prompt ('tri'):
  acc        accuracy of the order-averaged verdict on non-tie pairs
  inc        order inconsistency: verdict flips when the candidates are swapped
  first      mean P(choose the first slot) - .5  (position bias, signed)
  auroc      error AUROC of BPE = H(p_bar)  (= our 'U' / FC-swap; SCOPE's BPE)
  ece        ECE of max(p_bar, 1-p_bar)
  cov/risk/viol  SCOPE linearised conformal threshold on BPE, 500 random 50/50 cal/test splits of NON-TIE pairs
                 (exactly the published setting): mean test coverage, mean test risk, share of splits with risk > alpha
  tie-aware  (only datasets with human ties) the same pipeline on ALL pairs, a human tie counting as an error whenever the judge
             commits to a winner:  'drop'  threshold calibrated without ties (as published) but deployed on all pairs;
             'U'  calibrated with ties;  'U+P(T)'  abstain score = rank-mean of BPE and P(T)
Shared errors: share of a judge's errors that all judges evaluated on the dataset also make (non-tie pairs).
usage (judgebench dir, lis-omni env): python vlm_vs_llm.py --out paper_tables/vlm_vs_llm.md"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from eaglev_stats import auroc

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/vlm_vs_llm.md")
ap.add_argument("--splits", type=int, default=500)
ap.add_argument("--alphas", default="0.1,0.2")
a = ap.parse_args()
AL = [float(x) for x in a.alphas.split(",")]
VLM = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"]
LLM = ["qwen3_8b", "qwen25_7b"]
DS = {"text": ["mtbench_human", "arena55k", "arena_tt"],
      "resp": ["vlrb", "mmrb", "mllm_judge", "mmrlhf"],
      "img": ["genai_arena", "mjbench", "imagerewarddb", "mmrb2", "qbench2", "editreward", "pp"]}


def per_item(ds, j):
    I = items(ds)
    R = load(ds, j, conds={"bin", "tri"})
    out = {}
    for i, r in R.items():
        if i not in I or ("bin", "AB") not in r or ("bin", "BA") not in r:
            continue
        ab, ba = r[("bin", "AB")], r[("bin", "BA")]
        pab = ab["A"] / (ab["A"] + ab["B"])          # P(first slot) when cand0 is first
        pba = ba["A"] / (ba["A"] + ba["B"])          # P(first slot) when cand1 is first
        p = (pab + 1 - pba) / 2                      # P(cand0 wins)
        pt = np.nan
        if ("tri", "AB") in r and ("tri", "BA") in r:
            pt = (r[("tri", "AB")].get("T", 0) + r[("tri", "BA")].get("T", 0)) / 2
        out[i] = dict(lab=str(I[i]["label"]), p=p, u=H(p), inc=(pab > .5) == (pba > .5), first=(pab + pba) / 2 - .5, pt=pt)
    return out


def ece(conf, corr, b=10):
    e, n = 0, len(conf)
    for k in range(b):
        m = (conf >= .5 + k * .05) & (conf < .5 + (k + 1) * .05 + (k == b - 1))
        if m.any():
            e += m.sum() / n * abs(conf[m].mean() - corr[m].mean())
    return e


def scope(s, err, alpha, cal_mask_fn=None, splits=500, seed=0):
    """linearised conformal (SCOPE eq. 5-6): lambda = sup{l : sum_cal 1[s<=l](E-alpha) <= -1}; accept s <= lambda"""
    rng = np.random.default_rng(seed); n = len(s); cov, risk, viol = [], [], []
    for _ in range(splits):
        perm = rng.permutation(n); c, t = perm[: n // 2], perm[n // 2:]
        if cal_mask_fn is not None:
            c = c[cal_mask_fn[c]]
        o = np.argsort(s[c], kind="stable"); cs = np.cumsum(err[c][o] - alpha)
        ok = np.where(cs <= -1)[0]
        lam = s[c][o][ok.max()] if len(ok) else -np.inf
        acc = s[t] <= lam
        cov.append(acc.mean()); r = err[t][acc].mean() if acc.any() else 0.0
        risk.append(r); viol.append(r > alpha)
    return np.mean(cov), np.mean(risk), np.mean(viol)


def rank(x):
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


L = ["# Selective judging with guarantees: VLM judges vs LLM judges, same protocol", "",
     "kind: text = text candidates (LLM-judge benchmarks); resp = text answers about an image; img = candidates are images.",
     "BPE = order-averaged binary entropy (our U / FC-swap = SCOPE's BPE; not our method).  SCOPE calibration on non-tie pairs, "
     f"{a.splits} random 50/50 splits; cov / risk / viol = mean test coverage / mean test risk / share of splits with risk > alpha.", "",
     "| kind | dataset | judge | n | acc | inconsistent | first-slot bias | BPE err AUROC | ECE | "
     + " | ".join(f"cov@{x} | risk@{x} | viol@{x}" for x in AL) + " |",
     "|" + "---|" * (9 + 3 * len(AL))]
TIE = ["", "## Ties: what happens when the published 'drop ties' calibration meets a test stream that contains them", "",
       "Error on a human-tie pair = the judge commits to a winner.  alpha = 0.2.  drop = calibrate on non-tie pairs (as published), "
       "deploy on all; U = calibrate on all pairs with BPE; U+P(T) = calibrate on all pairs with rank-mean of BPE and P(T).", "",
       "| kind | dataset | judge | tie share | drop: cov | drop: risk | drop: viol | U: cov | U: risk | U+P(T): cov | U+P(T): risk |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
SH = ["", "## Shared errors (non-tie pairs): share of each judge's errors that every other judge also makes", "",
      "| kind | dataset | judges | mean acc | shared share of errors | expected if independent | shared errors with BPE below median |",
      "|---|---|---|---|---|---|---|"]
summ = {}
for kind, dss in DS.items():
    for ds in dss:
        per = {}
        for j in (VLM + LLM if kind == "text" else VLM):
            P = per_item(ds, j)
            if len(P) < 50:
                continue
            per[j] = P
            nt = [x for x in P.values() if x["lab"] in ("0", "1")]
            p = np.array([x["p"] for x in nt]); u = np.array([x["u"] for x in nt])
            corr = (p > .5) == np.array([x["lab"] == "0" for x in nt]); err = (~corr).astype(float)
            row = [kind, ds, j + (" (LLM)" if j in LLM else ""), len(nt), f"{corr.mean():.3f}",
                   f"{np.mean([x['inc'] for x in nt]):.3f}", f"{np.mean([x['first'] for x in nt]):+.3f}",
                   f"{auroc(u, err):.3f}", f"{ece(np.maximum(p, 1 - p), corr):.3f}"]
            for al in AL:
                c, r, v = scope(u, err, al, splits=a.splits)
                row += [f"{c:.2f}", f"{r:.3f}", f"{v:.2f}"]
                summ.setdefault((kind, j in LLM, al), []).append(c)
            L.append("| " + " | ".join(map(str, row)) + " |")
            # tie-aware
            allp = list(P.values())
            if any(x["lab"] == "tie" for x in allp) and not np.isnan([x["pt"] for x in allp]).all():
                lab = np.array([x["lab"] for x in allp]); pa = np.array([x["p"] for x in allp]); ua = np.array([x["u"] for x in allp])
                pt = np.nan_to_num(np.array([x["pt"] for x in allp]), nan=0.0)
                e = np.where(lab == "tie", 1.0, ((pa > .5) != (lab == "0")).astype(float))
                al = .2
                d = scope(ua, e, al, cal_mask_fn=(lab != "tie"), splits=a.splits)
                uu = scope(ua, e, al, splits=a.splits)
                up = scope((rank(ua) + rank(pt)) / 2, e, al, splits=a.splits)
                TIE.append(f"| {kind} | {ds} | {j} | {(lab == 'tie').mean():.2f} | {d[0]:.2f} | {d[1]:.3f} | {d[2]:.2f} | "
                           f"{uu[0]:.2f} | {uu[1]:.3f} | {up[0]:.2f} | {up[1]:.3f} |")
        # shared errors
        for grp, js in (("VLM", [j for j in VLM if j in per]), ("LLM+VLM", [j for j in per])):
            if len(js) < 3 or (grp == "LLM+VLM" and kind != "text"):
                continue
            ids = sorted(set.intersection(*[set(i for i, x in per[j].items() if x["lab"] in ("0", "1")) for j in js]))
            E = np.array([[(per[j][i]["p"] > .5) != (per[j][i]["lab"] == "0") for i in ids] for j in js])
            U = np.array([[per[j][i]["u"] for i in ids] for j in js])
            allw = E.all(0)
            sh = np.mean([allw.sum() / max(E[k].sum(), 1) for k in range(len(js))])
            ind = np.mean([np.prod([E[m].mean() for m in range(len(js)) if m != k]) for k in range(len(js))])
            conf = np.mean([(U[k][allw] < np.median(U[k])).mean() for k in range(len(js))]) if allw.any() else np.nan
            SH.append(f"| {kind} | {ds} | {grp}: {','.join(js)} | {1 - E.mean():.3f} | {sh:.2f} | {ind:.2f} | {conf:.2f} |")
        print(ds, "done", flush=True)
S = ["", "## Mean SCOPE coverage by candidate kind", "", "| kind | judge type | " + " | ".join(f"cov@{x}" for x in AL) + " |",
     "|---|---|" + "---|" * len(AL)]
for kind in DS:
    for llm in (False, True):
        if (kind, llm, AL[0]) in summ:
            S.append(f"| {kind} | {'LLM' if llm else 'VLM'} | " + " | ".join(f"{np.mean(summ[(kind, llm, x)]):.2f}" for x in AL) + " |")
txt = "\n".join(L + S + TIE + SH) + "\n"
open(a.out, "w").write(txt)
print(txt)
