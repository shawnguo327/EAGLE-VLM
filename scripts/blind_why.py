"""Where and why do VLM judges decide the same way without the image?  Is it guessing or a systematic text rule?
Per dataset x subset x judge (non-tie pairs), Z2 runs (A/B/T prompt renormalised to A vs B, both orders):
  acc img / acc blind / acc of simple text rules (pick longer; pick the one with more negations 'no/not/none/cannot')
  agree   share of verdicts unchanged without the image
  cons_b  blind order consistency (a guess flips with the order; a rule does not)
  conf_b  mean blind confidence |p_bar - .5|*2
  longer  share of blind verdicts choosing the longer response; neg = share choosing the more-negating one
  R2      5-fold CV accuracy of a logistic model predicting the BLIND verdict from text features alone
          [log length ratio, negation-count diff, digit-count diff, overlap with the question diff]
usage (judgebench dir): python blind_why.py --out paper_tables/blind_why.md"""
import argparse, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper_tables/blind_why.md"); a = ap.parse_args()
NEG = re.compile(r"\b(no|not|none|cannot|can't|isn't|aren't|doesn't|don't|there is no|without|neither|nor)\b", re.I)
W = lambda t: set(re.findall(r"[a-z]+", t.lower()))


def pc(r, c):
    ab, ba = r[(c, "AB")], r[(c, "BA")]
    f1, f2 = ab["A"] / (ab["A"] + ab["B"]), ba["A"] / (ba["A"] + ba["B"])
    return (f1 + 1 - f2) / 2, (f1 > .5) == (f2 > .5)   # second: order-INconsistent (verdict follows the slot)


L = ["# Blind (no image) verdicts: where, and guessing or rule?", "",
     "| dataset | subset | judge | n | acc img | acc blind | acc 'longer' | acc 'more negation' | agree img/blind | blind order-consistent | blind conf | blind picks longer | blind picks more-neg | CV acc predicting blind verdict from text features |",
     "|" + "---|" * 14]
for ds in ("vlrb", "mmrb", "mllm_judge", "mmrlhf"):
    I = items(ds)
    for j in ("qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"):
        R = load(ds, j, conds={"tri", "tri_blind"})
        G = collections.defaultdict(list)
        for i, r in R.items():
            if i not in I or str(I[i]["label"]) not in ("0", "1") or not all((c, o) in r for c in ("tri", "tri_blind") for o in ("AB", "BA")):
                continue
            it = I[i]; c0, c1 = it["cand"]
            p, _ = pc(r, "tri"); pb, inc = pc(r, "tri_blind")
            q = W(it["question"])
            f = [np.log((len(c0) + 1) / (len(c1) + 1)), len(NEG.findall(c0)) - len(NEG.findall(c1)),
                 len(re.findall(r"\d", c0)) - len(re.findall(r"\d", c1)), len(W(c0) & q) - len(W(c1) & q)]
            row = (str(it["label"]) == "0", p, pb, inc, f)
            G[it["subset"]].append(row); G["ALL"].append(row)
        if not G:
            continue
        for sub in sorted(G, key=lambda s: (s != "ALL", s)):
            X = G[sub]
            if len(X) < 60:
                continue
            y = np.array([x[0] for x in X]); p = np.array([x[1] for x in X]); pb = np.array([x[2] for x in X])
            inc = np.array([x[3] for x in X]); F = np.array([x[4] for x in X]); vb = pb > .5
            lon = F[:, 0] > 0; ng = F[:, 1] > 0
            cv = cross_val_score(LogisticRegression(max_iter=2000), F, vb, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean() if 10 < vb.sum() < len(vb) - 10 else np.nan
            valid_l = F[:, 0] != 0; valid_n = F[:, 1] != 0
            L.append(f"| {ds} | {sub} | {j} | {len(y)} | {((p > .5) == y).mean():.3f} | {(vb == y).mean():.3f} | "
                     f"{(lon == y)[valid_l].mean():.3f} | {(ng == y)[valid_n].mean() if valid_n.any() else float('nan'):.3f} | "
                     f"{((p > .5) == vb).mean():.3f} | {1 - inc.mean():.3f} | {np.mean(np.abs(pb - .5) * 2):.3f} | "
                     f"{(vb == lon)[valid_l].mean():.3f} | {(vb == ng)[valid_n].mean() if valid_n.any() else float('nan'):.3f} | {cv:.3f} |")
        print(ds, j, flush=True)
open(a.out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
