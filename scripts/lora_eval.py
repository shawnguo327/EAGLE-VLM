"""Evaluate the tie-channel LoRA diagnostic (out-of-fold predictions from train_pref_v2.py, Qwen3-VL-8B).
Variants: base (no fine-tuning, the judge's own tie prompt), lm (LoRA on LM attention, soft labels, ties = T),
connector (LoRA on the vision-language merger only, same targets), lm_hard (LoRA on LM, ties removed, one-hot winners).
Per dataset (pooled over the two folds; base restricted to the same items):
  tie AUROC of P(T); share of human ties on which T is the argmax; T share overall; 3-way accuracy;
  strict accuracy on decided pairs (argmax of P(A) vs P(B), order-averaged).
Pre-registered (playbook H6): tie AUROC >= .75, T chosen on >= 30% of human ties, strict accuracy within 1 point of
the hard-label control.  NB: the lm_hard run predicts decided pairs only (ties were dropped from its eval set).
usage (judgebench dir): python lora_eval.py --out paper_tables/lora_eval.md"""
import argparse, collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items
from label_noise_lib import per_judge
from eaglev_stats import auroc

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/lora_eval.md")
ap.add_argument("--dir", default="/media/shawn/C/papers/VLM/judgebench_local/a100_sync/out_lora")
ap.add_argument("--tags", default="lm,connector,lm_hard", help="fine-tuned variants to evaluate, e.g. lm,connector,lm_hard,lm_hard_all,lm_e3,connector_e3")
a = ap.parse_args()
SETS = {"mllm_judge_sp": ["mllm_judge_sp"], "mllm_judge": ["mllm_judge"], "genai_arena": ["genai_arena"]}


def read(tag, ds):
    R = collections.defaultdict(dict)
    for f in glob.glob(f"{a.dir}/{ds}__qwen3vl8b_{tag}.f*.r*.jsonl"):
        for l in open(f):
            try:
                x = json.loads(l)
            except ValueError:
                continue
            lp = x["lp"]; z = np.exp(np.array([lp["A"], lp["B"], lp["T"]])); R[x["id"]][x["order"]] = z / z.sum()
    out = {}
    for i, r in R.items():
        if "AB" in r and "BA" in r:
            q = np.array([(r["AB"][0] + r["BA"][1]) / 2, (r["AB"][1] + r["BA"][0]) / 2, (r["AB"][2] + r["BA"][2]) / 2]); out[i] = q / q.sum()
    return out


L = ["# Tie-channel LoRA diagnostic (Qwen3-VL-8B, out-of-fold)", "",
     "| dataset | variant | n | tie AUROC | T on human ties | T share | 3-way acc | strict acc (decided) |", "|---|---|---|---|---|---|---|---|"]
for ds in SETS:
    I = items(ds, rest=(ds != "mllm_judge_sp"))
    TAGS = a.tags.split(",")
    V = {t: read(t, ds) for t in TAGS}
    ids_ft = sorted(set(V["lm"]) & set(V["connector"]))
    if not ids_ft:
        continue
    P = per_judge(ds, "qwen3vl8b", {i: I[i] for i in ids_ft if i in I}); V["base"] = {i: P[i][1] for i in P}
    for tag in ["base"] + TAGS:
        ids = [i for i in (sorted(V[tag]) if tag.startswith("lm_hard") else ids_ft) if i in V[tag] and i in I]
        if not ids:
            continue
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids]); Q = np.array([V[tag][i] for i in ids])
        t = y == 2; am = Q.argmax(1); d = ~t
        tau = auroc(Q[:, 2], t.astype(float)) if 0 < t.mean() < 1 else np.nan
        tie_hit = (am[t] == 2).mean() if t.any() else np.nan
        strict = ((Q[d, 0] > Q[d, 1]) == (y[d] == 0)).mean()
        f = lambda v: "-" if v != v else f"{v:.3f}"
        L.append(f"| {ds} | {tag} | {len(ids)} | {f(tau)} | {f(tie_hit)} | {(am == 2).mean():.3f} | {f((am == y).mean() if t.any() else np.nan)} | {strict:.3f} |")
    print(ds, flush=True)
open(a.out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
