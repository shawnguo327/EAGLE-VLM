"""Cross-family test of shared errors. (1) InternVL error rate conditional on how many Qwen judges are wrong, and its
confidence on Qwen-unanimous errors. (2) AUROC of each judge with and without the pairs all judges get wrong.
usage (from $EAGLEVLM_DATA): python mechanism_cross.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import load_records, auroc
Q = ["qwen3vl8b", "qwen35_9b", "qwen25vl7b"]; I = "internvl35_8b"; J = Q + [I]; out = {}
for ds in "vlrb mmrb mmrlhf mllm_judge genai_arena mjbench mmrb2 imagerewarddb qbench2 editreward pp".split():
    R = {m: {r["id"]: r for r in load_records(f"items/{ds}.jsonl", f"results/{ds}__{m}.jsonl") if r["y"] is not None} for m in J}
    ids = sorted(set.intersection(*[set(v) for v in R.values()])); y = np.array([R[I][i]["y"] for i in ids])
    W = {m: np.array([(R[m][i]["pa"] > .5) != (y[k] == 1) for k, i in enumerate(ids)]) for m in J}
    kq = sum(W[m] for m in Q); kw = kq + W[I]; ci = np.array([abs(2 * R[I][i]["pa"] - 1) for i in ids])
    o = {"ivl_wrong_given_q": {k: float(W[I][kq == k].mean()) for k in range(4)},
         "ivl_conf_correct": float(ci[~W[I]].mean()), "ivl_conf_err_qunan": float(ci[W[I] & (kq == 3)].mean()),
         "auroc_all": {}, "auroc_excl_shared": {}}
    for m in J:
        c = np.abs([R[m][i]["lo"] for i in ids]); keep = kw < 4
        o["auroc_all"][m] = float(auroc(c, ~W[m])); o["auroc_excl_shared"][m] = float(auroc(c[keep], ~W[m][keep]))
    out[ds] = o; print(ds, json.dumps(o))
json.dump(out, open("paper_tables/mechanism_cross.json", "w"), indent=1)
