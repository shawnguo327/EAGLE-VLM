"""Escalation-ladder comparison per dataset: which rung order buys the most certified automation per unit cost?
Rungs: P = small-judge pairwise (|log-odds|), S = same judge pointwise (|expected-score margin|),
L = large judge (Qwen3-VL-32B) pairwise, H = human.  Cost units: one 8B forward pass = 1, one 32B pass = 4,
one human label = 100 (change with --human-cost).
usage (from $EAGLEVLM_DATA): python ladder_experiment.py --small qwen3vl8b --large qwen3vl32b [--ds ...]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import load_records, ladder_multi

ap = argparse.ArgumentParser(); ap.add_argument("--small", default="qwen3vl8b"); ap.add_argument("--large", default="qwen3vl32b")
ap.add_argument("--ds", default="pp,mmrlhf,editreward,qbench2,vlrb,mmrb,mllm_judge,genai_arena,mjbench,mmrb2,imagerewarddb")
ap.add_argument("--alphas", default=".1,.15,.2,.25,.3"); ap.add_argument("--human-cost", type=float, default=100.0)
a = ap.parse_args()
out = {}
for ds in a.ds.split(","):
    fs, fl = f"results/{ds}__{a.small}.jsonl", f"results/{ds}__{a.large}.jsonl"
    if not os.path.exists(fs):
        continue
    small = {r["id"]: r for r in load_records(f"items/{ds}.jsonl", fs) if r["y"] is not None and "ed" in r}
    large = {r["id"]: r for r in load_records(f"items/{ds}.jsonl", fl)} if os.path.exists(fl) else {}
    ids = [i for i in small if (not large or i in large)]
    if len(ids) < 50:
        continue
    y = np.array([small[i]["y"] for i in ids])
    P = (np.abs([small[i]["lo"] for i in ids]), ((np.array([small[i]["lo"] for i in ids]) > 0).astype(int) != y).astype(int))
    ed = np.array([small[i]["ed"] for i in ids])
    S = (np.abs(ed), np.where(ed == 0, 1, ((ed > 0).astype(int) != y).astype(int)))
    FIX, ADA = (1.0,), (0.0, 1.0)
    configs = {"P": ([P], [2], FIX), "P>S fix": ([P, S], [2, 2], FIX), "P>S": ([P, S], [2, 2], ADA)}
    if large:
        lo = np.array([large[i]["lo"] for i in ids])
        L = (np.abs(lo), ((lo > 0).astype(int) != y).astype(int))
        configs.update({"P>L": ([P, L], [2, 8], ADA), "P>S>L": ([P, S, L], [2, 2, 8], ADA), "L": ([L], [8], FIX)})
    res = {"n": len(ids), "acc": {"P": float(1 - P[1].mean()), "S": float(1 - S[1].mean())}}
    if large:
        res["acc"]["L"] = float(1 - L[1].mean())
    for al in map(float, a.alphas.split(",")):
        for name, (rungs, costs, rhos) in configs.items():
            r = ladder_multi(rungs, al, costs=costs + [a.human_cost], rhos=rhos)
            res[f"{name}@{al}"] = r
    out[ds] = res
    print(f"\n{ds} n={len(ids)} acc={ {k: round(v, 3) for k, v in res['acc'].items()} }")
    for al in map(float, a.alphas.split(",")):
        print(f"  alpha={al:.2f} " + "  ".join(f"{nm}: auto={res[f'{nm}@{al}']['auto']:.3f} cost={res[f'{nm}@{al}']['cost']:.1f}" for nm in configs))
os.makedirs("paper_tables", exist_ok=True)
json.dump(out, open(f"paper_tables/ladder_{a.small}_{a.large}.json", "w"), indent=1)
