"""Cross-judge error correlation and the order-consistency filter baseline (Seo et al. 2026), per dataset.
usage (from $EAGLEVLM_DATA): python extra_diag.py  -> paper_tables/extra_diag.json"""
import json, os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import load_records
DS = "vlrb,mmrb,mmrlhf,mllm_judge,genai_arena,mjbench,mmrb2,imagerewarddb,qbench2,editreward,pp".split(",")
M = ["qwen3vl8b", "qwen35_9b", "qwen25vl7b"]
out = {}
for d in DS:
    R = {m: {r["id"]: r for r in load_records(f"items/{d}.jsonl", f"results/{d}__{m}.jsonl") if r["y"] is not None} for m in M}
    ids = sorted(set.intersection(*[set(R[m]) for m in M]))
    E = {m: np.array([int((R[m][i]["pa"] > .5) != (R[m][i]["y"] == 1)) for i in ids]) for m in M}
    corr = {f"{a}~{b}": float(np.corrcoef(E[a], E[b])[0, 1]) for a, b in itertools.combinations(M, 2)}
    both = {f"{a}~{b}": float(((E[a] == 1) & (E[b] == 1)).sum() / max(1, (E[a] == 1).sum())) for a, b in itertools.combinations(M, 2)}
    filt = {}
    for m in M:
        cons = np.array([bool(R[m][i]["cons"]) for i in ids])
        filt[m] = {"coverage": float(cons.mean()), "risk": float(E[m][cons].mean()) if cons.any() else None, "risk_all": float(E[m].mean())}
    out[d] = {"n": len(ids), "err_corr": corr, "p_err_b_given_err_a": both, "order_filter": filt}
    print(d, len(ids), " ".join(f"{k}={v:.2f}" for k, v in corr.items()), "| filter " + " ".join(f"{m}:cov={v['coverage']:.2f},risk={v['risk']:.2f}/{v['risk_all']:.2f}" for m, v in filt.items()))
os.makedirs("paper_tables", exist_ok=True); json.dump(out, open("paper_tables/extra_diag.json", "w"), indent=1)
