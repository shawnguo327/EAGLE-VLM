import sys, json, glob, numpy as np
sys.argv = sys.argv[:1] + ["--out", "/tmp/_av_scratch"]
import action_value as av
av.DIRS = ["results", "results_A"]
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
def H(p): p = np.clip(p, 1e-9, 1-1e-9); return -(p*np.log(p)+(1-p)*np.log(1-p))
for ds in ["genai_arena", "mllm_judge", "imagerewarddb", "pp"]:
    I = {json.loads(l)["id"]: json.loads(l) for l in open(f"items/{ds}.jsonl")}
    for m in ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"]:
        R = av.load(ds, m)
        ids = [i for i in R if i in I and av.p_pair(R[i], "bin") and av.p_pair(R[i], "tri")]
        if not ids: continue
        t = np.array([I[i]["label"] == "tie" for i in ids])
        if t.sum() < 20: continue
        U = H(np.array([av.p_pair(R[i], "bin")[0] for i in ids])); pT = np.array([av.p_pair(R[i], "tri")[3] for i in ids])
        a1, a0 = roc_auc_score(t, pT), roc_auc_score(t, U); n = len(ids); bs = []
        for _ in range(1000):
            j = rng.integers(0, n, n)
            if 0 < t[j].sum() < n: bs.append(roc_auc_score(t[j], pT[j]) - roc_auc_score(t[j], U[j]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{ds:14s} {m:14s} n={n} ties={t.sum()} AUROC pT={a1:.3f} U={a0:.3f} d={a1-a0:+.3f} [{lo:+.3f},{hi:+.3f}] {'PASS' if lo>0 else ''}")
