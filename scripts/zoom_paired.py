import sys, json, glob, numpy as np
sys.path.insert(0, "/media/shawn/C/papers/EAGLE-VLM/scripts")
m = sys.argv[1] if len(sys.argv) > 1 else "qwen3vl8b"; sys.argv = sys.argv[:1]
import action_value as av
av.DIRS = ["results", "results_A"] + glob.glob("/media/shawn/C/papers/VLM/judgebench_local/runs/*/raw")
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
C = ["tri", "tri_zoom", "tri_zoomr", "tri_zoomlo", "tri_p1"]
for ds in ["genai_arena", "editreward", "mmrb2"]:
    I = {json.loads(l)["id"]: json.loads(l) for l in open(f"items/{ds}.jsonl")}
    R = av.load(ds, m)
    ids = [i for i in R if i in I and all(av.p_pair(R[i], c) for c in ["tri", "tri_zoom", "tri_zoomr"])]
    if not ids: continue
    have = [c for c in C if all(av.p_pair(R[i], c) for i in ids)]
    P = {c: np.array([av.p_pair(R[i], c)[0] for i in ids]) for c in have}
    T = {c: np.array([av.p_pair(R[i], c)[3] for i in ids]) for c in have}
    y = np.array([I[i]["label"] for i in ids], dtype=object)
    tie = y == "tie"; strict = ~tie
    yy = np.array([1.0 if v == 0 else 0.0 for v in y])
    pb = P["tri"]; h = -(pb*np.log(pb+1e-12)+(1-pb)*np.log(1-pb+1e-12))
    low = strict & (h >= np.quantile(h[strict], .7))
    print(f"== {m} {ds}: n={len(ids)} ties={tie.sum()} low={low.sum()} conds={have}")
    for c in have:
        acc = ((P[c] > .5) == (yy == 1))
        s = f"  {c:11s} acc_all={acc[strict].mean():.3f} acc_low={acc[low].mean():.3f}"
        if tie.sum() > 10: s += f" tieAUROC={roc_auc_score(tie, T[c]):.3f} P(T)|tie={T[c][tie].mean():.3f} P(T)|strict={T[c][strict].mean():.3f}"
        print(s)
    def boot(a, b, mask):
        d = (a[mask].astype(float) - b[mask].astype(float)); n = len(d)
        bs = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
        return f"{d.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]"
    A = {c: ((P[c] > .5) == (yy == 1)) for c in have}
    for ref in [c for c in have if c != "tri_zoom"]:
        print(f"  zoom-{ref:10s} dAcc all {boot(A['tri_zoom'], A[ref], strict)}  low {boot(A['tri_zoom'], A[ref], low)}")
