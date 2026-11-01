import sys, json, glob, numpy as np
sys.argv = sys.argv[:1] + ["--out", "/tmp/_av_scratch"]
import action_value as av
av.DIRS = ["results", "results_A"] + glob.glob("/media/shawn/C/papers/VLM/judgebench_local/runs/*/raw")
rng = np.random.default_rng(0)
M, O = "qwen3vl8b", "internvl35_8b"
def boot(x):
    x = np.asarray(x, float); n = len(x)
    if n == 0: return "-"
    b = [x[rng.integers(0, n, n)].mean() for _ in range(2000)]
    return f"{x.mean():.2f} [{np.percentile(b,2.5):.2f},{np.percentile(b,97.5):.2f}]"
print("ds | n | acc tri/blind/wrong | agree(blind,tri) | agree(wrong,tri) | conf-err n | kept blind | kept wrong | shared n | shared kept blind | shared kept wrong | conf-correct kept blind")
for ds in ["mllm_judge", "vlrb", "mmrb", "mmrlhf"]:
    I = {json.loads(l)["id"]: json.loads(l) for l in open(f"items/{ds}.jsonl")}
    R, Ro = av.load(ds, M), av.load(ds, O)
    ids = [i for i in R if i in I and I[i]["label"] in (0, 1) and all(av.p_pair(R[i], c) for c in ["tri", "tri_blind", "tri_wrong"])]
    p = {c: np.array([av.p_pair(R[i], c)[0] for i in ids]) for c in ["tri", "tri_blind", "tri_wrong"]}
    y = np.array([I[i]["label"] == 0 for i in ids])
    po = np.array([av.p_pair(Ro[i], "tri")[0] if i in Ro and av.p_pair(Ro[i], "tri") else np.nan for i in ids])
    acc = {c: ((p[c] > .5) == y).mean() for c in p}
    conf = np.abs(p["tri"] - .5) > .4
    err = (p["tri"] > .5) != y
    ce = conf & err
    sh = ce & ~np.isnan(po) & (np.abs(po - .5) > .4) & ((po > .5) == (p["tri"] > .5))
    kb = ((p["tri_blind"] > .5) == (p["tri"] > .5)); kw = ((p["tri_wrong"] > .5) == (p["tri"] > .5))
    cc = conf & ~err
    print(f"{ds} | {len(ids)} | {acc['tri']:.3f}/{acc['tri_blind']:.3f}/{acc['tri_wrong']:.3f} | {kb.mean():.2f} | {kw.mean():.2f} | {ce.sum()} | {boot(kb[ce])} | {boot(kw[ce])} | {sh.sum()} | {boot(kb[sh])} | {boot(kw[sh])} | {boot(kb[cc])}")
