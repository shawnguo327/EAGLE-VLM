import sys, json, glob, numpy as np
sys.argv = sys.argv[:1] + ["--out", "/tmp/_av_scratch"]
import action_value as av
av.DIRS = ["results", "results_A"] + glob.glob("/media/shawn/C/papers/VLM/judgebench_local/runs/*/raw")
I = {json.loads(l)["id"]: json.loads(l) for l in open("items/vlrb.jsonl")}
for m in ["qwen3vl8b", "internvl35_8b"]:
    R = av.load("vlrb", m)
    for s in ["hallucination", "reasoning", "general"]:
        ids = [i for i in R if I[i]["subset"] == s and all(av.p_pair(R[i], c) for c in ["tri", "tri_blind", "tri_wrong"])]
        y = np.array([I[i]["label"] in (0, "0") for i in ids])
        out = [f"{c}={(((np.array([av.p_pair(R[i], c)[0] for i in ids]) > .5) == y).mean()):.3f}" for c in ["tri", "tri_blind", "tri_wrong"]]
        print(m, s, len(ids), *out)
