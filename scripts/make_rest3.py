"""Complete ImageRewardDB (pairwise expansion, 7230) and MLLM-as-a-Judge (7756): <ds>_rest2 = every pair of
<ds>_full.jsonl not in the base or in <ds>_rest (ids disjoint).  After judging, base + rest + rest2 = the full benchmark
(base orientation kept for shared ids), so no proportional subsampling is needed.
usage: python make_rest3.py"""
import json
IT = "/media/shawn/C/papers/VLM/judgebench_local/items"
rd = lambda f: [json.loads(l) for l in open(f"{IT}/{f}.jsonl")]
for ds in ("imagerewarddb", "mllm_judge"):
    have = {x["id"] for x in rd(ds) + rd(ds + "_rest")}
    out = [x for x in rd(ds + "_full") if x["id"] not in have]
    with open(f"{IT}/{ds}_rest2.jsonl", "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(ds + "_rest2", len(out), "union", len(have) + len(out))
