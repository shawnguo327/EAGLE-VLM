"""VL-RewardBench (MMInstruction/VL-RewardBench) -> items/vlrb.jsonl
Subsets per the official eval: reasoning = mathverse*/mmmu*, hallucination = hallucination_pair/RLAIF-V/RLHF-V, general = rest.
human_ranking: lower rank = preferred."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from common import *

DS = "vlrb"
CRIT = {"general": "Prefer the response that is more helpful, accurate and appropriate for the image and question.",
        "hallucination": "Prefer the response that is accurate with respect to the image and free of hallucination.",
        "reasoning": "Prefer the response whose reasoning and final answer are correct."}


def subset(i):
    l = i.lower()
    if l.startswith(("mathverse", "mmmu")):
        return "reasoning"
    if l.startswith(("hallucination_pair", "rlaif", "rlhf")):
        return "hallucination"
    return "general"


d = pd.read_parquet(RAW / "MMInstruction__VL-RewardBench/data/test-00000-of-00001.parquet")
recs = []
for ix, r in d.iterrows():
    rk = list(r.human_ranking)
    if rk[0] == rk[1]:
        continue
    w = 0 if rk[0] < rk[1] else 1
    s = subset(r.id)
    recs.append(dict(id=f"{DS}/{ix:04d}_{r.id}", dataset=DS, subset=s, kind="resp", question=r.query, criterion=CRIT[s],
                     win=r.response[w], lose=r.response[1 - w], _img=r.image,
                     meta=dict(orig_id=r.id, row=int(ix), source=r.query_source, models=[r.models[w], r.models[1 - w]],
                               models_order="[chosen, rejected]", judge=r.judge, ground_truth=r.ground_truth)))
full = dict(pd.Series([x["subset"] for x in recs]).value_counts())
recs = orient(stratified_cap(recs))


def mat(r):
    r["context"] = [save_img(r["_img"], DS, f"{r['meta']['row']:04d}")]
    return r


write(DS, recs, mat, {k: int(v) for k, v in full.items()} | {"total": len(d)})
