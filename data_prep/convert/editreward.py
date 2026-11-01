"""EditReward-Bench (TIGER-Lab/EditReward-Bench), 2pair config only -> items/editreward.jsonl
ranking 'A>B' / 'B>A' / 'A=B' (explicit tie -> label "tie"). 3pair/4pair configs were not downloaded."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from common import *

DS = "editreward"
CRIT = "Prefer the edit that better follows the instruction while preserving the rest of the image, with good visual quality."
d = pd.read_parquet(RAW / "TIGER-Lab__EditReward-Bench/2pair/train-00000-of-00001.parquet")
recs = []
for ix, r in d.iterrows():
    c = [(r.candidate_1, 1, r.model_1, dict(overall=r.overall_score_1, instruction_following=r.instruction_following_1, visual_quality=r.visual_quality_1)),
         (r.candidate_2, 2, r.model_2, dict(overall=r.overall_score_2, instruction_following=r.instruction_following_2, visual_quality=r.visual_quality_2))]
    w = 1 if r.ranking == "B>A" else 0
    recs.append(dict(id=f"{DS}/{ix:03d}_{r.id}", dataset=DS, subset="2pair", kind="img", question=r.instruction, criterion=CRIT,
                     win=c[w], lose=c[1 - w], tie=r.ranking == "A=B", _src=r.source_image,
                     meta=dict(orig_id=r.id, row=int(ix), ranking=r.ranking)))
full = {"2pair": len(d), "total": len(d)}
recs = orient(stratified_cap(recs))


def mat(r):
    base = r["id"].split("/")[1]
    r["context"] = [save_img(r["_src"], DS, base + "_src")]
    r["meta"]["models"] = [c[2] for c in r["cand"]]
    r["meta"]["scores"] = [c[3] for c in r["cand"]]
    r["meta"]["orig_cand_index"] = [c[1] for c in r["cand"]]
    r["cand"] = [save_img(c[0], DS, f"{base}_c{c[1]}") for c in r["cand"]]
    return r


write(DS, recs, mat, full, "2pair only (3pair/4pair not downloaded). Explicit A=B ties kept as 'tie'.")
