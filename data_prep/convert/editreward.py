"""EditReward-Bench (TIGER-Lab/EditReward-Bench) -> items/editreward.jsonl
Configs 2pair (always) and, with EAGLEVLM_EDITREWARD_ALL=1, 3pair/4pair (pairwise comparisons drawn from 3- or
4-candidate rankings; comparison_type 'BvsC' names candidate_1/candidate_2). ranking 'X>Y' / 'X=Y' (explicit tie ->
label "tie"). 2pair ids are unchanged from the capped version."""
import sys, pathlib, os; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from common import *

DS = "editreward"
CRIT = "Prefer the edit that better follows the instruction while preserving the rest of the image, with good visual quality."
CFGS = ["2pair"] + (["3pair", "4pair"] if os.environ.get("EAGLEVLM_EDITREWARD_ALL") == "1" else [])
recs, full = [], {}
for cfg in CFGS:
    d = pd.concat([pd.read_parquet(f) for f in sorted((RAW / "TIGER-Lab__EditReward-Bench" / cfg).glob("*.parquet"))], ignore_index=True)
    full[cfg] = len(d)
    for ix, r in d.iterrows():
        letters = str(getattr(r, "comparison_type", "") or "AvsB").split("vs")
        if len(letters) != 2:
            letters = ["A", "B"]
        rk = str(r.ranking); tie = "=" in rk
        w = 0 if (tie or rk[0] == letters[0]) else 1
        c = [(r.candidate_1, 1, r.model_1, dict(overall=r.overall_score_1, instruction_following=r.instruction_following_1, visual_quality=r.visual_quality_1)),
             (r.candidate_2, 2, r.model_2, dict(overall=r.overall_score_2, instruction_following=r.instruction_following_2, visual_quality=r.visual_quality_2))]
        rid = f"{DS}/{ix:03d}_{r.id}" if cfg == "2pair" else f"{DS}/{cfg}_{ix:04d}_{r.id}"
        recs.append(dict(id=rid, dataset=DS, subset=cfg, kind="img", question=r.instruction, criterion=CRIT,
                         win=c[w], lose=c[1 - w], tie=tie, _src=r.source_image,
                         meta=dict(orig_id=r.id, row=int(ix), ranking=rk, config=cfg)))
full["total"] = sum(full.values())
recs = orient(stratified_cap(recs))


def mat(r):
    base = r["id"].split("/")[1]
    r["context"] = [save_img(r["_src"], DS, base + "_src")]
    r["meta"]["models"] = [c[2] for c in r["cand"]]
    r["meta"]["scores"] = [c[3] for c in r["cand"]]
    r["meta"]["orig_cand_index"] = [c[1] for c in r["cand"]]
    r["cand"] = [save_img(c[0], DS, f"{base}_c{c[1]}") for c in r["cand"]]
    return r


write(DS, recs, mat, full, f"configs {CFGS}. Explicit X=Y ties kept as 'tie'.")
