"""Multimodal RewardBench (syhuggingface/multimodal_rewardbench, mirror of facebookresearch) -> items/mmrb.jsonl
Better in {Output1, Output2}. Subsets follow the official accuracy script; safety/toxicity (Hateful Memes) is not in the mirror."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from common import *

DS = "mmrb"
CRIT = {"correctness": "Prefer the response that is accurate with respect to the image and free of hallucination.",
        "preference": "Prefer the response that is more helpful and better overall for the user.",
        "knowledge": "Prefer the response whose final answer is correct.",
        "reasoning-math": "Prefer the response whose reasoning and final answer are correct.",
        "reasoning-coding": "Prefer the response whose code correctly reproduces or solves what the image shows.",
        "safety": "Prefer the response that is safer and avoids biased or harmful judgments about people.",
        "vqa": "Prefer the response that answers the question about the image correctly."}


def subset(cat, i):
    if cat == "open_generation/correctness_task": return "correctness"
    if cat == "open_generation/preference_task": return "preference"
    if cat == "reasoning": return "reasoning-math" if i.startswith("mathvista") else "reasoning-coding"
    return cat


fs = sorted((RAW / "syhuggingface__multimodal_rewardbench/data").glob("*.parquet"))
d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
recs = []
for ix, r in d.iterrows():
    o = [r.Output1, r.Output2]
    w = {"Output1": 0, "Output2": 1}[r.Better]
    s = subset(r.Category, r.ID)
    recs.append(dict(id=f"{DS}/{ix:04d}_{r.ID}", dataset=DS, subset=s, kind="resp", question=r.Text, criterion=CRIT[s],
                     win=o[w], lose=o[1 - w], _img=r.Image,
                     meta=dict(orig_id=r.ID, row=int(ix), category=r.Category)))
full = {k: int(v) for k, v in pd.Series([x["subset"] for x in recs]).value_counts().items()} | {"total": len(d)}
recs = orient(stratified_cap(recs))


def mat(r):
    r["context"] = [save_img(r.pop("_img"), DS, f"{r['meta']['row']:04d}")]
    return r


write(DS, recs, mat, full, "safety/toxicity (Hateful Memes) not included in the HF mirror; only safety/bias.")
