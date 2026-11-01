"""GenAI-Arena-Bench (TIGER-Lab/GenAI-Arena-Bench), image_generation + image_edition -> items/genai_arena.jsonl
Votes: leftvote/rightvote -> preference; tievote (A=B=good) and bothbad_vote (A=B=bad) -> label "tie" (type in meta.vote_type).
Video generation split not downloaded."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from common import *

DS = "genai_arena"
SRC = RAW / "TIGER-Lab__GenAI-Arena-Bench"
CRIT = {"generation": "Prefer the image that better follows the prompt and has better overall quality.",
        "edition": "Prefer the edit that better follows the instruction while preserving the rest of the image."}
recs, full = [], {}
for sub, folder in [("generation", "image_generation"), ("edition", "image_edition")]:
    d = pd.read_parquet(SRC / folder / "test-00000-of-00001.parquet")
    full[sub] = len(d)
    full[sub + "_votes"] = {k: int(v) for k, v in d.vote_type.value_counts().items()}
    for ix, r in d.iterrows():
        if sub == "generation":
            L, R, q, src = r.left_image, r.right_image, r.prompt, None
            extra = {}
        else:
            L, R, q, src = r.left_output_image, r.right_output_image, r.instruct_prompt, r.source_image
            extra = dict(source_prompt=r.source_prompt, target_prompt=r.target_prompt)
        v = r.vote_type
        lw = v != "rightvote"  # left first unless right wins (ties: arbitrary)
        win, lose = ((L, r.left_model, "L"), (R, r.right_model, "R")) if lw else ((R, r.right_model, "R"), (L, r.left_model, "L"))
        recs.append(dict(id=f"{DS}/{sub}_{ix:04d}", dataset=DS, subset=sub, kind="img", question=q, criterion=CRIT[sub],
                         win=win, lose=lose, tie=v in ("tievote", "bothbad_vote"), _src=src,
                         meta=dict(row=int(ix), vote_type=v, n_votes=1, **extra)))
full["total"] = sum(full[k] for k in CRIT)
recs = orient(stratified_cap(recs))


def mat(r):
    base = r["id"].split("/")[1]
    r["context"] = [save_img(r.pop("_src"), DS, base + "_src")] if r["_src"] is not None else []
    r.pop("_src", None)
    c = r["cand"]
    r["meta"]["models"] = [c[0][1], c[1][1]]
    r["meta"]["orig_side"] = [c[0][2], c[1][2]]
    r["cand"] = [save_img(x[0], DS, f"{base}_{x[2]}") for x in c]
    return r


write(DS, recs, mat, full, "tievote and bothbad_vote both kept as label 'tie' (see meta.vote_type). Video split not converted.")
