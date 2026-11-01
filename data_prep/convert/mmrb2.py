"""Multimodal RewardBench 2 (rl-research/multimodal-rewardbench-2), t2i + edit -> items/mmrb2.jsonl
Already pairwise: chosen in {a,b}. '<image_k>' placeholders stripped from prompt; prompt_images -> context.
Items whose response has != 1 image are skipped."""
import sys, pathlib, json, re; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from common import *

DS = "mmrb2"
SRC = RAW / "rl-research__multimodal-rewardbench-2"
CRIT = {"t2i": "Prefer the image that better follows the prompt and has better overall quality.",
        "edit": "Prefer the output image that better follows the instruction while staying consistent with the input image(s)."}
recs, full, skipped = [], {}, 0
for sub in CRIT:
    d = pd.concat([pd.read_parquet(f) for f in sorted((SRC / sub).glob("*.parquet"))], ignore_index=True)
    full[sub] = len(d)
    for ix, r in d.iterrows():
        if len(r.response_a_images) != 1 or len(r.response_b_images) != 1:
            skipped += 1
            continue
        c = [(r.response_a_images[0], "a", r.response_a_model), (r.response_b_images[0], "b", r.response_b_model)]
        w = 0 if r.chosen == "a" else 1
        q = re.sub(r"\s*<image_\d+>\s*", " ", r.prompt_text).strip()
        recs.append(dict(id=f"{DS}/{sub}_{ix:04d}", dataset=DS, subset=sub, kind="img", question=q, criterion=CRIT[sub],
                         win=c[w], lose=c[1 - w], _src=list(r.prompt_images),
                         meta=dict(pair_id=r.pair_id, row=int(ix), prompt_source=r.prompt_source,
                                   human_annotations=r.human_annotations, prompt_metadata=r.prompt_metadata,
                                   raw_prompt=r.prompt_text)))
full["total"] = sum(full[k] for k in CRIT)
full["skipped_multi_image_response"] = skipped
recs = orient(stratified_cap(recs))


def mat(r):
    base = r["id"].split("/")[1]
    r["context"] = [save_img(im, DS, f"{base}_src{k}") for k, im in enumerate(r["_src"])]
    r["meta"]["models"] = [c[2] for c in r["cand"]]
    r["meta"]["orig_side"] = [c[1] for c in r["cand"]]
    r["cand"] = [save_img(c[0], DS, f"{base}_{c[1]}") for c in r["cand"]]
    return r


write(DS, recs, mat, full, f"{skipped} item(s) with multi-image responses skipped.")
