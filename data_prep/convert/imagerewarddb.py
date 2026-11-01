"""ImageRewardDB (THUDM/ImageRewardDB), test split -> items/imagerewarddb.jsonl
Per prompt, expert ranking over 4-9 images (rank 1 = best). All within-prompt pairs with distinct rank are formed
(equal ranks skipped: no explicit tie label), then stratified-sampled by prompt classification.
Images are read directly from the test zips; webp is re-encoded to JPEG (q95)."""
import sys, pathlib, itertools, zipfile; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from common import *

DS = "imagerewarddb"
SRC = RAW / "THUDM__ImageRewardDB"
CRIT = "Prefer the image that better follows the prompt and has better overall quality."
d = pd.read_parquet(SRC / "metadata-test.parquet")
KEEP_TIES = os.environ.get("EAGLEVLM_KEEP_TIES") == "1"   # full set: equal expert ranks become label "tie"
recs, ties = [], 0
for pid, g in d.groupby("prompt_id", sort=True):
    rows = list(g.itertuples())
    for a, b in itertools.combinations(rows, 2):
        if a.rank == b.rank:
            ties += 1
            if not KEEP_TIES:
                continue
        w, l = (a, b) if a.rank <= b.rank else (b, a)
        info = lambda x: dict(path=x.image_path, rank=int(x.rank), overall=int(x.overall_rating),
                              alignment=int(x.image_text_alignment_rating), fidelity=int(x.fidelity_rating))
        recs.append(dict(id=f"{DS}/{pid}_{pathlib.Path(w.image_path).stem[:8]}_{pathlib.Path(l.image_path).stem[:8]}",
                         dataset=DS, subset=w.classification, kind="img", question=w.prompt, context=[], criterion=CRIT,
                         win=info(w), lose=info(l), tie=a.rank == b.rank,
                         meta=dict(prompt_id=pid, n_images_for_prompt=int(w.image_amount_in_total))))
full = {"images": len(d), "prompts": int(d.prompt_id.nunique()), "pairs_distinct_rank": len(recs) - (ties if KEEP_TIES else 0), "pairs_equal_rank": ties, "equal_rank_kept_as_tie": KEEP_TIES,
        "by_class": {k: int(v) for k, v in pd.Series([r["subset"] for r in recs]).value_counts().items()}}
recs = orient(stratified_cap(recs))
zips = {}


def load(path):
    part, name = path.split("/")[2], path.split("/")[-1]
    if part not in zips:
        zips[part] = zipfile.ZipFile(SRC / "images/test" / f"{part}.zip")
    return zips[part].read(name)


def mat(r):
    r["meta"]["cand_info"] = r["cand"]
    r["cand"] = [save_img(load(c["path"]), DS, pathlib.Path(c["path"]).stem) for c in r["cand"]]
    return r


write(DS, recs, mat, full, "Pairs from expert rankings; equal-rank pairs skipped.")
