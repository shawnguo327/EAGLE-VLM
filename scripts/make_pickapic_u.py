"""Pick-a-Pic v1 standard evaluation splits (PickScore protocol): test_unique + validation_unique (500 + 500, one pair
per prompt).  label_0 = label_1 = 0.5 is an explicit user tie ("indicate a tie if they have no strong preference",
Kirstain et al. 2023).  A/B orientation randomised by a hash of the id (stable under re-runs).
Writes items/pickapic_u.jsonl (subset = split).  usage: python make_pickapic_u.py  (env with pyarrow)"""
import json, os, zlib
import pyarrow.parquet as pq

RAW = "/media/shawn/C/papers/VLM/judgebench_full/raw/pickapic_v1"
LOC = "/media/shawn/C/papers/VLM/judgebench_local"
CRIT = "Prefer the image that better follows the prompt and has better overall quality."
os.makedirs(f"{LOC}/images/pickapic", exist_ok=True)
out, seen = [], set()
for split in ("test_unique", "validation_unique"):
    for r in pq.read_table(f"{RAW}/{split}.parquet").to_pylist():
        u0, u1 = r["image_0_uid"], r["image_1_uid"]
        if u0 == u1 or frozenset((u0, u1)) in seen:
            continue
        seen.add(frozenset((u0, u1)))
        paths = []
        for s, u in (("0", u0), ("1", u1)):
            p = f"images/pickapic/{u}.jpg"
            if not os.path.exists(f"{LOC}/{p}"):
                open(f"{LOC}/{p}", "wb").write(r["jpg_" + s])
            paths.append(p)
        lab = "tie" if r["label_0"] == r["label_1"] else ("0" if r["label_0"] > r["label_1"] else "1")
        iid = f"pickapic_u/{split}_{r['ranking_id']}_{u0[:8]}_{u1[:8]}"
        flip = zlib.crc32(iid.encode()) % 2 == 1
        if flip:
            paths = paths[::-1]; lab = {"0": "1", "1": "0"}.get(lab, lab)
        out.append(dict(id=iid, dataset="pickapic_u", subset=split, kind="img", question=r["caption"], context=[],
                        cand=paths, criterion=CRIT, label=lab,
                        meta=dict(user_id=r["user_id"], ranking_id=r["ranking_id"], flipped=flip,
                                  models=[r["model_1"], r["model_0"]] if flip else [r["model_0"], r["model_1"]])))
with open(f"{LOC}/items/pickapic_u.jsonl", "w") as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
for sp in ("test_unique", "validation_unique"):
    X = [x for x in out if x["subset"] == sp]
    print(sp, len(X), "tie", round(sum(x["label"] == "tie" for x in X) / len(X), 3),
          "0/1", sum(x["label"] == "0" for x in X), sum(x["label"] == "1" for x in X))
