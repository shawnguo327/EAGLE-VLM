"""Pick-a-Pic v1 test split (Kirstain et al. 2023, the PickScore benchmark) -> pairwise judge items with human ties.
Each row: a prompt, two generated images, the user's choice; label_0 = label_1 = 0.5 marks a tie ('no preference').
We keep the natural tie rate: a simple random sample of N rows from the downloaded test shards (rows with
are_different == False, i.e. identical images, are dropped).  A/B orientation is randomised (in the raw data image 0
wins about 2:1, an interface artefact).
usage: python make_pickapic.py [N]  (env with pyarrow)"""
import glob, json, os, random, sys
import pyarrow.parquet as pq

RAW = "/media/shawn/C/papers/VLM/judgebench_full/raw/pickapic_v1"
LOC = "/media/shawn/C/papers/VLM/judgebench_local"
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3000
CRIT = "Prefer the image that better follows the prompt and has better overall quality."
os.makedirs(f"{LOC}/images/pickapic", exist_ok=True)
if os.path.exists(f"{LOC}/items/pickapic.jsonl") and "--force" not in sys.argv:
    sys.exit("items/pickapic.jsonl exists: re-running re-draws the A/B flips and would silently misalign existing judge "
             "outputs (ids do not encode orientation).  Pass --force only after deleting the pickapic judge outputs.")
rows = []
for f in sorted(glob.glob(f"{RAW}/test-*.parquet")):
    try:
        t = pq.read_table(f)
    except Exception as e:
        print("skip", f, e); continue
    rows += [(os.path.basename(f), r) for r in t.to_pylist()]
print("rows", len(rows))
rows = [x for x in rows if x[1]["are_different"] and x[1]["image_0_uid"] != x[1]["image_1_uid"]]
seen, uniq = set(), []
for src, r in rows:                       # one row per (ranking, image pair)
    k = (r["ranking_id"], r["image_0_uid"], r["image_1_uid"])
    if k not in seen:
        seen.add(k); uniq.append((src, r))
rng = random.Random(3)
pick = rng.sample(uniq, min(N, len(uniq)))
out = []
for src, r in pick:
    paths = []
    for s in ("0", "1"):
        p = f"images/pickapic/{r['image_' + s + '_uid']}.jpg"
        if not os.path.exists(f"{LOC}/{p}"):
            open(f"{LOC}/{p}", "wb").write(r["jpg_" + s])
        paths.append(p)
    lab = "tie" if r["label_0"] == r["label_1"] else ("0" if r["label_0"] > r["label_1"] else "1")
    flip = rng.random() < .5
    if flip:
        paths = paths[::-1]; lab = {"0": "1", "1": "0"}.get(lab, lab)
    out.append(dict(id=f"pickapic/{r['ranking_id']}_{r['image_0_uid'][:8]}_{r['image_1_uid'][:8]}", dataset="pickapic",
                    subset=" vs ".join(sorted([r["model_0"], r["model_1"]])), kind="img", question=r["caption"], context=[],
                    cand=paths, criterion=CRIT, label=lab,
                    meta=dict(user_id=r["user_id"], ranking_id=r["ranking_id"], flipped=flip, src=src,
                              models=[r["model_0"], r["model_1"]][::-1] if flip else [r["model_0"], r["model_1"]])))
with open(f"{LOC}/items/pickapic.jsonl", "w") as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
nt = sum(x["label"] == "tie" for x in out)
print(f"pickapic {len(out)} items (pool {len(uniq)}), tie share {nt / len(out):.3f}, "
      f"label 0/1 {sum(x['label'] == '0' for x in out)}/{sum(x['label'] == '1' for x in out)}, "
      f"prompts {len(set(x['question'] for x in out))}, users {len(set(x['meta']['user_id'] for x in out))}")
