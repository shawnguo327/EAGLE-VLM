"""GenAI-Bench (Lin et al., VQAScore; BaiqiL/GenAI-Bench): 1600 prompts x 6 T2I models, each image rated 1-5 by 3 humans.
Pairs within prompt (all 15 model pairs) for a random sample of prompts; meta.scores = mean human rating per image,
label = higher mean better, equal means = tie.  A/B orientation by a hash of the id.  Writes items/genaibench.jsonl.
usage: python make_genaibench.py [n_prompts]"""
import glob, io, itertools, json, os, random, sys, zlib
import pyarrow.parquet as pq
from PIL import Image
RAW = "/media/vipuser/Data1/raw/GenAI-Bench/data"; LOC = "/media/vipuser/Data/eaglevlm/judgebench_local"
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 600
MODELS = ["DALLE_3", "DeepFloyd_I_XL_v1", "Midjourney_6", "SDXL_2_1", "SDXL_Base", "SDXL_Turbo"]
CRIT = "Prefer the image that better follows the prompt and has better overall quality."
os.makedirs(f"{LOC}/images/genaibench", exist_ok=True)
idx = []
for f in sorted(glob.glob(f"{RAW}/*.parquet")):
    t = pq.read_table(f, columns=["Index"]).column(0).to_pylist(); idx += [(f, i) for i in t]
pick = set(i for _, i in random.Random(0).sample(idx, min(NP, len(idx))))
out = []
for f in sorted(glob.glob(f"{RAW}/*.parquet")):
    for r in pq.read_table(f).to_pylist():
        if r["Index"] not in pick:
            continue
        sc, paths = {}, {}
        for m in MODELS:
            hr = (r["HumanRatings"] or {}).get(m); im = r.get(m)
            if not hr or not im or not im.get("bytes"):
                continue
            p = f"images/genaibench/{r['Index']}_{m}.jpg"
            if not os.path.exists(f"{LOC}/{p}"):
                Image.open(io.BytesIO(im["bytes"])).convert("RGB").save(f"{LOC}/{p}", quality=95)
            sc[m] = sum(hr) / len(hr); paths[m] = p
        for a, b in itertools.combinations(sorted(sc), 2):
            iid = f"genaibench/{r['Index']}_{a}_{b}"
            if zlib.crc32(iid.encode()) % 2:
                a, b = b, a
            lab = "tie" if abs(sc[a] - sc[b]) < 1e-9 else ("0" if sc[a] > sc[b] else "1")
            out.append(dict(id=iid, dataset="genaibench", subset=",".join((r.get("Tags") or {}).get("basic") or []) or "basic",
                            kind="img", question=r["Prompt"], context=[], cand=[paths[a], paths[b]], criterion=CRIT, label=lab,
                            meta=dict(scores=[sc[a], sc[b]], models=[a, b], ratings=[r["HumanRatings"][a], r["HumanRatings"][b]])))
with open(f"{LOC}/items/genaibench.jsonl", "w") as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
import collections
c = collections.Counter(x["label"] for x in out); s = [x["meta"]["scores"] for x in out]
print(len(out), "pairs", dict(c), "no-clear-winner(|d|<=1)", round(sum(abs(a - b) <= 1 for a, b in s) / len(s), 3),
      "both<=2", round(sum(max(a, b) <= 2 for a, b in s) / len(s), 3))
