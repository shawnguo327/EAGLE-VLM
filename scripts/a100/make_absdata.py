"""Two more pair sets with HUMAN ABSOLUTE SCORES for the pairwise -> pointwise router (same conventions as genaibench):
  agiqa3k  AGIQA-3K (Li et al., IEEE TCSVT 2023; images from chaofengc/IQA-PyTorch-Datasets, MOS from the authors' repo
           lcysyzxdxc/AGIQA-3k-Database data.csv): 300 prompts x ~10 generated images (6 generators). Pairs within prompt
           for a random sample of prompts; score = mean of mos_quality and mos_align (0-5 scale).
  polaris  Polaris test (yuwd/Polaris; Wada et al., CVPR 2024): image + candidate caption + human score in {0,.25,.5,.75,1}.
           Pairs of candidate captions of the same image (grouped by image bytes).
label = higher human score better, equal scores = tie; A/B orientation by a hash of the id.
meta.scores = the two human scores, meta.scale = [lo, hi] of the score scale (router level claims use a quarter of the range:
'no clear winner' = |gap| <= (hi-lo)/4, 'both unacceptable' = both <= lo + (hi-lo)/4 - the same rule as |gap| <= 1, both <= 2 on
1-5 scales).
usage: python make_absdata.py [n_prompts_agiqa=all] [max_pairs_polaris=all]  (ids are stable across sizes)"""
import collections, csv, glob, hashlib, io, itertools, json, os, random, sys, zipfile, zlib
import pyarrow.parquet as pq
from PIL import Image
RAW = "/media/vipuser/Data1/raw"; LOC = "/media/vipuser/Data/eaglevlm/judgebench_local"
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 100000     # default: all prompts
NPOL = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9     # default: all pairs


def orient(iid, a, b):
    return (b, a) if zlib.crc32(iid.encode()) % 2 else (a, b)


def write(name, out):
    with open(f"{LOC}/items/{name}.jsonl", "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    lo, hi = out[0]["meta"]["scale"]; q = (hi - lo) / 4
    s = [x["meta"]["scores"] for x in out]
    print(name, len(out), "pairs", dict(collections.Counter(x["label"] for x in out)),
          "no-clear-winner", round(sum(abs(a - b) <= q for a, b in s) / len(s), 3),
          "both-unacceptable", round(sum(max(a, b) <= lo + q for a, b in s) / len(s), 3), flush=True)


# ---- AGIQA-3K
os.makedirs(f"{LOC}/images/agiqa3k", exist_ok=True)
R = list(csv.DictReader(open(f"{RAW}/IQA/meta/AGIQA3K_data.csv")))
groups = collections.defaultdict(list)
for r in R:
    groups[r["prompt"]].append(r)
prompts = sorted(groups); pidx = {p: k for k, p in enumerate(prompts)}       # stable id = index in the sorted prompt list
pick = sorted(random.Random(0).sample(prompts, min(NP, len(prompts))), key=pidx.get)
zf = zipfile.ZipFile(f"{RAW}/IQA/AGIQA-3K.zip"); names = {os.path.basename(n): n for n in zf.namelist()}
CRIT = "Prefer the image that better follows the prompt and has better overall quality."
out = []
for p in pick:
    k = pidx[p]
    rows = [r for r in groups[p] if r["name"] in names]
    for r in rows:
        dst = f"{LOC}/images/agiqa3k/{r['name']}"
        if not os.path.exists(dst):
            open(dst, "wb").write(zf.read(names[r["name"]]))          # original bytes
    sc = {r["name"]: (float(r["mos_quality"]) + float(r["mos_align"])) / 2 for r in rows}
    for a, b in itertools.combinations(sorted(sc), 2):
        iid = f"agiqa3k/{k}_{a.rsplit('.', 1)[0]}_{b.rsplit('.', 1)[0]}"; a, b = orient(iid, a, b)
        lab = "tie" if abs(sc[a] - sc[b]) < 1e-9 else ("0" if sc[a] > sc[b] else "1")
        out.append(dict(id=iid, dataset="agiqa3k", subset=a.split("_")[0] + "|" + b.split("_")[0], kind="img", question=p,
                        context=[], cand=[f"images/agiqa3k/{a}", f"images/agiqa3k/{b}"], criterion=CRIT, label=lab,
                        meta=dict(scores=[sc[a], sc[b]], scale=[0.0, 5.0], models=[a.split("_")[0], b.split("_")[0]])))
write("agiqa3k", out)

# ---- Polaris (test shards present locally)
os.makedirs(f"{LOC}/images/polaris", exist_ok=True)
rows = []
for f in sorted(glob.glob(f"{RAW}/Polaris/data/test-*.parquet")):
    rows += pq.read_table(f).to_pylist()
byimg = collections.defaultdict(list)
for r in rows:
    h = hashlib.md5(r["img"]["bytes"]).hexdigest()[:16]; byimg[h].append(r)
CRITC = "Prefer the caption that describes the image more accurately and completely."
pairs = []
for h, rs in sorted(byimg.items()):
    dst = f"{LOC}/images/polaris/{h}.jpg"
    if not os.path.exists(dst):
        im = Image.open(io.BytesIO(rs[0]["img"]["bytes"]))
        if im.format == "JPEG":
            open(dst, "wb").write(rs[0]["img"]["bytes"])
        else:
            im.convert("RGB").save(dst, quality=95)
    cands = {}
    for r in rs:
        cands.setdefault(r["cand"].strip(), float(r["human_score"]))
    for a, b in itertools.combinations(sorted(cands), 2):
        pairs.append((h, a, b, cands[a], cands[b]))
random.Random(0).shuffle(pairs); pairs = pairs[:NPOL]
out = []
for h, a, b, sa, sb in pairs:
    iid = "polaris/" + h + "_" + hashlib.md5((a + "||" + b).encode()).hexdigest()[:10]
    (a, sa), (b, sb) = orient(iid, (a, sa), (b, sb))
    lab = "tie" if abs(sa - sb) < 1e-9 else ("0" if sa > sb else "1")
    out.append(dict(id=iid, dataset="polaris", subset="caption", kind="resp", question="Describe this image.",
                    context=[f"images/polaris/{h}.jpg"], cand=[a, b], criterion=CRITC, label=lab,
                    meta=dict(scores=[sa, sb], scale=[0.0, 1.0])))
print("polaris images", len(byimg), "candidate pairs available", len(pairs))
write("polaris", out)
