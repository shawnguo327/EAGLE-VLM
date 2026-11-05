"""Resolution ablation subsets: items with at least one image (question image / source / candidates) whose long side
exceeds MAXSIDE (768), i.e. the items the default judge input downsamples.  Writes items/<ds>_big.jsonl (same ids and
labels; the full-resolution runs go to this name, the 768-px reference outputs already exist under <ds>).
usage: python make_big.py <items dir> ds1 ds2 ..."""
import json, os, sys, collections
from PIL import Image
D = os.path.abspath(sys.argv[1]); root = os.path.dirname(D)
for ds in sys.argv[2:]:
    out = []; tot = 0
    for name in (ds, ds + "_rest", ds + "_rest2"):
        p = f"{D}/{name}.jsonl"
        if not os.path.exists(p):
            continue
        for l in open(p):
            r = json.loads(l); tot += 1
            ims = list(r["context"]) + (list(r["cand"]) if r["kind"] == "img" else [])
            if ims and max(max(Image.open(f"{root}/{q}").size) for q in ims) > 768:
                out.append(r)
    with open(f"{D}/{ds}_big.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(str(r["label"]) for r in out)
    print(f"{ds}: {len(out)} / {tot} items have an image > 768 px; labels {dict(c)}; subsets {collections.Counter(r['subset'] for r in out).most_common(5)}")
