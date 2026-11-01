"""Sanity check: schema of every item + for 2 random items per dataset every image path exists and opens with PIL.
Also reports orphan image files not referenced by any item."""
import sys, pathlib, json, random; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT
from PIL import Image

rng = random.Random(0)
refd, ok = set(), True
for f in sorted((ROOT / "items").glob("*.jsonl")):
    L = [json.loads(l) for l in open(f)]
    for r in L:
        paths = r["context"] + (r["cand"] if r["kind"] == "img" else [])
        refd.update(paths)
        assert r["label"] in (0, 1, "tie") and len(r["cand"]) == 2 and r["kind"] in ("resp", "img"), r["id"]
        assert all(not p.startswith("/") for p in paths), r["id"]
    for r in rng.sample(L, 2):
        paths = r["context"] + (r["cand"] if r["kind"] == "img" else [])
        for p in paths:
            try:
                with Image.open(ROOT / p) as im:
                    im.load(); sz = im.size
                print(f"  ok {f.stem:12s} {r['id'][:40]:40s} {p} {sz}")
            except Exception as e:
                ok = False; print("  FAIL", r["id"], p, e)
    print(f"{f.stem}: {len(L)} items schema ok")
on_disk = {str(p.relative_to(ROOT)) for p in (ROOT / "images").rglob("*") if p.is_file()}
print("missing referenced:", len(refd - on_disk), "orphans on disk:", len(on_disk - refd))
print("ALL OK" if ok and not (refd - on_disk) else "PROBLEMS")
