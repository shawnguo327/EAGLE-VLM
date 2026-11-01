"""Shared helpers for judge-bench converters. Run scripts from anywhere: python -I convert/<x>.py"""
import io, json, os, pathlib, random, base64, hashlib, collections
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
CAP = 1500


def save_img(src, dataset, name):
    """src: bytes | base64 str | PIL.Image | dict{'bytes','path'} | file path. Returns path relative to ROOT."""
    if isinstance(src, dict):
        src = src.get("bytes") if src.get("bytes") is not None else src.get("path")
    if isinstance(src, (str, pathlib.Path)) and os.path.exists(str(src)):
        src = open(src, "rb").read()
    elif isinstance(src, str):
        s = src.split(",", 1)[1] if src.startswith("data:") else src
        src = base64.b64decode(s)
    d = ROOT / "images" / dataset
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(src, (bytes, bytearray)):
        im = Image.open(io.BytesIO(src))
        fmt = (im.format or "").upper()
        if fmt in ("JPEG", "PNG"):
            ext = "jpg" if fmt == "JPEG" else "png"
            p = d / f"{name}.{ext}"
            if not p.exists():
                p.write_bytes(src)
            return str(p.relative_to(ROOT))
    else:
        im = src
    alpha = im.mode in ("RGBA", "LA", "P") and ("transparency" in im.info or im.mode != "P")
    ext = "png" if alpha else "jpg"
    p = d / f"{name}.{ext}"
    if not p.exists():
        if ext == "jpg":
            im.convert("RGB").save(p, quality=95)
        else:
            im.save(p)
    return str(p.relative_to(ROOT))


def safe(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))[:120]


def stratified_cap(recs, key=lambda r: r["subset"], cap=CAP, seed=0):
    """Water-fill: every subset gets an equal share; small subsets are kept whole."""
    groups = collections.defaultdict(list)
    for r in recs:
        groups[key(r)].append(r)
    if len(recs) <= cap:
        return list(recs)
    rng = random.Random(seed)
    alloc, left, keys = {}, cap, sorted(groups, key=lambda k: len(groups[k]))
    for i, k in enumerate(keys):
        share = left // (len(keys) - i)
        alloc[k] = min(share, len(groups[k]))
        left -= alloc[k]
    out = []
    for k in sorted(groups):
        g = groups[k]
        out += rng.sample(g, alloc[k]) if alloc[k] < len(g) else g
    return out


def orient(recs, seed=0):
    """recs carry 'win' (preferred cand), 'lose' and optional 'tie'. Randomized A/B placement, balanced within each
    subset: a seeded shuffle of an alternating 0/1 list, so the gold sits at A for ~50% of items -> cand, label."""
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for r in recs:
        groups[r["subset"]].append(r)
    for k in sorted(groups):
        g = [r for r in groups[k] if not r.get("tie")] + [r for r in groups[k] if r.get("tie")]
        n = sum(1 for r in g if not r.get("tie"))
        pos = [i % 2 for i in range(n)]
        rng.shuffle(pos)
        pos += [rng.randint(0, 1) for _ in range(len(g) - n)]
        for r, p in zip(g, pos):
            w, l = r.pop("win"), r.pop("lose")
            tie = r.pop("tie", False)
            r["cand"] = [w, l] if p == 0 else [l, w]
            r["label"] = "tie" if tie else p
    return recs


FIELDS = ["id", "dataset", "subset", "kind", "question", "context", "cand", "criterion", "label", "meta"]


def write(dataset, recs, materialize=None, full_sizes=None, notes=""):
    """materialize(rec) -> rec : extracts images lazily (only for sampled items). Writes items + stats json."""
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids)), "duplicate ids"
    out = ROOT / "items" / f"{dataset}.jsonl"
    stats = collections.Counter()
    with open(out, "w") as f:
        for r in recs:
            if materialize:
                r = materialize(r)
            r = {k: r[k] for k in FIELDS if k in r} | {k: v for k, v in r.items() if k not in FIELDS and not k.startswith("_")}
            for k in ["id", "dataset", "subset", "kind", "question", "context", "cand", "criterion", "label"]:
                assert k in r, (k, r.get("id"))
            assert len(r["cand"]) == 2
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats[(r["subset"], str(r["label"]))] += 1
    summ = {"dataset": dataset, "written": len(recs), "full_sizes": full_sizes or {}, "notes": notes,
            "by_subset_label": {f"{s}|{l}": n for (s, l), n in sorted(stats.items())}}
    (ROOT / "items" / f"{dataset}.stats.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
