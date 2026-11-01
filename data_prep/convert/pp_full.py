"""Place Pulse 2.0 tie-rich sample -> items/pp_full.jsonl (human aleatoric target for street-view perception).
From the author archive votes: N_TIE 'equal' votes and N_STRICT left/right votes per category (6 categories), one vote
per unordered image pair, both images present. Label: left/right -> preference, equal -> "tie". The 1,860-pair
pilot items (items/pp.jsonl) are excluded so the two sets do not overlap.
usage: EAGLEVLM_PREP_ROOT=<data root> python pp_full.py [n_tie_per_cat] [n_strict_per_cat]"""
import sys, pathlib, gzip, json, os, random, collections; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import *

DS = "pp_full"
PPR = pathlib.Path("/media/shawn/C/papers/VLM/place-pulse-2.0")
N_TIE = int(sys.argv[1]) if len(sys.argv) > 1 else 500
N_STRICT = int(sys.argv[2]) if len(sys.argv) > 2 else 500
CRIT = "Judge only the visual appearance of the street view."
pilot = set()
pp_items = pathlib.Path(os.environ.get("EAGLEVLM_DATA", "/media/shawn/C/papers/VLM/judgebench")) / "items/pp.jsonl"
if pp_items.exists():
    for l in open(pp_items):
        r = json.loads(l); pilot.add(frozenset(os.path.basename(p) for p in r["cand"]) | {r["subset"]})
rng = random.Random(0)
pools = collections.defaultdict(list)
for fn in ("pairs_ties.jsonl.gz", "pairs_binary.jsonl.gz"):
    with gzip.open(PPR / "annotations" / fn, "rt") as f:
        for l in f:
            v = json.loads(l)
            pools[(v["category"], v["choice"] == "equal")].append(v)
recs, seen, full = [], set(), {}
for (cat, tie), V in sorted(pools.items()):
    full[f"{cat}|{'tie' if tie else 'strict'}"] = len(V)
    rng.shuffle(V); k = 0
    for v in V:
        key = frozenset((os.path.basename(v["left_image"]), os.path.basename(v["right_image"]))) | {cat}
        if key in seen or key in pilot:
            continue
        L, R = PPR / v["left_image"], PPR / v["right_image"]
        if not (L.exists() and R.exists()):
            continue
        seen.add(key)
        w, lo = (L, R) if v["choice"] != "right" else (R, L)
        recs.append(dict(id=f"{DS}/{cat}_{v['vote_id']}", dataset=DS, subset=cat, kind="img", question=v["question"],
                         context=[], criterion=CRIT, win=str(w), lose=str(lo), tie=tie,
                         meta=dict(vote_id=v["vote_id"], choice=v["choice"], study_id=v["study_id"], timestamp=v["timestamp"])))
        k += 1
        if k >= (N_TIE if tie else N_STRICT):
            break
recs = orient(recs)
write(DS, recs, None, full, f"{N_TIE} equal + {N_STRICT} strict votes per category; one vote per image pair; pilot pairs excluded; absolute image paths (localised by build_pack).")
