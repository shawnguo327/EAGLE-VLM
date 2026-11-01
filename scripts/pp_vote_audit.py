"""Place Pulse vote audit: are repeated votes on the same pair independent people, or duplicate submissions?
Then re-test 'Δp vs human disagreement' on the pilot's multi-vote pairs after collapsing near-duplicates.
usage: python pp_vote_audit.py [annotations_dir] [pilot_pp_dir]"""
import collections, gzip, json, math, sys
from datetime import datetime
import numpy as np
from scipy import stats

ANN = sys.argv[1] if len(sys.argv) > 1 else "/media/shawn/C/papers/VLM/place-pulse-2.0/annotations"
PP = sys.argv[2] if len(sys.argv) > 2 else "/media/shawn/C/papers/VLM/urban-judge-20261030/pp"
GAP = 86400  # votes on the same pair within a day (same orientation) are treated as one submission

V = collections.defaultdict(list)
for fn in ("pairs_binary.jsonl.gz", "pairs_ties.jsonl.gz"):
    for l in gzip.open(f"{ANN}/{fn}", "rt"):
        d = json.loads(l); a, b = d["left_id"], d["right_id"]
        k = (d["category"], min(a, b), max(a, b))
        w = "tie" if d["choice"] == "equal" else (a if d["choice"] == "left" else b)
        V[k].append((datetime.fromisoformat(d["timestamp"][:19]).timestamp(), w, a))

# 1. gap structure of repeated votes
gaps = collections.Counter(); agree = collections.defaultdict(list)
for k, v in V.items():
    if len(v) < 2:
        continue
    v.sort()
    for i in range(1, len(v)):
        g = v[i][0] - v[i - 1][0]
        b = "<10s" if g < 10 else "<1min" if g < 60 else "<1h" if g < 3600 else "<1d" if g < 86400 else ">=1d"
        gaps[b] += 1
        if "tie" not in (v[i][1], v[i - 1][1]):
            agree[b].append(v[i][1] == v[i - 1][1])
print("consecutive repeated votes by gap:", dict(gaps))
print("agreement by gap:", {b: (len(x), round(float(np.mean(x)), 3)) for b, x in agree.items()})


def dedup(v):
    out = []; last = {}
    for t, w, left in sorted(v):
        if left in last and t - last[left] < GAP:
            continue
        last[left] = t; out.append(w)
    return out

# 2. pilot multi-vote pairs: raw vs deduplicated vote counts, and Δp vs human split
items = [json.loads(l) for l in open(f"{PP}/multi.jsonl")]
for m in ("qwen3vl8b", "qwen35_9b", "qwen25vl7b"):
    R = collections.defaultdict(dict)
    for l in open(f"{PP}/raw_multi_{m}.jsonl"):
        d = json.loads(l)
        if d.get("cond") == "bin" and "lp" in d:
            R[d["id"]][d["order"]] = d["lp"]
    rows = []
    for it in items:
        r = R.get(it["id"])
        if not r or "AB" not in r or "BA" not in r:
            continue
        def p(lp, key):
            z = math.exp(lp["A"]) + math.exp(lp["B"]); return math.exp(lp[key]) / z
        pa = (p(r["AB"], "A") + p(r["BA"], "B")) / 2
        k = (it["category"], min(it["a_id"], it["b_id"]), max(it["a_id"], it["b_id"]))
        raw = [w for _, w, _ in V[k]]; dd = dedup(V[k])
        def split(ws):
            na = sum(w == it["a_id"] for w in ws); nb = sum(w == it["b_id"] for w in ws)
            return na, nb
        rows.append(dict(dp=abs(2 * pa - 1), pa=pa, raw=split(raw), dd=split(dd)))
    n_raw = np.array([sum(r["raw"]) for r in rows]); n_dd = np.array([sum(r["dd"]) for r in rows])
    ok = [r for r in rows if sum(r["dd"]) >= 2]
    cons = np.array([abs(r["dd"][0] - r["dd"][1]) / sum(r["dd"]) for r in ok])  # human consensus in [0,1]
    dp = np.array([r["dp"] for r in ok])
    rho = stats.spearmanr(dp, cons) if len(ok) > 5 else None
    maj = np.array([r["dd"][0] > r["dd"][1] for r in ok]); strict = np.array([r["dd"][0] != r["dd"][1] for r in ok])
    acc = np.mean(((np.array([r["pa"] for r in ok]) > .5) == maj)[strict]) if strict.any() else float("nan")
    print(f"{m}: pairs={len(rows)} raw votes/pair={n_raw.mean():.2f} dedup votes/pair={n_dd.mean():.2f} "
          f"pairs with >=2 independent strict votes={len(ok)} | spearman(dp, human consensus)="
          f"{rho.statistic if rho else float('nan'):.3f} (p={rho.pvalue if rho else float('nan'):.3f}) | acc vs dedup majority={acc:.3f}")
