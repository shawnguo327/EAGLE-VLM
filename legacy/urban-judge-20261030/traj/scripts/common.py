import numpy as np, re, math, json, random
DATA = "/media/shawn/C/papers/VLM/urban-judge-20261030/data"
WS = "/media/shawn/C/papers/VLM/urban-judge-20261030/traj"

import os
CITY = os.environ.get("CITY", "D")
SUF = "" if CITY == "D" else f"_{CITY}"


def load_poi():
    import pandas as pd
    names = [l.strip().lstrip("﻿") for l in open(f"{DATA}/POI_datacategories.csv", encoding="utf-8") if l.strip()]
    p = pd.read_csv(f"{DATA}/POIdata_city{CITY}.csv")
    top = {}
    for (x, y), g in p.groupby(["x", "y"]):
        g = g.sort_values("POI_count", ascending=False)
        top[(int(x), int(y))] = [names[int(c) - 1] for c in g.category.values[:2]]
    return top

def hhmm(t):
    return f"{t//2:02d}:{30*(t%2):02d}"

def segments(recs):
    """recs: sorted list of (t,x,y) -> list of (ts,te,x,y), merging consecutive same cell."""
    segs = []
    for t, x, y in recs:
        if segs and segs[-1][2] == x and segs[-1][3] == y:
            segs[-1][1] = t
        else:
            segs.append([t, t, x, y])
    return [tuple(s) for s in segs]

def render(recs, poi, tags=True, km=True):
    segs = segments(recs)
    lines = []
    prev = None
    for ts, te, x, y in segs:
        tm = hhmm(ts) if ts == te else f"{hhmm(ts)}-{hhmm(te)}"
        s = f"{tm} ({x},{y})"
        if tags:
            c = poi.get((x, y))
            s += " [" + ("; ".join(c) if c else "no POI") + "]"
        if km and prev is not None:
            d = 0.5 * math.hypot(x - prev[0], y - prev[1])
            s += f" moved {d:.1f}km"
        prev = (x, y)
        lines.append(s)
    return "\n".join(lines)

LINE_RE = re.compile(r"(\d{1,2}):(\d{2})(?:\s*[-–~]\s*(\d{1,2}):(\d{2}))?\s*\(?\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)?")

def parse_gen(text):
    recs = {}
    for m in LINE_RE.finditer(text):
        h1, m1, h2, m2, x, y = m.groups()
        x, y = int(x), int(y)
        if not (1 <= x <= 200 and 1 <= y <= 200):
            continue
        t1 = int(h1) * 2 + (1 if int(m1) >= 30 else 0)
        if t1 > 47: continue
        recs[t1] = (t1, x, y)
        if h2 is not None:
            t2 = int(h2) * 2 + (1 if int(m2) >= 30 else 0)
            if t1 < t2 <= 47:
                recs[t2] = (t2, x, y)
    return [recs[k] for k in sorted(recs)]

# ---------- perturbations ----------
def gauss(recs, sigma, rng):
    out = []
    for t, x, y in recs:
        nx = int(np.clip(round(x + rng.normal(0, sigma)), 1, 200))
        ny = int(np.clip(round(y + rng.normal(0, sigma)), 1, 200))
        out.append((t, nx, ny))
    return out

def tshuffle(recs, rng):
    ts = [r[0] for r in recs]
    locs = [(r[1], r[2]) for r in recs]
    perm = rng.permutation(len(locs))
    return [(ts[i], locs[perm[i]][0], locs[perm[i]][1]) for i in range(len(ts))]

def uniform(recs, rng):
    return [(t, int(rng.integers(1, 201)), int(rng.integers(1, 201))) for t, _, _ in recs]

def markov_sample(times, trans, starts, rng, start=None):
    keys, probs = starts
    cur = start if start is not None else keys[rng.choice(len(keys), p=probs)]
    out = [(times[0], cur[0], cur[1])]
    for t in times[1:]:
        if cur in trans:
            ks, ps = trans[cur]
            cur = ks[rng.choice(len(ks), p=ps)]
        out.append((t, cur[0], cur[1]))
    return out

# ---------- metrics ----------
def cdist_km(a, b):
    return 0.5 * math.hypot(a[0] - b[0], a[1] - b[1])

def dtw(seqa, seqb):
    n, m = len(seqa), len(seqb)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = cdist_km(seqa[i - 1], seqb[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / max(n, m)   # km per aligned step (normalized)

def geobleu(gen, ref, N=3, beta=0.5):
    """GEO-BLEU-like: soft n-gram precision with sim=exp(-beta*dist_km), greedy matching,
    uniform geometric mean over n=1..N, brevity penalty. seqs are lists of (x,y)."""
    if not gen or not ref: return 0.0
    logs = []
    for n in range(1, N + 1):
        G = [gen[i:i + n] for i in range(len(gen) - n + 1)]
        R = [ref[i:i + n] for i in range(len(ref) - n + 1)]
        if not G or not R:
            logs.append(math.log(1e-3)); continue
        sims = np.array([[math.prod(math.exp(-beta * cdist_km(g[k], r[k])) for k in range(n)) for r in R] for g in G])
        used = set(); tot = 0.0
        for gi in np.argsort(-sims.max(1)):
            order = np.argsort(-sims[gi])
            for rj in order:
                if rj not in used:
                    used.add(rj); tot += sims[gi, rj]; break
            if len(used) == len(R): break
        logs.append(math.log(max(tot / len(G), 1e-3)))
    bp = 1.0 if len(gen) >= len(ref) else math.exp(1 - len(ref) / len(gen))
    return bp * math.exp(sum(logs) / N)

def slot_err(gen_recs, ref_recs):
    """Mean km error at each reference record slot, using the gen location at the nearest earlier-or-equal gen record (else first)."""
    if not gen_recs: return float("nan")
    gt = [r[0] for r in gen_recs]
    errs = []
    for t, x, y in ref_recs:
        idx = 0
        for i, tt in enumerate(gt):
            if tt <= t: idx = i
        g = gen_recs[idx]
        errs.append(cdist_km((g[1], g[2]), (x, y)))
    return float(np.mean(errs))

def seq_of(recs):
    return [(s[2], s[3]) for s in segments(recs)]

def all_metrics(gen_recs, ref_recs):
    a, b = seq_of(gen_recs), seq_of(ref_recs)
    if not a: return {"dtw": float("nan"), "geobleu": 0.0, "slot_km": float("nan")}
    return {"dtw": dtw(a, b), "geobleu": geobleu(a, b), "slot_km": slot_err(gen_recs, ref_recs)}

# ---------- features for LR baseline ----------
def feats(recs):
    segs = segments(recs)
    xy = np.array([[r[1], r[2]] for r in recs], float) * 0.5
    c = xy.mean(0)
    rg = math.sqrt(((xy - c) ** 2).sum(1).mean())
    jumps = [cdist_km(segs[i][2:], segs[i + 1][2:]) for i in range(len(segs) - 1)] or [0.0]
    cells = [(r[1], r[2]) for r in recs]
    from collections import Counter
    cnt = Counter(cells)
    top = cnt.most_common(1)[0][1] / len(cells)
    return [len(recs), len(segs), len(cnt), rg, np.mean(jumps), np.max(jumps), np.median(jumps),
            np.mean(np.array(jumps) > 5), top, len(segs) / len(recs)]
