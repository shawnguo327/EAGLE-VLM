import pandas as pd, numpy as np, pickle, random
from collections import defaultdict, Counter
from common import *
rng = np.random.default_rng(0)
df = pd.read_csv(f"{DATA}/city{CITY}-dataset.csv")
print("masked 999 rows:", int((df.x == 999).sum())); df = df[(df.x != 999) & (df.y != 999)]
if CITY != "D":  # subsample users for speed
    keep = set(np.random.default_rng(5).choice(df.uid.unique(), 6000, replace=False).tolist()); df = df[df.uid.isin(keep)]
df = df.sort_values(["uid", "d", "t"])
uids = df.uid.unique(); rng.shuffle(uids)
ntr = int(0.8 * len(uids)); train_u = set(uids[:ntr].tolist()); test_u = uids[ntr:].tolist()
poi = load_poi()
# Markov fit on train users
tr = df[df.uid.isin(train_u)]
trans = defaultdict(Counter); starts = Counter()
for (u, d), g in tr.groupby(["uid", "d"], sort=False):
    c = list(zip(g.x.values.tolist(), g.y.values.tolist()))
    starts[c[0]] += 1
    for a, b in zip(c[:-1], c[1:]): trans[a][b] += 1
transP = {}
for a, cn in trans.items():
    ks = list(cn.keys()); v = np.array([cn[k] for k in ks], float); transP[a] = (ks, v / v.sum())
sk = list(starts.keys()); sv = np.array([starts[k] for k in sk], float); startsP = (sk, sv / sv.sum())
print("markov states", len(transP))
te = df[df.uid.isin(set(test_u))]
ud = {}
for (u, d), g in te.groupby(["uid", "d"], sort=False):
    ud[(int(u), int(d))] = list(zip(g.t.values.tolist(), g.x.values.tolist(), g.y.values.tolist()))
home = {}
for u in test_u:
    c = Counter()
    for d in range(75):
        for t, x, y in ud.get((u, d), []):
            if t < 14 or t >= 44: c[(x, y)] += 1
    if not c:
        for d in range(75):
            for t, x, y in ud.get((u, d), []): c[(x, y)] += 1
    home[u] = c.most_common(1)[0][0] if c else None
# ladder pool: one user-day per test user, >=6 records
pool = []
for u in test_u:
    days = [d for d in range(75) if len(ud.get((u, d), [])) >= 6]
    if days: pool.append((u, int(rng.choice(days))))
rng.shuffle(pool); pool = pool[:600]
ladder = {"real": [ud[k] for k in pool]}
for s in [1, 2, 4, 8]:
    ladder[f"gauss{s}"] = [gauss(r, s, rng) for r in ladder["real"]]
ladder["tshuffle"] = [tshuffle(r, rng) for r in ladder["real"]]
ladder["markov"] = [markov_sample([t for t, _, _ in r], transP, startsP, rng) for r in ladder["real"]]
ladder["uniform"] = [uniform(r, rng) for r in ladder["real"]]
# generation targets
targets = []
cand_u = list(test_u); random.Random(1).shuffle(cand_u)
for u in cand_u:
    ok = [d for d in range(8, 70) if len(ud.get((u, d), [])) >= 6 and len(ud.get((u, d - 7), [])) >= 3
          and sum(len(ud.get((u, d - k), [])) >= 3 for k in range(1, 8)) >= 5]
    if not ok: continue
    d = int(rng.choice(ok))
    later = [d2 for d2 in range(d + 1, 75) if len(ud.get((u, d2), [])) >= 6]
    if not later: continue
    d2 = int(rng.choice(later))
    ctx = [(k, ud.get((u, d - k), [])) for k in range(7, 0, -1)]
    prev = ud[(u, d - 7)]
    times = [t for t, _, _ in prev]
    h = home[u]
    targets.append(dict(uid=u, d=d, ctx=ctx, real=ud[(u, d)], other_day=ud[(u, d2)], other_d=d2,
                        copy7=prev, copy1=ud.get((u, d - 1)) or prev,
                        markov=markov_sample(times, transP, startsP, rng, start=h),
                        stayhome=[(t, h[0], h[1]) for t in times], home=h))
    if len(targets) >= 240: break
print("pool", len(pool), "targets", len(targets))
pickle.dump(dict(pool=pool, ladder=ladder, targets=targets, poi=poi, home=home), open(f"{WS}/out/prep{SUF}.pkl", "wb"))
t0 = targets[0]
print("\n".join(f"Day -{k}:\n" + render(r, poi) for k, r in t0["ctx"]))
print("REAL\n" + render(t0["real"], poi))
print("G4\n" + render(ladder["gauss4"][0], poi))
