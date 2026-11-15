"""Addendum 20, section 2: confirmatory triad test on the three agent judges that were not analysed when it was written.
(C-1) strict-cycle rate of the order-averaged forced choice: triads with three failed trajectories minus mixed triads; lower bound > 0.
(C-2) mean tie probability of cyclic minus non-cyclic triads; upper bound <= +.02.
Explicit tie answers and order disagreement of the tie question are reported in separate columns.  Intervals: task bootstrap (2,000),
on the mean over judges.  usage: python a20_triads.py <items_dir> <raw_dir> <out.json>"""
import sys, json, itertools, collections
import numpy as np
sys.path.insert(0, "/media/shawn/C/papers/VLM/agent_reward_bench")
import arb_analysis_v3 as A
items, raw, out = sys.argv[1:4]; JUD = ["internvl35_8b", "qwen35_9b", "qwen3vl32b"]; cyc = lambda xy, yz, xz: (xy and yz and not xz) or (not xy and not yz and xz)
def triad_rows(D):
    rel = {}
    for i in range(len(D["id"])):
        a, b = D["agents"][i]; p = (D["f1"][i] + D["f2"][i]) / 2; tri = D["tri_arg"][i]
        rel[(D["task"][i], a, b)] = (p, bool(D["acc"][i][0]), bool(D["acc"][i][1]), float(D["Q"][i][2]), sum(t == "T" for t in tri) > 0, (tri[0] != "T" and tri[1] != "T" and {"A": 0, "B": 1}[tri[0]] != {"A": 1, "B": 0}[tri[1]]))
    by = collections.defaultdict(set)
    for (task, a, b) in rel: by[task] |= {a, b}
    rows = []
    for task, ags in by.items():
        for x, y, z in itertools.combinations(sorted(ags), 3):
            e = []
            for u, v in ((x, y), (y, z), (x, z)):
                if (task, u, v) in rel: e.append(rel[(task, u, v)])
                elif (task, v, u) in rel: q = rel[(task, v, u)]; e.append((1 - q[0], q[2], q[1], q[3], q[4], q[5]))
                else: break
            if len(e) < 3: continue
            ns = e[0][1] + e[1][1] + e[1][2]
            rows.append((task, {0: "fail", 3: "succ"}.get(ns, "mixed"), cyc(e[0][0] > .5, e[1][0] > .5, e[2][0] > .5), np.mean([q[3] for q in e]), np.mean([q[4] for q in e]), np.mean([q[5] for q in e])))
    return rows
T = {}; info = {}
for j in JUD:
    D, n_all, n_err = A.load(items, raw, j); T[j] = triad_rows(D); info[j] = dict(pairs=len(D["id"]), of=n_all, error_rows=n_err, triads=len(T[j]))
    print(j, info[j])
tasks = sorted({r[0] for j in JUD for r in T[j]}); idx = {t: i for i, t in enumerate(tasks)}
def stats(w):                    # w: weight per task (bootstrap multiplicity)
    c1, c2, ex, od = [], [], [], []
    for j in JUD:
        W = np.array([w[idx[r[0]]] for r in T[j]], float); typ = np.array([r[1] for r in T[j]]); cy = np.array([r[2] for r in T[j]], float); pt = np.array([r[3] for r in T[j]]); e_ = np.array([r[4] for r in T[j]]); o_ = np.array([r[5] for r in T[j]])
        m = lambda v, k: (W[k] * v[k]).sum() / max(W[k].sum(), 1e-9)
        f, mx = typ == "fail", typ == "mixed"; c1.append((m(cy, f), m(cy, mx))); cyb = cy > 0
        c2.append((m(pt, cyb) if (W[cyb]).sum() > 0 else np.nan, m(pt, ~cyb))); ex.append((m(e_, f), m(e_, mx))); od.append((m(o_, f), m(o_, mx)))
    return np.array(c1), np.array(c2), np.array(ex), np.array(od)
c1, c2, ex, od = stats(np.ones(len(tasks))); rng = np.random.default_rng(0); B1, B2 = [], []
for _ in range(2000):
    w = np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)); a, b, _, _ = stats(w); B1.append(np.mean(a[:, 0] - a[:, 1])); B2.append(np.nanmean(b[:, 0] - b[:, 1]))
ci = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]
res = dict(info=info, per_judge={j: dict(cycle_all_failed=float(c1[k, 0]), cycle_mixed=float(c1[k, 1]), tie_prob_cyclic=float(c2[k, 0]), tie_prob_noncyclic=float(c2[k, 1]),
           explicit_tie_all_failed=float(ex[k, 0]), explicit_tie_mixed=float(ex[k, 1]), order_disagree_all_failed=float(od[k, 0]), order_disagree_mixed=float(od[k, 1]),
           n_fail=int(sum(r[1] == "fail" for r in T[j])), n_mixed=int(sum(r[1] == "mixed" for r in T[j])), n_cyclic=int(sum(r[2] for r in T[j]))) for k, j in enumerate(JUD)},
           C1=dict(diff=float(np.mean(c1[:, 0] - c1[:, 1])), ci=ci(B1)), C2=dict(diff=float(np.nanmean(c2[:, 0] - c2[:, 1])), ci=ci(B2)))
res["C1"]["holds"] = res["C1"]["ci"][0] > 0; res["C2"]["holds"] = res["C2"]["ci"][1] <= .02
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
