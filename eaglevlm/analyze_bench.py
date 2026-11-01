"""Summarize judge_bench.py outputs: accuracy, swap consistency, Δp calibration, pointwise, EAGLE gate, ties.
usage: python analyze_bench.py results/   (expects results/<dataset>__<model>.jsonl and items/<dataset>.jsonl)
"""
import collections, glob, json, math, os, sys
import numpy as np

rng = np.random.default_rng(0)


def sm(lp):
    m = max(lp.values())
    e = {k: math.exp(v - m) for k, v in lp.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.r_[p, n])) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def ci(x, B=1000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 2
    v = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def analyze(items_path, res_path):
    items = {json.loads(l)["id"]: json.loads(l) for l in open(items_path)}
    R = collections.defaultdict(dict)
    nerr = 0
    for l in open(res_path):
        d = json.loads(l)
        if "lp" not in d:
            nerr += 1
            continue
        R[d["id"]][(d["cond"], d["order"])] = d["lp"]
    rows = []
    for i, it in items.items():
        r = R.get(i, {})
        if ("bin", "AB") not in r or ("bin", "BA") not in r:
            continue
        ab, ba = sm(r[("bin", "AB")]), sm(r[("bin", "BA")])
        pa = (ab["A"] + ba["B"]) / 2  # P(cand[0] better)
        d = dict(id=i, subset=it.get("subset", ""), label=it["label"], pa=pa, dp=abs(2 * pa - 1),
                 cons=(ab["A"] > .5) == (ba["B"] > .5), pfirst=(ab["A"] + ba["A"]) / 2)
        if ("score", "A") in r and ("score", "B") in r:
            ea = sum(int(k) * v for k, v in sm(r[("score", "A")]).items())
            eb = sum(int(k) * v for k, v in sm(r[("score", "B")]).items())
            d["ed"] = ea - eb
        rows.append(d)
    out = {"n_items": len(items), "n_judged": len(rows), "errors": nerr}
    B = [d for d in rows if d["label"] != "tie"]
    if not B:
        return out
    y = np.array([1 if d["label"] == 0 else 0 for d in B])  # 1 = cand[0] better
    pa = np.array([d["pa"] for d in B]); dp = np.array([d["dp"] for d in B])
    corr = ((pa > .5) == (y == 1)).astype(int)
    out.update(n_binary=len(B), acc=round(corr.mean(), 3), acc_ci=ci(corr),
               swap_consistency=round(np.mean([d["cons"] for d in B]), 3),
               mean_P_first=round(np.mean([d["pfirst"] for d in B]), 3),
               auroc_dp=round(auroc(dp, corr), 3),
               share_dp_ge_0_95=round(float((dp >= .95).mean()), 3))
    qb = np.quantile(dp, [.2, .4, .6, .8]); b = np.digitize(dp, qb)
    out["acc_by_dp_quintile"] = [round(float(corr[b == k].mean()), 3) if (b == k).any() else None for k in range(5)]
    # simple temperature-free ECE using p_correct = (1+dp)/2
    conf = (1 + dp) / 2
    bins = np.clip((conf * 10).astype(int), 0, 9)
    out["ece"] = round(float(sum(abs(corr[bins == k].mean() - conf[bins == k].mean()) * (bins == k).mean()
                                 for k in range(10) if (bins == k).any())), 3)
    if all("ed" in d for d in B):
        ed = np.array([d["ed"] for d in B])
        cp = np.where(ed == 0, .5, ((ed > 0) == (y == 1)).astype(float))
        agree = (ed > 0) == (pa > .5)
        out["pointwise"] = dict(acc=round(cp.mean(), 3), agree_with_pairwise=round(agree.mean(), 3),
                                pairwise_acc_when_disagree=round(corr[~agree].mean(), 3) if (~agree).any() else None)
        # EAGLE gate with crossfit tau (choose on even ids, report on odd, and vice versa)
        idx = np.arange(len(B)); taus = np.linspace(0, 1, 21)
        def gacc(m, t):
            return np.where(dp[m] >= t, corr[m], cp[m]).mean()
        res = []
        for fit, ev in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
            t = taus[int(np.argmax([gacc(fit, t) for t in taus]))]
            res.append((gacc(ev, t) * ev.sum(), ev.sum(), t))
        out["eagle_gate_crossfit"] = dict(acc=round(sum(r[0] for r in res) / sum(r[1] for r in res), 3),
                                          taus=[round(float(r[2]), 2) for r in res],
                                          oracle_best=round(max(np.where(dp >= t, corr, cp).mean() for t in taus), 3))
    subs = sorted(set(d["subset"] for d in B))
    if len(subs) > 1:
        out["by_subset"] = {}
        for s in subs:
            m = np.array([d["subset"] == s for d in B])
            out["by_subset"][s] = dict(n=int(m.sum()), acc=round(corr[m].mean(), 3),
                                       swap=round(np.mean([d["cons"] for d, k in zip(B, m) if k]), 3),
                                       auroc_dp=round(auroc(dp[m], corr[m]), 3))
    T = [d for d in rows if d["label"] == "tie"]
    if T:
        alld = np.array([d["dp"] for d in rows]); ist = np.array([d["label"] == "tie" for d in rows]).astype(int)
        out["ties"] = dict(n=len(T), mean_dp_ties=round(float(np.mean([d["dp"] for d in T])), 3),
                           mean_dp_nonties=round(float(dp.mean()), 3), auroc_lowdp_detects_tie=round(auroc(-alld, ist), 3))
    return out


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else "results"
    S = {}
    for f in sorted(glob.glob(os.path.join(rd, "*.jsonl"))):
        ds, model = os.path.basename(f)[:-6].split("__")
        S[f"{ds}__{model}"] = analyze(f"items/{ds}.jsonl", f)
    json.dump(S, open(os.path.join(rd, "summary.json"), "w"), indent=1)
    hdr = "| dataset | model | n | acc [CI] | swap | P(first) | acc by Δp quintile | AUROC Δp | ECE | pointwise | gate (crossfit) |"
    print(hdr); print("|" + "---|" * 11)
    for k, o in S.items():
        if "acc" not in o:
            continue
        ds, m = k.split("__")
        pw = o.get("pointwise", {}).get("acc", "")
        g = o.get("eagle_gate_crossfit", {}).get("acc", "")
        print(f"| {ds} | {m} | {o['n_binary']} | {o['acc']} {o['acc_ci']} | {o['swap_consistency']} | {o['mean_P_first']} | "
              f"{' '.join(str(x) for x in o['acc_by_dp_quintile'])} | {o['auroc_dp']} | {o['ece']} | {pw} | {g} |")
