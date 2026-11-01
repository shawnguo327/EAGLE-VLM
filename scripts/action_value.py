"""Which extra computation fixes which kind of low confidence? (review response: TrustJudge-style protocol conflict
analysis + "value of computation" per uncertainty type). Post-hoc on existing predictions, no GPU.

Base decision P = order-averaged forced choice (bin). Its predictive entropy splits exactly into
    H(p_bar) = U + V,   U = [H(p_AB) + H(p_BA)] / 2  (hesitation inside each order),
                        V = JS divergence between the two orders (order conflict).
For every follow-up action a (pointwise score with expected / argmax readout, tri, str7, paraphrase, zoom variants,
blind / wrong image, 32B) on strict (non-tie) pairs:
    correction = P(P wrong, a right),  damage = P(P right, a wrong),  net = correction - damage
overall, and inside the 30% least-confident pairs split at the median of V / H(p_bar) into "conflict" (order
disagreement dominates) and "hesitant" (both orders unsure). Paired bootstrap CI for net(conflict) - net(hesitant).
For tri-family actions also: tie AUROC of P(T) (human ties vs strict) and mean P(T) on ties vs on low-confidence errors.
TrustJudge baselines: score-comparison inconsistency (P vs S, expected vs argmax score) and bidirectional tri
aggregation m[k] = p_AB[k] + p_BA[-k] (argmax incl. T).
Transitivity (ImageRewardDB, same-prompt triangles): rate of cyclic triangles and edge agreement with humans for
P, S_exp, S_arg, tri-B, and the mixed rule "P, switch to S when P is in its least-confident 30%".

usage (from $EAGLEVLM_DATA): python action_value.py --dirs results,results_A,../judgebench_local/runs/*/raw
       [--judges ...] [--out paper_tables/action_value]"""
import argparse, collections, glob, itertools, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import _sm, auroc

ap = argparse.ArgumentParser()
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen35_9b,qwen25vl7b")
ap.add_argument("--dirs", default="results,results_A")
ap.add_argument("--ds", default="genai_arena,mllm_judge,pp,editreward,mmrb2,vlrb,mmrb,mjbench,imagerewarddb,qbench2,mmrlhf")
ap.add_argument("--big", default="qwen3vl32b", help="larger judge used as the 'L' action (bin, wherever present)")
ap.add_argument("--low", type=float, default=0.3)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--out", default="paper_tables/action_value")
a = ap.parse_args()
H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))
DIRS = [d for pat in a.dirs.split(",") for d in sorted(glob.glob(pat))]
TRI = ["tri", "tri_p1", "tri_p2", "tri_p3", "tri_jpeg", "tri_crop", "tri_blind", "tri_wrong", "tri_zoom", "tri_zoomr", "tri_zoomlo"]


def load(ds, m):
    R = collections.defaultdict(dict)
    for d in DIRS:
        for suf in ("", "_edit", "_s32", "_full"):
            for f in glob.glob(f"{d}/{ds}{suf}__{m}.jsonl") + glob.glob(f"{d}/{ds}{suf}__{m}.s*.jsonl"):
                for l in open(f):
                    x = json.loads(l)
                    if "lp" in x:
                        R[x["id"]][(x["cond"], x["order"])] = x["lp"]
    return R


def p_pair(r, c):
    """P(cand0 better) from a pairwise condition, both orders; for tri renormalized over A/B. Also per-order values."""
    if (c, "AB") not in r or (c, "BA") not in r:
        return None
    ab, ba = _sm(r[(c, "AB")]), _sm(r[(c, "BA")])
    if c == "str7":
        d = np.array([(ab[str(k)] + ba[str(8 - k)]) / 2 for k in range(1, 8)])
        return d[:3].sum() / max(1e-9, d[:3].sum() + d[4:].sum()), None, None, d[3]
    pab = ab["A"] / (ab["A"] + ab["B"]); pba = ba["B"] / (ba["A"] + ba["B"])
    t = (ab["T"] + ba["T"]) / 2 if "T" in ab else None
    return (pab + pba) / 2, pab, pba, t


def score_pred(r):
    if ("score", "A") not in r or ("score", "B") not in r:
        return None
    qa, qb = _sm(r[("score", "A")]), _sm(r[("score", "B")])         # normalized over the digits 1..9
    ea = sum(int(k) * v for k, v in qa.items()); eb = sum(int(k) * v for k, v in qb.items())
    aa = int(max(qa, key=qa.get)); ab_ = int(max(qb, key=qb.get))
    return ea - eb, aa - ab_


def tj_tri(r):
    """TrustJudge Option B on the tri readout: m[k] = p_AB[k] + p_BA[-k]; returns 0 (cand0), 1 (cand1) or 'T'."""
    if ("tri", "AB") not in r or ("tri", "BA") not in r:
        return None
    ab, ba = _sm(r[("tri", "AB")]), _sm(r[("tri", "BA")])
    m = {0: ab["A"] + ba["B"], 1: ab["B"] + ba["A"], "T": ab["T"] + ba["T"]}
    return max(m, key=m.get)


def boot_ci(fn, n, B):
    g = np.random.default_rng(0); v = [fn(g.integers(0, n, n)) for _ in range(B)]
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


items = {}
for ds in a.ds.split(","):
    f = f"items/{ds}.jsonl"
    if os.path.exists(f):
        items[ds] = {json.loads(l)["id"]: json.loads(l) for l in open(f)}
big = {ds: load(ds, a.big) for ds in items} if a.big else {}

rows_out, cons_out, tj_out, trans_out = [], [], [], []
for m in a.judges.split(","):
    for ds, I in items.items():
        R = load(ds, m)
        ids = [i for i in I if i in R and p_pair(R[i], "bin")]
        if len(ids) < 50:
            continue
        y = {i: I[i]["label"] for i in ids}
        strict = [i for i in ids if y[i] in (0, 1)]
        ties = [i for i in ids if y[i] == "tie"]
        P = {i: p_pair(R[i], "bin") for i in ids}
        pbar = np.array([P[i][0] for i in strict]); Hb = H(pbar)
        U = np.array([(H(P[i][1]) + H(P[i][2])) / 2 for i in strict]); V = Hb - U
        yy = np.array([y[i] for i in strict]); base = ((pbar < .5).astype(int) == yy)
        low = Hb >= np.quantile(Hb, 1 - a.low)
        share = np.where(Hb > 1e-9, V / np.maximum(Hb, 1e-9), 0)
        med = np.median(share[low]); conf = low & (share >= med); hes = low & (share < med)
        # --- actions
        acts = {}
        S = {i: score_pred(R[i]) for i in strict}
        if sum(v is not None for v in S.values()) > .8 * len(strict):
            acts["S_exp"] = np.array([(0 if S[i][0] > 0 else 1) if S[i] and S[i][0] != 0 else -1 for i in strict])
            acts["S_arg"] = np.array([(0 if S[i][1] > 0 else 1) if S[i] and S[i][1] != 0 else -1 for i in strict])
        for c in ["str7"] + TRI:
            v = {i: p_pair(R[i], c) for i in strict}
            if sum(x is not None for x in v.values()) >= 50:
                acts[c] = np.array([(0 if v[i][0] > .5 else 1) if v[i] else -2 for i in strict])
        if ds in big:
            v = {i: p_pair(big[ds][i], "bin") if i in big[ds] else None for i in strict}
            if sum(x is not None for x in v.values()) >= 50:
                acts["L32"] = np.array([(0 if v[i][0] > .5 else 1) if v[i] else -2 for i in strict])
        for name, pred in acts.items():
            ok = pred != -2                                   # action not run on this pair
            ac = np.where(pred == -1, 0.5, (pred == yy).astype(float))       # undecided (equal scores) = half
            def stats(mask):
                mk = mask & ok
                if mk.sum() < 10:
                    return dict(n=int(mk.sum()))
                cor = float(((~base) & (ac == 1))[mk].mean() + .5 * ((~base) & (ac == .5))[mk].mean())
                dmg = float((base & (ac == 0))[mk].mean() + .5 * (base & (ac == .5))[mk].mean())
                return dict(n=int(mk.sum()), base=float(base[mk].mean()), acc=float(ac[mk].mean()), corr=cor, dmg=dmg,
                            net=cor - dmg, cr=float(((pred != (pbar < .5).astype(int)))[mk].mean()))
            st = {"all": stats(np.ones_like(low)), "low": stats(low), "conflict": stats(conf), "hesitant": stats(hes)}
            net = np.array(ac - base, float)
            def dnet(ix):
                c_, h_ = conf[ix] & ok[ix], hes[ix] & ok[ix]
                return net[ix][c_].mean() - net[ix][h_].mean() if c_.sum() and h_.sum() else np.nan
            if st["conflict"].get("n", 0) >= 10 and st["hesitant"].get("n", 0) >= 10:
                st["diff"] = st["conflict"]["net"] - st["hesitant"]["net"]; st["diff_ci"] = boot_ci(dnet, len(net), a.boot)
            if name.startswith("tri") and ties:
                v = {i: p_pair(R[i], name) for i in ties + strict}
                tt = [i for i in ties if v[i]]; ss = [i for i in strict if v[i]]
                if len(tt) >= 10:
                    sc = [v[i][3] for i in tt + ss]; lab = [1] * len(tt) + [0] * len(ss)
                    st["tieAUROC_pT"] = auroc(sc, lab)
                    st["pT_ties"] = float(np.mean([v[i][3] for i in tt]))
                    er = [i for i, b, l_ in zip(strict, base, low) if (not b) and l_ and v[i]]
                    st["pT_lowconf_err"] = float(np.mean([v[i][3] for i in er])) if er else None
            rows_out.append(dict(judge=m, ds=ds, action=name, **st))
        # --- TrustJudge: score-comparison inconsistency, bidirectional tri
        if "S_exp" in acts:
            pdec = (pbar < .5).astype(int)
            cons_out.append(dict(judge=m, ds=ds, n=len(strict),
                                 CR_exp=float((acts["S_exp"] != pdec).mean()), CR_arg=float((acts["S_arg"] != pdec).mean()),
                                 tie_arg=float((acts["S_arg"] == -1).mean()), tie_exp=float((acts["S_exp"] == -1).mean()),
                                 acc_P=float(base.mean()),
                                 acc_exp=float(np.where(acts["S_exp"] == -1, .5, acts["S_exp"] == yy).mean()),
                                 acc_arg=float(np.where(acts["S_arg"] == -1, .5, acts["S_arg"] == yy).mean()),
                                 U_lowconf=float(U[low].mean()), V_lowconf=float(V[low].mean()),
                                 conflict_share_low=float((share[low] >= .5).mean())))
        tj = {i: tj_tri(R[i]) for i in ids}
        if sum(v is not None for v in tj.values()) > .8 * len(ids):
            d = dict(judge=m, ds=ds, T_rate=float(np.mean([tj[i] == "T" for i in ids if tj[i] is not None])),
                     acc_strict=float(np.mean([tj[i] == y[i] for i in strict if tj[i] is not None])))
            if ties:
                tp = sum(tj[i] == "T" for i in ties if tj[i] is not None)
                pt = sum(tj[i] == "T" for i in ids if tj[i] is not None)
                d.update(tie_recall=tp / len(ties), tie_precision=tp / pt if pt else None)
            tj_out.append(d)
        # --- transitivity on same-prompt triangles (ImageRewardDB)
        if ds == "imagerewarddb":
            E = {}
            for i in ids:
                c0, c1 = I[i]["cand"]
                pr = P[i][0]; s = score_pred(R[i]); t = tj.get(i)
                E[(c0, c1)] = dict(y=y[i], P=0 if pr > .5 else 1, Hb=H(pr),
                                   S_exp=None if not s or s[0] == 0 else (0 if s[0] > 0 else 1),
                                   S_arg=None if not s or s[1] == 0 else (0 if s[1] > 0 else 1), triB=None if t == "T" else t)
            thr = np.quantile([e["Hb"] for e in E.values()], 1 - a.low)
            for e in E.values():
                e["mixed"] = e["S_exp"] if (e["Hb"] >= thr and e["S_exp"] is not None) else e["P"]
                e["human"] = e["y"] if e["y"] in (0, 1) else None
            byp = collections.defaultdict(dict)
            for i in ids:
                c0, c1 = I[i]["cand"]; byp[I[i]["meta"]["prompt_id"]][(c0, c1)] = E[(c0, c1)]
            res = {}
            for proto in ("human", "P", "S_exp", "S_arg", "triB", "mixed"):
                ncyc = ntri = 0
                for G in byp.values():
                    win = {}
                    for (c0, c1), e in G.items():
                        if e[proto] is not None:
                            win[frozenset((c0, c1))] = c0 if e[proto] == 0 else c1
                    nodes = set().union(*[set(k) for k in G])
                    for x, yv, z in itertools.combinations(sorted(nodes), 3):
                        ks = [frozenset((x, yv)), frozenset((yv, z)), frozenset((x, z))]
                        if all(k in win for k in ks):
                            ntri += 1
                            w = collections.Counter(win[k] for k in ks)
                            ncyc += max(w.values()) == 1          # each node wins once -> cycle
                agree = np.mean([e[proto] == e["y"] for e in E.values() if e[proto] is not None and e["y"] in (0, 1)])
                res[proto] = dict(triangles=ntri, cyclic=ncyc / ntri if ntri else None, edge_acc=float(agree))
            trans_out.append(dict(judge=m, **res))

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(dict(actions=rows_out, consistency=cons_out, trustjudge_tri=tj_out, transitivity=trans_out),
          open(a.out + ".json", "w"), indent=1)
f = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}" if isinstance(v, float) else str(v)
L = ["## Net correction of each follow-up action (strict pairs). low = 30% least-confident P; conflict/hesitant = "
     "median split of V/H(p_bar) inside low", "",
     "| judge | ds | action | n | net all | net low | net conflict | net hesitant | diff [95% CI] | tieAUROC P(T) | P(T) ties | P(T) low-conf err |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows_out:
    ci = r.get("diff_ci"); d = f"{f(r.get('diff'))} [{f(ci[0])}, {f(ci[1])}]" if ci else "-"
    L.append(f"| {r['judge']} | {r['ds']} | {r['action']} | {r['all'].get('n')} | {f(r['all'].get('net'))} | "
             f"{f(r['low'].get('net'))} | {f(r['conflict'].get('net'))} | {f(r['hesitant'].get('net'))} | {d} | "
             f"{f(r.get('tieAUROC_pT'))} | {f(r.get('pT_ties'))} | {f(r.get('pT_lowconf_err'))} |")
L += ["", "## Score-comparison inconsistency (TrustJudge) and readout", "",
      "| judge | ds | n | CR exp | CR argmax | equal-argmax | acc P | acc S exp | acc S argmax | U low | V low | conflict-dominated share (low) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
L += [f"| {r['judge']} | {r['ds']} | {r['n']} | {f(r['CR_exp'])} | {f(r['CR_arg'])} | {f(r['tie_arg'])} | {f(r['acc_P'])} | "
      f"{f(r['acc_exp'])} | {f(r['acc_arg'])} | {f(r['U_lowconf'])} | {f(r['V_lowconf'])} | {f(r['conflict_share_low'])} |" for r in cons_out]
L += ["", "## TrustJudge bidirectional aggregation on tri (argmax incl. T)", "",
      "| judge | ds | T rate | acc strict | tie recall | tie precision |", "|---|---|---|---|---|---|"]
L += [f"| {r['judge']} | {r['ds']} | {f(r['T_rate'])} | {f(r['acc_strict'])} | {f(r.get('tie_recall'))} | {f(r.get('tie_precision'))} |" for r in tj_out]
L += ["", "## Transitivity on ImageRewardDB same-prompt triangles", "",
      "| judge | protocol | triangles | cyclic rate | edge acc |", "|---|---|---|---|---|"]
for r in trans_out:
    for k, v in r.items():
        if k != "judge":
            L.append(f"| {r['judge']} | {k} | {v['triangles']} | {f(v['cyclic'])} | {f(v['edge_acc'])} |")
open(a.out + ".md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
