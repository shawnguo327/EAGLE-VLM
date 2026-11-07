"""Quick probes for the idea branches (2026-11-07; descriptive, from saved judge outputs only, no new inference).

  b1 tie aversion      per benchmark x judge: human tie rate, mean P(T), share of (pair, order) whose greedy first token
                       under the A/B/T prompt is T ("verbal" tie rate: what the judge would say), recall of T on human ties;
                       includes general instruction models next to judge-trained models built on them
                       (Qwen2.5-VL-7B -> R1-Reward, LLaVA-OV-7B -> LLaVA-Critic-7B) where both were run
  b2 forced choice     on typed benchmarks, decided vs 'both good' vs 'both bad' pairs: order flips of the A/B verdict,
     without a tie     first-slot preference, forced-choice confidence |p - 1/2|; what a judge that cannot say 'tie' does
                       when there is no winner
  b3 rank vs score     benchmarks with per-candidate human ratings: within-source ranking accuracy of the pointwise expected
                       score vs cross-source comparability (two candidates from different sources), Spearman with ratings
  b4 transitivity      sources with >= 3 candidates and all three pairs judged: strict cycles and tie intransitivity
                       (x~y, y~z, x strictly vs z) for humans, the pairwise A/B/T judge, pointwise with a gap threshold, and
                       pointwise with interval overlap (central 50% interval of the 1-9 score distribution)
usage (repo root): python scripts/branch_probes.py b1|b2|b3|b4|all --out <dir> [--workers 16]"""
import argparse, ast, collections, itertools, os, sys
from multiprocessing import Pool
import numpy as np
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eaglevlm.pipeline.run import load_cfg                         # noqa: E402
from eaglevlm.pipeline.io import Store, dataset_items              # noqa: E402
from eaglevlm.pipeline.certify import signals, source_key          # noqa: E402

SPECS = ["genai_arena/edition", "imagenhub/edit", "genai_arena/generation", "genaibench", "evalmuse", "imagenhub/t2i",
         "imagerewarddb", "pickapic_u", "visionarena", "mllm_judge", "mllm_judge_sp", "mtbench_human", "arena55k"]
JUDGES = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b", "internvl35_38b", "r1reward", "llavacritic7b",
          "llavaov7b", "mmrlhfrm7b", "qwen3_8b", "qwen25_7b"]
SCALE = {"genaibench": (1, 5), "evalmuse": (1, 5), "imagenhub": (0, 1), "imagerewarddb": (1, 7)}
D = np.arange(1, 10)


def lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def human_scores(spec, x):
    ds = spec.split("/")[0]; m = x.get("meta", {}) or {}
    if ds == "imagerewarddb":
        return [float(c["overall"]) for c in lit(m["cand_info"])]
    if ds in SCALE:
        return [float(s) for s in lit(m["scores"])]
    return None


def tie_type(spec, x):
    if str(x["label"]) != "tie":
        return "decided"
    m = x.get("meta", {}) or {}
    if "vote_type" in m:
        return {"bothbad_vote": "both bad", "tievote": "both good"}.get(m["vote_type"], "tie")
    sc = human_scores(spec, x)
    if sc is None:
        return "tie"
    lo_s, hi_s = SCALE[spec.split("/")[0]]
    if min(sc) >= lo_s + .75 * (hi_s - lo_s):
        return "both good"
    if max(sc) <= lo_s + .25 * (hi_s - lo_s):
        return "both bad"
    return "tie (middle)"


def cand_keys(x):
    m = x.get("meta", {}) or {}
    if "models" in m:
        return [str(v) for v in lit(m["models"])]
    return [str(c) for c in x["cand"]]


def load_cell(spec, j, P, need_score=False):
    st = Store(P["items"], P["outputs"], P.get("items_override"), P.get("snapshot"))
    ds, I = dataset_items(st, spec)
    S = signals(st, ds, j, I); ids = sorted(S)
    if len(ids) < 200 or len(ids) < .95 * len(I):
        return None
    out = dict(ds=ds, I=I, ids=ids, S=S)
    if need_score:
        R = st.load(ds, j, conds={"score"}); dist = {}
        for i in ids:
            r = R.get(i, {})
            if ("score", "A") in r and ("score", "B") in r:
                dd = []
                for o in "AB":
                    q = np.array([r[("score", o)].get(str(d), 0.0) for d in D]); dd.append(q / q.sum() if q.sum() > 0 else None)
                if all(v is not None for v in dd):
                    dist[i] = dd
        out["dist"] = dist if len(dist) >= .95 * len(ids) else None
    return out


def b1(args):
    spec, j, P = args
    c = load_cell(spec, j, P)
    if c is None:
        return None
    I, ids, S = c["I"], c["ids"], c["S"]
    y = np.array([str(I[i]["label"]) for i in ids]); tie = y == "tie"
    q1 = np.array([S[i][2] for i in ids]); q2 = np.array([S[i][3] for i in ids]); Q = (q1 + q2) / 2
    verbal = ((q1.argmax(1) == 2).astype(float) + (q2.argmax(1) == 2)) / 2
    return dict(dataset=spec, judge=j, n=len(ids), human_tie=tie.mean(), mean_PT=Q[:, 2].mean(), verbal_T=verbal.mean(),
                verbal_T_on_ties=verbal[tie].mean() if tie.any() else np.nan, verbal_T_on_decided=verbal[~tie].mean(),
                PT_on_ties=Q[tie, 2].mean() if tie.any() else np.nan, PT_on_decided=Q[~tie, 2].mean())


def b2(args):
    spec, j, P = args
    c = load_cell(spec, j, P)
    if c is None:
        return None
    I, ids, S = c["I"], c["ids"], c["S"]
    f1 = np.array([S[i][0] for i in ids]); f2 = np.array([S[i][1] for i in ids]); p = (f1 + f2) / 2
    flip = ((f1 > .5) != (f2 > .5)).astype(float)
    first = (f1 + (1 - f2)) / 2                          # P(first slot) averaged over the two orders
    ty = np.array([tie_type(spec, I[i]) for i in ids]); rows = []
    for t in ("decided", "both good", "both bad", "tie (middle)", "tie"):
        m = ty == t
        if m.sum() < 20:
            continue
        rows.append(dict(dataset=spec, judge=j, type=t, n=int(m.sum()), order_flip=flip[m].mean(), first_slot=first[m].mean(),
                         confidence=np.abs(p[m] - .5).mean()))
    return rows


def b3(args):
    spec, j, P = args
    if spec.split("/")[0] not in SCALE:
        return None
    c = load_cell(spec, j, P, need_score=True)
    if c is None or c.get("dist") is None:
        return None
    I, ids, S, dist = c["I"], c["ids"], c["S"], c["dist"]
    ids = [i for i in ids if i in dist]
    E = np.array([[dist[i][0] @ D, dist[i][1] @ D] for i in ids]); H = np.array([human_scores(spec, I[i]) for i in ids])
    p = np.array([(S[i][0] + S[i][1]) / 2 for i in ids])
    dec = H[:, 0] != H[:, 1]
    win = (H[:, 0] > H[:, 1]).astype(int)                   # 1 = candidate 0 better (human)
    within_pw = np.mean((E[dec, 0] > E[dec, 1]) == win[dec].astype(bool))
    within_pair = np.mean((p[dec] > .5) == win[dec].astype(bool))
    # cross-source: candidate 0 of one pair against candidate 1 of another pair from a different source
    g = np.array([source_key(I[i]) for i in ids]); rng = np.random.default_rng(0); acc = []
    for _ in range(5):
        perm = rng.permutation(len(ids)); ok = (g != g[perm]) & (H[:, 0] != H[perm, 1])
        acc.append(np.mean((E[ok, 0] > E[perm][ok, 1]) == (H[ok, 0] > H[perm][ok, 1])))
    rho = spearmanr(np.r_[E[:, 0], E[:, 1]], np.r_[H[:, 0], H[:, 1]]).correlation
    return dict(dataset=spec, judge=j, n=len(ids), within_source_pointwise=within_pw, within_source_pairwise_prompt=within_pair,
                cross_source_pointwise=float(np.mean(acc)), spearman_all=rho)


def interval(q, lo=.25, hi=.75):
    c = np.cumsum(q); return D[min(int(np.searchsorted(c, lo)), 8)], D[min(int(np.searchsorted(c, hi)), 8)]


def b4(args):
    spec, j, P = args
    c = load_cell(spec, j, P, need_score=True)
    if c is None:
        return None
    I, ids, S, dist = c["I"], c["ids"], c["S"], c["dist"]
    by_src = collections.defaultdict(dict)
    for i in ids:
        k = cand_keys(I[i])
        if len(k) == 2 and k[0] != k[1]:
            by_src[source_key(I[i])][frozenset(k)] = i
    eps = None
    if dist is not None:                                  # gap threshold matching the human tie rate (descriptive)
        gaps = np.array([abs(dist[i][0] @ D - dist[i][1] @ D) for i in ids if i in dist])
        tr = np.mean([str(I[i]["label"]) == "tie" for i in ids]); eps = np.quantile(gaps, tr) if tr > 0 else 0.0

    def rel(i, a, b, how):
        """relation of candidate a vs b on pair i: +1 a better, -1 b better, 0 tie"""
        k = cand_keys(I[i]); s = 1 if k[0] == a else -1
        if how == "human":
            lab = str(I[i]["label"]); r = 0 if lab == "tie" else (1 if lab == "0" else -1)
        elif how == "pair3":
            Q = (np.asarray(S[i][2]) + np.asarray(S[i][3])) / 2; t = Q.argmax(); r = 0 if t == 2 else (1 if t == 0 else -1)
        elif how == "forced":
            p = (S[i][0] + S[i][1]) / 2; r = 1 if p > .5 else -1
        elif how == "pw_gap":
            if i not in dist:
                return None
            dE = dist[i][0] @ D - dist[i][1] @ D; r = 0 if abs(dE) <= eps else (1 if dE > 0 else -1)
        else:                                             # pw_interval: tie iff the central 50% intervals overlap
            if i not in dist:
                return None
            (l0, h0), (l1, h1) = interval(dist[i][0]), interval(dist[i][1])
            r = 0 if not (h0 < l1 or h1 < l0) else (1 if l0 > h1 else -1)
        return r * s
    cnt = collections.Counter(); hows = ["human", "pair3", "forced"] + (["pw_gap", "pw_interval"] if dist is not None else [])
    for src, P2 in by_src.items():
        cands = sorted({c_ for k in P2 for c_ in k})
        for a, b, cc in itertools.combinations(cands, 3):
            ks = [frozenset((a, b)), frozenset((b, cc)), frozenset((a, cc))]
            if not all(k in P2 for k in ks):
                continue
            for how in hows:
                r1, r2, r3 = rel(P2[ks[0]], a, b, how), rel(P2[ks[1]], b, cc, how), rel(P2[ks[2]], a, cc, how)
                if None in (r1, r2, r3):
                    continue
                cnt[(how, "triads")] += 1
                # strict cycle a>b>c>a or a<b<c<a  (r3 is a vs c)
                if (r1 == 1 and r2 == 1 and r3 == -1) or (r1 == -1 and r2 == -1 and r3 == 1):
                    cnt[(how, "cycle")] += 1
                # tie intransitivity: two ties and one strict relation (in any position)
                if sorted([r1 == 0, r2 == 0, r3 == 0]) == [False, True, True]:
                    cnt[(how, "tie_intrans")] += 1
                if (r1, r2, r3).count(0) >= 1:
                    cnt[(how, "with_tie")] += 1
    if not cnt:
        return None
    return [dict(dataset=spec, judge=j, relation=h, triads=cnt[(h, "triads")], cycle=cnt[(h, "cycle")] / max(cnt[(h, "triads")], 1),
                 tie_intrans=cnt[(h, "tie_intrans")] / max(cnt[(h, "triads")], 1), any_tie=cnt[(h, "with_tie")] / max(cnt[(h, "triads")], 1))
            for h in hows if cnt[(h, "triads")]]


def md_table(rows, cols, fmt=".3f"):
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(f"{r[c]:{fmt}}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what"); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    paths = load_cfg("configs/exp/certify_source.yaml")["paths"]
    os.makedirs(a.out, exist_ok=True)
    jobs = [(s, j, paths) for s in SPECS for j in JUDGES]
    probes = {"b1": b1, "b2": b2, "b3": b3, "b4": b4}
    for name in (["b1", "b2", "b3", "b4"] if a.what == "all" else a.what.split(",")):
        with Pool(a.workers) as pool:
            res = pool.map(probes[name], jobs, chunksize=1)
        rows = [r for x in res if x for r in (x if isinstance(x, list) else [x])]
        import csv
        with open(f"{a.out}/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        cols = list(rows[0]); L = [f"# {name}", "", [l for l in __doc__.split("\n") if l.strip().startswith(name)][0].strip(), ""]
        L += md_table(rows, cols)
        open(f"{a.out}/{name}.md", "w").write("\n".join(L) + "\n")
        print(f"{name}: {len(rows)} rows")
