"""Typed human ties across benchmarks and visual tasks: what does 'same' mean, and which judge signal can see it?
(descriptive; no certification.  Branch 'tie semantics by visual task', 2026-11-07)

For every benchmark whose human data say WHY a pair is a tie, the ties are split into types (from the human data only,
never from a judge) and each type is contrasted with the DECIDED pairs of the same benchmark / subset by four signals:
  P(T)    probability of 'T' under the A/B/T prompt, averaged over the two orders (the tie signal of the graded claims)
  close   forced-choice closeness -|p - 1/2|, p the order-averaged A/B probability (what tie-dropping protocols threshold)
  level   pointwise level from the 1-9 score prompt: -(higher expected score) for a 'bad' type, (lower expected score)
          for a 'good' type (not defined for the other types)
  pwgap   pointwise closeness -|E_A - E_B|
Reported: AUROC(type vs decided) of each signal (.5 = the signal cannot tell the type from a decided pair), with a 95%
bootstrap interval over sources (source = question / prompt + context images, as in the certification).
Tie types:
  vote   GenAI-Arena, VisionArena (arena votes): 'both bad' (bothbad_vote), 'equally good' (tievote); decided = left /
         right votes.  VisionArena is also reported per question category.
  level  rating benchmarks (GenAI-Bench, EvalMuse, ImagenHub: tie = equal mean rating; ImageRewardDB: tie = equal rank):
         'both high' / 'both low' when both candidates' mean ratings are in the top / bottom quarter of the rating scale,
         else 'middle'
  agree  rating benchmarks with several raters: 'unanimous' (every rating of both candidates identical) / 'mixed'
         (equal means from differing ratings); MT-Bench (expert votes per pair): 'unanimous' (>= 2 votes, all tie) /
         'mixed' (some non-tie votes) / 'single' (one vote)
Benchmarks without tie types (MLLM-as-a-Judge, Pick-a-Pic, Arena 55k) are listed with all ties as one type, for reference.
Judges: those of the certification runs; a (benchmark, judge) cell needs both prompts in both orders on >= 95% of the
items (as in the certification), and pointwise scores on >= 95% for the pointwise columns.
usage (repo root): python scripts/tie_types_all.py --out <dir> [--boot 200] [--workers 12]"""
import argparse, ast, os, sys
from multiprocessing import Pool
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eaglevlm.pipeline.run import load_cfg                      # noqa: E402
from eaglevlm.pipeline.io import Store, dataset_items           # noqa: E402
from eaglevlm.pipeline.certify import signals, source_key       # noqa: E402
from eaglevlm.pipeline.router import pointwise                  # noqa: E402

TASK = {"genai_arena/edition": "edit", "imagenhub/edit": "edit",
        "genai_arena/generation": "t2i", "genaibench": "t2i", "evalmuse": "t2i", "imagenhub/t2i": "t2i",
        "imagerewarddb": "t2i", "pickapic_u": "t2i",
        "visionarena": "understanding", "mllm_judge": "understanding", "mllm_judge_sp": "understanding",
        "mtbench_human": "text", "arena55k": "text"}
SCALE = {"genaibench": (1, 5), "evalmuse": (1, 5), "imagenhub": (0, 1), "imagerewarddb": (1, 7)}
GOOD, BAD = {"equally good", "both high"}, {"both bad", "both low"}
JUDGES = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b", "internvl35_38b", "r1reward", "llavacritic7b"]
CFGS = ["configs/exp/certify_source.yaml", "configs/exp/newdata_certify_source.yaml"]


def lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def types_of(spec, x):
    """{family: type} of a TIE item (human data only)"""
    ds = spec.split("/")[0]; m = x.get("meta", {}) or {}
    if "vote_type" in m:
        return {"vote": {"bothbad_vote": "both bad", "tievote": "equally good"}.get(m["vote_type"], "other")}
    if ds == "mtbench_human":
        v = lit(m["votes"]); n = int(m["n_votes"])
        return {"agree": "single" if n == 1 else "unanimous" if v.get("tie", 0) == n else "mixed"}
    if ds in SCALE:
        lo_s, hi_s = SCALE[ds]; lo, hi = lo_s + .25 * (hi_s - lo_s), lo_s + .75 * (hi_s - lo_s)
        if ds == "imagerewarddb":
            ci = lit(m["cand_info"]); sc = [float(c["overall"]) for c in ci]; R = None
        else:
            sc = [float(s) for s in lit(m["scores"])]; R = lit(m.get("ratings"))
        out = {"level": "both high" if min(sc) >= hi else "both low" if max(sc) <= lo else "middle"}
        if R is not None and len(R[0]) > 1:
            out["agree"] = "unanimous" if len({float(r) for rr in R for r in rr}) == 1 else "mixed"
        return out
    return {"all ties": "tie"}


def auroc(s, pos):
    n1 = int(pos.sum()); n0 = len(s) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(s, pos, grp, B, seed=0):
    """95% percentile interval of the AUROC, resampling sources with replacement"""
    ug, gi = np.unique(grp, return_inverse=True); members = np.split(np.argsort(gi, kind="stable"), np.cumsum(np.bincount(gi))[:-1])
    rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(ug), len(ug))])
        vals.append(auroc(s[idx], pos[idx]))
    vals = np.array(vals); vals = vals[~np.isnan(vals)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) >= .9 * B else (np.nan, np.nan)


def cell(args):
    spec, j, B, P = args
    st = Store(P["items"], P["outputs"], P.get("items_override"), P.get("snapshot"))
    ds, I = dataset_items(st, spec)
    S = signals(st, ds, j, I); ids = sorted(S)
    if len(ids) < 200 or len(ids) < .95 * len(I):
        return [], f"skip {spec} {j}: {len(ids)} / {len(I)} items with both prompts in both orders"
    f1 = np.array([S[i][0] for i in ids]); f2 = np.array([S[i][1] for i in ids])
    Q = (np.array([S[i][2] for i in ids]) + np.array([S[i][3] for i in ids])) / 2; p = (f1 + f2) / 2
    sig = {"P(T)": Q[:, 2], "close": -np.abs(p - .5)}
    M = pointwise(st, ds, j, ids)
    if M is not None and not np.isnan(M).any(1).mean() > .05:
        EA, EB = M[:, 0], M[:, 3]
        sig["pwgap"] = -np.abs(EA - EB); sig["_lowmax"] = -np.fmax(EA, EB); sig["_highmin"] = np.fmin(EA, EB)
    y = np.array([str(I[i]["label"]) for i in ids]); tie = y == "tie"
    grp = np.array([source_key(I[i]) for i in ids])
    subs = [(None, np.ones(len(ids), bool))]
    if spec == "visionarena":
        cats = np.array([I[i]["subset"] for i in ids])
        subs += [(c, cats == c) for c in sorted(set(cats))]
    T = [types_of(spec, I[i]) if t else {} for i, t in zip(ids, tie)]
    fams = sorted({f for t in T for f in t})
    rows = []
    for sub, msk in subs:
        dec = msk & ~tie
        for fam in fams:
            ty = np.array([t.get(fam, "") for t in T])
            for tname in sorted(set(ty[msk & tie]) - {""}):
                pos_m = msk & tie & (ty == tname); use = pos_m | dec
                d = dict(dataset=spec, subset=sub or "all", task=TASK.get(spec, "?"), judge=j, family=fam, type=tname,
                         n_type=int(pos_m.sum()), n_decided=int(dec.sum()), n_pairs=int(msk.sum()),
                         share_of_ties=float(pos_m.sum() / max((msk & tie).sum(), 1)), share_of_pairs=float(pos_m.sum() / msk.sum()))
                if pos_m.sum() < 20:
                    rows.append(d); continue
                for name in ("P(T)", "close", "pwgap", "level"):
                    if name == "level":
                        key = "_lowmax" if tname in BAD else "_highmin" if tname in GOOD else None
                        if key is None or key not in sig:
                            continue
                        s = sig[key]
                    elif name in sig:
                        s = sig[name]
                    else:
                        continue
                    ok = use & ~np.isnan(s)
                    d[f"auc_{name}"] = auroc(s[ok], pos_m[ok])
                    if name in ("P(T)", "close", "level") and B:
                        d[f"lo_{name}"], d[f"hi_{name}"] = boot_ci(s[ok], pos_m[ok], grp[ok], B)
                rows.append(d)
    return rows, f"done {spec} {j}: {len(ids)} pairs, pointwise {'yes' if 'pwgap' in sig else 'no'}"


def fmt(v, lo=None, hi=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.2f}" if lo is None or np.isnan(lo) else f"{v:.2f} [{lo:.2f}, {hi:.2f}]"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--only", default="")
    a = ap.parse_args()
    specs, paths = [], None
    for c in CFGS:
        cfg = load_cfg(c); specs += [s for s in cfg["datasets"] if s not in specs]
        assert paths is None or paths == cfg["paths"], "the two configs read different data paths"
        paths = cfg["paths"]
    if a.only:
        specs = [s for s in specs if s in a.only.split(",")]
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        res = pool.map(cell, [(s, j, a.boot, paths) for s in specs for j in JUDGES], chunksize=1)
    rows = [r for rr, _ in res for r in rr]; logs = [m for _, m in res]
    import csv
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{a.out}/tie_types_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    # summary: per benchmark / subset / type, the mean AUROC over judges (min - max over judges)
    L = ["# Typed human ties by visual task", "", __doc__.split("Judges:")[0].strip(), "",
         "Mean AUROC over judges (min-max over judges); n = pairs of the type; share = of the benchmark's ties.", "",
         "| task | benchmark | subset | family | type | n | share | judges | P(T) | close | level | pwgap |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    import collections
    G = collections.OrderedDict()
    order = {"edit": 0, "t2i": 1, "understanding": 2, "text": 3}
    for r in sorted(rows, key=lambda r: (order.get(r["task"], 9), r["dataset"], r["subset"] != "all", r["subset"], r["family"], r["type"])):
        G.setdefault((r["task"], r["dataset"], r["subset"], r["family"], r["type"]), []).append(r)
    for (task, ds, sub, fam, tn), rr in G.items():
        def agg(k):
            v = np.array([r.get(k, np.nan) for r in rr], float); v = v[~np.isnan(v)]
            return "-" if not len(v) else f"{v.mean():.2f}" + (f" ({v.min():.2f}-{v.max():.2f})" if len(v) > 1 else "")
        L.append(f"| {task} | {ds} | {sub} | {fam} | {tn} | {rr[0]['n_type']} | {rr[0]['share_of_ties']:.2f} | {len(rr)} | "
                 f"{agg('auc_P(T)')} | {agg('auc_close')} | {agg('auc_level')} | {agg('auc_pwgap')} |")
    L += ["", "## Per judge (95% source-bootstrap intervals)", "",
          "| benchmark | subset | judge | family | type | n | P(T) | close | level | pwgap |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (order.get(r["task"], 9), r["dataset"], r["subset"] != "all", r["subset"], r["family"], r["type"], r["judge"])):
        L.append(f"| {r['dataset']} | {r['subset']} | {r['judge']} | {r['family']} | {r['type']} | {r['n_type']} | "
                 f"{fmt(r.get('auc_P(T)'), r.get('lo_P(T)'), r.get('hi_P(T)'))} | {fmt(r.get('auc_close'), r.get('lo_close'), r.get('hi_close'))} | "
                 f"{fmt(r.get('auc_level'), r.get('lo_level'), r.get('hi_level'))} | {fmt(r.get('auc_pwgap'))} |")
    L += ["", "## Log", ""] + [f"- {m}" for m in logs]
    open(f"{a.out}/tie_types_all.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:len(G) + 8]))
