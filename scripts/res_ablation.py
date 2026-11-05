"""Resolution ablation (768 px vs full resolution, long side <= 1536).
Only items with an image > 768 px were re-judged (<ds>_big); every other item is pixel-identical at both caps.
 (1) merge:   writes runs/HR_merged/raw/<ds>__<judge>_hr.jsonl = the 768-px outputs with the >768-px items replaced by their
              full-resolution outputs (cond names mapped back to bin / tri / score), so every downstream script (chain,
              router_v2, conv_metrics, c3_diag) can run on judge '<judge>_hr' unchanged.
 (2) paired:  on the re-judged items only, 768 vs full resolution: accuracy (ties dropped / ties = wrong), 3-way accuracy,
              tie AUROC of P(T), error AUROC of BPE, pointwise Spearman with the human score (score-pair set), share of
              flipped verdicts; paired bootstrap 95% CI of the accuracy difference; per source task.
usage (judgebench dir): python res_ablation.py --judges qwen3vl8b --out paper_tables/res_ablation.md"""
import argparse, collections, json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eaglevlm"))
from tie_mech import items, load, H
from eaglev_stats import auroc

ap = argparse.ArgumentParser()
ap.add_argument("--judges", default="qwen3vl8b")
ap.add_argument("--sets", default="mllm_judge,mllm_judge_sp,genai_arena,imagerewarddb,genaibench")
ap.add_argument("--out", default="paper_tables/res_ablation.md")
ap.add_argument("--merged", default="/media/shawn/C/papers/VLM/judgebench_local/runs/HR_merged/raw")
a = ap.parse_args()
os.makedirs(a.merged, exist_ok=True)
MAP = {"bin_hires": "bin", "tri_hires": "tri", "score_hires": "score"}
PAIRC = [(c, o) for c in ("bin", "tri") for o in ("AB", "BA")]


def sig(r, sfx=""):
    b1, b2, t1, t2 = (r[(c + sfx, o)] for c, o in PAIRC)
    p = (b1["A"] / (b1["A"] + b1["B"]) + b2["B"] / (b2["A"] + b2["B"])) / 2
    q = np.array([(t1["A"] + t2["B"]) / 2, (t1["B"] + t2["A"]) / 2, (t1["T"] + t2["T"]) / 2]); return p, q / q.sum()


def es(d):
    D = np.arange(1, 10); q = np.array([d.get(str(k), 0) for k in D]); return float(q @ D / q.sum()) if q.sum() else np.nan


def boot(x, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(x); m = [x[rng.integers(0, n, n)].mean() for _ in range(B)]
    return np.percentile(m, [2.5, 97.5])


f = lambda v: "-" if v != v else f"{v:.3f}"
L = ["# Resolution ablation: 768 px (default) vs full resolution (long side <= 1536 px)", "",
     "Paired, on the items that contain an image > 768 px (all other items are identical at both caps).", "",
     "| set | judge | n (re-judged / total) | tie | acc 768 -> full (ties dropped) [95% CI of diff] | acc ties = wrong | 3-way acc | "
     "tie AUROC P(T) | err AUROC BPE | verdict flips | pointwise rho with human 768 -> full |", "|" + "---|" * 11]
TASK = []
for ds in a.sets.split(","):
    I = items(ds, rest=ds not in ("mllm_judge_sp", "genaibench"))
    for j in a.judges.split(","):
        base = load(ds, j); hr = load(ds + "_big", j, conds=set(MAP))
        ids = [i for i in hr if i in base and i in I and all((c + "_hires", o) in hr[i] for c, o in PAIRC) and all(k in base[i] for k in PAIRC)]
        if not ids:
            continue
        # (1) merged file
        n_rep = 0
        with open(f"{a.merged}/{ds}__{j}_hr.jsonl", "w") as fo:
            for i, r in base.items():
                src = dict(r)
                if i in hr and all((c + "_hires", o) in hr[i] for c, o in PAIRC):
                    n_rep += 1
                    for (c, o) in list(src):
                        if c in ("bin", "tri", "score"):
                            del src[(c, o)]
                    for (c, o), v in hr[i].items():
                        src[(MAP[c], o)] = v
                for (c, o), v in src.items():
                    fo.write(json.dumps({"id": i, "cond": c, "order": o, "lp": {k: float(np.log(max(p, 1e-30))) for k, p in v.items()}}) + "\n")
        # (2) paired comparison on the re-judged items
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids]); t = y == 2; d = ~t
        S0 = [sig(base[i]) for i in ids]; S1 = [sig(hr[i], "_hires") for i in ids]
        p0 = np.array([s[0] for s in S0]); p1 = np.array([s[0] for s in S1]); Q0 = np.array([s[1] for s in S0]); Q1 = np.array([s[1] for s in S1])
        c0 = (np.where(p0 > .5, 0, 1) == y).astype(float); c1 = (np.where(p1 > .5, 0, 1) == y).astype(float)
        diff = c1[d] - c0[d]; lo, hi = boot(diff) if d.sum() > 20 else (np.nan, np.nan)
        flips = ((p0 > .5) != (p1 > .5)).mean()
        rho = "-"
        if ds in ("mllm_judge_sp", "genaibench"):
            xs0, xs1, hs = [], [], []
            for i in ids:
                hsc = I[i].get("meta", {}).get("human_scores") or I[i].get("meta", {}).get("scores")
                if not hsc:
                    continue
                for k, o in enumerate("AB"):
                    if ("score", o) in base[i] and ("score_hires", o) in hr[i]:
                        xs0.append(es(base[i][("score", o)])); xs1.append(es(hr[i][("score_hires", o)])); hs.append(float(hsc[k]))
            if hs:
                rho = f"{spearmanr(xs0, hs)[0]:.3f} -> {spearmanr(xs1, hs)[0]:.3f} (n={len(hs)})"
        L.append(f"| {ds} | {j} | {len(ids)} / {len(I)} (merged {n_rep}) | {t.mean():.2f} | {c0[d].mean():.3f} -> {c1[d].mean():.3f} "
                 f"[{f(lo)}, {f(hi)}] | {f(np.where(t, 0, c0).mean())} -> {f(np.where(t, 0, c1).mean())} | "
                 f"{f((Q0.argmax(1) == y).mean())} -> {f((Q1.argmax(1) == y).mean())} | "
                 f"{f(auroc(Q0[:, 2], t) if 0 < t.mean() < 1 else np.nan)} -> {f(auroc(Q1[:, 2], t) if 0 < t.mean() < 1 else np.nan)} | "
                 f"{f(auroc(H(p0[d]), 1 - c0[d]))} -> {f(auroc(H(p1[d]), 1 - c1[d]))} | {flips:.3f} | {rho} |")
        print(L[-1], flush=True)
        sub = np.array([I[i].get("subset", "-") for i in ids])
        for s in sorted(set(sub)):
            m = (sub == s) & d
            if m.sum() >= 30:
                TASK.append(f"| {ds} | {j} | {s} | {m.sum()} | {c0[m].mean():.3f} -> {c1[m].mean():.3f} | {c1[m].mean() - c0[m].mean():+.3f} |")
L += ["", "## Per source task (decided pairs, >= 30): accuracy 768 -> full", "", "| set | judge | task | n | acc | diff |", "|---|---|---|---|---|---|"] + TASK
open(a.out, "w").write("\n".join(L) + "\n")
