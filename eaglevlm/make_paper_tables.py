"""Turn results/<ds>__<model>.jsonl into the paper's LaTeX tables (diagnostic + selective/ladder).
usage: python make_paper_tables.py results/ ../eaglev-paper/tables/   (run from judgebench/)
"""
import glob, json, os, sys
import numpy as np
from eaglev_stats import load_records, conf_scores, auroc, selective_eval, ladder_eval

NAMES = {"vlrb": "VL-RewardBench", "mmrb": "Multimodal RB", "mmrlhf": "MM-RLHF-RB", "mllm_judge": "MLLM-as-a-Judge",
         "genai_arena": "GenAI-Arena", "mjbench": "MJ-Bench", "mmrb2": "MMRB2", "imagerewarddb": "ImageRewardDB",
         "qbench2": "Q-Bench2", "editreward": "EditReward-Bench", "pp": "Place Pulse 2.0"}
ALPHAS = (0.10, 0.15, 0.20, 0.25, 0.30)


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}".lstrip("0").replace("-0.", "-.")


def diag_row(recs):
    B = [r for r in recs if r["y"] is not None]
    lo = np.array([r["lo"] for r in B]); y = np.array([r["y"] for r in B])
    corr = ((lo > 0).astype(int) == y).astype(int)
    S = conf_scores(B); dp = S["dp"]
    q = np.quantile(dp, [.2, .8])
    out = dict(n=len(B), acc=corr.mean(), swap=np.mean([r["cons"] for r in B]),
               q1=corr[dp <= q[0]].mean(), q5=corr[dp >= q[1]].mean(),
               au_dp=auroc(dp, corr), au_lo=auroc(S["abs_logodds"], corr), sat=(dp >= .95).mean())
    conf = (1 + dp) / 2; bins = np.clip((conf * 10).astype(int), 0, 9)
    out["ece"] = sum(abs(corr[bins == k].mean() - conf[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any())
    if "dp_x_pw_agree" in S:
        out["au_comb"] = auroc(S["dp_x_pw_agree"], corr)
    ties = [r for r in recs if r["tie"]]
    if ties:
        alld = np.array([abs(2 * r["pa"] - 1) for r in recs]); ist = np.array([r["tie"] for r in recs]).astype(int)
        out["tie_auroc"] = auroc(-alld, ist)
    return out, S, 1 - corr


def main(rd, od):
    os.makedirs(od, exist_ok=True)
    allres = {}
    for f in sorted(glob.glob(os.path.join(rd, "*.jsonl"))):
        ds, model = os.path.basename(f)[:-6].split("__")
        recs = load_records(f"items/{ds}.jsonl", f)
        if not recs:
            continue
        d, S, err = diag_row(recs)
        sel = {}
        for a in ALPHAS:
            sel[a] = {"dp": selective_eval(S["dp"], err, a), "lo": selective_eval(S["abs_logodds"], err, a)}
            if "dp_x_pw_agree" in S:
                sel[a]["ladder"] = ladder_eval(recs, a)
        allres[f"{ds}__{model}"] = dict(diag=d, selective={str(k): v for k, v in sel.items()})
    json.dump(allres, open(os.path.join(od, "paper_results.json"), "w"), indent=1, default=float)
    for model in sorted({k.split("__")[1] for k in allres}):
        L = [r"\begin{tabular}{lrcccccccc}", r"\toprule",
             r"Dataset & $n$ & Acc & Swap & Q1 & Q5 & AUROC $\Delta p$ & AUROC $\lambda$ & Sat. & ECE\\", r"\midrule"]
        for k, v in allres.items():
            ds, m = k.split("__")
            if m != model:
                continue
            d = v["diag"]
            L.append(f"{NAMES.get(ds, ds)} & {d['n']} & {f3(d['acc'])} & {f3(d['swap'])} & {f3(d['q1'])} & {f3(d['q5'])} & "
                     f"{f3(d['au_dp'])} & {f3(d['au_lo'])} & {f3(d['sat'])} & {f3(d['ece'])}\\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(od, f"diag_{model}.tex"), "w").write("\n".join(L) + "\n")
        L = [r"\begin{tabular}{lc" + "ccc" + "}", r"\toprule", r"Dataset & $\alpha$ & SCOPE ($\Delta p$) & $\lambda$ & \textsc{Eagle-V} ladder\\", r"\midrule"]
        for k, v in allres.items():
            ds, m = k.split("__")
            if m != model:
                continue
            for a in ALPHAS:
                s = v["selective"][str(a)]
                lad = s.get("ladder")
                L.append(f"{NAMES.get(ds, ds)} & {a:.2f} & {100*s['dp']['coverage']:.1f} & {100*s['lo']['coverage']:.1f} & "
                         f"{(100*lad['auto']) if lad else float('nan'):.1f}\\\\")
            L.append(r"\midrule")
        L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
        open(os.path.join(od, f"selective_{model}.tex"), "w").write("\n".join(L) + "\n")
    print("wrote", od)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results", sys.argv[2] if len(sys.argv) > 2 else "paper_tables")
