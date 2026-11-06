"""Conventional (non-certification) metrics of the pairwise judges, with and without human ties, so the paper can report the
numbers the judge literature uses next to the selective-prediction metrics (SCOPE: coverage + empirical FDR; Trust-or-
Escalate: coverage + empirical human agreement + guarantee success rate).
Per set x judge (order-averaged forced choice p = P(cand 0 better), tie prompt Q = order-averaged A/B/T probs):
  acc (ties dropped)      accuracy on decided pairs - the number benchmarks usually report
  acc (ties = wrong)      accuracy on all pairs, a human tie counts as an error for any forced choice
  acc (half credit)       tie pairs score .5
  acc 3-way               argmax of Q against {0, 1, tie}
  T share / tie F1        how often the judge says T, F1 of 'T' against human ties
  err AUROC (decided)     binary entropy of p ranks errors on decided pairs (SCOPE's uncertainty-quality metric)
  AURC decided / all      area under the selective risk-coverage curve (confidence = |p - .5|), decided pairs only vs all
                          pairs with ties = errors (lower is better; the published setting vs deployment with ties)
  sel. acc @50% dec / all accuracy of the 50% most confident pairs, decided only vs all pairs
  ECE decided / all       expected calibration error of the forced-choice confidence max(p, 1-p) (SCOPE / ToE Table 1)
  err AUPRC / AUROC all   BPE as an error detector; 'all' counts a human tie as an error of any forced choice
usage (judgebench dir): python conv_metrics.py --out paper_tables/conv_metrics.md"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eaglevlm"))
from tie_mech import items, load, H
from label_noise_lib import eval_items
from eaglev_stats import auroc

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/conv_metrics.md")
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen35_9b,qwen25vl7b,qwen3vl32b,internvl35_38b")
ap.add_argument("--sets", default="mllm_judge,mllm_judge_sp,genaibench,mtbench_human,arena55k,genai_arena/generation,"
                                  "genai_arena/edition,pickapic_u,imagerewarddb")
a = ap.parse_args()


def sig(ds, j, I):
    R = load(ds, j, conds={"bin", "tri"}); out = {}
    for i, r in R.items():
        if i not in I or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
        p = (b1["A"] / (b1["A"] + b1["B"]) + b2["B"] / (b2["A"] + b2["B"])) / 2
        q = np.array([(t1["A"] + t2["B"]) / 2, (t1["B"] + t2["A"]) / 2, (t1["T"] + t2["T"]) / 2]); out[i] = (p, q / q.sum())
    return out


def aurc(conf, err):
    o = np.argsort(-conf, kind="stable"); e = err[o]; return float(np.mean(np.cumsum(e) / np.arange(1, len(e) + 1)))


def ece(conf, correct, bins=10):
    """expected calibration error of the forced-choice confidence max(p, 1-p) (10 equal-width bins on [.5, 1])"""
    e = np.linspace(.5, 1, bins + 1); out = 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        m = (conf >= lo) & ((conf < hi) | (hi == 1))
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return out


def auprc(score, pos):
    """average precision of `score` for detecting the positive class (errors)"""
    o = np.argsort(-score, kind="stable"); y = pos[o].astype(float)
    if y.sum() == 0:
        return np.nan
    prec = np.cumsum(y) / np.arange(1, len(y) + 1); return float((prec * y).sum() / y.sum())


def selacc(conf, err, cov=.5):
    o = np.argsort(-conf, kind="stable")[: max(1, int(cov * len(conf)))]; return 1 - err[o].mean()


f = lambda v: "-" if v != v else f"{v:.3f}"
L = ["# Conventional judge metrics with and without human ties", "",
     "Order-averaged forced choice / tie prompt.  AURC: lower is better.  sel.acc@50%: accuracy of the most confident half.", "",
     "| set | judge | n | human tie | acc (ties dropped) | acc (ties = wrong) | acc (half credit) | acc 3-way | T share | tie F1 | "
     "err AUROC (decided) | AURC decided | AURC all | sel.acc@50% decided | sel.acc@50% all | ECE decided | ECE all | err AUPRC decided | "
     "err AUROC all | err AUPRC all |", "|" + "---|" * 20]
for spec in a.sets.split(","):
    ds, sub = (spec.split("/") + [None])[:2]
    I = eval_items(ds, items(ds, rest=ds not in ("mllm_judge_sp", "genaibench")))
    I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    for j in a.judges.split(","):
        S = sig(ds, j, I); ids = sorted(S)
        if len(ids) < 200 or len(ids) < .95 * len(I):
            continue
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids]); t = y == 2; d = ~t
        p = np.array([S[i][0] for i in ids]); Q = np.array([S[i][1] for i in ids])
        fc = np.where(p > .5, 0, 1); err_d = (fc[d] != y[d]).astype(float); err_all = np.where(t, 1.0, (fc != y).astype(float))
        conf = np.abs(p - .5); am = Q.argmax(1)
        tp = ((am == 2) & t).sum(); prec = tp / max((am == 2).sum(), 1); rec = tp / max(t.sum(), 1)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        L.append(f"| {spec} | {j} | {len(ids)} | {t.mean():.2f} | {f(1 - err_d.mean())} | {f(1 - err_all.mean())} | "
                 f"{f(np.where(t, .5, 1 - err_all).mean())} | {f((am == y).mean())} | {(am == 2).mean():.3f} | {f(f1)} | "
                 f"{f(auroc(H(p[d]), err_d) if 0 < err_d.mean() < 1 else np.nan)} | {f(aurc(conf[d], err_d))} | {f(aurc(conf, err_all))} | "
                 f"{f(selacc(conf[d], err_d))} | {f(selacc(conf, err_all))} | {f(ece(.5 + conf[d], 1 - err_d))} | {f(ece(.5 + conf, 1 - err_all))} | "
                 f"{f(auprc(H(p[d]), err_d))} | {f(auroc(H(p), err_all))} | {f(auprc(H(p), err_all))} |")
        print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
