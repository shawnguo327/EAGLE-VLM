"""Modality-controlled comparison: the same pairs and human labels, judged from the IMAGES vs from fixed text DESCRIPTIONS
of the images (caption_cands.py; captioner Qwen3-VL-8B, prompt-agnostic).  Per set x judge, on pairs present in both:
  acc        forced-choice accuracy on non-tie labels (order-averaged p_bar)
  inc        order inconsistency (verdict follows the slot)
  tieAU      AUROC of P(T) (A/B/T prompt) for human tie vs decided
  strict / graded coverage at alpha (tie_cert protocol: 25% fit / 25% cal / 50% test, SCOPE threshold)
A text LLM (Qwen3-8B) judging the descriptions is included: if it matches the VLM judging pixels, the visual comparison
step adds nothing that the description lacks.
usage (judgebench dir): python cap_compare.py --out paper_tables/cap_compare.md"""
import argparse, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from eaglev_stats import auroc
from label_noise_lib import per_judge, evaluate_cov

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/cap_compare.md")
ap.add_argument("--splits", type=int, default=100)
ap.add_argument("--alphas", default="0.1,0.2,0.3")
a = ap.parse_args()
AL = [float(x) for x in a.alphas.split(",")]
SETS = [("genai_arena", "generation"), ("imagerewarddb", None), ("pp", None)]
JUD = ["qwen3vl8b", "internvl35_8b", "qwen3_8b"]


def stats(P, ids, y):
    p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
    nt = y != 2
    r = dict(acc=((p[nt] > .5) == (y[nt] == 0)).mean(), inc=1 - fl.mean(),
             tieAU=auroc(Q[:, 2], (y == 2).astype(float)) if 0 < (y == 2).mean() < 1 else np.nan)
    F = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), fl]
    return r | evaluate_cov(F, y, AL, a.splits, 0)


L = ["# Images vs descriptions of the same images (modality-controlled)", "",
     "Same pairs, same human labels.  'image' = the judge sees the two images; 'desc' = it sees two fixed text descriptions "
     "(Qwen3-VL-8B captioner, prompt-agnostic).  Coverage: tie_cert protocol.", "",
     "| set | judge | n | tie | input | acc | inc | tieAU | " + " | ".join(f"strict @{x} | graded @{x}" for x in AL) + " |",
     "|" + "---|" * (8 + 2 * len(AL))]
for ds, sub in SETS:
    I = items(ds); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    Ic = items(ds + "_cap", rest=False)
    for j in JUD:
        Pc = per_judge(ds + "_cap", j, Ic, lambda i: i.split("#")[0])
        Pi = per_judge(ds, j, I) if j != "qwen3_8b" else {}
        ref = Pi if Pi else per_judge(ds, "qwen3vl8b", I)        # LLM rows: the same pairs as the VLM image rows
        ids = sorted(i for i in Pc if i in I and i in ref)
        if len(ids) < 200:
            continue
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        for tag, P in (("image", Pi), ("desc", Pc)):
            if not P:
                continue
            r = stats(P, ids, y)
            L.append(f"| {ds}{'/' + sub if sub else ''} | {j} | {len(ids)} | {(y == 2).mean():.2f} | {tag} | {r['acc']:.3f} | "
                     f"{r['inc']:.2f} | {r['tieAU']:.3f} | " + " | ".join(f"{r[('strict', x)]:.2f} | {r[('graded', x)]:.2f}" for x in AL) + " |")
            print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
