"""Tie behaviour of general instruction-tuned VLMs vs judge-trained models built on them (same A/B/T prompt, order-
averaged): LLaVA-OneVision-7B -> LLaVA-Critic-7B, Qwen2.5-VL-7B -> R1-Reward.  Per dataset: share of pairs with T as the
argmax, mean P(T), AUROC of P(T) for human ties (on the pairs every compared model has).
usage (judgebench dir): python tie_basevsjudge.py --out paper_tables/tie_basevsjudge.md"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eaglevlm"))
from tie_mech import items, load
from label_noise_lib import eval_items
from eaglev_stats import auroc
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper_tables/tie_basevsjudge.md")
ap.add_argument("--sets", default="mllm_judge,genai_arena/generation,genai_arena/edition,pickapic_u,imagerewarddb")
ap.add_argument("--models", default="llavaov7b,llavacritic7b,qwen25vl7b,r1reward"); a = ap.parse_args()
M = a.models.split(",")
L = ["# Tie behaviour: general instruction-tuned VLMs vs judge-trained models built on them", "",
     "Cell = share of pairs with T as argmax / mean P(T) / AUROC of P(T) for human ties.  LLaVA-OV-7B -> LLaVA-Critic-7B; Qwen2.5-VL-7B -> R1-Reward.", "",
     "| set | n | human tie | " + " | ".join(M) + " |", "|" + "---|" * (3 + len(M))]
for spec in a.sets.split(","):
    ds, sub = (spec.split("/") + [None])[:2]
    I = eval_items(ds, items(ds)); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    R = {m: load(ds, m, conds={"tri"}) for m in M}
    common = [i for i in sorted(I) if all(("tri", "AB") in R[m].get(i, {}) and ("tri", "BA") in R[m].get(i, {}) for m in M)]
    if len(common) < .9 * len(I):
        continue
    y = np.array([str(I[i]["label"]) == "tie" for i in common]); cells = []
    for m in M:
        Q = np.array([[(R[m][i][("tri", "AB")]["A"] + R[m][i][("tri", "BA")]["B"]) / 2, (R[m][i][("tri", "AB")]["B"] + R[m][i][("tri", "BA")]["A"]) / 2,
                       (R[m][i][("tri", "AB")]["T"] + R[m][i][("tri", "BA")]["T"]) / 2] for i in common])
        cells.append(f"{(Q.argmax(1) == 2).mean():.3f} / {Q[:, 2].mean():.3f} / {auroc(Q[:, 2], y):.3f}")
    L.append(f"| {spec} | {len(common)} | {y.mean():.2f} | " + " | ".join(cells) + " |"); print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
