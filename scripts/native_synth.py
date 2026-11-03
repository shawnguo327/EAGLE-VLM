"""Decisive test for 'judge fine-tuning erases the equivalence tie': on the controlled synthetic set, does each
fine-tuned judge, run under its NATIVE protocol (native_judge.py / mmrlhf_score.py), still give paraphrase pairs (TG)
a higher tie/uncertainty score than decided pairs (DV, DT)?  Compared with the same judge and its base under our
A/B/T prompt (P(T), U).  Tie score per native protocol (same as baseline_table.py):
  r1reward       1 - top vote share over greedy + samples, both orders (ties not allowed by its prompt)
  llavacritic7b  share of its own tie verdicts (+ small vote-split tiebreak)
  mmrlhfrm7b     -|reward(cand0) - reward(cand1)|
usage (from judgebench dir): python native_synth.py --native .../runs/N_native_synth --out paper_tables/native_synth.md"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, signals, boot_auc
from eaglev_stats import auroc

ap = argparse.ArgumentParser()
ap.add_argument("--native", default="/media/shawn/C/papers/VLM/judgebench_local/runs/N_native_synth")
ap.add_argument("--out", default="paper_tables/native_synth.md")
a = ap.parse_args()
I = items("synth_tie")


def native(m):
    out = {}
    if m == "mmrlhfrm7b":
        p = os.path.join(a.native, "mmrlhf_score.jsonl")
        if not os.path.exists(p):
            return out
        Rw = {}
        for l in open(p):
            x = json.loads(l); Rw.setdefault(x["id"], {})[x["cand"]] = x["reward"]
        return {i: (d[0] > d[1], -abs(d[0] - d[1])) for i, d in Rw.items() if len(d) == 2}
    p = os.path.join(a.native, f"{m}.jsonl")
    if not os.path.exists(p):
        return out
    V = {}
    for l in open(p):
        x = json.loads(l); flip = x["order"] == "BA"
        for k, z in enumerate([x["parsed"]["greedy"]] + x["parsed"]["samples"]):
            if z is None:
                continue
            V.setdefault(x["id"], []).append(("T" if z == "T" else ("0" if (z == "1") != flip else "1"), k == 0))
    for i, vs in V.items():
        n = len(vs); n0 = sum(c == "0" for c, _ in vs); n1 = sum(c == "1" for c, _ in vs); nT = n - n0 - n1
        g = [c for c, gr in vs if gr and c != "T"]
        v = n0 > n1 if n0 != n1 else (g[0] == "0" if g else True)
        top = max(n0, n1, nT) / n
        out[i] = (v, (nT / n + 1e-3 * (1 - top)) if m == "llavacritic7b" else 1 - top)
    return out


def cmp(score, lab_of, ids):
    """AUROC (95% CI) of score for TG vs decided (DV+DT) and TB vs decided"""
    res = []
    for pos in ("TG", "TB"):
        z = [(score[i], I[i]["subset"] == pos) for i in ids if I[i]["subset"] in (pos, "DV", "DT")]
        s = np.array([x[0] for x in z], float); y = np.array([x[1] for x in z], int)
        lo, hi = boot_auc(s, y) if "boot_auc" in globals() else (np.nan, np.nan)
        res.append(f"{auroc(s, y):.3f} [{lo:.2f}, {hi:.2f}]")
    return res


L = ["# Native protocol on the controlled tie set: is the equivalence tie gone, or only unused under our prompt?", "",
     "AUROC of the tie / uncertainty score, positives = tie pairs, negatives = decided pairs (DV + DT); 95% bootstrap CI.", "",
     "| judge | protocol | n | TG (paraphrase) vs decided | TB (both wrong) vs decided | acc DV | acc DT |", "|---|---|---|---|---|---|---|"]
for base, m in (("qwen25vl7b", "r1reward"), ("llavaov7b", "llavacritic7b"), ("llavaov7b", "mmrlhfrm7b")):
    for j, tag in ((base, "base, A/B/T P(T)"), (base, "base, forced-choice U"), (m, "fine-tuned, A/B/T P(T)"), (m, "fine-tuned, forced-choice U")):
        R = load("synth_tie", j)
        S = {i: signals(r) for i, r in R.items() if i in I}
        S = {i: s for i, s in S.items() if s}
        k = 0 if "P(T)" in tag else 1
        ids = list(S)
        accs = []
        for sub in ("DV", "DT"):
            z = [(S[i][2] > .5) == (I[i]["label"] == 0) for i in ids if I[i]["subset"] == sub]
            accs.append(f"{np.mean(z):.3f}" if z else "--")
        L.append(f"| {j} | {tag} | {len(ids)} | " + " | ".join(cmp({i: S[i][k] for i in ids}, None, ids)) + " | " + " | ".join(accs) + " |")
    N = native(m)
    ids = [i for i in N if i in I]
    if len(ids) < 100:
        L.append(f"| {m} | **native** | {len(ids)} | (not finished) | | | |"); continue
    accs = []
    for sub in ("DV", "DT"):
        z = [N[i][0] == (I[i]["label"] == 0) for i in ids if I[i]["subset"] == sub]
        accs.append(f"{np.mean(z):.3f}" if z else "--")
    L.append(f"| {m} | **native** | {len(ids)} | " + " | ".join(cmp({i: N[i][1] for i in ids}, None, ids)) + " | " + " | ".join(accs) + " |")
open(a.out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
