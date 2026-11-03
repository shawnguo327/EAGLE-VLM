"""Position bias of pairwise judges by candidate modality, and what it does to selective judging.
Conditions (A/B forced choice, both orders): bin (normal), bin_same (the same candidate in both slots: any preference is
pure position prior), bin_grid (img items: both candidates composed into one side-by-side image), bin_lowres (112 px).
Per dataset x judge (non-tie pairs for accuracy / AUROC / coverage):
  prior      mean P(first slot) under bin_same - .5   (negative = prefers the second / later candidate)
  bias       mean P(first slot) under the condition - .5 ; inc = verdict flips when the order is swapped
  acc, BPE error AUROC, SCOPE coverage at alpha = .2 (300 splits) per condition
  debiased   bin with the per-item prior removed in logit space: logit p_first - logit p_first(bin_same, same slot item)
usage (judgebench dir): python posbias.py --out paper_tables/posbias.md"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from eaglev_stats import auroc

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper_tables/posbias.md")
ap.add_argument("--splits", type=int, default=300)
a = ap.parse_args()
SETS = {"text": ["mtbench_human", "arena55k"], "resp": ["vlrb", "mmrb", "mllm_judge"],
        "img": ["genai_arena", "mjbench", "imagerewarddb", "mmrb2", "qbench2", "editreward"]}
JUD = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"]; LLM = ["qwen3_8b", "qwen25_7b"]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
sg = lambda z: 1 / (1 + np.exp(-z))


def scope_cov(s, e, al=.2, splits=300):
    rng = np.random.default_rng(0); n = len(s); cov = []
    for _ in range(splits):
        perm = rng.permutation(n); c, t = perm[: n // 2], perm[n // 2:]
        o = np.argsort(s[c], kind="stable"); cs = np.cumsum(e[c][o] - al); ok = np.where(cs <= -1)[0]
        lam = s[c][o][ok.max()] if len(ok) else -np.inf; cov.append((s[t] <= lam).mean())
    return np.mean(cov)


def first(r, c, o):
    d = r[(c, o)]; return d["A"] / (d["A"] + d["B"])


L = ["# Position bias by candidate modality", "",
     "prior = mean P(first slot) - .5 with the SAME candidate in both slots (pure position prior).  bias / inc / acc / AUROC / "
     "cov@.2 per condition (non-tie pairs).  debiased = logit P(first) minus the item's own same-candidate prior.", "",
     "| kind | dataset | judge | n | prior (same) | bias bin | bias grid | bias lowres | inc bin | inc grid | inc lowres | inc debiased | "
     "acc bin | acc grid | acc lowres | acc debiased | AUROC bin | AUROC grid | AUROC debiased | cov bin | cov grid | cov lowres | cov debiased |",
     "|" + "---|" * 23]
for kind, dss in SETS.items():
    for ds in dss:
        I = items(ds)
        for j in JUD + (LLM if kind == "text" else []):
            R = load(ds, j, conds={"bin", "bin_same", "bin_grid", "bin_lowres"})
            have = lambda r, c: (c, "AB") in r and (c, "BA") in r
            rows = [(i, r) for i, r in R.items() if i in I and have(r, "bin") and have(r, "bin_same")]
            if len(rows) < 50:
                continue
            prior = np.mean([(first(r, "bin_same", "AB") + first(r, "bin_same", "BA")) / 2 - .5 for _, r in rows])
            nt = [(i, r) for i, r in rows if str(I[i]["label"]) in ("0", "1")]
            y = np.array([str(I[i]["label"]) == "0" for i, _ in nt])
            out = {}
            for c in ("bin", "bin_grid", "bin_lowres", "deb"):
                sub = [(k, r) for k, (i, r) in enumerate(nt) if have(r, "bin" if c == "deb" else c)]
                if len(sub) < 50:
                    out[c] = None; continue
                idx = np.array([k for k, _ in sub]); rr = [r for _, r in sub]
                if c == "deb":   # remove the slot prior measured on the same item: AB order -> cand0 duplicate prior, BA -> cand1
                    f1 = sg(lg(np.array([first(r, "bin", "AB") for r in rr])) - lg(np.array([first(r, "bin_same", "AB") for r in rr])))
                    f2 = sg(lg(np.array([first(r, "bin", "BA") for r in rr])) - lg(np.array([first(r, "bin_same", "BA") for r in rr])))
                else:
                    f1 = np.array([first(r, c, "AB") for r in rr]); f2 = np.array([first(r, c, "BA") for r in rr])
                p = (f1 + 1 - f2) / 2; e = ((p > .5) != y[idx]).astype(float); u = H(p)
                out[c] = dict(bias=np.mean((f1 + f2) / 2 - .5), inc=np.mean((f1 > .5) == (f2 > .5)), acc=1 - e.mean(),
                              au=auroc(u, e), cov=scope_cov(u, e, splits=a.splits))
            g = lambda c, k, f="{:.3f}": f.format(out[c][k]) if out.get(c) else "--"
            L.append(f"| {kind} | {ds} | {j}{' (LLM)' if j in LLM else ''} | {len(nt)} | {prior:+.3f} | {g('bin','bias','{:+.3f}')} | "
                     f"{g('bin_grid','bias','{:+.3f}')} | {g('bin_lowres','bias','{:+.3f}')} | {g('bin','inc')} | {g('bin_grid','inc')} | "
                     f"{g('bin_lowres','inc')} | {g('deb','inc')} | {g('bin','acc')} | {g('bin_grid','acc')} | {g('bin_lowres','acc')} | "
                     f"{g('deb','acc')} | {g('bin','au')} | {g('bin_grid','au')} | {g('deb','au')} | {g('bin','cov','{:.2f}')} | "
                     f"{g('bin_grid','cov','{:.2f}')} | {g('bin_lowres','cov','{:.2f}')} | {g('deb','cov','{:.2f}')} |")
            print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
