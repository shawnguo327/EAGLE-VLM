"""Tie types on the FULL GenAI-Arena benchmark (4175 pairs; meta.vote_type: leftvote / rightvote / tievote / bothbad_vote).
Per subset x judge:
  share     of each tie type
  AUROC     P(T) (A/B/T prompt, order-averaged) for 'both bad' vs decided and for 'tie (equally good)' vs decided;
            and -|p_bar - .5| (forced-choice closeness) for the same contrasts
  pi        published protocol (BPE, SCOPE calibrated on non-tie pairs, alpha = .2, 200 splits): share of accepted test
            verdicts whose human label is 'both bad' / 'tie', next to the overall shares (tie-blindness per type)
usage (judgebench dir): python tie_types.py --out paper_tables/tie_types.md"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, H
from eaglev_stats import auroc
from label_noise_lib import per_judge, thr

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/tie_types.md")
ap.add_argument("--splits", type=int, default=200)
a = ap.parse_args()
JUD = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b"]
I0 = items("genai_arena")
L = ["# Tie types on the full GenAI-Arena benchmark", "",
     "bb = 'both bad' votes, eq = 'tie' (equally good) votes, dec = left/right votes.  AUROC of P(T) and of forced-choice "
     "closeness -|p-.5| for bb vs dec and eq vs dec.  pi = share among verdicts accepted by the published protocol "
     "(alpha .2, calibrated without ties), next to the overall share.", "",
     "| subset | judge | n | bb share | eq share | AUROC P(T) bb | AUROC P(T) eq | AUROC close bb | AUROC close eq | pi bb (overall) | pi eq (overall) |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for sub in ("generation", "edition", None):
    I = {i: x for i, x in I0.items() if sub is None or x["subset"] == sub}
    for j in JUD:
        P = per_judge("genai_arena", j, I)
        ids = sorted(P)
        if len(ids) < 200:
            continue
        vt = np.array([I[i]["meta"]["vote_type"] for i in ids])
        bb, eq = vt == "bothbad_vote", vt == "tievote"; dec = ~(bb | eq)
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        p = np.array([P[i][0] for i in ids]); pt = np.array([P[i][1][2] for i in ids]); close = -np.abs(p - .5)
        au = lambda s, m: auroc(s[m | dec], m[m | dec].astype(float))
        u = H(p); e = ((p > .5) != (y == 0)).astype(float)
        rng = np.random.default_rng(0); n = len(ids); pib, pie = [], []
        for _ in range(a.splits):
            pm = rng.permutation(n); cal, te = pm[: n // 2], pm[n // 2:]; nt = cal[dec[cal]]
            lam = thr(u[nt], e[nt], .2); acc = te[u[te] <= lam]
            if len(acc):
                pib.append(bb[acc].mean()); pie.append(eq[acc].mean())
        L.append(f"| {sub or 'all'} | {j} | {n} | {bb.mean():.2f} | {eq.mean():.2f} | {au(pt, bb):.3f} | {au(pt, eq):.3f} | "
                 f"{au(close, bb):.3f} | {au(close, eq):.3f} | {np.mean(pib):.2f} ({bb.mean():.2f}) | {np.mean(pie):.2f} ({eq.mean():.2f}) |")
        print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
