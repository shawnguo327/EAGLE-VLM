"""Go / no-go test for pointwise routing (pre-registered criteria, 2026-11-04):
  (2) does the pointwise absolute score detect 'both bad' ties?  AUROC of -max(E[sA], E[sB]) for bothbad vs decided on
      GenAI-Arena (target >= .70 on generation), next to the pairwise P(T)
  (3) does adding pointwise features to the 3-class head raise certified coverage at the same alpha?
      base features  = cert_main head features [log P0, log P1, log PT, BPE, |p-.5|, consistency]
      +point         = base + [E[sA], E[sB], min, max, E[sA]-E[sB], P(sA<=3), P(sB<=3)]
      strict-LTT and graded-LTT (fixed-sequence, delta .1), 25% fit / 25% cal / 50% test, same splits for both.
E[s] = expectation of the 1-9 digit distribution from the 'score' cond (order A = cand 0, B = cand 1).
usage (judgebench dir): python route_test.py --out paper_tables/route_test.md"""
import argparse, collections, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from eaglev_stats import auroc
from label_noise_lib import per_judge, eval_items, ltt, _claims

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/route_test.md")
ap.add_argument("--splits", type=int, default=100)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b")
ap.add_argument("--base-only", action="store_true", help="no pointwise runs: report the base head only (+point columns = base)")
a = ap.parse_args()
AL = [0.1, 0.2, 0.3]
SETS = [("genai_arena", None), ("genai_arena", "generation"), ("genai_arena", "edition"), ("pickapic_u", None),
        ("imagerewarddb", None), ("mllm_judge", None)]
D = np.arange(1, 10)


def point(ds, j, ids):
    R = load(ds, j, conds={"score"}); out = {}
    for i in ids:
        r = R.get(i, {})
        if ("score", "A") in r and ("score", "B") in r:
            pa = np.array([r[("score", "A")].get(str(d), 0) for d in D]); pb = np.array([r[("score", "B")].get(str(d), 0) for d in D])
            pa, pb = pa / pa.sum(), pb / pb.sum()
            out[i] = (pa @ D, pb @ D, pa[:3].sum(), pb[:3].sum())
    return out


def run(F, y, splits):
    n = len(y); rng = np.random.default_rng(0); R = collections.defaultdict(list)
    for _ in range(splits):
        pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
        m = LogisticRegression(max_iter=5000).fit(F[fit], y[fit])
        Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(F)
        k = Pr[:, :2].argmax(1); o = 1 - k; pk = Pr[np.arange(n), k]; po = Pr[np.arange(n), o]
        S1 = (1 - pk)[:, None]; E1 = (y != k).astype(float)[:, None]
        S3 = np.c_[1 - pk, 1 - Pr[:, 2], po]; E3 = np.c_[y != k, y != 2, y == o].astype(float)
        for al in AL:
            lam = ltt(S1[cal], E1[cal], al, a.delta, S1[fit], E1[fit]); acc = S1[te, 0] <= lam
            R[("strict", al)].append((acc.mean(), E1[te, 0][acc].mean() if acc.any() else np.nan))
            lam = ltt(S3[cal], E3[cal], al, a.delta, S3[fit], E3[fit]); acc, ch = _claims(S3[te], lam)
            e = E3[te][np.arange(len(te)), np.maximum(ch, 0)]
            R[("graded", al)].append((acc.mean(), e[acc].mean() if acc.any() else np.nan))
            size = np.where(~acc, 3, np.where(ch == 2, 2, 1))          # {k} / {T} = 1, weak {k, T} = 2, abstain = 3
            R[("size", al)].append((size.mean(), (acc & (ch == 0)).mean(), (acc & (ch == 1)).mean()))
    return R


cell = lambda Z: "{:.2f} / {:.3f}".format(np.mean([z[0] for z in Z]), np.nanmean([z[1] for z in Z]) if any(
    not np.isnan(z[1]) for z in Z) else np.nan)
L = ["# Pointwise routing: go / no-go test", "",
     "Cells = mean acceptance / mean realised selective risk (LTT, delta .1); set size: strict or tie claim = 1, weak claim = 2, abstain = 3 (mean over test pairs); decisive = share of strict claims; tie-claim = share of certified tie claims.  AUROC bb = 'both bad' vs decided pairs.", "",
     "| set | judge | n | tie | AUROC bb: P(T) | AUROC bb: point | AUROC tie: P(T) | AUROC tie: +point head | "
     + " | ".join(f"strict @{x} base / +point | graded @{x} base / +point | set size / decisive / tie-claim @{x} base -> +point" for x in AL) + " |", "|" + "---|" * (8 + 3 * len(AL))]
for ds, sub in SETS:
    I = eval_items(ds, items(ds)); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    for j in a.judges.split(","):
        P = per_judge(ds, j, I); S = {i: (0., 0., 0., 0.) for i in P} if a.base_only else point(ds, j, sorted(P)); ids = sorted(i for i in P if i in S)
        if len(ids) < 200 or len(ids) < .95 * len(I):
            print(f"[skip] {ds}/{sub} {j}: {len(ids)}/{len(I)}", file=sys.stderr); continue
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
        sa, sb, la, lb = (np.array([S[i][c] for i in ids]) for c in range(4))
        Fb = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), fl]
        Fp = np.c_[Fb, sa, sb, np.minimum(sa, sb), np.maximum(sa, sb), sa - sb, la, lb]
        vt = np.array([I[i].get("meta", {}).get("vote_type", "") for i in ids]); bb = vt == "bothbad_vote"; dec = y != 2
        if bb.any():
            m = bb | dec; aub_t = auroc(Q[m, 2], bb[m].astype(float)); aub_p = auroc(-np.maximum(sa, sb)[m], bb[m].astype(float))
        else:
            aub_t = aub_p = np.nan
        rng = np.random.default_rng(1); pm = rng.permutation(len(y)); h = len(y) // 2
        mt = LogisticRegression(max_iter=5000).fit(Fp[pm[:h]], (y[pm[:h]] == 2).astype(int))
        au_tie_p = auroc(mt.predict_proba(Fp[pm[h:]])[:, 1], (y[pm[h:]] == 2).astype(float))
        Rb, Rp = run(Fb, y, a.splits), run(Fp, y, a.splits)
        L.append(f"| {ds}{'/' + sub if sub else ''} | {j} | {len(ids)} | {(y == 2).mean():.2f} | {aub_t:.3f} | {aub_p:.3f} | "
                 f"{auroc(Q[:, 2], (y == 2).astype(float)):.3f} | {au_tie_p:.3f} | "
                 + " | ".join(f"{cell(Rb[('strict', al)])} → {cell(Rp[('strict', al)])} | {cell(Rb[('graded', al)])} → {cell(Rp[('graded', al)])} | "
                              + "{:.2f}/{:.2f}/{:.2f} → {:.2f}/{:.2f}/{:.2f}".format(*np.mean(Rb[('size', al)], 0), *np.mean(Rp[('size', al)], 0))
                              for al in AL) + " |")
        print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
