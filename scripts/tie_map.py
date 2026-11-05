"""Tie map + explicit pairwise->pointwise router.
Map: pairwise uncertainty tercile (BPE of the order-averaged forced choice) x pointwise LEVEL (from the judge's own
expected 1-9 scores, rank-normalised within the judge to avoid scale compression): both-low (higher answer in the bottom
third of all answers), both-high (lower answer in the top third), mixed; x pointwise GAP (rank gap < .15 = close).
Per cell (fit + cal + test pooled, descriptive): n, human tie rate, composition by human type.
Router (certified, same 25/25/50 splits, 100 splits):
  pairwise-confident pairs (BPE below the top tercile, threshold from the fit split) -> pairwise graded claims (head on
  pairwise features only); pointwise is NOT queried
  pairwise-uncertain pairs -> pointwise is queried; candidate claims: 'tie' with estimated error 1 - tie rate of the pair's
  map cell (cell tie rates estimated on the fit split), else the pairwise claims
  certification: every claim level <= alpha (intersection-union fixed-sequence LTT) and the overall-risk version.
Reported: acceptance / set size / decisive / tie-claim share / tie-claim precision / risk / share of pairs routed.
usage (judgebench dir): python tie_map.py --out paper_tables/tie_map.md"""
import argparse, collections, os, sys
import numpy as np
from scipy.stats import binom, rankdata
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from label_noise_lib import per_judge

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/tie_map.md")
ap.add_argument("--splits", type=int, default=100)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--judges", default="qwen3vl32b,qwen3vl8b,internvl35_8b")
ap.add_argument("--level-claims", action="store_true", help="router issues absolute-level claims instead of tie")
a = ap.parse_args()
AL = [0.2, 0.3]; GRID = np.linspace(0, 1, 201)[1:]; D = np.arange(1, 10)


def ptype(ds, x):
    if ds == "mllm_judge_sp":
        s = x["meta"]["scores"]
        if s[0] == s[1]:
            return "tie-good" if min(s) >= 4 else "tie-bad" if max(s) <= 2 else "tie-mid"
        return "decided-clear" if abs(s[0] - s[1]) >= 2 else "decided-subtle"
    v = x["meta"]["vote_type"]
    return "tie-bad" if v == "bothbad_vote" else "tie-equal" if v == "tievote" else "decided"


def claims(S, lam):
    ok = S <= lam; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def pj(S, E, lam, al, joint):
    ok, ch = claims(S, lam)
    if not ok.any():
        return 1.0
    if not joint:
        return binom.cdf(int(E[np.arange(len(E)), np.maximum(ch, 0)][ok].sum()), int(ok.sum()), al)
    ps = [0.0]
    for c in range(S.shape[1]):
        mk = ok & (ch == c)
        if mk.any():
            ps.append(binom.cdf(int(E[mk, c].sum()), int(mk.sum()), al))
    return max(ps)


def fst(Sc, Ec, Sf, Ef, al, joint):
    s0 = int(np.argmin([pj(Sf, Ef, g, al, joint) for g in GRID])); best = -np.inf
    for g in GRID[s0:]:
        if pj(Sc, Ec, g, al, joint) <= a.delta:
            best = g
        else:
            break
    return best


L = ["# Tie map and explicit pairwise -> pointwise router", ""]
for ds, sub in (("mllm_judge_sp", None), ("genai_arena", "generation"), ("genai_arena", "edition")):
    I = items(ds, rest=(ds != "mllm_judge_sp")); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    name = ds + (f"/{sub}" if sub else "")
    for j in a.judges.split(","):
        P = per_judge(ds, j, I); R = load(ds, j, conds={"score"})
        ids = sorted(i for i in P if ("score", "A") in R.get(i, {}) and ("score", "B") in R.get(i, {}))
        if len(ids) < .9 * len(I):
            continue
        n = len(ids); y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        T = np.array([ptype(ds, I[i]) for i in ids])
        p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
        es = np.array([[np.array([R[i][("score", o)].get(str(d), 0) for d in D]) @ D /
                        sum(R[i][("score", o)].get(str(d), 0) for d in D) for o in "AB"] for i in ids])
        rk = (rankdata(es.ravel()) / es.size).reshape(es.shape)                     # rank within the judge
        lvl = np.where(rk.max(1) <= 1 / 3, "both-low", np.where(rk.min(1) >= 2 / 3, "both-high", "mixed"))
        gap = np.where(np.abs(rk[:, 0] - rk[:, 1]) < .15, "close", "far")
        u = H(p); terc = np.digitize(u, np.quantile(u, [1 / 3, 2 / 3]))
        tname = np.array(["confident", "middle", "uncertain"])[terc]
        # ---- descriptive map
        kinds = sorted(set(T))
        L += [f"## {name} / {j}: n = {n}, tie {np.mean(y == 2):.2f}", "",
              "| pairwise | pointwise level | gap | n | tie rate | " + " | ".join(kinds) + " |", "|" + "---|" * (5 + len(kinds))]
        for t in ("confident", "middle", "uncertain"):
            base = tname == t
            L.append(f"| **{t}** | (all) | | {base.sum()} | **{np.mean(y[base] == 2):.2f}** | " + " | ".join(f"{np.mean(T[base] == k):.2f}" for k in kinds) + " |")
            for lv in ("both-low", "both-high", "mixed"):
                for g in ("close", "far"):
                    m = base & (lvl == lv) & (gap == g)
                    if m.sum() >= 30:
                        L.append(f"| {t} | {lv} | {g} | {m.sum()} | {np.mean(y[m] == 2):.2f} | " + " | ".join(f"{np.mean(T[m] == k):.2f}" for k in kinds) + " |")
        # ---- certified router
        cell = np.array([f"{l}|{g}" for l, g in zip(lvl, gap)])
        if ds == "mllm_judge_sp":
            hs = np.array([I[i]["meta"]["scores"] for i in ids], float)
            LV = {"ncw": np.abs(hs[:, 0] - hs[:, 1]) <= 1, "bun": hs.max(1) <= 2}
        else:
            vt = np.array([I[i]["meta"]["vote_type"] for i in ids])
            LV = {"ncw": np.isin(vt, ["tievote", "bothbad_vote"]), "bun": vt == "bothbad_vote"}
        Fp = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), fl]
        Rs = collections.defaultdict(list); rng = np.random.default_rng(0)
        for _ in range(a.splits):
            pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
            m = LogisticRegression(max_iter=5000).fit(Fp[fit], y[fit]); Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(Fp)
            k = Pr[:, :2].argmax(1); o = 1 - k
            S0 = np.c_[1 - Pr[np.arange(n), k], 1 - Pr[:, 2], Pr[np.arange(n), o]]; E = np.c_[y != k, y != 2, y == o].astype(float)
            uth = np.quantile(u[fit], 2 / 3); routed = u >= uth
            tr = {c: (np.mean(y[fit][cell[fit] == c] == 2) if (cell[fit] == c).sum() >= 20 else 0.0) for c in set(cell)}
            ct = np.array([1 - tr[c] for c in cell])                                    # estimated error of a routed tie claim
            S1 = S0.copy(); S1[routed, 1] = np.minimum(S0[routed, 1], ct[routed])      # router: tie claim from the map
            if a.level_claims:
                # level claims (only where a human absolute reference defines them):
                #   'no clear winner' = human score gap <= 1 (score pairs) / human tie or both-bad vote (GenAI)
                #   'both unacceptable' = both human scores <= 2 (score pairs) / both-bad vote (GenAI)
                ncw, bun = LV["ncw"], LV["bun"]
                rn = {c: (np.mean(ncw[fit][cell[fit] == c]) if (cell[fit] == c).sum() >= 20 else 0.0) for c in set(cell)}
                rb = {c: (np.mean(bun[fit][cell[fit] == c]) if (cell[fit] == c).sum() >= 20 else 0.0) for c in set(cell)}
                cn = np.array([1 - rn[c] for c in cell]); cb = np.array([1 - rb[c] for c in cell])
                inf = np.full(n, np.inf)
                # claim order: strict, both-unacceptable, no-clear-winner, weak (strongest first); pairwise tie claim dropped
                S1 = np.c_[S0[:, 0], np.where(routed, cb, inf), np.where(routed, cn, inf), S0[:, 2]]
                E1 = np.c_[E[:, 0], ~bun, ~ncw, E[:, 2]].astype(float)
                S0x = np.c_[S0[:, 0], inf, inf, S0[:, 2]]
            for al in AL:
                runs = (("pairwise only", S0, E), ("router", S1, E)) if not a.level_claims else (("pairwise strict + weak", S0x, E1), ("router, level claims", S1, E1))
                for key, S, EE in runs:
                    for rule, joint in (("overall", False), ("every level", True)):
                        ok, ch = claims(S[te], fst(S[cal], EE[cal], S[fit], EE[fit], al, joint))
                        e = EE[te][np.arange(len(te)), np.maximum(ch, 0)]; tc = ok & (ch >= 1) & (ch < S.shape[1] - 1)
                        Rs[(key, rule, al)].append((ok.mean(), np.where(~ok, 3, np.where(ch == 2, 2, 1)).mean(), (ok & (ch == 0)).mean(),
                                                    tc.mean(), (1 - e[tc]).sum(), tc.sum(), e[ok].mean() if ok.any() else np.nan, routed[te].mean()))
        L += ["", "| claims | rule | " + " | ".join(f"alpha {x}: accept / size / decisive / tie-or-level claims (precision) / risk" for x in AL) + " | routed |",
              "|---|---|" + "---|" * len(AL) + "---|"]
        for key in (("pairwise only", "router") if not a.level_claims else ("pairwise strict + weak", "router, level claims")):
            for rule in ("overall", "every level"):
                cells = []
                for al in AL:
                    Z = np.array(Rs[(key, rule, al)], float)
                    pr = Z[:, 4].sum() / Z[:, 5].sum() if Z[:, 5].sum() else np.nan
                    cells.append(f"{Z[:, 0].mean():.2f} / {Z[:, 1].mean():.2f} / {Z[:, 2].mean():.2f} / {Z[:, 3].mean():.2f} "
                                 f"({'none' if np.isnan(pr) else f'{pr:.2f}'}) / {np.nanmean(Z[:, 6]):.3f}")
                L.append(f"| {key} | {rule} | " + " | ".join(cells) + f" | {np.mean(Z[:, 7]):.2f} |")
        L.append(""); print(name, j, flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
