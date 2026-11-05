"""Task-conditional (Mondrian) certification + why tie claims are never certified.
Sets: mllm_judge_sp (14 source tasks), mllm_judge (pair split, 14 tasks).  Feature sets: single judge (32B) and the jury
of all VLM judges + their pointwise scores (best routing variant).  Same 25/25/50 splits and graded claims as chain.py.
  global   one LTT threshold for all tasks (reference)
  mondrian one LTT threshold per task group (each group certified at alpha with its own delta: group-conditional risk)
           groups = source tasks with >= 300 pairs, the rest pooled
Per task we also report the global rule's realised risk (conditional-coverage check: does the global threshold over-
or under-shoot on hard tasks?).
Tie claims: precision of 'tie' among the top-k% pairs by head P(tie) (k = 1, 5, 10, 20) vs the tie base rate - a tie
claim can only be certified at alpha if some top-k slice reaches precision >= 1 - alpha.
usage (judgebench dir): python mondrian.py --out paper_tables/mondrian.md"""
import argparse, collections, os, sys
import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from label_noise_lib import per_judge, eval_items

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/mondrian.md")
ap.add_argument("--splits", type=int, default=50)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--grouping", default="task", choices=["task", "qa_vs_open"],
                help="task: one group per source task (>= 300 pairs); qa_vs_open: two groups fixed by task definition")
a = ap.parse_args()
QA = {"ChartQA", "infographicsVQA", "textVQA", "mathvista", "ScienceQA", "mind2web"}     # visual QA with a reference answer
AL = [0.2, 0.3]; GRID = np.linspace(0, 1, 201)[1:]; D = np.arange(1, 10)
VLM = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b", "internvl35_38b"]


def claims(S, lam):
    ok = S <= lam; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def pval(S, E, lam, al):
    ok, ch = claims(S, lam); n = int(ok.sum())
    return 1.0 if n == 0 else binom.cdf(int(E[np.arange(len(E)), np.maximum(ch, 0)][ok].sum()), n, al)


def fst(Sc, Ec, Sf, Ef, al):
    if len(Sc) == 0 or len(Sf) == 0:
        return -np.inf
    s0 = int(np.argmin([pval(Sf, Ef, g, al) for g in GRID])); best = -np.inf
    for g in GRID[s0:]:
        if pval(Sc, Ec, g, al) <= a.delta:
            best = g
        else:
            break
    return best


def feats(ds, j, ids, I):
    P = per_judge(ds, j, I)
    if not all(i in P for i in ids):
        return None
    p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
    return np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), fl]


def pfeats(ds, j, ids):
    R = load(ds, j, conds={"score"}); rows = []
    for i in ids:
        r = R.get(i, {})
        if not (("score", "A") in r and ("score", "B") in r):
            return None
        row = []
        for o in "AB":
            q = np.array([r[("score", o)].get(str(d), 0) for d in D]); q = q / q.sum(); mu = q @ D
            row += [mu, q @ (D - mu) ** 2, q[:3].sum()]
        rows.append(row + [min(row[0], row[3]), max(row[0], row[3])])
    return np.array(rows)


L = ["# Task-conditional (Mondrian) certification and tie-claim precision", "",
     f"{a.splits} splits.  Cell = acceptance / set size / decisive / strict error / risk.", ""]
for ds in ("mllm_judge_sp", "mllm_judge"):
    I = eval_items(ds, items(ds, rest=(ds != "mllm_judge_sp"))); ids = sorted(I)
    y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids]); n = len(y)
    sub = np.array([I[i]["subset"] for i in ids]); cnt = collections.Counter(sub)
    grp = (np.array([s if cnt[s] >= 300 else "other" for s in sub]) if a.grouping == "task" else
           np.array(["visual QA (reference answer)" if s in QA else "open-ended" for s in sub]))
    G = sorted(set(grp)); tsk = np.array([s if cnt[s] >= 300 else "other" for s in sub])
    FJ = {j: feats(ds, j, ids, I) for j in VLM}; FJ = {j: v for j, v in FJ.items() if v is not None}
    PW = {j: pfeats(ds, j, ids) for j in FJ}; PW = {j: v for j, v in PW.items() if v is not None}
    variants = []
    if "qwen3vl32b" in FJ:
        variants.append(("32B", FJ["qwen3vl32b"]))
    variants.append((f"jury {len(FJ)} + pointwise {len(PW)}", np.concatenate([FJ[j] for j in sorted(FJ)] + [PW[j] for j in sorted(PW)], 1)))
    L += [f"## {ds}: n = {n}, tie {np.mean(y == 2):.2f}, judges {sorted(FJ)}, pointwise {sorted(PW)}", ""]
    for vname, F in variants:
        R = collections.defaultdict(list); per_task = collections.defaultdict(list); tieprec = collections.defaultdict(list)
        rng = np.random.default_rng(0)
        for _ in range(a.splits):
            pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
            m = LogisticRegression(max_iter=5000).fit(F[fit], y[fit]); Pr = np.zeros((n, 3)); Pr[:, m.classes_] = m.predict_proba(F)
            k = Pr[:, :2].argmax(1); o = 1 - k
            S = np.c_[1 - Pr[np.arange(n), k], 1 - Pr[:, 2], Pr[np.arange(n), o]]; E = np.c_[y != k, y != 2, y == o].astype(float)
            pt = Pr[te, 2]; srt = np.argsort(-pt)
            for q in (.01, .05, .10, .20):
                top = srt[: max(1, int(q * len(te)))]; tieprec[q].append((y[te][top] == 2).mean())
            for al in AL:
                for rule in ("global", "mondrian"):
                    ok = np.zeros(len(te), bool); ch = np.full(len(te), -1)
                    if rule == "global":
                        ok, ch = claims(S[te], fst(S[cal], E[cal], S[fit], E[fit], al))
                    else:
                        for g in G:
                            c, f_, t_ = cal[grp[cal] == g], fit[grp[fit] == g], np.where(grp[te] == g)[0]
                            o_, c_ = claims(S[te][t_], fst(S[c], E[c], S[f_], E[f_], al)); ok[t_] = o_; ch[t_] = c_
                    e = E[te][np.arange(len(te)), np.maximum(ch, 0)]; st = ok & (ch == 0)
                    R[(rule, al)].append((ok.mean(), np.where(~ok, 3, np.where(ch == 2, 2, 1)).mean(), st.mean(), e[st].sum(), st.sum(),
                                          e[ok].mean() if ok.any() else np.nan))
                    if al == .2:
                        for g in sorted(set(tsk)):
                            mk = ok & (tsk[te] == g)
                            per_task[(rule, g)].append((mk.sum() / max((tsk[te] == g).sum(), 1), e[mk].mean() if mk.any() else np.nan))
        L += [f"### {vname}", "", "| rule | " + " | ".join(f"alpha {x}" for x in AL) + " |", "|---|---|---|"]
        for rule in ("global", "mondrian"):
            cells = []
            for al in AL:
                Z = np.array(R[(rule, al)], float)
                es = Z[:, 3].sum() / Z[:, 4].sum() if Z[:, 4].sum() else np.nan
                cells.append(f"{Z[:, 0].mean():.2f} / {Z[:, 1].mean():.2f} / {Z[:, 2].mean():.2f} / "
                             f"{'none' if np.isnan(es) else f'{es:.2f}'} / {np.nanmean(Z[:, 5]):.3f}")
            L.append(f"| {rule} | " + " | ".join(cells) + " |")
        L += ["", f"per source task (alpha .2): acceptance / realised risk, global rule vs group rule (grouping = {a.grouping})", "",
              "| task | group | n | global: acceptance | global: risk | group rule: acceptance | group rule: risk |", "|---|---|---|---|---|---|---|"]
        for g in sorted(set(tsk)):
            cells = []
            for rule in ("global", "mondrian"):
                Z = np.array(per_task[(rule, g)], float)
                cells += [f"{Z[:, 0].mean():.2f}", f"{np.nanmean(Z[:, 1]) if (~np.isnan(Z[:, 1])).any() else float('nan'):.3f}"]
            L.append(f"| {g} | {grp[tsk == g][0]} | {(tsk == g).sum()} | " + " | ".join(cells) + " |")
        L += ["", f"tie precision among the top-k% pairs by head P(tie) (base rate {np.mean(y == 2):.2f}): "
              + ", ".join(f"top {int(q * 100)}%: {np.mean(v):.2f}" for q, v in tieprec.items()), ""]
        print(vname, ds, flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
