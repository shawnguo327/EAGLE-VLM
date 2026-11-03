"""Three-way selective judging: ties are a valid, certifiable output, not something to drop.
Human label in {cand0, cand1, tie}.  Error = judge output != human label.  SCOPE linearised conformal threshold on an
uncertainty score; random splits.  Methods:
  bin-drop     published setting: forced choice (A/B prompt, both orders), BPE score, calibrated on non-tie pairs only,
               deployed on the full stream (ties included)  -> guarantee not valid
  bin-all      same judge and score, calibrated on the full stream (tie = error whenever a winner is named)  -> valid
  tri-argmax   A/B/T prompt, both orders aligned and averaged, output = argmax, score = 1 - max prob
  tri-cal      a 3-class multinomial logistic head on [log P0, log P1, log PT (tri), BPE, |p_bar-.5| (bin)] fitted on
               a disjoint quarter; output = argmax, score = 1 - max prob; calibrate on the next quarter, test on the half
All methods share identical test halves.  Cells: mean test coverage / mean test risk at alpha; share of accepted
outputs that are 'tie'.
usage (judgebench dir): python tie3.py --out paper_tables/tie3.md"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/tie3.md")
ap.add_argument("--splits", type=int, default=200)
ap.add_argument("--alphas", default="0.2,0.3")
a = ap.parse_args()
AL = [float(x) for x in a.alphas.split(",")]
SETS = {"text": ["mtbench_human", "arena55k", "arena_tt"], "resp": ["mllm_judge"], "img": ["genai_arena", "editreward", "pp"]}
VLM = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"]; LLM = ["qwen3_8b", "qwen25_7b"]


def thr(s, e, alpha):
    o = np.argsort(s, kind="stable"); cs = np.cumsum(e[o] - alpha); ok = np.where(cs <= -1)[0]
    return s[o][ok.max()] if len(ok) else -np.inf


def data(ds, j):
    I = items(ds); R = load(ds, j, conds={"bin", "tri"}); X = []
    for i, r in R.items():
        if i not in I or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
        p = (b1["A"] / (b1["A"] + b1["B"]) + b2["B"] / (b2["A"] + b2["B"])) / 2
        q = np.array([(t1["A"] + t2["B"]) / 2, (t1["B"] + t2["A"]) / 2, (t1["T"] + t2["T"]) / 2]); q = q / q.sum()
        lab = {"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])]
        X.append((lab, p, q))
    return X


L = ["# Three-way selective judging: certify ties instead of dropping them", "",
     f"SCOPE calibration, {a.splits} random splits; error = output != human label (cand0 / cand1 / tie).  "
     "Cells = coverage / risk; 'tie%' = share of accepted outputs that are ties.  bin-drop = published protocol (calibrated "
     "without ties, deployed with them).", ""]
hdr = "| kind | dataset | judge | n | tie share | " + " | ".join(
    f"bin-drop @{x} | bin-all @{x} | tri-argmax @{x} | tri-cal @{x} | tri-cal tie% @{x}" for x in AL) + " |"
L += [hdr, "|" + "---|" * (5 + 5 * len(AL))]
agg = {}
for kind, dss in SETS.items():
    for ds in dss:
        for j in VLM + (LLM if kind == "text" else []):
            X = data(ds, j)
            if len(X) < 100:
                continue
            y = np.array([x[0] for x in X]); p = np.array([x[1] for x in X]); Q = np.array([x[2] for x in X])
            u = H(p); yb = np.where(p > .5, 0, 1)
            eb = (yb != y).astype(float)
            q3 = Q.argmax(1); e3 = (q3 != y).astype(float); s3 = 1 - Q.max(1)
            F = np.c_[np.log(Q + 1e-9), u, np.abs(p - .5)]
            rng = np.random.default_rng(0); n = len(y)
            res = {al: {k: [] for k in ("drop", "all", "arg", "cal", "calT")} for al in AL}
            for _ in range(a.splits):
                perm = rng.permutation(n); fit, cal, te = perm[: n // 4], perm[n // 4: n // 2], perm[n // 2:]
                calb = perm[: n // 2]                           # binary / argmax methods need no fit set: use both quarters
                m = LogisticRegression(max_iter=2000, C=1.0).fit(F[fit], y[fit])
                Pc = m.predict_proba(F); cls = m.classes_[Pc.argmax(1)]; sc = 1 - Pc.max(1); ec = (cls != y).astype(float)
                for al in AL:
                    nt = calb[y[calb] != 2]
                    for key, s, e, cset in (("drop", u, eb, nt), ("all", u, eb, calb), ("arg", s3, e3, calb), ("cal", sc, ec, cal)):
                        lam = thr(s[cset], e[cset], al); acc = s[te] <= lam
                        res[al][key].append((acc.mean(), e[te][acc].mean() if acc.any() else 0.0))
                        if key == "cal":
                            res[al]["calT"].append((cls[te][acc] == 2).mean() if acc.any() else 0.0)
            row = [kind, ds, j + (" (LLM)" if j in LLM else ""), n, f"{(y == 2).mean():.2f}"]
            for al in AL:
                for key in ("drop", "all", "arg", "cal"):
                    c, r = np.mean(res[al][key], 0)
                    row.append(f"{c:.2f} / {r:.3f}")
                    agg.setdefault((kind, key, al), []).append(c)
                row.append(f"{np.mean(res[al]['calT']):.2f}")
            L.append("| " + " | ".join(map(str, row)) + " |"); print(L[-1], flush=True)
L += ["", "## Mean coverage by kind", "", "| kind | method | " + " | ".join(f"@{x}" for x in AL) + " |", "|---|---|" + "---|" * len(AL)]
for kind in SETS:
    for key in ("drop", "all", "arg", "cal"):
        if (kind, key, AL[0]) in agg:
            L.append(f"| {kind} | {key} | " + " | ".join(f"{np.mean(agg[(kind, key, x)]):.2f}" for x in AL) + " |")
open(a.out, "w").write("\n".join(L) + "\n"); print("\n".join(L[-14:]))
