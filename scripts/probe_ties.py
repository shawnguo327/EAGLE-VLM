"""Does the judge know more than it says?  Linear probes on the last-token hidden state at the verdict position
(judge_bench.py --hidden), compared with the output signals P(T) and U.
Targets (per dataset): human tie; tie type (both bad / equally good vs decided, GenAI-Arena and the Arena text set);
judge error on decided pairs; for the controlled set (synth_tie) pair type and which candidate is correct.
Features: hidden state of the tri prompt (A/B/T) and of the bin prompt, each order-averaged (AB and BA concatenated
after aligning nothing: the probe sees [h_AB, h_BA]); 5-fold CV logistic regression (L2, C chosen by inner CV) on
PCA-256; reported per layer fraction.  Baseline row: the same probe on [log P(T), U] only.
usage (from $EAGLEVLM_DATA): python probe_ties.py --runs RUNDIR[,RUNDIR] --judge qwen3vl8b --ds genai_arena,... --out F"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, signals, ttype
from eaglev_stats import auroc
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True, help="dirs holding <ds>__<judge>*.jsonl.hid")
ap.add_argument("--judge", required=True)
ap.add_argument("--ds", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pca", type=int, default=256)
ap.add_argument("--blind", action="store_true", help="hidden states from the no-image conds (tri_blind, bin_blind): surface-text control")
a = ap.parse_args()


def hidden(ds):
    H = {}
    for d in a.runs.split(","):
        for f in sorted(glob.glob(f"{d}/{ds}__{a.judge}*.jsonl.hid/*.npz")):
            z = np.load(f)
            for i, c, o, h in zip(z["id"], z["cond"], z["order"], z["h"]):
                H[(str(i), str(c), str(o))] = h.astype(np.float32)
            L = z["layers"]
    return H, L


G = None


def cv(X, y, seed=0, g=None):
    """grouped by base item when given (synth: the TG/TB/DV/DT pairs of one base never straddle train/test)"""
    out = np.zeros(len(y))
    g = G if g is None else g
    sp = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g) if g is not None else \
        StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    for tr, te in sp:
        steps = [StandardScaler()]
        if X.shape[1] > a.pca:
            steps.append(PCA(min(a.pca, len(tr) - 1), random_state=0))
        m = make_pipeline(*steps, LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1], max_iter=3000, cv=3))
        m.fit(X[tr], y[tr]); out[te] = m.predict_proba(X[te])[:, 1]
    return auroc(out, y)


def targets(ds, I, R):
    """name -> function(item, signals) -> 0/1/None"""
    T = {"human tie": lambda it, s: int(it["label"] == "tie")}
    if ds in ("genai_arena", "arena_tt"):
        T["both bad vs decided"] = lambda it, s: 1 if ttype(it) == "bothbad_vote" else 0 if ttype(it) in ("leftvote", "rightvote") else None
        T["equally good vs decided"] = lambda it, s: 1 if ttype(it) == "tievote" else 0 if ttype(it) in ("leftvote", "rightvote") else None
        T["both bad vs equally good"] = lambda it, s: 1 if ttype(it) == "bothbad_vote" else 0 if ttype(it) == "tievote" else None
    if ds == "synth_tie":
        T = {"paraphrase tie vs decided": lambda it, s: 1 if it["subset"] == "TG" else 0 if it["subset"] in ("DV", "DT") else None,
             "both-wrong tie vs decided": lambda it, s: 1 if it["subset"] == "TB" else 0 if it["subset"] in ("DV", "DT") else None,
             "both-wrong vs paraphrase": lambda it, s: 1 if it["subset"] == "TB" else 0 if it["subset"] == "TG" else None,
             "both-wrong vs one-wrong (TB vs DV, same edit type)": lambda it, s: 1 if it["subset"] == "TB" else 0 if it["subset"] == "DV" else None,
             "DV: cand0 correct": lambda it, s: (1 - it["label"]) if it["subset"] == "DV" else None,
             "judge error, DV only": lambda it, s: int((s[2] > .5) != (it["label"] == 0)) if it["subset"] == "DV" else None}
    T["judge error (decided pairs)"] = lambda it, s: None if it["label"] == "tie" else int((s[2] > .5) != (it["label"] == 0))
    return T


L = [f"# Hidden-state probes: {a.judge}", "",
     "AUROC, 5-fold CV logistic on PCA-256 of the verdict-position hidden state ([AB, BA] concatenated).",
     "'output' = the same probe on the output signals [log P(T), U, p̄] only.", ""]
import warnings
warnings.filterwarnings("ignore")
for dss in a.ds.split(","):
    ds, _, subset = dss.partition(":")         # "genai_arena:edition" = probe within one subset
    I = {k: v for k, v in items(ds).items() if not subset or v["subset"] == subset}
    R = load(ds, a.judge)
    H, layers = hidden(ds)
    if not H:
        print("no hidden for", ds); continue
    T = targets(ds, I, R)
    L += [f"## {dss}", "", "| target | n (pos) | P(T) | U | output | subset only | " + " | ".join(f"tri h@{l:.2f}" for l in layers) +
          " | " + " | ".join(f"bin h@{l:.2f}" for l in layers) + " |", "|---|" + "---|" * (5 + 2 * len(layers))]
    for tn, tf in T.items():
        rows = []
        for i, r in R.items():
            if i not in I:
                continue
            s = signals(r)
            keys = [(i, c + ("_blind" if a.blind else ""), o) for c in ("tri", "bin") for o in ("AB", "BA")]
            if not s or not all(k in H for k in keys):
                continue
            y = tf(I[i], s)
            if y is None:
                continue
            rows.append((y, s, [H[k] for k in keys], I[i].get("subset", ""), (I[i].get("meta") or {}).get("base", i)))
        if len(rows) < 60 or len({x[0] for x in rows}) < 2 or min(sum(x[0] for x in rows), sum(1 - x[0] for x in rows)) < 10:
            continue
        y = np.array([x[0] for x in rows])
        G = np.array([str(x[4]) for x in rows]) if ds.startswith("synth") else None
        pt = np.array([x[1][0] for x in rows]); u = np.array([x[1][1] for x in rows]); p = np.array([x[1][2] for x in rows])
        subs = sorted({x[3] for x in rows})
        S1h = np.array([[x[3] == v for v in subs] for x in rows], float)      # source-subset one-hot: metadata baseline
        cells = [f"{len(y)} ({y.sum()})", f"{auroc(pt, y):.3f}", f"{auroc(u, y):.3f}",
                 f"{cv(np.c_[np.log(pt + 1e-4), np.log(u + 1e-4), p], y):.3f}",
                 f"{cv(S1h, y):.3f}" if len(subs) > 1 else "--"]
        for ci in (0, 1):                      # tri, bin
            for li in range(len(layers)):
                X = np.stack([np.concatenate([x[2][2 * ci][li], x[2][2 * ci + 1][li]]) for x in rows])
                cells.append(f"{cv(X, y):.3f}")
        L.append(f"| {tn} | " + " | ".join(cells) + " |")
        print(dss, tn, cells, flush=True)
    L.append("")
open(a.out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
