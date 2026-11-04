"""Why are edit 'both bad' ties detectable?  Hypothesis: an edit can be checked against its source image - a candidate
that barely changed the source (instruction not applied) or wrecked it is visibly bad, so 'both bad' is an absolute,
reference-checkable judgment; text-to-image has no reference.
Tests on full GenAI-Arena (vote types from the benchmark; candidate-source and candidate-candidate similarities from
pair_sim.py: DINOv2-L CLS cosine, CLIP-L cosine, grey 256px pixel L1 similarity):
  1. distribution of source similarity for bothbad / tie / decided edit pairs (both candidates unchanged? either?)
  2. AUROC of similarity features alone for bothbad vs decided (5-fold CV logistic) - how much of 'both bad' is
     explained by 'did the edit change the source'
  3. judge signals (P(T), pointwise max score): AUROC for bothbad vs decided, overall and WITHIN source-similarity
     terciles; and CV AUROC of sims + judge vs sims alone (incremental information beyond the reference check)
  4. same contrasts on generation pairs (no source; candidate-candidate similarity only)
usage (judgebench dir): python edit_bothbad.py --out paper_tables/edit_bothbad.md"""
import argparse, json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load
from eaglev_stats import auroc
from label_noise_lib import per_judge

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/edit_bothbad.md")
ap.add_argument("--sims", default="paper_tables/pair_sim.jsonl,paper_tables/pair_sim_genai_rest.jsonl")
a = ap.parse_args()
JUD = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b"]
S = {}
for f in a.sims.split(","):
    for l in open(f):
        d = json.loads(l)
        if d["id"].startswith("genai_arena/"):
            S.setdefault(d["id"], d)
I = items("genai_arena")
D = np.arange(1, 10)


def pmax(j, ids):
    R = load("genai_arena", j, conds={"score"}); out = {}
    for i in ids:
        r = R.get(i, {})
        if ("score", "A") in r and ("score", "B") in r:
            e = [np.array([r[("score", o)].get(str(d), 0) for d in D]) for o in "AB"]
            out[i] = max(x @ D / x.sum() for x in e)
    return out


def cv(X, t):
    return auroc(cross_val_predict(LogisticRegression(max_iter=3000), X, t, cv=5, method="predict_proba")[:, 1], t)


L = ["# Edit 'both bad' ties: reference check or something else?", ""]
for sub in ("edition", "generation"):
    ids = sorted(i for i, x in I.items() if x["subset"] == sub and i in S)
    vt = np.array([I[i]["meta"]["vote_type"] for i in ids]); bb = vt == "bothbad_vote"; eq = vt == "tievote"; dec = ~(bb | eq)
    m = bb | dec; t = bb[m].astype(float)
    cc = np.array([[S[i]["dino"], S[i]["clip"], S[i]["pix"]] for i in ids])
    L += [f"## {sub}: n = {len(ids)} with similarities ({len(I) and sum(x['subset'] == sub for x in I.values())} in set); "
          f"bothbad {bb.mean():.2f}, tie {eq.mean():.2f}, decided {dec.mean():.2f}", ""]
    feats = {"cand-cand dino": cc[:, 0], "cand-cand clip": cc[:, 1], "cand-cand pix": cc[:, 2]}
    X = cc
    if sub == "edition":
        src = np.array([S[i]["dino_src"] for i in ids]); srcc = np.array([S[i]["clip_src"] for i in ids])
        hi = np.quantile(src, .9)                      # 'unchanged' = top decile of candidate-source DINO similarity
        L += ["| group | mean max src-sim (dino) | mean min src-sim | share with BOTH cands unchanged | share with EITHER unchanged |",
              "|---|---|---|---|---|"]
        for name, g in (("bothbad", bb), ("tie", eq), ("decided", dec)):
            L.append(f"| {name} | {src[g].max(1).mean():.3f} | {src[g].min(1).mean():.3f} | {(src[g].min(1) > hi).mean():.2f} | "
                     f"{(src[g].max(1) > hi).mean():.2f} |")
        L.append(f"('unchanged' = candidate-source DINO cosine above the 90th percentile, {hi:.3f})")
        L.append("")
        feats.update({"max src-sim dino": src.max(1), "min src-sim dino": src.min(1), "max src-sim clip": srcc.max(1),
                      "|src-sim diff| dino": np.abs(src[:, 0] - src[:, 1])})
        X = np.c_[cc, np.sort(src, 1), np.sort(srcc, 1), np.abs(src[:, 0] - src[:, 1])]
    L += ["| similarity feature | AUROC bothbad vs decided (higher = feature larger for bothbad) |", "|---|---|"]
    for k, v in feats.items():
        L.append(f"| {k} | {auroc(v[m], t):.3f} |")
    au_sims = cv(X[m], t)
    L += [f"| **all similarities (5-fold CV logistic)** | **{au_sims:.3f}** |", ""]
    key = np.max(src, 1) if sub == "edition" else cc[:, 0]
    qs = np.quantile(key[m], [1 / 3, 2 / 3]); strata = np.digitize(key, qs)
    L += [f"| judge | AUROC P(T) | AUROC pointwise -max | P(T) within {'max src-sim' if sub == 'edition' else 'cand-cand dino'} terciles (low/mid/high) | "
          "CV sims | CV sims + P(T) + pointwise |", "|---|---|---|---|---|---|"]
    for j in JUD:
        P = per_judge("genai_arena", j, {i: I[i] for i in ids}); W = pmax(j, ids)
        ok = np.array([i in P for i in ids])
        if ok.mean() < .95:
            continue
        pt = np.array([P[i][1][2] if i in P else np.nan for i in ids])
        pw = np.array([-W[i] if i in W else np.nan for i in ids]); has_w = ~np.isnan(pw)
        within = []
        for s in range(3):
            g = m & (strata == s) & ok
            within.append(f"{auroc(pt[g], bb[g].astype(float)):.2f}" if 0 < bb[g].sum() < g.sum() else "-")
        mm = m & ok & has_w
        aw = f"{auroc(pw[mm], bb[mm].astype(float)):.3f}" if mm.sum() > 100 else "-"
        both = f"{cv(np.c_[X, np.log(pt + 1e-9), pw][mm], bb[mm].astype(float)):.3f}" if mm.sum() > 100 else "-"
        L.append(f"| {j} | {auroc(pt[m & ok], bb[m & ok].astype(float)):.3f} | {aw} | {' / '.join(within)} | {au_sims:.3f} | {both} |")
        print(L[-1], flush=True)
    L.append("")
open(a.out, "w").write("\n".join(L) + "\n")
