"""Is 'pairwise-uncertain' (high forced-choice entropy) one thing?  Decompose it with HUMAN references, and test whether
pointwise signals resolve the parts that pairwise cannot.
Pair types (human references):
  GenAI-Arena    decided | tie-equal (tievote) | tie-bothbad
  ImageRewardDB  decided-clear (overall gap >= 2) | decided-subtle (rank differs, overall gap <= 1) | tie-good (both >= 5) |
                 tie-other
  MLLM-as-a-Judge (pair split joined with the score split: human 1-5 per answer; matched by question id + model)
                 decided-clear (gap >= 2) | decided-subtle (gap <= 1) | tie-good (both >= 4) | tie-bad (both <= 2) | tie-mixed
  MT-Bench       decided | tie-consensus (all experts tie) | tie-disagreement (tie label without consensus)
For each set x judge:
  (a) composition of ALL pairs and of the most pairwise-uncertain third (highest BPE) by type
  (b) inside the uncertain third: CV AUROC for each type vs the rest, pairwise-only vs pointwise-only vs both
  (c) inside the uncertain third and overall: 3-way (A / B / tie) accuracy and tie F1 of a CV logistic head with
      pairwise-only vs pairwise + pointwise features (routing = use pointwise only where pairwise is uncertain)
  (d) Kumar et al. replication (MLLM-as-a-Judge): judge pointwise E[s] vs human score - Pearson, bias per human level
usage (judgebench dir): python decompose.py --out paper_tables/decompose.md"""
import argparse, json, os, sys, collections
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from eaglev_stats import auroc
from label_noise_lib import per_judge

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/decompose.md")
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen35_9b,qwen25vl7b,qwen3vl32b")
ap.add_argument("--mj-raw", default="/media/shawn/C/papers/VLM/judgebench_full/raw/ONE-Lab__MLLM-as-a-Judge/Benchmark")
a = ap.parse_args()
JUD = a.judges.split(","); D = np.arange(1, 10)
f3 = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def point(ds, j, ids):
    R = load(ds, j, conds={"score"}); out = {}
    for i in ids:
        r = R.get(i, {})
        if ("score", "A") in r and ("score", "B") in r:
            row = []
            for o in "AB":
                q = np.array([r[("score", o)].get(str(d), 0) for d in D]); q = q / q.sum(); mu = q @ D
                row += [mu, q @ (D - mu) ** 2, q[:3].sum(), q[6:].sum()]
            out[i] = row
    return out


def mj_scores():
    """pair item id -> (human score cand0, cand1) via the MLLM-as-a-Judge score split"""
    S = {}
    for l in open(f"{a.mj_raw}/score.jsonl"):
        d = json.loads(l)
        try:
            S[(str(d["id"]), d["name"], d["answer"].strip())] = float(d["human"])
        except Exception:
            pass
    byname = collections.defaultdict(list)
    for (qid, name, ans), v in S.items():
        byname[(qid, name)].append((ans, v))
    I = items("mllm_judge"); out = {}
    for i, x in I.items():
        qid = str(x["meta"].get("orig_id")); ms = x["meta"].get("models") or []
        if len(ms) != 2:
            continue
        sc = []
        for k in range(2):
            c = byname.get((qid, ms[k]), [])
            m = [v for ans, v in c if ans == x["cand"][k].strip()] or ([c[0][1]] if len(c) == 1 else [])
            sc.append(m[0] if m else None)
        if None not in sc:
            out[i] = sc
    return out


def types(ds, I, ids):
    if ds == "genai_arena":
        vt = [I[i]["meta"]["vote_type"] for i in ids]
        return np.array(["tie-bothbad" if v == "bothbad_vote" else "tie-equal" if v == "tievote" else "decided" for v in vt])
    if ds == "imagerewarddb":
        ov = np.array([[c["overall"] for c in I[i]["meta"]["cand_info"]] for i in ids], float)
        tie = np.array([str(I[i]["label"]) == "tie" for i in ids]); gap = np.abs(ov[:, 0] - ov[:, 1])
        return np.where(tie, np.where(ov.min(1) >= 5, "tie-good", "tie-other"), np.where(gap >= 2, "decided-clear", "decided-subtle"))
    if ds == "mllm_judge":
        sc = np.array([MJ[i] for i in ids]); tie = np.array([str(I[i]["label"]) == "tie" for i in ids]); gap = np.abs(sc[:, 0] - sc[:, 1])
        return np.where(tie, np.where(sc.min(1) >= 4, "tie-good", np.where(sc.max(1) <= 2, "tie-bad", "tie-mixed")),
                        np.where(gap >= 2, "decided-clear", "decided-subtle"))
    if ds == "mllm_judge_sp":
        sc = np.array([I[i]["meta"]["scores"] for i in ids], float); tie = sc[:, 0] == sc[:, 1]; gap = np.abs(sc[:, 0] - sc[:, 1])
        return np.where(tie, np.where(sc.min(1) >= 4, "tie-good", np.where(sc.max(1) <= 2, "tie-bad", "tie-mid")),
                        np.where(gap >= 2, "decided-clear", "decided-subtle"))
    if ds == "mtbench_human":
        vt = [I[i]["meta"]["votes"] for i in ids]; nv = [I[i]["meta"]["n_votes"] for i in ids]
        lab = [str(I[i]["label"]) for i in ids]
        return np.array(["decided" if l != "tie" else ("tie-consensus" if v["tie"] == n and n >= 2 else "tie-disagreement")
                         for l, v, n in zip(lab, vt, nv)])


def cvp(X, y):
    return cross_val_predict(LogisticRegression(max_iter=5000), X, y, cv=5, method="predict_proba")


L = ["# Decomposing pairwise uncertainty with human references", "",
     "MLLM-as-a-Judge score split made pairwise (mllm_judge_sp): the pair split and the score split share no questions, so the "
     "score split is paired within question instead.", ""]
# (d) Kumar et al. replication
L += ["## (d) Kumar et al. replication on the MLLM-as-a-Judge score split (unique answers): judge pointwise E[s] (1-9) vs human score (1-5)", "",
      "| judge | n answers | Pearson | judge E[s] mean by human score 1 / 2 / 3 / 4 / 5 |", "|---|---|---|---|"]
Isp = items("mllm_judge_sp", rest=False)
for j in JUD:
    W = point("mllm_judge_sp", j, sorted(Isp))
    if len(W) < 500:
        continue
    seen, hs, js = set(), [], []
    for i in W:
        for k in range(2):
            sid = Isp[i]["meta"]["score_ids"][k]
            if sid not in seen:
                seen.add(sid); hs.append(Isp[i]["meta"]["scores"][k]); js.append(W[i][4 * k])
    hs, js = np.array(hs), np.array(js)
    L.append(f"| {j} | {len(hs)} | {pearsonr(hs, js)[0]:.3f} | " + " / ".join(f"{js[hs == g].mean():.2f}" for g in (1, 2, 3, 4, 5)) + " |")
L.append("")
for ds, sub in (("mllm_judge_sp", None), ("genai_arena", "generation"), ("genai_arena", "edition"), ("imagerewarddb", None), ("mtbench_human", None)):
    I = items(ds, rest=(ds != "mllm_judge_sp")); I = {i: x for i, x in I.items() if (sub is None or x["subset"] == sub)}
    name = ds + (f"/{sub}" if sub else "")
    L += [f"## {name}", ""]
    tabA, tabB, tabC = [], [], []
    for j in JUD:
        P = per_judge(ds, j, I); W = point(ds, j, sorted(P)) if ds != "mtbench_human" else {}
        ids = sorted(i for i in P if (i in W or ds == "mtbench_human"))
        if len(ids) < .9 * len(I) or len(ids) < 200:
            continue
        T = types(ds, I, ids); y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids]); fl = np.array([P[i][2] for i in ids])
        Xp = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), fl, np.log(p / (1 - p) + 1e-9)]
        Xw = np.array([W[i] for i in ids]) if W else None
        if Xw is not None:
            Xw = np.c_[Xw, Xw[:, 0] - Xw[:, 4], np.minimum(Xw[:, 0], Xw[:, 4]), np.maximum(Xw[:, 0], Xw[:, 4])]
        u = H(p); unc = u >= np.quantile(u, 2 / 3)
        kinds = sorted(set(T))
        tabA.append(f"| {j} | all: " + ", ".join(f"{k} {np.mean(T == k):.2f}" for k in kinds) +
                    f" | uncertain third: " + ", ".join(f"{k} {np.mean(T[unc] == k):.2f}" for k in kinds) + " |")
        if Xw is not None:
            for k in kinds:
                t = (T[unc] == k).astype(int)
                if 15 <= t.sum() <= len(t) - 15:
                    r = [auroc(cvp(X[unc], t)[:, 1], t.astype(float)) for X in (Xp, Xw, np.c_[Xp, Xw])]
                    tabB.append(f"| {j} | {k} ({t.mean():.2f}) | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} |")
        for region, m in (("uncertain third", unc), ("all pairs", np.ones(len(y), bool))):
            if Xw is None:
                continue
            res = []
            for X in (Xp, np.c_[Xp, Xw]):
                pr = cvp(X[m], y[m]); v = np.array(sorted(set(y[m])))[pr.argmax(1)]
                tp = ((v == 2) & (y[m] == 2)).sum(); f1 = 2 * tp / max((v == 2).sum() + (y[m] == 2).sum(), 1)
                res.append(((v == y[m]).mean(), f1))
            tabC.append(f"| {j} | {region} | {m.sum()} | {res[0][0]:.3f} -> {res[1][0]:.3f} | {res[0][1]:.3f} -> {res[1][1]:.3f} |")
        print(name, j, flush=True)
    L += ["### (a) composition: all pairs vs the most pairwise-uncertain third", "", "| judge | all | uncertain third |", "|---|---|---|"] + tabA + [""]
    if tabB:
        L += ["### (b) inside the uncertain third: CV AUROC for each type vs rest", "", "| judge | type (share) | pairwise | pointwise | both |", "|---|---|---|---|---|"] + tabB + [""]
    if tabC:
        L += ["### (c) 3-way accuracy and tie F1: pairwise-only -> pairwise + pointwise (CV)", "", "| judge | region | n | 3-way acc | tie F1 |", "|---|---|---|---|---|"] + tabC + [""]
open(a.out, "w").write("\n".join(L) + "\n"); print("\n".join(L[:12]))
