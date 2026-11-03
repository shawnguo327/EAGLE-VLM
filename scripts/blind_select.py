"""Does selective judging certify visual judgement or text priors?  Z2 runs: same A/B/T prompt with the image (tri), without it
(tri_blind), and with another item's image (tri_wrong).  Forced choice = A vs B renormalised, averaged over both orders.
Per dataset x judge (non-tie pairs): accuracy and BPE error AUROC with / without image; verdict agreement; SCOPE coverage @.2 for
both; share of the with-image accepted set whose verdict equals the blind verdict; error AUROC of image-sensitivity
d = |p_img - p_blind| (higher = more image-dependent); 5-fold CV logistic AUROC of [BPE] vs [BPE, d, BPE_blind].
usage (judgebench dir): python blind_select.py --out paper_tables/blind_select.md"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from eaglev_stats import auroc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper_tables/blind_select.md")
a, _ = ap.parse_known_args()


def scope(s, err, alpha, splits=300, seed=0):
    """SCOPE linearised conformal threshold (same as vlm_vs_llm.scope); returns mean test coverage, risk, violation share"""
    rng = np.random.default_rng(seed); n = len(s); cov, risk, viol = [], [], []
    for _ in range(splits):
        perm = rng.permutation(n); c, t = perm[: n // 2], perm[n // 2:]
        o = np.argsort(s[c], kind="stable"); cs = np.cumsum(err[c][o] - alpha); ok = np.where(cs <= -1)[0]
        lam = s[c][o][ok.max()] if len(ok) else -np.inf
        acc = s[t] <= lam; cov.append(acc.mean()); r = err[t][acc].mean() if acc.any() else 0.0
        risk.append(r); viol.append(r > alpha)
    return np.mean(cov), np.mean(risk), np.mean(viol)


def pc(r, c):
    ab, ba = r[(c, "AB")], r[(c, "BA")]
    return (ab["A"] / (ab["A"] + ab["B"]) + ba["B"] / (ba["A"] + ba["B"])) / 2


L = ["# With image vs without image: what does the selective judge certify?", "",
     "Non-tie pairs, A/B/T prompt renormalised to A vs B, both orders.  cov = SCOPE coverage at alpha=.2 (300 splits).", "",
     "| dataset | judge | n | acc img | acc blind | acc wrong-img | agree img/blind | BPE AUROC img | BPE AUROC blind | cov img | cov blind | accepted & same as blind | AUROC d | CV [BPE] | CV [BPE,d,BPE_blind] |",
     "|" + "---|" * 15]
for ds in ("vlrb", "mmrb", "mllm_judge", "mmrlhf"):
    I = items(ds)
    for j in ("qwen3vl8b", "internvl35_8b"):
        R = load(ds, j, conds={"tri", "tri_blind", "tri_wrong"})
        rows = []
        for i, r in R.items():
            if i in I and str(I[i]["label"]) in ("0", "1") and all((c, o) in r for c in ("tri", "tri_blind", "tri_wrong") for o in ("AB", "BA")):
                rows.append((str(I[i]["label"]) == "0", pc(r, "tri"), pc(r, "tri_blind"), pc(r, "tri_wrong")))
        if len(rows) < 50:
            continue
        y = np.array([x[0] for x in rows]); p, pb, pw = (np.array([x[k] for x in rows]) for k in (1, 2, 3))
        e, eb = ((p > .5) != y).astype(float), ((pb > .5) != y).astype(float)
        u, ub, d = H(p), H(pb), np.abs(p - pb)
        c1 = scope(u, e, .2, splits=300)[0]; c2 = scope(ub, eb, .2, splits=300)[0]
        # accepted set at alpha .2 on full data (single calibration on all, descriptive)
        o = np.argsort(u); cs = np.cumsum(e[o] - .2); ok = np.where(cs <= -1)[0]
        acc_set = u <= (u[o][ok.max()] if len(ok) else -np.inf)
        same = ((p > .5) == (pb > .5))[acc_set].mean() if acc_set.any() else np.nan
        cv = lambda X: auroc(cross_val_predict(LogisticRegression(max_iter=2000), X, e, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1], e)
        L.append(f"| {ds} | {j} | {len(y)} | {1-e.mean():.3f} | {1-eb.mean():.3f} | {((pw>.5)==y).mean():.3f} | {((p>.5)==(pb>.5)).mean():.3f} | "
                 f"{auroc(u,e):.3f} | {auroc(ub,eb):.3f} | {c1:.2f} | {c2:.2f} | {same:.2f} | {auroc(-d,e):.3f} | {cv(u[:,None]):.3f} | {cv(np.c_[u,d,ub]):.3f} |")
        print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
