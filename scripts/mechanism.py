"""Mechanism analysis: why is order-symmetric confidence weak for VLM judges?
(1) Error anatomy across judges: share of strict pairs on which k of the J judges are wrong; are unanimous errors
    confident?  A large confident-unanimous-error mass means confidence cannot separate it (shared bias or contested label).
(2) Per-subset rate of unanimous errors.
usage (from $EAGLEVLM_DATA): python mechanism.py [--judges qwen3vl8b,qwen35_9b,qwen25vl7b,internvl35_8b]"""
import argparse, collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import load_records, _sm

ap = argparse.ArgumentParser(); ap.add_argument("--judges", default="qwen3vl8b,qwen35_9b,qwen25vl7b")
ap.add_argument("--ds", default="vlrb,mmrb,mmrlhf,mllm_judge,genai_arena,mjbench,mmrb2,imagerewarddb,qbench2,editreward,pp")
a = ap.parse_args(); J = a.judges.split(",")
out = {}
for ds in a.ds.split(","):
    if not all(os.path.exists(f"results/{ds}__{m}.jsonl") for m in J):
        continue
    sub = {json.loads(l)["id"]: json.loads(l).get("subset", "") for l in open(f"items/{ds}.jsonl")}
    R = {m: {r["id"]: r for r in load_records(f"items/{ds}.jsonl", f"results/{ds}__{m}.jsonl") if r["y"] is not None} for m in J}
    ids = sorted(set.intersection(*[set(R[m]) for m in J]))
    y = np.array([R[J[0]][i]["y"] for i in ids])
    wrong = np.array([[(R[m][i]["pa"] > .5) != (y[k] == 1) for k, i in enumerate(ids)] for m in J])  # J x n
    conf = np.array([[abs(2 * R[m][i]["pa"] - 1) for i in ids] for m in J])
    kw = wrong.sum(0)
    res = {"n": len(ids), "share_k_wrong": [float((kw == k).mean()) for k in range(len(J) + 1)]}
    # of each judge's errors, how many are unanimous, and how confident are they vs its correct verdicts
    res["unanimous_share_of_errors"] = float(np.mean([(wrong[j] & (kw == len(J))).sum() / max(1, wrong[j].sum()) for j in range(len(J))]))
    res["conf_unanimous_err"] = float(conf[:, kw == len(J)].mean()) if (kw == len(J)).any() else None
    res["conf_idiosyncratic_err"] = float(np.mean([conf[j][wrong[j] & (kw == 1)].mean() for j in range(len(J)) if (wrong[j] & (kw == 1)).any()]))
    res["conf_correct"] = float(np.mean([conf[j][~wrong[j]].mean() for j in range(len(J))]))
    # best achievable by majority vote of the judges
    res["acc_majority"] = float((kw < len(J) / 2).mean())
    # per-subset unanimous-error rate
    S = collections.defaultdict(list)
    for k, i in enumerate(ids):
        S[sub[i]].append(kw[k] == len(J))
    res["unanimous_err_by_subset"] = {s: round(float(np.mean(v)), 3) for s, v in S.items()}
    out[ds] = res
    print(f"{ds:13s} n={len(ids)} k-wrong shares={[round(x, 2) for x in res['share_k_wrong']]} "
          f"unanimous/errors={res['unanimous_share_of_errors']:.2f} conf: correct={res['conf_correct']:.2f} "
          f"idio-err={res['conf_idiosyncratic_err']:.2f} unan-err={res['conf_unanimous_err'] or 0:.2f} maj-acc={res['acc_majority']:.3f}")
    if len(S) > 1:
        print("      by subset:", res["unanimous_err_by_subset"])
os.makedirs("paper_tables", exist_ok=True)
json.dump(out, open("paper_tables/mechanism.json", "w"), indent=1)
