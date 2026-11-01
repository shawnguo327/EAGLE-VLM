"""Label-efficient model leaderboards on GenAI-Arena: human-only vs PPI (uniform) vs PPI (confidence-routed).
Universe = the judged battles in items/genai_arena.jsonl (one subset at a time). Reference = BT fitted on ALL human
labels of that universe (ties = 0.5). For a human budget b, each estimator sees b·N human labels (in expectation)
plus judge soft labels p̄ on every battle. Reports mean |θ − θ_ref|, Kendall τ to the reference ranking, and
the PPI++ λ chosen.
usage (from $EAGLEVLM_DATA): python ppi_leaderboard.py qwen3vl8b [--reps 200]"""
import argparse, json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import bt_fit, ppi_bt, ppi_bt_tuned, routed_propensity, _sm
import collections

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--budgets", default=".05,.1,.2,.3"); ap.add_argument("--ds", default="genai_arena")
a = ap.parse_args()
items = [json.loads(l) for l in open(f"items/{a.ds}.jsonl")]
R = collections.defaultdict(dict)
for l in open(f"results/{a.ds}__{a.model}.jsonl"):
    d = json.loads(l)
    if d.get("cond") == "bin" and "lp" in d:
        R[d["id"]][d["order"]] = d["lp"]
out = {}
for sub in sorted({it["subset"] for it in items}):
    rows = [it for it in items if it["subset"] == sub and "AB" in R[it["id"]] and "BA" in R[it["id"]]]
    models = sorted({m for it in rows for m in it["meta"]["models"]}); mi = {m: i for i, m in enumerate(models)}
    P = np.array([[mi[it["meta"]["models"][0]], mi[it["meta"]["models"][1]]] for it in rows])
    y = np.array([0.5 if it["label"] == "tie" else float(it["label"] == 0) for it in rows])
    pa = np.array([(_sm(R[it["id"]]["AB"])["A"] + _sm(R[it["id"]]["BA"])["B"]) / 2 for it in rows])
    K = len(models); N = len(rows)
    ref = bt_fit(P, y, np.ones(N), K)
    judge_only = bt_fit(P, pa, np.ones(N), K)
    res = {"n_battles": N, "n_models": K,
           "judge_only": dict(mae=float(np.abs(judge_only - ref).mean()), kendall=float(stats.kendalltau(judge_only, ref).statistic)),
           }
    res["judge_acc_vs_human_strict"] = float(np.mean(((pa > .5) == (y == 1))[y != .5]))
    rng = np.random.default_rng(0)
    for b in map(float, a.budgets.split(",")):
        acc = collections.defaultdict(list)
        for rep in range(a.reps):
            for name, pi in (("human_only", np.full(N, b)), ("ppi_uniform", np.full(N, b)), ("ppi_routed", routed_propensity(pa, b))):
                lab = rng.random(N) < pi
                if lab.sum() < 3:
                    continue
                if name == "human_only":
                    th = bt_fit(P[lab], y[lab], np.ones(lab.sum()), K); lam = 0.0
                else:
                    th, lam = ppi_bt_tuned(P, pa, y, lab, pi, K, B=6, seed=rep)
                acc[name + "_mae"].append(np.abs(th - ref).mean())
                acc[name + "_kendall"].append(stats.kendalltau(th, ref).statistic)
                if name != "human_only":
                    acc[name + "_lam"].append(lam)
        res[f"budget_{b}"] = {k: float(np.nanmean(v)) for k, v in acc.items()}
    out[sub] = res
    print(sub, json.dumps(res, indent=1))
os.makedirs("paper_tables", exist_ok=True)
json.dump(out, open(f"paper_tables/ppi_{a.ds}_{a.model}.json", "w"), indent=1)
