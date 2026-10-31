import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import WS
S = json.load(open(f"{WS}/out/summary.json"))
rungs = ["gauss1", "gauss2", "gauss4", "gauss8", "uniform", "tshuffle", "markov", "llmgen"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
x = range(len(rungs))
acc = [S["cf_real_vs"][r]["acc"] for r in rungs]
ax.errorbar(x, [a[0] for a in acc], yerr=[[a[0] - a[1] for a in acc], [a[2] - a[0] for a in acc]], fmt="o", color="#2a78d6", ms=8, lw=2, capsize=0, label="LLM judge (unmatched pairs)")
if "cfm" in S:
    xm = [i for i, r in enumerate(rungs) if r in S["cfm"]]
    am = [S["cfm"][rungs[i]]["acc"] for i in xm]
    ax.errorbar([i + 0.15 for i in xm], [a[0] for a in am], yerr=[[a[0] - a[1] for a in am], [a[2] - a[0] for a in am]], fmt="s", color="#1baf7a", ms=8, lw=2, capsize=0, label="LLM judge (matched: same source day)")
ax.plot([i - 0.15 for i in x], [S["lr_auc"][r] for r in rungs], "D", color="#eb6834", ms=8, label="LR on 10 features (AUC)")
ax.axhline(0.5, color="#999", lw=1, ls="--")
ax.set_xticks(list(x)); ax.set_xticklabels(rungs, rotation=30)
ax.set_ylabel("P(real ranked above fake)"); ax.set_ylim(0.3, 1.02)
ax.set_title("Real vs degraded: pairwise accuracy (95% CI)", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="lower right")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax = axes[1]
bt = S["bt"]; order = sorted(bt, key=lambda r: -bt[r][0])
ax.errorbar([bt[r][0] for r in order], range(len(order)), xerr=[[bt[r][0] - bt[r][1] for r in order], [bt[r][2] - bt[r][0] for r in order]], fmt="o", color="#2a78d6", ms=8, lw=2, capsize=0)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
ax.set_xlabel("Bradley-Terry 'realness' score (context-free judge)")
ax.set_title(f"Judge ladder; Kendall tau on real>g1>g2>g4>g8>uniform = {S['ladder_kendall_tau'][0]:.2f}", fontsize=10)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{WS}/ladder.png", dpi=130)
print("ok")
