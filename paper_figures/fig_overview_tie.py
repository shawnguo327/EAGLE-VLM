"""Figure 1 of the tie paper (2026-11-10): one human tie on ImagenHub editing, the judge's signals, and what each certified
output says about it.  Inputs: results/fig1_pair_qwen3vl8b.json (scripts: VLM/... fig1_pair.py, certify / absolute_actions
functions, 100 source-level splits, alpha = .2) and the three images.  Output: <draft>/figs_new/fig_overview.{pdf,png}.
usage: python fig_overview_tie.py <draft dir>"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

D = "/media/shawn/C/papers/VLM"; OUT = sys.argv[1] + "/figs_new/fig_overview"
d = json.load(open(f"{D}/results/fig1_pair_qwen3vl8b.json"))
imgs = [Image.open(f"{D}/judgebench_local/{p}").convert("RGB") for p in d["images"]]
ORANGE, BLUE, GRAY, RED, GREEN = "#d9822b", "#2f6db5", "#6b6b6b", "#c0392b", "#2e8b57"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2})
fig = plt.figure(figsize=(7.0, 2.55))

# (a) the pair
fig.text(0.005, 0.955, "(a) A human tie: two failed edits", fontsize=8, weight="bold")
fig.text(0.005, 0.875, f"Instruction: \u201c{d['instruction']}\u201d", fontsize=6.8, style="italic")
for k, (im, lab) in enumerate(zip(imgs, ["Source", "Edit A", "Edit B"])):
    ax = fig.add_axes([0.005 + k * 0.108, 0.25, 0.104, 0.54]); ax.imshow(im.resize((256, 256))); ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_color("#999999"); sp_.set_linewidth(.6)
    ax.set_title(lab, fontsize=6.8, pad=2)
fig.text(0.005, 0.165, "People: tie \u2014 both edits rated 0 (failed)", fontsize=7, color=RED, weight="bold")
fig.text(0.005, 0.085, "ImagenHub editing; A: Pix2PixZero, B: SDEdit", fontsize=6.1, color=GRAY)

# (b) the judge's signals
x0 = 0.352
fig.text(x0, 0.955, "(b) The judge's signals", fontsize=8, weight="bold")
fig.text(x0, 0.875, "Qwen3-VL-8B, answer-token probabilities", fontsize=6.4, color=GRAY)
pa_ab, pc0_ba = d["forced_AB_P_A"], d["forced_BA_P_cand0"]
rows = [(0.775, "A/B, order A then B", f"picks B  {1 - pa_ab:.2f}", RED), (0.675, "A/B, order B then A", f"picks A  {pc0_ba:.2f}", RED),
        (0.475, "A/B/tie prompt", f"P(tie)  {d['tri_AB'][2]:.2f} / {d['tri_BA'][2]:.2f}", BLUE),
        (0.375, "1\u20139 score, alone", f"A {d['score_E'][0]:.1f}   B {d['score_E'][1]:.1f}", BLUE)]
for yy, lab, val, col in rows:
    fig.text(x0, yy, lab, fontsize=6.7); fig.text(x0 + 0.135, yy, val, fontsize=7, color=col, weight="bold")
fig.text(x0 + 0.004, 0.585, "\u2192 the \u201cwinner\u201d is the edit shown second", fontsize=6.2, color=RED, style="italic")
fig.text(x0, 0.13, "Position decides the forced choice;\nthe tie probability and the low\nscores carry the shared failure.",
         fontsize=6.2, color=GRAY, va="bottom")

# (c) certified outputs
x1 = 0.632; W = 0.363
fig.text(x1, 0.955, "(c) What a certified judge says, \u03b1 = .2", fontsize=8, weight="bold")
n = sum(d["graded_claims"].values())
fig.text(x1, 0.875, f"its output in the {n} test splits that contain this pair", fontsize=6.4, color=GRAY)
gc, sc, ab = d["graded_claims"], d["scope_rule"], d["absolute"]
boxes = [("Winner-only certificate (SCOPE / LEC rule)", f"\u201cB is better\u201d  {sc.get('answers B', 0)}/{n}", "\u2717 wrong: people tied", ORANGE, RED,
          "all benchmarks: 39\u201366% accurate at a promised 80%"),
         ("Graded claims (relative layer, ours)", f"\u201ctie\u201d  {gc.get('tie', 0)}/{n}", "\u2713 correct", BLUE, GREEN,
          "edit benchmarks: .95\u2013.97 of sources, 86% accurate"),
         ("Absolute layer (ours)", f"\u201creject both\u201d  {ab.get('reject both', 0)}/{n}, else escalate", "\u2713 correct", BLUE, GREEN,
          "a ship-only policy automates none of these sources")]
yb = 0.645
for title, out, verdict, col, vcol, foot in boxes:
    fig.patches.append(FancyBboxPatch((x1, yb - 0.04), W, 0.2, boxstyle="round,pad=0.004,rounding_size=0.01",
                                      transform=fig.transFigure, facecolor="white", edgecolor=col, linewidth=1.0))
    fig.text(x1 + 0.007, yb + 0.112, title, fontsize=6.6, color=col, weight="bold")
    fig.text(x1 + 0.007, yb + 0.045, out, fontsize=7, weight="bold")
    fig.text(x1 + W - 0.007, yb + 0.045, verdict, fontsize=6.6, color=vcol, weight="bold", ha="right")
    fig.text(x1 + 0.007, yb - 0.018, foot, fontsize=5.8, color=GRAY)
    yb -= 0.245
fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT)
