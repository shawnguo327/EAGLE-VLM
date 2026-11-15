"""Review 22, figure 3 (DRAFT, 2026-11-08; not in the paper until the authors decide): coverage must be read together with
claim specificity.  Top row: at alpha = .2, the share of test pairs given each claim type (strict {k}, tie {T}, weak
{k,T}) by every output family certified with the same source-level LTT protocol (six open judges; tie_claims J6 and
weak_cap): graded (ours), strict-safe graded (ours), the restricted ordinal minimum set, filtered ordinal APS, singletons
{A}/{B}/{T}, a fit-learned tie band on the forced-choice margin and on the pointwise score gap, weak-only, constant tie and
the fit-split label prior; the text gives the source-level coverage and pooled risk.  Bottom row: a predeclared cap rho on
the weak share AMONG ANSWERED pairs, certified jointly with the claim risk on independent calibration sources
(intersection-union; rho = 0: singletons, rho = 1: no cap): total, strict and tie coverage as rho falls.
usage (repo root): python paper_figures/fig_claim_composition.py --out <dir> --tc <tie_claims_summary.csv>
                   --wc <weak_cap_summary.csv> --wc2 <weak_cap_summary.csv with caps .25/.5/.75>"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRIDC, BG = "#1a1a1a", "#52514e", "#e1e0d9", "#fbfbf8"
C_STRICT, C_TIE, C_WEAK = "#2a78d6", "#7d5bc4", "#9dc0ec"
DATA = [("imagenhub/edit", "ImagenHub edit"), ("genai_arena/edition", "GenAI-Arena edit"), ("genaibench", "GenAI-Bench"),
        ("bon_scienceqa", "ScienceQA Best-of-8 (base-rate check)")]
METH = [("G2", "Graded (ours)"), ("G5", "Strict-safe (ours)"), ("ORD", "Ordinal minimum set"), ("SAo", "Filtered ordinal APS"),
        ("G2s", "Singletons A / B / T"), ("TCL", "Tie band, margin"), ("PWTL", "Tie band, pointwise"), ("WO", "Weak only"),
        ("CT", "Constant tie"), ("FMJ", "Fit label prior")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
f = lambda x: float(x) if x not in ("", "nan", None) else np.nan

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--out", "--tc", "--wc", "--wc2"):
        ap.add_argument(k, required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    T = {(r["dataset"], r["method"]): r for r in csv.DictReader(open(a.tc)) if abs(f(r["alpha"]) - .2) < 1e-9}
    W = {}
    for p in (a.wc, a.wc2):
        for r in csv.DictReader(open(p)):
            if abs(f(r["alpha"]) - .2) < 1e-9:
                W[(r["dataset"], r["method"])] = r
    fig = plt.figure(figsize=(15, 8.6))
    top = fig.add_gridspec(1, 4, left=.13, right=.985, top=.93, bottom=.47, wspace=.08)
    bot = fig.add_gridspec(1, 4, left=.13, right=.985, top=.33, bottom=.08, wspace=.25)
    rows_out = []
    for k, (ds, nm) in enumerate(DATA):
        ax = fig.add_subplot(top[0, k]); y = np.arange(len(METH))[::-1]
        for yi, (m, lab) in zip(y, METH):
            if m == "WO":
                r = W.get((ds, "WO"))
                if r is None:
                    continue
                st_, ti_, wk_, cov, risk = 0.0, 0.0, f(r["weak"]), f(r["coverage"]), f(r["risk"])
            else:
                r = T.get((ds, m))
                if r is None:
                    continue
                st_, ti_, wk_, cov, risk = f(r["strict_share"]), f(r["tie_share"]), f(r["weak_share"]), f(r["coverage"]), f(r["risk"])
            rows_out.append(dict(dataset=ds, method=m, strict=st_, tie=ti_, weak=wk_, coverage=cov, risk=risk))
            left = 0
            for v, c in ((st_, C_STRICT), (ti_, C_TIE), (wk_, C_WEAK)):
                v = 0 if v != v else v; ax.barh(yi, v, left=left, color=c, height=.62, edgecolor=BG, linewidth=1.1); left += v
            txt = "no answer" if cov != cov or cov < .005 else f"{cov:.2f}" + ("" if risk != risk else f"  risk {risk:.2f}")
            ax.text(min(left, 1) + .02, yi, txt, va="center", fontsize=7.6, color=INK2)
        ax.set_xlim(0, 1.45); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in METH] if k == 0 else []); ax.grid(axis="x", color=GRIDC); ax.set_axisbelow(True)
        hum = f(T[(ds, "G2")]["human_tie"]) if (ds, "G2") in T else np.nan
        ax.set_title(f"{nm}\nhuman ties {hum:.2f}", fontsize=9.5, color=INK, loc="left"); ax.set_xlabel("share of test pairs")
        bx = fig.add_subplot(bot[0, k]); rho = [0, .25, .5, .75, 1]; tot, stc, tic = [], [], []
        for r_ in rho:
            if r_ == 0:
                r = T.get((ds, "G2s")); g = (lambda key: f(r[key])) if r else None
                vals = (f(r["coverage"]), f(r["strict_share"]), f(r["tie_share"])) if r else (np.nan,) * 3
            elif r_ == 1:
                r = W.get((ds, "G2")); vals = (f(r["coverage"]), f(r["strict"]), f(r["tie"])) if r else (np.nan,) * 3
            else:
                r = W.get((ds, f"WC@{r_}")); vals = (f(r["coverage"]), f(r["strict"]), f(r["tie"])) if r else (np.nan,) * 3
            tot.append(vals[0]); stc.append(vals[1]); tic.append(vals[2])
            rows_out.append(dict(dataset=ds, method=f"cap rho={r_}", strict=vals[1], tie=vals[2], weak=np.nan, coverage=vals[0], risk=np.nan))
        bx.plot(rho, tot, "-o", color=INK, ms=5, lw=1.6, label="total coverage")
        bx.plot(rho, stc, "-s", color=C_STRICT, ms=5, lw=1.6, label="strict claims")
        bx.plot(rho, tic, "-^", color=C_TIE, ms=5, lw=1.6, label="tie claims")
        bx.set_xticks(rho); bx.set_xticklabels(["0\n(singletons)", ".25", ".5", ".75", "1\n(no cap)"]); bx.set_ylim(-.02, 1.02)
        bx.grid(color=GRIDC); bx.set_axisbelow(True); bx.set_title(nm, fontsize=10, color=INK, loc="left")
        bx.set_xlabel("cap on the weak share among answered pairs"); bx.set_ylabel("share of test pairs" if k == 0 else "")
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_STRICT, C_TIE, C_WEAK)]
    fig.legend(hs, ["strict claim {k}", "tie claim {T}", "weak claim {k, T}"], loc="upper center", bbox_to_anchor=(.56, .415),
               ncol=3, frameon=False, fontsize=8.5)
    fig.axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"fig3_claim_composition.{ext}"), dpi=170, bbox_inches="tight", facecolor=BG)
    with open(os.path.join(a.out, "fig3_claim_composition.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    print("wrote", a.out)
