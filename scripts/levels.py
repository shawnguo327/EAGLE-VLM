"""Where does a VLM judge fail on a pair that differs in ONE visual fact?  (make_synth_levels.py sets)
For every DV pair (correct R vs W1 = R with one visual phrase replaced):
  L1 acquisition : synth_perc/e1 — does the judge pick the true sentence (order-averaged p > .5)?  blind = language prior
  L3 verdict     : synth_tie DV  — bin verdict correct?   P(T) under the tie prompt
  L2 localised   : synth_loc DV  — verdict correct when the differing phrases are pointed out?
Attribution:  acq  = L1 wrong;  ok = L1 right & L3 right;  loc = L1 right & L3 wrong & L2 right;  wt = L1 right & L2 wrong & L3 wrong
TB pairs (two different single-fact errors): if the judge perceives both facts (L1 right on e1 and e2) the pair is
'equally bad' to it; does P(T) rise, before and after localisation?
usage (from $EAGLEVLM_DATA): python levels.py --judges a,b --out paper_tables/levels.md"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load
from eaglev_stats import auroc as _auroc


def auroc(x, y):
    return _auroc(x, y) if len(y) and y.min() != y.max() else float('nan')


def mn(x):
    return float(np.nanmean(x)) if np.size(x) and not np.all(np.isnan(x)) else float('nan')

ap = argparse.ArgumentParser()
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen25vl7b,qwen35_9b,r1reward,llavaov7b,llavacritic7b,mmrlhfrm7b")
ap.add_argument("--out", default="paper_tables/levels.md")
a = ap.parse_args()


def p0(r, c):
    """order-averaged P(cand0 better) and P(T) for condition c, or None"""
    if (c, "AB") not in r or (c, "BA") not in r:
        return None
    ab, ba = r[(c, "AB")], r[(c, "BA")]
    p = (ab["A"] / (ab["A"] + ab["B"]) + ba["B"] / (ba["A"] + ba["B"])) / 2
    return p, (ab.get("T", 0) + ba.get("T", 0)) / 2


IP, IT, IL = items("synth_perc"), items("synth_tie"), items("synth_loc")
L = ["# Levels of judge failure on single-fact pairs", "",
     "acc = order-averaged choice correct.  P(T) from the tie prompt.  Attribution over DV pairs (shares).", "",
     "| judge | L1 perception acc (img / blind) | L3 verdict acc DV | L2 localised acc DV | share acq / ok / loc / wt | "
     "DV P(T): acq vs perceived | AUROC P(T)→acq | AUROC U→acq | TB P(T) (both perceived): orig → localised | TG P(T) |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for j in a.judges.split(","):
    RP, RT, RL = load("synth_perc", j), load("synth_tie", j), load("synth_loc", j)
    if not RP or not RT:
        continue
    perc, pblind = {}, {}
    for i, r in RP.items():
        if i not in IP:
            continue
        it = IP[i]; base = it["meta"]["base"]; e = it["subset"]
        for c, D in (("bin", perc), ("bin_blind", pblind)):
            x = p0(r, c)
            if x:
                pc = x[0] if it["label"] == 0 else 1 - x[0]          # P(true sentence)
                D[(base, e)] = pc
    rows = []
    for i, r in RT.items():
        it = IT.get(i)
        if not it or it["subset"] != "DV":
            continue
        b = it["meta"]["base"]
        xb, xt = p0(r, "bin"), p0(r, "tri")
        rl = RL.get(f"synth_loc/DV/{b}", {})
        lb, lt = p0(rl, "bin"), p0(rl, "tri")
        if xb is None or (b, "e1") not in perc:
            continue
        good = lambda p: (p > .5) == (it["label"] == 0)
        pc = perc[(b, "e1")]
        H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))
        rows.append(dict(acq=pc <= .5, pc=pc, v=good(xb[0]), loc=None if lb is None else good(lb[0]),
                         pt=None if xt is None else xt[1], u=H(xb[0])))
    if not rows:
        continue
    n = len(rows)
    acq = np.array([x["acq"] for x in rows]); v = np.array([x["v"] for x in rows])
    lc = np.array([np.nan if x["loc"] is None else x["loc"] for x in rows], float)
    sh_acq = acq.mean(); sh_ok = (~acq & v).mean()
    sh_loc = np.nanmean(np.where(~acq & ~v, lc == 1, np.nan)) * (~acq & ~v).mean() if (~acq & ~v).any() else 0
    sh_wt = (~acq & ~v).mean() - sh_loc
    pt = np.array([np.nan if x["pt"] is None else x["pt"] for x in rows]); u = np.array([x["u"] for x in rows])
    m = ~np.isnan(pt)
    tb = []
    for i, r in RT.items():
        it = IT.get(i)
        if it and it["subset"] == "TB":
            b = it["meta"]["base"]
            if perc.get((b, "e1"), 0) > .5 and perc.get((b, "e2"), 0) > .5:
                xo = p0(r, "tri"); xl = p0(RL.get(f"synth_loc/TB/{b}", {}), "tri")
                if xo and xl:
                    tb.append((xo[1], xl[1]))
    tg = [p0(r, "tri")[1] for i, r in RT.items() if IT.get(i, {}).get("subset") == "TG" and p0(r, "tri")]
    pb = np.mean([x > .5 for x in pblind.values()]) if pblind else np.nan
    L.append(f"| {j} (n={n}) | {np.mean([x > .5 for x in perc.values()]):.3f} / {pb:.3f} | {v.mean():.3f} | {mn(lc):.3f} | "
             f"{sh_acq:.2f} / {sh_ok:.2f} / {sh_loc:.2f} / {sh_wt:.2f} | "
             f"{mn(pt[acq & m]):.3f} vs {mn(pt[~acq & m]):.3f} | "
             f"{auroc(pt[m], acq[m].astype(int)):.3f} | {auroc(u, acq.astype(int)):.3f} | "
             f"{np.mean([x[0] for x in tb]) if tb else float('nan'):.3f} → {np.mean([x[1] for x in tb]) if tb else float('nan'):.3f} (n={len(tb)}) | "
             f"{np.mean(tg) if tg else float('nan'):.3f} |")
open(a.out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
