"""Addendum 22: the head-to-head table at alpha .1/.2/.3 from the finished certification runs, with each baseline family
at its most favourable variant.  Reads summaries only.  usage (repo root): python multi_alpha.py <out_dir>"""
import sys, os, json
import numpy as np
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
src_code = open("/media/shawn/C/papers/EAGLE-VLM/paper_figures/make_tie_paper_assets.py").read()
M = {"__name__": "mtpa"}; sys.argv = ["mtpa"]; exec(compile(src_code, "mtpa", "exec"), M)
rows, latest, jm, f, toe_rows, HEAD_ROWS, keep_ds, TIEAWARE = (M[k] for k in ("rows", "latest", "jm", "f", "toe_rows", "HEAD_ROWS", "keep_ds", "TIEAWARE"))
src = {"main": ["certify_source", "certify_baselines", "certify_baselines_pw"], "new": ["newdata_certify_source", "newdata_certify_baselines"],
       "bon_mmmu": ["downstream_bon_mmmu", "downstream_bon_mmmu_baselines"], "bon": ["downstream_bon", "downstream_bon_baselines"]}
cache = {}
def load(name):
    if name not in cache:
        p = latest(name) + "/summary.csv"; cache[name] = rows(p) if os.path.exists(p) else []
    return cache[name]
TR = toe_rows(); res = {}
def best(vs, al):
    """vs: list of (label, cov, risk).  Most favourable variant: highest coverage among those that keep the promise; else the most accurate."""
    vs = [v for v in vs if v[1] == v[1]]
    ok = [v for v in vs if v[1] >= .005 and (v[2] != v[2] or v[2] <= al + 1e-9)]
    if ok: v = max(ok, key=lambda x: (x[1], -(x[2] if x[2] == x[2] else 1))); return dict(variant=v[0], cov=v[1], risk=v[2], keeps=True)
    an = [v for v in vs if v[1] >= .005 and v[2] == v[2]]
    if an: v = min(an, key=lambda x: x[2]); return dict(variant=v[0], cov=v[1], risk=v[2], keeps=False)
    return dict(variant=None, cov=0.0, risk=float("nan"), keeps=None)        # answers nothing
for al in (.1, .2, .3):
    def stat(R, ds, m):
        c, n = jm(R, ds, m, "source_coverage", al); r = jm(R, ds, m, "pooled_risk_source", al)[0]; return c, r, n
    def fs(ds, m):
        R = [r for r in TR if r["dataset"] == ds and r["method"] == m and abs(f(r["alpha"]) - al) < 1e-9]
        if not R: return None
        rk = [f(r["pooled_risk_source"]) for r in R]; rk = [x for x in rk if x == x]
        return float(np.mean([f(r["source_coverage"]) for r in R])), (float(np.mean(rk)) if rk else float("nan")), len(R)
    for ds, nm, kind in HEAD_ROWS:
        if not keep_ds(ds): continue
        Rc = load(src[kind][0]); g2 = stat(Rc, ds, "G2")
        if g2[2] == 0: continue
        d = res.setdefault(nm, dict(dataset=ds, judges=g2[2], tie_rate=float(np.mean([f(r["human_tie"]) for r in Rc if r["dataset"] == ds and r["method"] == "G2"])), alpha={}))
        sc = [("ties dropped",) + stat(Rc, ds, "P1")[:2], ("ties kept",) + stat(Rc, ds, "P2")[:2]]
        toe = [(lab,) + v[:2] for lab, m in (("personas, ties dropped", "T1"), ("personas, ties kept", "T2"), ("few-shot, ties dropped", "T4"), ("few-shot, ties kept", "T5")) for v in [fs(ds, m)] if v]
        ta = [(m,) + stat(load(name), ds, m)[:2] for m in TIEAWARE for name in src[kind] if stat(load(name), ds, m)[2]]
        od = [(m,) + stat(Rc, ds, m)[:2] for m in ("SAo", "SLo") if stat(Rc, ds, m)[2]]
        g5 = stat(Rc, ds, "G5")
        d["alpha"][str(al)] = dict(scope_variants=sc, toe_variants=toe, scope=best(sc, al), toe=best(toe, al), tieaware=best(ta, al), ordinal=best(od, al),
                                   ours=dict(cov=g2[0], risk=g2[1], keeps=bool(g2[1] != g2[1] or g2[1] <= al + 1e-9)), strict_safe=dict(cov=g5[0], risk=g5[1]))
json.dump(res, open(f"{OUT}/multi_alpha.json", "w"), indent=1, default=float)
fm = lambda x: "  –  " if x is None or x != x else f"{x:.2f}"
L = ["# Addendum 22: coverage / accuracy (1 − risk) at three targets; † = promise broken; each baseline at its most favourable variant\n"]
for al in ("0.1", "0.2", "0.3"):
    L += [f"\n## alpha = {al}\n", "| benchmark (judges) | ties | SCOPE/LEC | Trust-or-Esc. | tie-aware, no weak claim | ordinal sets | ours | strict-safe |", "|---|---|---|---|---|---|---|---|"]
    for nm, d in res.items():
        a = d["alpha"][al]; cell = lambda v: "answers nothing" if v["keeps"] is None else f"{fm(v['cov'])} / {fm(1 - v['risk'])}{'' if v['keeps'] else '†'} ({v['variant']})"
        o = a["ours"]; s = a["strict_safe"]
        L.append(f"| {nm} ({d['judges']}) | {d['tie_rate']:.2f} | {cell(a['scope'])} | {cell(a['toe'])} | {cell(a['tieaware'])} | {cell(a['ordinal'])} | {fm(o['cov'])} / {fm(1 - o['risk'])}{'' if o['keeps'] else '†'} | {fm(s['cov'])} / {fm(1 - s['risk'])} |")
open(f"{OUT}/multi_alpha.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
