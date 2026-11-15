"""Main decision table with source-bootstrap intervals (descriptive, 2026-11-15).  Three settings (ImagenHub edit at the
midpoint bar, GenAI-Bench at the top-quarter bar, web-agent runs), four policies (ship only on all signals; ship or reject
on comparison, pointwise, all signals), alpha = .2, 100 source splits, the certified pipeline of arb_analysis_v3.run_cert.
Per source: outcomes averaged over the splits in which it is a test source and over judges; 95 % bootstrap over sources
(2,000 resamples).  Per 100 test sources: correct ships, correct rejects, wrong decisions, correct total.
usage: OMP_NUM_THREADS=1 python decision_ci.py <agent items dir> <agent raw dir> <out.json>"""
import json, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/media/shawn/C/papers/VLM/agent_reward_bench")
import arb_analysis_v3 as A, label_budget as LB
AQ, C = LB.AQ, LB.C
POL = [("both|ship", "Ship only"), ("pair|sr", "Ship or reject, comparison"), ("pw|sr", "Ship or reject, pointwise"), ("both|sr", "Ship or reject, all (ours)")]
IMG = [("ImagenHub edit", "imagenhub/edit", .5), ("GenAI-Bench, top quarter", "genaibench", .75)]


def img_D(a):
    """as run_cert, with missing pointwise scores imputed from the fit split of each split (as absolute_actions.py)"""
    import collections
    from absolute_actions import decide, fixed_seq
    spec, frac, j = a; st = LB.store(spec); ds, I = AQ.dataset_items(st, spec); S = C.signals(st, ds, j, I); ids = sorted(S)
    PWF = AQ.pointwise_feats(st, ds, j, ids)
    if len(ids) < .95 * len(I) or PWF is None: return None
    n = len(ids); f1 = np.array([S[i][0] for i in ids]); f2 = np.array([S[i][1] for i in ids]); p = (f1 + f2) / 2
    Q = (np.array([S[i][2] for i in ids]) + np.array([S[i][3] for i in ids])) / 2; cons = ((f1 > .5) == (f2 > .5)).astype(float)
    R = np.array([AQ.ratings(spec, I[i]) for i in ids], float); lo, hi = AQ.SCALE[spec.split("/")[0]]; acc = R >= lo + frac * (hi - lo) - 1e-9; anyok = acc[:, 0] | acc[:, 1]
    y4 = np.where(acc[:, 0] & acc[:, 1], 0, np.where(acc[:, 0], 1, np.where(acc[:, 1], 2, 3)))
    Fpair = np.c_[np.log(Q + 1e-9), C.H(p), np.abs(p - .5), cons]; T_ = np.log(np.maximum(Q[:, 2], 1e-9) / np.maximum(1 - Q[:, 2], 1e-9))
    ug, gi = np.unique(np.array([C.source_key(I[i]) for i in ids]), return_inverse=True)
    Acc = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(8))); rng = np.random.default_rng(0)
    for sp in range(100):
        fit, cal, te = AQ.splits(n, gi, {"unit": "source1", "split": {"fit": .25, "cal": .5}}, rng)
        pw = PWF.copy(); bad = np.isnan(pw)
        if bad.any(): pw[bad] = np.nanmean(PWF[fit], 0)[np.where(bad)[1]]
        _, inv_te, cnt_te = np.unique(gi[te], return_inverse=True, return_counts=True); src_te = ug[np.unique(gi[te])]; sw = lambda v: np.bincount(inv_te, v.astype(float)) / cnt_te
        F = {"pair": Fpair, "pw": pw, "both": np.c_[Fpair, pw], "B+T": np.c_[pw, T_]}
        for sig, aset in (("both", "ship"), ("pair", "sr"), ("pw", "sr"), ("both", "sr"), ("B+T", "sr")):
            q, _ = A.head4(F[sig], y4, fit); a_, s_ = decide(q[:, 0] + q[:, 1], q[:, 0] + q[:, 2], q[:, 3], aset)
            ship = a_ < 2; k = np.minimum(a_, 1); err = np.where(ship, ~acc[np.arange(n), k], anyok).astype(float); branch = np.where(ship, 0, 1)
            lam = fixed_seq(s_, err, branch, aset, .2, "all", cal, fit); ok = s_[te] <= lam
            gs, bs_ = sw(ship[te] & ok & (err[te] == 0)), sw(ship[te] & ok & (err[te] == 1)); gr, br = sw(~ship[te] & ok & (err[te] == 0)), sw(~ship[te] & ok & (err[te] == 1))
            for sid, a1, a2, a3, a4 in zip(src_te, gs, bs_, gr, br):
                Acc[f"{sig}|{aset}@0.2"][sid] += (a1, a2, a3, a4, a1 + a2 + a3 + a4, 1, 0, 0)
    return dict(A={k: dict(d) for k, d in Acc.items()}, pi=float((np.bincount(gi, (~anyok).astype(float)) / np.bincount(gi)).mean()))


def agent_D(a):
    items, raw, j = a; D, n_all, n_err = A.load(items, raw, j); print(j, "agent pairs", len(D["id"]), "of", n_all, "error rows", n_err, flush=True)
    return A.run_cert((D, np.ones(len(D["id"]), bool), 100, [.2], "agent"))


if __name__ == "__main__":
    items, raw, outp = sys.argv[1:4]; JI = LB.JUD; JA = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen3vl32b"]
    with Pool(14) as pool:
        r1 = pool.map_async(img_D, [(s, f, j) for _, s, f in IMG for j in JI], chunksize=1); r2 = pool.map_async(agent_D, [(items, raw, j) for j in JA], chunksize=1)
        O1, O2 = r1.get(), r2.get()
    rng = np.random.default_rng(0); out = {}
    sets = [(nm, [o for o in O1[k * len(JI):(k + 1) * len(JI)] if o]) for k, (nm, _, _) in enumerate(IMG)] + [("Web-agent runs", O2)]
    for nm, O in sets:
        T = [A.summarize(o, rng) for o in O]; out[nm] = dict(judges=len(O), pi=float(np.mean([o["pi"] for o in O])), policies={})
        for key, lab in POL:
            k = f"{key}@0.2"; common = sorted(set.intersection(*[set(t[k]["src"]) for t in T]))
            def vec(q): return np.mean([np.asarray(t[k][q])[[t[k]["src"].index(s) for s in common]] if len(common) != len(t[k]["src"]) else np.asarray(t[k][q]) for t in T], 0)
            gs, gr, bad = vec("good_ship"), vec("good_rej"), vec("bad_ship") + vec("bad_rej")
            out[nm]["policies"][lab] = dict(sources=len(common), correct=A.boot(gs + gr, rng), ship=A.boot(gs, rng), reject=A.boot(gr, rng), wrong=A.boot(bad, rng))
            p = out[nm]["policies"][lab]; print("%-26s %-28s correct %5.1f [%5.1f, %5.1f]  ship %5.1f  reject %5.1f  wrong %4.1f [%4.1f, %4.1f]  (sources %d, judges %d)" % (nm, lab, *p["correct"], p["ship"][0], p["reject"][0], *p["wrong"], len(common), len(O)), flush=True)
    # what the tie probability adds to the pointwise scores, by action (B+T|sr minus pw|sr)
    for nm, O in sets:
        T = [A.summarize(o, rng) for o in O]; common = sorted(set.intersection(*[set(t["pw|sr@0.2"]["src"]) for t in T]))
        def dv(q): return np.mean([(np.asarray(t["B+T|sr@0.2"][q]) - np.asarray(t["pw|sr@0.2"][q]))[[t["pw|sr@0.2"]["src"].index(s) for s in common]] for t in T], 0)
        out[nm]["tie_adds"] = dict(ship=A.boot(dv("good_ship"), rng), reject=A.boot(dv("good_rej"), rng), wrong=A.boot(dv("bad_ship") + dv("bad_rej"), rng))
        t_ = out[nm]["tie_adds"]; print("%-26s tie probability added to the scores: correct ships %+.1f [%+.1f, %+.1f]  correct rejects %+.1f [%+.1f, %+.1f]  wrong %+.1f [%+.1f, %+.1f]" % (nm, *t_["ship"], *t_["reject"], *t_["wrong"]), flush=True)
    json.dump(out, open(outp, "w"), indent=1)
