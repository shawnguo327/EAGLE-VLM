"""Certified triage with typed human ties (branch B pilot, 2026-11-07; descriptive pilot, not yet a paper table).

Pipeline actions for a pair (two candidates for one source): ship A, ship B, reject both, or escalate (abstain).
Typed human ground truth y4: A better / B better / tie with both acceptable (TG) / both bad (TB):
  arena votes (GenAI-Arena, VisionArena): leftvote / rightvote / tievote / bothbad_vote
  ImagenHub (ratings in [0, 1]): decided pairs by the mean rating; a tie with both means <= .25 is TB, any other tie TG
Losses (fixed before the results):
  selection  pick one of the two; wrong only if the human preferred the other (a tie of either type is never an error)
  ship       ship k is wrong if the human preferred the other OR said both bad; 'reject both' is wrong unless both bad
Action spaces, all from the same 4-class calibrator (multinomial logistic on the fit split, standardised features), the same
fixed-sequence LTT on one pair per calibration source (certify.ltt, delta .1), and the same test sources:
  BIN  ship A / ship B only (the claim space of tie-dropping certified judges, here calibrated with every pair)
  T3   three human classes (A, B, tie) as in a Ties Matter-style judge: ship A / ship B / a tie action that the fit split
       maps to 'reject both' when most fit ties are both-bad (a 3-way judge cannot tell a both-good tie from a both-bad one)
  T4   typed (GenAI-Arena's 4-way label space): ship A / ship B / reject both, a both-good tie credited to either ship
  action = the one with the highest estimated probability of being correct; confidence score = 1 - that probability
Feature sets for T4 (signal ablation): pairwise (A/B and A/B/T prompts, both orders), pointwise (1-9 scores), both.
Reference: SCOPE as published (P1: forced-choice BPE, threshold on non-tie calibration pairs) scored with the same losses.
Per method / alpha: automated share = mean over test sources of the accepted fraction; pooled source risk over 100 splits;
share of splits whose test risk exceeds alpha; action mix; error among ship and among reject actions.
usage (repo root): python scripts/typed_triage.py --out <dir> [--splits 100] [--workers 16]"""
import argparse, ast, collections, csv, os, sys
from multiprocessing import Pool
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eaglevlm.pipeline.run import load_cfg                                   # noqa: E402
from eaglevlm.pipeline.io import Store, dataset_items                        # noqa: E402
from eaglevlm.pipeline.certify import signals, source_key, pointwise_feats, ltt, thr, H   # noqa: E402
from eaglevlm.pipeline.router import splits                                  # noqa: E402

SPECS = ["genai_arena/edition", "imagenhub/edit", "genai_arena/generation", "visionarena"]
JUDGES = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b", "internvl35_38b", "r1reward", "llavacritic7b"]
AL, DELTA = [.1, .2, .3], .1
A_, B_, TG, TB = 0, 1, 2, 3


def typed_label(spec, x):
    lab = str(x["label"])
    if lab in ("0", "1"):
        return int(lab)
    m = x.get("meta", {}) or {}
    if "vote_type" in m:
        return TB if m["vote_type"] == "bothbad_vote" else TG
    sc = [float(s) for s in (ast.literal_eval(m["scores"]) if isinstance(m["scores"], str) else m["scores"])]
    return TB if max(sc) <= .25 else TG


def probs(F, y, fit, ncls):
    sc = StandardScaler().fit(F[fit]); m = LogisticRegression(max_iter=5000).fit(sc.transform(F[fit]), y[fit])
    P = np.zeros((len(y), ncls)); P[:, m.classes_] = m.predict_proba(sc.transform(F)); return P


def policy(kind, P4, P3, qtb):
    """-> action (0 ship A, 1 ship B, 3 reject), confidence score (lower = more confident)"""
    if kind == "BIN":
        c = np.c_[P4[:, A_] + P4[:, TG], P4[:, B_] + P4[:, TG]]
    elif kind == "T4":
        c = np.c_[P4[:, A_] + P4[:, TG], P4[:, B_] + P4[:, TG], P4[:, TB]]
    else:                                   # T3: the tie action is 'reject' only if most fit ties are both-bad
        c = np.c_[P3[:, 0] + P3[:, 2] * (1 - qtb), P3[:, 1] + P3[:, 2] * (1 - qtb), P3[:, 2] * qtb if qtb >= .5 else np.zeros(len(P3))]
    a = c.argmax(1); return np.where(a == 2, 3, a), 1 - c.max(1)


def losses(act, y4):
    ship = np.where(act == 3, (y4 != TB), (y4 == 1 - np.minimum(act, 1)) | (y4 == TB))      # 'ship' loss
    sel = np.where(act == 3, np.nan, (y4 == 1 - np.minimum(act, 1)).astype(float))           # 'selection' loss
    return ship.astype(float), sel


def cell(args):
    spec, j, n_splits, P = args
    st = Store(P["items"], P["outputs"], P.get("items_override"), P.get("snapshot"))
    ds, I = dataset_items(st, spec)
    S = signals(st, ds, j, I); ids = sorted(S)
    if len(ids) < 200 or len(ids) < .95 * len(I):
        return [], f"skip {spec} {j}: {len(ids)} / {len(I)} items with both prompts in both orders"
    f1 = np.array([S[i][0] for i in ids]); f2 = np.array([S[i][1] for i in ids])
    Q = (np.array([S[i][2] for i in ids]) + np.array([S[i][3] for i in ids])) / 2; p = (f1 + f2) / 2
    cons = ((f1 > .5) == (f2 > .5)).astype(float)
    Fpair = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), cons]
    PWF = pointwise_feats(st, ds, j, ids)
    if PWF is not None and np.isnan(PWF).any(1).mean() > .05:
        PWF = None
    y4 = np.array([typed_label(spec, I[i]) for i in ids]); y3 = np.minimum(y4, 2)
    gi = np.unique(np.array([source_key(I[i]) for i in ids]), return_inverse=True)[1]
    fc = np.where(p > .5, 0, 1); u = H(p)
    feats = {"pairwise": Fpair}
    if PWF is not None:
        feats["pointwise"] = PWF; feats["both"] = None
    rng = np.random.default_rng(0); out = []
    for sp in range(n_splits):
        fit, cal, te = splits(len(ids), gi, {"unit": "source1", "split": {"fit": .25, "cal": .5}}, rng)
        Fs = {}
        if PWF is not None:
            pw = PWF.copy(); bad = np.isnan(pw)
            if bad.any():
                pw[bad] = np.nanmean(PWF[fit], 0)[np.where(bad)[1]]
            Fs = {"pointwise": pw, "both": np.c_[Fpair, pw]}
        Fs["pairwise"] = Fpair
        tie_fit = fit[y4[fit] >= TG]; qtb = float((y4[tie_fit] == TB).mean()) if len(tie_fit) else 0.0
        pols = []
        for fs in Fs:
            P4 = probs(Fs[fs], y4, fit, 4); P3 = probs(Fs[fs], y3, fit, 3)
            pols += [(f"T4[{fs}]", *policy("T4", P4, None, qtb)), (f"T3[{fs}]", *policy("T3", None, P3, qtb)),
                     (f"BIN[{fs}]", *policy("BIN", P4, None, qtb))]
        for al in AL:
            calnt = cal[y3[cal] != 2]           # SCOPE as published: calibrate on decided pairs only, ship the forced choice
            lam1 = thr(u[calnt], (fc != y3)[calnt].astype(float), al)
            cand = [("P1 (SCOPE as published)", fc, u, lam1)]
            for name, act, s in pols:
                for loss in ("ship", "selection"):
                    if loss == "selection" and not name.startswith("BIN"):
                        continue
                    e = losses(act, y4)[0 if loss == "ship" else 1]
                    lam = ltt(s[cal][:, None], e[cal][:, None], al, DELTA, s[fit][:, None], e[fit][:, None])
                    cand.append((f"{name}|{loss}", act, s, lam))
            for name, act, s, lam in cand:
                ok = s[te] <= lam; act_t = act[te]; y_t = y4[te]
                ship_e, sel_e = losses(act_t, y_t)
                for loss in (("ship", "selection") if name.startswith("P1") else (name.split("|")[1],)):
                    e = ship_e if loss == "ship" else np.nan_to_num(sel_e)
                    _, inv, cnt = np.unique(gi[te], return_inverse=True, return_counts=True)
                    ag = np.bincount(inv, ok.astype(float)) / cnt; eg = np.bincount(inv, (e * ok)) / cnt
                    rej = ok & (act_t == 3); shp = ok & (act_t != 3)
                    out.append(dict(dataset=spec, judge=j, method=name.split("|")[0], loss=loss, alpha=al, split=sp,
                                    automated=float(ag.mean()), acc_mass=float(ag.sum()), err_mass=float(eg.sum()),
                                    reject_share=float(np.bincount(inv, rej.astype(float)).sum() / len(te)),
                                    ship_share=float(np.bincount(inv, shp.astype(float)).sum() / len(te)),
                                    ship_err=float(e[shp].mean()) if shp.any() else np.nan,
                                    reject_err=float(e[rej].mean()) if rej.any() else np.nan,
                                    n=len(ids), tb_rate=float((y4 == TB).mean()), tg_rate=float((y4 == TG).mean())))
    return out, f"done {spec} {j}: {len(ids)} pairs, pointwise {'yes' if PWF is not None else 'no'}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--only", default="")
    a = ap.parse_args()
    paths = load_cfg("configs/exp/certify_source.yaml")["paths"]
    assert paths == load_cfg("configs/exp/newdata_certify_source.yaml")["paths"]
    specs = [s for s in SPECS if not a.only or s in a.only.split(",")]
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        res = pool.map(cell, [(s, j, a.splits, paths) for s in specs for j in JUDGES], chunksize=1)
    rows = [r for rr, _ in res for r in rr]
    with open(f"{a.out}/typed_triage_splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    G = collections.defaultdict(list)
    for r in rows:
        G[(r["dataset"], r["loss"], r["alpha"], r["method"], r["judge"])].append(r)
    S = collections.defaultdict(list)          # per judge: mean automated share, pooled source risk, violation share
    for (ds, loss, al, m, j), rr in G.items():
        acc = sum(r["acc_mass"] for r in rr); err = sum(r["err_mass"] for r in rr)
        rk = [r["err_mass"] / r["acc_mass"] for r in rr if r["acc_mass"] > 0]
        S[(ds, loss, al, m)].append(dict(judge=j, automated=np.mean([r["automated"] for r in rr]), risk=err / acc if acc else np.nan,
                                         viol=np.mean([x > al for x in rk]) if rk else np.nan,
                                         reject=np.mean([r["reject_share"] for r in rr]), ship=np.mean([r["ship_share"] for r in rr]),
                                         reject_err=np.nanmean([r["reject_err"] for r in rr]) if any(not np.isnan(r["reject_err"]) for r in rr) else np.nan,
                                         ship_err=np.nanmean([r["ship_err"] for r in rr]) if any(not np.isnan(r["ship_err"]) for r in rr) else np.nan))
    L = ["# Certified triage with typed ties (pilot)", "", __doc__.split("usage")[0].strip(), "",
         "Mean over judges (min-max over judges).  automated = share of test pairs auto-resolved (source-weighted); risk = "
         "pooled source risk over splits (worst judge in brackets); viol = share of splits whose test risk exceeds alpha "
         "(mean over judges); reject / ship = share of all test pairs; errors among the accepted ship / reject actions.", ""]
    for ds in specs:
        L += [f"## {ds}", "", "| loss | alpha | method | J | automated | risk [worst] | viol | ship | reject | ship err | reject err |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for (d, loss, al, m), v in sorted(S.items(), key=lambda k: (k[0][1] != "ship", k[0][2], k[0][3])):
            if d != ds:
                continue
            g = lambda k: np.array([x[k] for x in v], float)
            au = g("automated"); rk = g("risk")
            L.append(f"| {loss} | {al} | {m} | {len(v)} | {au.mean():.3f} ({au.min():.3f}-{au.max():.3f}) | {np.nanmean(rk):.3f} [{np.nanmax(rk) if (~np.isnan(rk)).any() else float('nan'):.3f}] | "
                     f"{np.nanmean(g('viol')):.2f} | {g('ship').mean():.3f} | {g('reject').mean():.3f} | {np.nanmean(g('ship_err')):.3f} | {np.nanmean(g('reject_err')) if (~np.isnan(g('reject_err'))).any() else float('nan'):.3f} |")
        L.append("")
    L += ["## Log", ""] + [f"- {m}" for _, m in res]
    open(f"{a.out}/typed_triage.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
