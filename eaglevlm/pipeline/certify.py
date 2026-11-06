"""Certification experiment ('certify'): published tie-dropping protocols -> our graded claims, on the same splits.
Ported unchanged from scripts/chain.py (P1..G3) and scripts/c3_diag.py (G4, G5, significance-corrected violation).
Every method is deployed on ALL test pairs (ties included); a tie is an error for a 'k better' claim, a non-tie is an
error for a 'tie' claim, and only the opposite verdict is an error for a weak 'k not worse' claim.
Methods:
  P1  SCOPE as published (BPE of order-averaged forced choice; calibration WITHOUT ties; SCOPE linearised threshold)
  P1s same, single order          P2  SCOPE, ties kept in calibration       P3  strict + LTT (BPE), ties kept
  P4  strict + LTT with all our signal components and the 3-class calibrator
  T1  Trust-or-Escalate as published (4 simulated annotators; calibration without ties)   T2  ToE, ties kept
  G0  graded, no calibrator       G1  graded + calibrator, single-order features      G2  graded + calibrator (ours)
  G3  graded, every claim level <= alpha (intersection-union)
  G4  graded, overall AND strict-claim risk <= alpha, one threshold
  G5  graded, overall AND strict-claim risk <= alpha, separate strict / tie-weak thresholds (Bonferroni over 10 strict values)
  G2pw / G5pw  as G2 / G5 with the judge's pointwise (single-candidate 1-9 score) features added to the calibrator
  judge "jury:<m1>+<m2>+...": the calibrator sees the features of all member judges (pairs judged by every member);
  P-type baselines use the members' averaged probabilities; T methods are not defined for a jury
Per split and alpha one CSV row: acceptance, mean set size, decisive share, strict errors / strict claims, accepted,
errors, risk, violated (risk > alpha), significant (one-sided binomial p < .05 that risk > alpha), empty."""
import collections
import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression

GRID = np.linspace(0, 1, 201)[1:]
LS = np.linspace(.04, .40, 10)
SA = ["tri_sa1", "tri_sa2", "tri_sa3", "tri_sa4"]
H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))


# ---------------- thresholds (scripts/label_noise_lib.py)
def thr(s, e, al):
    """SCOPE linearised threshold, tie-aware (checked at the last position of each distinct score)"""
    o = np.argsort(s, kind="stable"); ss = s[o]; cs = np.cumsum(e[o] - al)
    last = np.r_[ss[1:] != ss[:-1], True]; ok = np.where((cs <= -1) & last)[0]
    return ss[ok.max()] if len(ok) else -np.inf


def claims(S, lam):
    ok = S <= lam; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def _pval(S, E, lam, al):
    acc, ch = claims(S, lam); n = int(acc.sum())
    if n == 0:
        return 1.0
    return binom.cdf(int(E[np.arange(len(E)), np.maximum(ch, 0)][acc].sum()), n, al)


def ltt(S, E, al, delta, Sf=None, Ef=None, grid=GRID):
    """fixed-sequence Learn-then-Test; start = grid value with the smallest p-value on the FIT split"""
    s0 = int(np.argmin([_pval(Sf, Ef, lam, al) for lam in grid])) if Sf is not None else 0
    best = -np.inf
    for lam in grid[s0:]:
        if _pval(S, E, lam, al) <= delta:
            best = lam
        else:
            break
    return best


def pjoint(S, E, lam, al):
    ok, ch = claims(S, lam); ps = [0.0]
    for c in range(S.shape[1]):
        mk = ok & (ch == c)
        if mk.any():
            ps.append(binom.cdf(int(E[mk, c].sum()), int(mk.sum()), al))
    return max(ps) if ok.any() else 1.0


def pstrict(S, E, lam, al):
    ok, ch = claims(S, lam)
    if not ok.any():
        return 1.0
    e = E[np.arange(len(E)), np.maximum(ch, 0)]
    ps = [binom.cdf(int(e[ok].sum()), int(ok.sum()), al)]
    mk = ok & (ch == 0)
    if mk.any():
        ps.append(binom.cdf(int(E[mk, 0].sum()), int(mk.sum()), al))
    return max(ps)


def ltt_p(Sc, Ec, Sf, Ef, al, delta, pf):
    s0 = int(np.argmin([pf(Sf, Ef, g, al) for g in GRID])); best = -np.inf
    for g in GRID[s0:]:
        if pf(Sc, Ec, g, al) <= delta:
            best = g
        else:
            break
    return best


def claims2(S, ls, lw):
    ok = np.c_[S[:, 0] <= ls, S[:, 1] <= lw, S[:, 2] <= lw]; anyok = ok.any(1); return anyok, np.where(anyok, ok.argmax(1), -1)


def p2(S, E, ls, lw, al):
    ok, ch = claims2(S, ls, lw)
    if not ok.any():
        return 1.0
    e = E[np.arange(len(E)), np.maximum(ch, 0)]
    ps = [binom.cdf(int(e[ok].sum()), int(ok.sum()), al)]
    mk = S[:, 0] <= ls
    if mk.any():
        ps.append(binom.cdf(int(E[mk, 0].sum()), int(mk.sum()), al))
    return max(ps)


def ltt_two(Sc, Ec, Sf, Ef, al, delta):
    best, bacc = (-np.inf, -np.inf), -1.0
    for ls in LS:
        s0 = int(np.argmin([p2(Sf, Ef, ls, g, al) for g in GRID])); lw = None
        for g in GRID[s0:]:
            if p2(Sc, Ec, ls, g, al) <= delta / len(LS):
                lw = g
            else:
                break
        if lw is not None:
            acc = claims2(Sc, ls, lw)[0].mean()
            if acc > bacc:
                best, bacc = (ls, lw), acc
    return best


# ---------------- signals
def signals(store, ds, j, I):
    """per id: forced-choice P(cand 0 better) per order, tie-prompt probs per order (aligned to cand 0 / cand 1 / T)"""
    R = store.load(ds, j, conds={"bin", "tri"}); out = {}
    for i, r in R.items():
        if i not in I or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
        f1 = b1["A"] / (b1["A"] + b1["B"]); f2 = b2["B"] / (b2["A"] + b2["B"])
        q1 = np.array([t1["A"], t1["B"], t1["T"]]); q2 = np.array([t2["B"], t2["A"], t2["T"]])
        out[i] = (f1, f2, q1 / q1.sum(), q2 / q2.sum())
    return out


def persona(store, ds, j, ids):
    R = store.load(ds, j, conds=set(SA)); out = {}
    for i in ids:
        r = R.get(i, {})
        if all((c, o) in r for c in SA for o in ("AB", "BA")):
            ps = []
            for c in SA:
                t1, t2 = r[(c, "AB")], r[(c, "BA")]
                pa = (t1["A"] + t2["B"]) / 2; pb = (t1["B"] + t2["A"]) / 2; ps.append(pa / (pa + pb))
            out[i] = float(np.mean(ps))
    return out


def pointwise_feats(store, ds, j, ids):
    """[E_A, E_B, Var_A, Var_B, P(s<=3)_A, P(s<=3)_B, min E, max E, |E_A - E_B|] from the 1-9 score prompt, or None"""
    from eaglevlm.pipeline.router import pointwise
    M = pointwise(store, ds, j, ids)
    if M is None:
        return None
    return np.c_[M[:, 0], M[:, 3], M[:, 1], M[:, 4], M[:, 2], M[:, 5], np.minimum(M[:, 0], M[:, 3]), np.maximum(M[:, 0], M[:, 3]),
                 np.abs(M[:, 0] - M[:, 3])]


def head(F, y, fit):
    m = LogisticRegression(max_iter=5000).fit(F[fit], y[fit]); Pr = np.zeros((len(y), 3)); Pr[:, m.classes_] = m.predict_proba(F)
    return Pr


def graded_scores(Pr, y):
    n = len(y); k = Pr[:, :2].argmax(1); o = 1 - k
    S = np.c_[1 - Pr[np.arange(n), k], 1 - Pr[:, 2], Pr[np.arange(n), o]]; E = np.c_[y != k, y != 2, y == o].astype(float)
    return S, E


def row(ok, ch, err, al):
    size = np.where(~ok, 3, np.where(ch == 2, 2, 1)); st = ok & (ch == 0); k = int(ok.sum()); e = int(err[ok].sum()) if k else 0
    risk = e / k if k else np.nan
    return dict(acceptance=float(ok.mean()), set_size=float(size.mean()), decisive=float(st.mean()), strict_errors=int(err[st].sum()),
                strict_claims=int(st.sum()), accepted=k, errors=e, risk=risk, violated=(float(risk > al) if k else np.nan),
                significant=(float(binom.sf(e - 1, k, al) < .05) if k else np.nan), empty=float(not ok.any()))


def run(cfg, store, log):
    methods = cfg["methods"]; AL = cfg["alphas"]; delta = cfg["delta"]; fit_f, cal_f = cfg["split"]["fit"], cfg["split"]["cal"]
    out = []
    for spec in cfg["datasets"]:
        ds, I = __import__("eaglevlm.pipeline.io", fromlist=["dataset_items"]).dataset_items(store, spec)
        for j in cfg["judges"]:
            members = j.split(":", 1)[1].split("+") if j.startswith("jury:") else [j]
            SG = {m: signals(store, ds, m, I) for m in members}
            ids = sorted(set.intersection(*[set(v) for v in SG.values()]))
            if len(ids) < 200 or len(ids) < cfg.get("min_coverage", .95) * len(I):
                log(f"skip {spec} {j}: {len(ids)} / {len(I)} items with both prompts in both orders"); continue
            n = len(ids); y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
            parts_full, parts_one, P_, Q_, F1_, Q1_, C_ = [], [], [], [], [], [], []
            for m in members:
                Sg = SG[m]
                f1 = np.array([Sg[i][0] for i in ids]); f2 = np.array([Sg[i][1] for i in ids])
                q1 = np.array([Sg[i][2] for i in ids]); q2 = np.array([Sg[i][3] for i in ids])
                p = (f1 + f2) / 2; Q = (q1 + q2) / 2; cons = ((f1 > .5) == (f2 > .5)).astype(float)
                parts_full.append(np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), cons])
                parts_one.append(np.c_[np.log(q1 + 1e-9), H(f1), np.abs(f1 - .5)])
                P_.append(p); Q_.append(Q); F1_.append(f1); Q1_.append(q1); C_.append(cons)
            if len(members) == 1:
                F_full, F_one = parts_full[0], parts_one[0]
            else:
                F_full, F_one = np.concatenate(parts_full, 1), np.concatenate(parts_one, 1)
                p, Q, f1, q1 = np.mean(P_, 0), np.mean(Q_, 0), np.mean(F1_, 0), np.mean(Q1_, 0)
            PWF = None
            if {"G2pw", "G5pw"} & set(methods):
                pws = [pointwise_feats(store, ds, m, ids) for m in members]; pws = [x for x in pws if x is not None]
                PWF = np.mean(pws, 0) if pws else None
            fc = np.where(p > .5, 0, 1); eb = (fc != y).astype(float); et = np.where(y == 2, 1.0, eb)
            fc1 = np.where(f1 > .5, 0, 1); et1 = np.where(y == 2, 1.0, (fc1 != y).astype(float))
            nsa = 0
            if {"T1", "T2"} & set(methods) and len(members) == 1:
                PS = persona(store, ds, j, ids); has = np.array([i in PS for i in ids]); nsa = int(has.sum())
                if nsa >= 200:
                    ps = np.array([PS.get(i, .5) for i in ids]); sa_s = 1 - np.maximum(ps, 1 - ps)
                    sa_fc = np.where(ps > .5, 0, 1); sa_et = np.where(y == 2, 1.0, (sa_fc != y).astype(float))
            rng = np.random.default_rng(cfg.get("seed", 0))
            for sp in range(cfg["n_splits"]):
                pm = rng.permutation(n)
                a1, a2 = int(round(fit_f * n)), int(round((fit_f + cal_f) * n))
                if fit_f == .25 and cal_f == .25:
                    a1, a2 = n // 4, n // 2                     # identical to the original scripts
                fit, cal, te = pm[:a1], pm[a1:a2], pm[a2:]
                calnt = cal[y[cal] != 2]
                Prf = head(F_full, y, fit); S3, E3 = graded_scores(Prf, y)
                need1 = "G1" in methods
                if need1:
                    Pr1 = head(F_one, y, fit); S31, E31 = graded_scores(Pr1, y)
                S0, E0 = graded_scores(Q, y)
                if PWF is not None:
                    Prp = head(np.c_[F_full, PWF], y, fit); S3p, E3p = graded_scores(Prp, y)
                u, u1 = H(p), H(f1); zeros = np.zeros(len(te), int)
                for al in AL:
                    res = {}
                    if "P1" in methods:
                        res["P1"] = row(u[te] <= thr(u[calnt], eb[calnt], al), zeros, et[te], al)
                    if "P1s" in methods:
                        res["P1s"] = row(u1[te] <= thr(u1[calnt], (fc1 != y)[calnt].astype(float), al), zeros, et1[te], al)
                    if "P2" in methods:
                        res["P2"] = row(u[te] <= thr(u[cal], et[cal], al), zeros, et[te], al)
                    if "P3" in methods:
                        lam = ltt(u[cal][:, None], et[cal][:, None], al, delta, u[fit][:, None], et[fit][:, None])
                        res["P3"] = row(u[te] <= lam, zeros, et[te], al)
                    if "P4" in methods:
                        lam = ltt(S3[cal][:, [0]], E3[cal][:, [0]], al, delta, S3[fit][:, [0]], E3[fit][:, [0]])
                        res["P4"] = row(S3[te, 0] <= lam, zeros, E3[te, 0], al)
                    if nsa >= 200:
                        ct, tt, ft = cal[has[cal]], te[has[te]], fit[has[fit]]
                        ctn = ct[y[ct] != 2]
                        for key, cc in (("T1", ctn), ("T2", ct)):
                            if key not in methods:
                                continue
                            ec = np.where(y[cc] == 2, 1.0, (sa_fc[cc] != y[cc]).astype(float))
                            ef = np.where(y[ft] == 2, 1.0, (sa_fc[ft] != y[ft]).astype(float)) if key == "T2" else (sa_fc[ft] != y[ft]).astype(float)
                            lam = ltt(sa_s[cc][:, None], ec[:, None], al, delta, sa_s[ft][:, None], ef[:, None])
                            res[key] = row(sa_s[tt] <= lam, np.zeros(len(tt), int), sa_et[tt], al)
                    for key, S, E in (("G0", S0, E0), ("G1", S31 if need1 else None, E31 if need1 else None), ("G2", S3, E3)):
                        if key in methods:
                            ok, ch = claims(S[te], ltt(S[cal], E[cal], al, delta, S[fit], E[fit]))
                            res[key] = row(ok, ch, E[te][np.arange(len(te)), np.maximum(ch, 0)], al)
                    if "G3" in methods:
                        ok, ch = claims(S3[te], ltt_p(S3[cal], E3[cal], S3[fit], E3[fit], al, delta, pjoint))
                        res["G3"] = row(ok, ch, E3[te][np.arange(len(te)), np.maximum(ch, 0)], al)
                    if "G4" in methods:
                        ok, ch = claims(S3[te], ltt_p(S3[cal], E3[cal], S3[fit], E3[fit], al, delta, pstrict))
                        res["G4"] = row(ok, ch, E3[te][np.arange(len(te)), np.maximum(ch, 0)], al)
                    if "G5" in methods:
                        ls, lw = ltt_two(S3[cal], E3[cal], S3[fit], E3[fit], al, delta); ok, ch = claims2(S3[te], ls, lw)
                        res["G5"] = row(ok, ch, E3[te][np.arange(len(te)), np.maximum(ch, 0)], al)
                    if PWF is not None and "G2pw" in methods:
                        ok, ch = claims(S3p[te], ltt(S3p[cal], E3p[cal], al, delta, S3p[fit], E3p[fit]))
                        res["G2pw"] = row(ok, ch, E3p[te][np.arange(len(te)), np.maximum(ch, 0)], al)
                    if PWF is not None and "G5pw" in methods:
                        ls, lw = ltt_two(S3p[cal], E3p[cal], S3p[fit], E3p[fit], al, delta); ok, ch = claims2(S3p[te], ls, lw)
                        res["G5pw"] = row(ok, ch, E3p[te][np.arange(len(te)), np.maximum(ch, 0)], al)
                    for mth, r in res.items():
                        out.append(dict(dataset=spec, judge=j, n=n, human_tie=float((y == 2).mean()),
                                        n_persona=(nsa if mth in ("T1", "T2") else None), method=mth, alpha=al, split=sp, **r))
            log(f"done {spec} {j} (n={n})")
    return out


def summarize(rows):
    """per dataset x judge x method x alpha: means over splits; strict-claim error pooled over splits (as the original)"""
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["judge"], r["method"], r["alpha"])].append(r)
    out = []
    for (ds, j, m, al), R in g.items():
        nn = lambda k: [r[k] for r in R if r[k] == r[k] and r[k] is not None]
        sc = sum(r["strict_claims"] for r in R); risk = nn("risk")
        out.append(dict(dataset=ds, judge=j, method=m, alpha=al, n=R[0]["n"], human_tie=R[0]["human_tie"], n_persona=R[0]["n_persona"],
                        splits=len(R), acceptance=np.mean([r["acceptance"] for r in R]), decisive=np.mean([r["decisive"] for r in R]),
                        set_size=np.mean([r["set_size"] for r in R]),
                        strict_error=(sum(r["strict_errors"] for r in R) / sc if sc else np.nan),
                        risk=(np.mean(risk) if risk else np.nan), risk_sd=(np.std(risk) if risk else np.nan),
                        human_agreement=(1 - np.mean(risk) if risk else np.nan),
                        violation=(np.mean(nn("violated")) if risk else np.nan),
                        guarantee_success=(1 - np.mean(nn("violated")) if risk else np.nan),
                        significant_violation=(np.mean(nn("significant")) if risk else np.nan),
                        empty_splits=np.mean([r["empty"] for r in R])))
    return out
