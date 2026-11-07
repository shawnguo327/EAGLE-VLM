"""Reviewer-plan ablations ('ablate'; review improvement plan of 2026-11-06, experiments 2, 3 and 7), all on saved judge
outputs.  Every method is certified with the same fixed-sequence LTT (certify.ltt: fit-chosen start, first failure stops)
on the same splits; default unit: source1 (split by source, one random pair per calibration source - the binomial LTT is
exact for the risk of a pair drawn uniformly from a uniformly drawn new source).  The split sequence is the one of
certify (same seed -> the same splits as certify_source, so F00 / F01 / F02 reproduce P4 / G2s / G2 there).

Experiment 2 - signal x claim family (2 x 3 factorial); one 3-class logistic head per signal row, one claim rule:
  S0 = the pairwise features of G2 (log tie-prompt probabilities, forced-choice entropy / margin, order consistency);
  S1 = S0 + the judge's pointwise (1-9 single-candidate score) features (fit-only imputation; S1 rows only where the
       pointwise run exists; S1 costs two extra calls per pair).
  family 0: {A}, {B}        family 1: {A}, {B}, {T}        family 2: family 1 + the weak sets {A,T}, {B,T}
  rule at threshold lam: the estimated error of a set C is 1 - sum_{y in C} p(y); among the family's sets whose estimated
  error is <= lam, the most specific one is issued (a singleton before a weak set), ties by the smaller estimated error,
  then the fixed order A, B, T.  (G2's priority rule coincides with this one whenever lam < .5.)
  F00 F01 F02 = S0 x family 0 1 2;   F10 F11 F12 = S1 x family 0 1 2.
  Primary endpoint: mean number of excluded labels 3 - |C| (abstention 0, weak set 1, singleton 2).  Pre-specified
  contrasts: F01 - F00 (adding the tie singleton), F02 - F01 (adding weak sets), F12 - F02 (adding pointwise signals).
Experiment 3 - does the instance's p(T) carry information beyond a tie prior?  Claim family 2 throughout:
  B0  binary-prompt features only (entropy, margin, order consistency, p) -> 3-class head (its intercept learns the prior)
  B1  B0 + the tie prompt's CONDITIONAL direction share qA / (qA + qB) per order, averaged (raw log qA / log qB would
      imply p(T)); the same head
  B2  constant tie prior: p(T) = tau (fit tie rate; per-source average under a source unit) and a direction head b(x) on
      the B1 features fitted on the fit non-tie pairs; p = ((1 - tau) b, (1 - tau)(1 - b), tau), no 3-class head on top
  B3  B1 + the instance's log p(T) (order-averaged tie-prompt probability); same head and supervision as B1
  B4  B1 + a pseudo log p(T) drawn i.i.d. from the fit rows' empirical distribution (the same extra dimension without the
      correspondence to the instance; fixed seed per split)
  TCs selective tie calibration: Ties Matter's tie band |p - .5| <= eps with eps fitted on the FIT split (3-way accuracy);
      confidence = depth inside the band (tie claims) or beyond it (strict claims), certified with the same LTT
  Primary contrast B3 - B1 (excluded labels); secondary B3 - B4, B3 - B2, B1 - B0.  Test tie AUROC (p(T) against the
  human tie) and 3-class log loss of every head are reported (columns tie_auroc, logloss).
Experiment 7 - two-candidate selection with escalation: the policy picks one candidate or escalates to a human.  Loss:
  picking the candidate the human label calls strictly worse; on a human tie either pick is loss-free.  Each policy is
  certified on THIS loss (selection risk <= alpha among the automated pairs; 'acceptance' = automation rate):
  U1  binary selector: forced-choice winner, confidence max(p, 1 - p)
  U1c binary selector with its own calibrator: direction head on the binary features, fitted on the fit non-tie pairs
  U1t binary signals with a 3-class head (G2b's head): pick the likelier direction k, estimated loss p(other side)
  U1s U1 certified on the STRICT loss (a human tie counts as an error, as when ties are dropped from the loss), deployed
      on the selection loss: what the loss definition alone changes
  U2  graded action policy (full pairwise head S0): pick the likelier direction k; the claims {k}, {T}, {k,T} all map to
      'pick k', and the smallest of their estimated errors is p(other side); recalibrated on the selection loss
  U2i F02's claim certificate inherited without recalibration: a correct claim is a loss-free pick, so the claim-risk
      certificate also bounds the selection risk
  U3  U2 with the pointwise-augmented head (S1)
  U0 (everything to the human) automates nothing.  Primary endpoint: automation at certified selection risk; contrasts
  U2 - U1, U2 - U1c, U2 - U1t, U3 - U2, U2 - U2i, U1 - U1s.  Also reported: automation among human ties (auto_on_ties)
  and the selection regret over all test pairs with escalated pairs resolved by the human reference (regret).
Optional cfg 'cond_map' / 'pointwise: false' as in certify (image-removed signals, review W13).
Rows: per split x alpha one row per method (certify.row fields + excluded, tie_auroc, logloss, auto_on_ties, regret; on
split 0 a 95% source-bootstrap interval of the source-weighted test risk) and one row per contrast 'D:<a>-<b>' with the
source-weighted test difference (per-source mean, then mean over the test sources) in acceptance and excluded labels; on
split 0 - the fixed holdout - also a paired 95% source-bootstrap interval (the same resampled sources for both methods).
The other splits describe stability only; they are overlapping re-splits of one benchmark, not replications."""
import collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from eaglevlm.pipeline.certify import H, claims, head, ltt, pointwise_feats, row, signals, source_key

FAMS = {"F00": ("S0", 0), "F01": ("S0", 1), "F02": ("S0", 2), "F10": ("S1", 0), "F11": ("S1", 1), "F12": ("S1", 2),
        "B0": ("B0", 2), "B1": ("B1", 2), "B2": ("B2", 2), "B3": ("B3", 2), "B4": ("B4", 2)}
CONTRASTS = [("F01", "F00"), ("F02", "F01"), ("F12", "F02"), ("F10", "F00"), ("F11", "F01"),
             ("B3", "B1"), ("B3", "B4"), ("B3", "B2"), ("B1", "B0"), ("F02", "B3"), ("B3", "TCs"),
             ("U2", "U1"), ("U2", "U1c"), ("U2", "U1t"), ("U3", "U2"), ("U2", "U2i"), ("U1", "U1s")]
KEYS = ["dataset", "judge", "n", "human_tie", "method", "alpha", "split", "acceptance", "set_size", "excluded", "decisive",
        "strict_errors", "strict_claims", "accepted", "errors", "risk", "violated", "significant", "empty", "tie_claims",
        "weak_claims", "nondirectional", "risk_source", "sources", "violated_source", "source_coverage", "src_acc_mass",
        "src_err_mass", "accepted_sources", "accepted_ties", "strict_on_ties",
        "auto_on_ties", "regret", "tie_auroc", "logloss", "ci_risk_lo", "ci_risk_hi",
        "d_acceptance", "d_excluded", "ci_acc_lo", "ci_acc_hi", "ci_excl_lo", "ci_excl_hi"]


def family_scores(Pr, y, fam):
    """score / error / claim-type matrices for certify.claims (the first passing column is issued): column 0 = the best
    singleton of the family ({A}, {B}; family >= 1 also {T}), column 1 (family 2) = the best weak set {k, T}.
    Claim types as in certify.row: 0 strict, 1 tie, 2 weak."""
    n = len(y); r = np.arange(n)
    k = Pr[:, :2].argmax(1) if fam == 0 else Pr.argmax(1)            # argmax ties -> the first of A, B, T
    s0 = 1 - Pr[r, k]; e0 = (y != k).astype(float); t0 = (k == 2).astype(int)
    if fam < 2:
        return s0[:, None], e0[:, None], t0[:, None]
    kw = np.where(Pr[:, 1] <= Pr[:, 0], 0, 1)                         # {A,T} errs only on B: estimated error p(B)
    s1 = Pr[r, 1 - kw]; e1 = (y == 1 - kw).astype(float)
    return np.c_[s0, s1], np.c_[e0, e1], np.c_[t0, np.full(n, 2)]


def pick(Pr, y):
    """selection policy from 3-class probabilities: pick the likelier direction k; estimated loss p(other side)"""
    kw = np.where(Pr[:, 1] <= Pr[:, 0], 0, 1); n = len(y)
    return Pr[np.arange(n), 1 - kw], (y == 1 - kw).astype(float), kw


def diag(Pr, y, te):
    t = y[te] == 2
    au = float(roc_auc_score(t, Pr[te, 2])) if 0 < t.sum() < len(t) else np.nan
    return au, float(-np.mean(np.log(np.clip(Pr[te, y[te]], 1e-12, 1))))


def boot(vals, B, rng, ratio=None):
    """95% percentile interval of the mean over sources (or of sum(vals) / sum(ratio) when ratio is given)"""
    G = len(vals)
    if G == 0:
        return np.nan, np.nan
    idx = rng.integers(0, G, size=(B, G))
    if ratio is None:
        s = vals[idx].mean(1)
    else:
        den = ratio[idx].sum(1); s = np.where(den > 0, vals[idx].sum(1) / np.where(den > 0, den, 1), np.nan)
        s = s[np.isfinite(s)]
        if len(s) == 0:
            return np.nan, np.nan
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def run(cfg, store, log):
    from eaglevlm.pipeline.io import dataset_items
    AL = cfg["alphas"]; delta = cfg["delta"]; fit_f, cal_f = cfg["split"]["fit"], cfg["split"]["cal"]
    unit = cfg.get("unit", "source1"); NB = cfg.get("bootstrap", 2000); out = []
    for spec in cfg["datasets"]:
        ds, I = dataset_items(store, spec)
        for j in cfg["judges"]:
            Sg = signals(store, ds, j, I, cfg.get("cond_map")); ids = sorted(Sg)
            if len(ids) < 200 or len(ids) < cfg.get("min_coverage", .95) * len(I):
                log(f"skip {spec} {j}: {len(ids)} / {len(I)} items with both prompts in both orders"); continue
            n = len(ids); y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids]); Ty = y == 2
            ug, gi = np.unique(np.array([source_key(I[i]) for i in ids]), return_inverse=True)
            f1 = np.array([Sg[i][0] for i in ids]); f2 = np.array([Sg[i][1] for i in ids])
            q1 = np.array([Sg[i][2] for i in ids]); q2 = np.array([Sg[i][3] for i in ids])
            p = (f1 + f2) / 2; Q = (q1 + q2) / 2; cons = ((f1 > .5) == (f2 > .5)).astype(float)
            F_full = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), cons]
            F_bin = np.c_[H(p), np.abs(p - .5), cons, p]
            cc = (q1[:, 0] / (q1[:, 0] + q1[:, 1]) + q2[:, 0] / (q2[:, 0] + q2[:, 1])) / 2
            F_b1 = np.c_[F_bin, cc, np.abs(cc - .5)]; lpT = np.log(Q[:, 2] + 1e-9)
            PWF = pointwise_feats(store, ds, j, ids) if cfg.get("pointwise", True) else None
            fc = np.where(p > .5, 0, 1); mg = np.abs(p - .5)
            e_sel1 = (y == 1 - fc).astype(float); e_str1 = np.where(Ty, 1.0, (fc != y).astype(float))
            rng = np.random.default_rng(cfg.get("seed", 0))            # the split sequence of certify
            for sp in range(cfg["n_splits"]):
                if unit == "pair":
                    pm = rng.permutation(n); a1, a2 = int(round(fit_f * n)), int(round((fit_f + cal_f) * n))
                    fit, cal, te = pm[:a1], pm[a1:a2], pm[a2:]
                else:
                    rk = np.empty(len(ug), int); rk[rng.permutation(len(ug))] = np.arange(len(ug)); r = rk[gi]
                    g1, g2 = int(round(fit_f * len(ug))), int(round((fit_f + cal_f) * len(ug)))
                    fit, cal, te = np.where(r < g1)[0], np.where((r >= g1) & (r < g2))[0], np.where(r >= g2)[0]
                    if unit == "source1":
                        pc = rng.permutation(cal); _, first = np.unique(gi[pc], return_index=True); cal = np.sort(pc[first])
                r2 = np.random.default_rng([cfg.get("seed", 0), sp, 7])      # B4 draws / bootstrap; keeps the split stream
                _, inv_te, cnt_te = np.unique(gi[te], return_inverse=True, return_counts=True); G = len(cnt_te)
                smean = lambda x: np.bincount(inv_te, x.astype(float), minlength=G) / cnt_te
                # ---- heads (fit only)
                PR = {"S0": head(F_full, y, fit), "B0": head(F_bin, y, fit), "B1": head(F_b1, y, fit),
                      "B3": head(np.c_[F_b1, lpT], y, fit),
                      "B4": head(np.c_[F_b1, r2.choice(lpT[fit], size=n, replace=True)], y, fit)}
                fnt = fit[y[fit] != 2]
                _, fi, fcnt = np.unique(gi[fit], return_inverse=True, return_counts=True)
                tau = float((np.bincount(fi, Ty[fit].astype(float)) / fcnt).mean()) if unit != "pair" else float(Ty[fit].mean())
                dh = LogisticRegression(max_iter=5000).fit(F_b1[fnt], (y[fnt] == 0).astype(int)); b = dh.predict_proba(F_b1)[:, 1]
                PR["B2"] = np.c_[(1 - tau) * b, (1 - tau) * (1 - b), np.full(n, tau)]
                if PWF is not None:
                    PWFs = PWF.copy(); bad = np.isnan(PWFs)
                    if bad.any():
                        PWFs[bad] = np.nanmean(PWF[fit], 0)[np.where(bad)[1]]
                    PR["S1"] = head(np.c_[F_full, PWFs], y, fit)
                bh = LogisticRegression(max_iter=5000).fit(F_bin[fnt], (y[fnt] == 0).astype(int)); bb = bh.predict_proba(F_bin)[:, 1]
                DG = {h: diag(P, y, te) for h, P in PR.items()}
                # ---- scored policies: name -> (S, E_cal, T, E_test)
                POL = {}
                for m, (h, fam) in FAMS.items():
                    if h in PR:
                        S, E, T = family_scores(PR[h], y, fam); POL[m] = (S, E, T, E)
                of = fit[np.argsort(mg[fit], kind="stable")]; mf = mg[of]          # TCs: tie band fitted on the fit split
                ht = np.cumsum(y[of] == 2); hs = np.cumsum(fc[of] == y[of]); tot = hs[-1]
                last = np.r_[mf[1:] != mf[:-1], True]; acc3 = (ht + tot - hs)[last]
                eps = float(mf[last][int(np.argmax(acc3))]) if acc3.max() > tot else -1.0
                is_t = mg <= eps
                s_tc = np.where(is_t, mg / max(eps, 1e-12), (.5 - mg) / max(.5 - max(eps, 0.0), 1e-12))
                e_tc = np.where(is_t, ~Ty, fc != y).astype(float)
                POL["TCs"] = (s_tc[:, None], e_tc[:, None], is_t.astype(int)[:, None], e_tc[:, None])
                z = np.zeros(n, int)[:, None]
                e_c = (y == np.where(bb >= .5, 1, 0)).astype(float)          # U1c picks A iff b >= .5; loss = B better
                SEL = {"U1": (1 - np.maximum(p, 1 - p), e_sel1, e_sel1),
                       "U1c": (1 - np.maximum(bb, 1 - bb), e_c, e_c),
                       "U1s": (1 - np.maximum(p, 1 - p), e_str1, e_sel1)}     # certified on the strict loss
                for nm, h in (("U1t", "B0"), ("U2", "S0"), ("U3", "S1")):
                    if h in PR:
                        s, e, _ = pick(PR[h], y); SEL[nm] = (s, e, e)
                for nm, (s, ec, et) in SEL.items():
                    POL[nm] = (s[:, None], ec[:, None], z, et[:, None])
                _, _, kw0 = pick(PR["S0"], y); e_u2i = (y == 1 - kw0).astype(float)
                for al in AL:
                    keep = {}; rows = {}
                    for m, (S, E, T, Et) in POL.items():
                        lam = ltt(S[cal], E[cal], al, delta, S[fit], E[fit])
                        ok, col = claims(S[te], lam); c = np.maximum(col, 0); rr = np.arange(len(te))
                        ch, err = T[te][rr, c], Et[te][rr, c]
                        d = row(ok, ch, err, al, Ty[te], gi[te])
                        sel = m.startswith("U")
                        excl = np.where(~ok, 0.0, np.where(ch == 2, 1.0, 2.0)) if not sel else np.full(len(te), np.nan)
                        h = FAMS.get(m, (None,))[0]
                        d.update(excluded=float(excl.mean()) if not sel else np.nan,
                                 auto_on_ties=float(ok[Ty[te]].mean()) if Ty[te].any() else np.nan,
                                 regret=float((err * ok).sum() / len(te)),
                                 tie_auroc=DG[h][0] if h in DG else np.nan, logloss=DG[h][1] if h in DG else np.nan)
                        if m == "F02":                       # U2i: F02's claims mapped to picks (inherited certificate)
                            ok2 = ok; e2 = e_u2i[te]
                            d2 = row(ok2, np.zeros(len(te), int), e2, al, Ty[te], gi[te])
                            d2.update(excluded=np.nan, auto_on_ties=float(ok2[Ty[te]].mean()) if Ty[te].any() else np.nan,
                                      regret=float((e2 * ok2).sum() / len(te)), tie_auroc=np.nan, logloss=np.nan)
                            rows["U2i"] = (d2, ok2, e2); keep["U2i"] = (ok2.astype(float), np.full(len(te), np.nan))
                        rows[m] = (d, ok, err); keep[m] = (ok.astype(float), excl)
                    for m, (d, ok, err) in rows.items():
                        if sp == 0:
                            lo, hi = boot(smean(err * ok), NB, r2, ratio=smean(ok))
                            d.update(ci_risk_lo=lo, ci_risk_hi=hi)
                        out.append(_full(dict(dataset=spec, judge=j, n=n, human_tie=float(Ty.mean()), method=m, alpha=al, split=sp, **d)))
                    for a, b_ in CONTRASTS:
                        if a not in keep or b_ not in keep:
                            continue
                        da = smean(keep[a][0] - keep[b_][0]); xe = keep[a][1] - keep[b_][1]
                        de = smean(xe) if np.isfinite(xe).all() else None
                        d = dict(dataset=spec, judge=j, n=n, human_tie=float(Ty.mean()), method=f"D:{a}-{b_}", alpha=al, split=sp,
                                 d_acceptance=float(da.mean()), d_excluded=(float(de.mean()) if de is not None else np.nan))
                        if sp == 0:
                            d["ci_acc_lo"], d["ci_acc_hi"] = boot(da, NB, r2)
                            if de is not None:
                                d["ci_excl_lo"], d["ci_excl_hi"] = boot(de, NB, r2)
                        out.append(_full(d))
            log(f"done {spec} {j} (n={n}, sources={len(ug)})")
    return out


def _full(d):
    return {k: d.get(k, np.nan) for k in KEYS}


SUMKEYS = ["dataset", "judge", "method", "alpha", "n", "human_tie", "splits", "acceptance", "excluded", "set_size", "decisive",
           "tie_claim_share", "weak_share", "strict_error", "risk", "pooled_risk", "risk_source", "source_coverage",
           "pooled_risk_source", "guarantee_success_all",
           "guarantee_success_source", "splits_answered", "accepted_median", "auto_on_ties", "regret", "tie_auroc", "logloss",
           "holdout_acceptance", "holdout_excluded", "holdout_risk_source", "holdout_risk_lo", "holdout_risk_hi",
           "d_acceptance", "d_acc_q025", "d_acc_q975", "d_excluded", "d_excl_q025", "d_excl_q975",
           "holdout_d_acceptance", "holdout_acc_lo", "holdout_acc_hi", "holdout_d_excluded", "holdout_excl_lo", "holdout_excl_hi",
           "share_splits_positive"]


def summarize(rows):
    """per dataset x judge x method x alpha: means over splits (stability), the split-0 holdout values with their
    source-bootstrap intervals, and for contrasts the 2.5 / 97.5 percentiles over splits"""
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["judge"], r["method"], r["alpha"])].append(r)
    out = []
    for (ds, j, m, al), R in g.items():
        f = lambda k: np.array([np.nan if r[k] is None else r[k] for r in R], float)
        mean = lambda k: float(np.nanmean(f(k))) if np.isfinite(f(k)).any() else np.nan
        pct = lambda k, q: float(np.nanpercentile(f(k), q)) if np.isfinite(f(k)).any() else np.nan
        h0 = next((r for r in R if r["split"] == 0), {})
        d = dict(dataset=ds, judge=j, method=m, alpha=al, n=R[0]["n"], human_tie=R[0]["human_tie"], splits=len(R))
        if m.startswith("D:"):
            d.update(d_acceptance=mean("d_acceptance"), d_acc_q025=pct("d_acceptance", 2.5), d_acc_q975=pct("d_acceptance", 97.5),
                     d_excluded=mean("d_excluded"), d_excl_q025=pct("d_excluded", 2.5), d_excl_q975=pct("d_excluded", 97.5),
                     holdout_d_acceptance=h0.get("d_acceptance", np.nan), holdout_acc_lo=h0.get("ci_acc_lo", np.nan),
                     holdout_acc_hi=h0.get("ci_acc_hi", np.nan), holdout_d_excluded=h0.get("d_excluded", np.nan),
                     holdout_excl_lo=h0.get("ci_excl_lo", np.nan), holdout_excl_hi=h0.get("ci_excl_hi", np.nan),
                     share_splits_positive=float(np.mean(f("d_acceptance") > 0)))
        else:
            sc = np.nansum(f("strict_claims")); ka = np.nansum(f("accepted"))
            d.update(acceptance=mean("acceptance"), excluded=mean("excluded"), set_size=mean("set_size"), decisive=mean("decisive"),
                     tie_claim_share=mean("tie_claims"), weak_share=mean("weak_claims"),
                     strict_error=(float(np.nansum(f("strict_errors")) / sc) if sc else np.nan),
                     risk=mean("risk"), pooled_risk=(float(np.nansum(f("errors")) / ka) if ka else np.nan),
                     risk_source=mean("risk_source"), source_coverage=mean("source_coverage"),
                     pooled_risk_source=(float(np.nansum(f("src_err_mass")) / np.nansum(f("src_acc_mass")))
                                         if np.nansum(f("src_acc_mass")) > 0 else np.nan),
                     guarantee_success_all=float(1 - np.mean(f("violated") == 1.0)),
                     guarantee_success_source=float(1 - np.mean(f("violated_source") == 1.0)),
                     splits_answered=int(np.sum(f("accepted") > 0)), accepted_median=float(np.nanmedian(f("accepted"))),
                     auto_on_ties=mean("auto_on_ties"), regret=mean("regret"), tie_auroc=mean("tie_auroc"), logloss=mean("logloss"),
                     holdout_acceptance=h0.get("acceptance", np.nan), holdout_excluded=h0.get("excluded", np.nan),
                     holdout_risk_source=h0.get("risk_source", np.nan), holdout_risk_lo=h0.get("ci_risk_lo", np.nan),
                     holdout_risk_hi=h0.get("ci_risk_hi", np.nan))
        out.append({k: d.get(k, np.nan) for k in SUMKEYS})
    return out
