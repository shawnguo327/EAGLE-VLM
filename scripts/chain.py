"""Controlled chain: from the published tie-dropping protocols to our method, one step at a time, on the same splits.
Every variant uses the same 25% fit / 25% cal / 50% test split (fit = head training + LTT start point; methods without a
head ignore the fit labels except for the LTT start).  Deployment is always on ALL test pairs (ties included); a tie is
an error for a 'k better' claim, a non-tie is an error for a 'tie' claim, and only the opposite verdict is an error for
a weak 'k not worse' claim.
Chain (rows):
  P1  SCOPE as published        BPE of the order-averaged forced choice; calibration set = cal pairs WITHOUT ties;
                                SCOPE linearised threshold (expectation guarantee)
  P1s SCOPE, single order       same, but only the A-first call (no order averaging)
  P2  SCOPE + ties in calib.    minimal fix: same score/threshold rule, calibration keeps ties (tie = error)
  P3  strict + LTT (BPE)        same score, high-probability LTT instead of SCOPE, ties kept
  P4  strict + head + LTT       all our signal components (both prompts, both orders, consistency, 3-class head),
                                but strict 'k better' claims only
  T1  ToE as published          Trust-or-Escalate: 4 simulated-annotator personas, confidence = max persona-averaged
                                A/B prob, fixed-sequence test, calibration WITHOUT ties   (only where persona runs exist)
  T2  ToE + ties in calib.
  G0  graded, no head           strict / tie / weak claims from the raw order-averaged tie-prompt probabilities
  G1  graded, single order      our head on A-first-call features only
  G2  graded + head (ours)      overall LTT
  G3  graded + head, joint LTT  every claim level <= alpha simultaneously (intersection-union)
Metrics (test, mean over splits): acceptance (any claim), mean set size ({k},{T}=1, weak=2, abstain=3), decisive =
strict share, err(strict) pooled, overall selective risk, violation = share of splits with risk > alpha (non-empty
splits), empty = share of splits accepting nothing.
usage (judgebench dir): python chain.py --out paper_tables/chain.md"""
import argparse, collections, os, sys
import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from label_noise_lib import eval_items, ltt, thr, _claims

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/chain.md")
ap.add_argument("--splits", type=int, default=100)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--alphas", default="0.1,0.2,0.3")
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen35_9b,qwen25vl7b,qwen3vl32b")
ap.add_argument("--sets", default="mllm_judge,mtbench_human,arena55k,genai_arena/generation,genai_arena/edition,pickapic_u,imagerewarddb")
a = ap.parse_args()
AL = [float(x) for x in a.alphas.split(",")]
GRID = np.linspace(0, 1, 201)[1:]
SA = ["tri_sa1", "tri_sa2", "tri_sa3", "tri_sa4"]


def signals(ds, j, I):
    """per id: forced-choice P(first cand) per order, tie-prompt probs per order (order-aligned to cand 0 / cand 1 / T)"""
    R = load(ds, j, conds={"bin", "tri"}); out = {}
    for i, r in R.items():
        if i not in I or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
        f1 = b1["A"] / (b1["A"] + b1["B"]); f2 = b2["B"] / (b2["A"] + b2["B"])        # both = P(cand 0 better)
        q1 = np.array([t1["A"], t1["B"], t1["T"]]); q2 = np.array([t2["B"], t2["A"], t2["T"]])
        out[i] = (f1, f2, q1 / q1.sum(), q2 / q2.sum())
    return out


def persona(ds, j, ids):
    R = load(ds, j, conds=set(SA)); out = {}
    for i in ids:
        r = R.get(i, {})
        if all((c, o) in r for c in SA for o in ("AB", "BA")):
            ps = []
            for c in SA:
                t1, t2 = r[(c, "AB")], r[(c, "BA")]
                pa = (t1["A"] + t2["B"]) / 2; pb = (t1["B"] + t2["A"]) / 2; ps.append(pa / (pa + pb))
            out[i] = float(np.mean(ps))
    return out


def pjoint(S, E, lam, al):
    ok, ch = _claims(S, lam); ps = [0.0]
    for c in range(3):
        mk = ok & (ch == c)
        if mk.any():
            ps.append(binom.cdf(int(E[mk, c].sum()), int(mk.sum()), al))
    return max(ps) if ok.any() else 1.0


def ltt_joint(Sc, Ec, Sf, Ef, al):
    s0 = int(np.argmin([pjoint(Sf, Ef, g, al) for g in GRID])); best = -np.inf
    for g in GRID[s0:]:
        if pjoint(Sc, Ec, g, al) <= a.delta:
            best = g
        else:
            break
    return best


def head(F, y, fit):
    m = LogisticRegression(max_iter=5000).fit(F[fit], y[fit]); Pr = np.zeros((len(y), 3)); Pr[:, m.classes_] = m.predict_proba(F)
    return Pr


def graded_scores(Pr, y):
    n = len(y); k = Pr[:, :2].argmax(1); o = 1 - k
    S = np.c_[1 - Pr[np.arange(n), k], 1 - Pr[:, 2], Pr[np.arange(n), o]]; E = np.c_[y != k, y != 2, y == o].astype(float)
    return S, E


class Acc:
    def __init__(self):
        self.R = collections.defaultdict(list)

    def add(self, key, al, ok, ch, err):
        """ok: accepted mask on test; ch: claim level 0 strict / 1 tie / 2 weak; err: per-test-pair error of its claim"""
        size = np.where(~ok, 3, np.where(ch == 2, 2, 1)); st = ok & (ch == 0)
        r = err[ok].mean() if ok.any() else np.nan
        self.R[(key, al)].append((ok.mean(), size.mean(), st.mean(), err[st].sum(), st.sum(), r,
                                  float(r > al) if ok.any() else np.nan, float(not ok.any())))

    def cell(self, key, al):
        Z = np.array(self.R[(key, al)], float)
        if not len(Z):
            return "-"
        es = Z[:, 3].sum() / Z[:, 4].sum() if Z[:, 4].sum() else np.nan
        nz = ~np.isnan(Z[:, 5])
        f = lambda v: "-" if np.isnan(v) else f"{v:.2f}"
        return (f"{Z[:, 0].mean():.2f} / {Z[:, 1].mean():.2f} / {Z[:, 2].mean():.2f} / {f(es)} / "
                f"{f(np.nanmean(Z[:, 5]) if nz.any() else np.nan)} / {f(np.nanmean(Z[:, 6]) if nz.any() else np.nan)}"
                + (f" ({Z[:, 7].mean():.0%} empty)" if Z[:, 7].mean() > 0 else ""))


ROWS = [("P1", "SCOPE as published (BPE, calib. w/o ties)"), ("P1s", "  same, single order"),
        ("P2", "SCOPE, ties kept in calibration"), ("P3", "strict + LTT, BPE, ties kept"),
        ("P4", "strict + LTT, all our signal components + head"), ("T1", "Trust-or-Escalate as published (calib. w/o ties)"),
        ("T2", "Trust-or-Escalate, ties kept"), ("G0", "graded + LTT, no head (raw tie-prompt probs)"),
        ("G1", "graded + LTT + head, single-order features"), ("G2", "graded + LTT + head (ours, overall risk)"),
        ("G3", "graded + joint LTT + head (ours, every level <= alpha)")]
L = ["# Controlled chain: published tie-dropping protocols -> our method", "",
     f"{a.splits} splits (25% fit / 25% cal / 50% test), delta {a.delta}.  Cell = acceptance / mean set size / decisive "
     "(strict) share / strict-claim error / overall selective risk / violation rate (+ share of empty splits).  "
     "Deployment on all test pairs, ties included.  T1/T2 only where persona runs exist (n in brackets).", "",
     "| set | judge | n | tie | step | " + " | ".join(f"alpha {x}" for x in AL) + " |", "|" + "---|" * (5 + len(AL))]
for spec in a.sets.split(","):
    ds, sub = (spec.split("/") + [None])[:2]
    I = eval_items(ds, items(ds)); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    for j in a.judges.split(","):
        Sg = signals(ds, j, I); ids = sorted(Sg)
        if len(ids) < 200 or len(ids) < .95 * len(I):
            continue
        n = len(ids); y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        f1 = np.array([Sg[i][0] for i in ids]); f2 = np.array([Sg[i][1] for i in ids])
        q1 = np.array([Sg[i][2] for i in ids]); q2 = np.array([Sg[i][3] for i in ids])
        p = (f1 + f2) / 2; Q = (q1 + q2) / 2; cons = ((f1 > .5) == (f2 > .5)).astype(float)
        F_full = np.c_[np.log(Q + 1e-9), H(p), np.abs(p - .5), cons]
        F_one = np.c_[np.log(q1 + 1e-9), H(f1), np.abs(f1 - .5)]
        fc = np.where(p > .5, 0, 1); eb = (fc != y).astype(float); et = np.where(y == 2, 1.0, eb)
        fc1 = np.where(f1 > .5, 0, 1); et1 = np.where(y == 2, 1.0, (fc1 != y).astype(float))
        PS = persona(ds, j, ids); has = np.array([i in PS for i in ids]); nsa = int(has.sum())
        if nsa >= 200:
            ps = np.array([PS.get(i, .5) for i in ids]); sa_s = 1 - np.maximum(ps, 1 - ps)
            sa_fc = np.where(ps > .5, 0, 1); sa_et = np.where(y == 2, 1.0, (sa_fc != y).astype(float))
        acc = Acc(); rng = np.random.default_rng(0)
        for _ in range(a.splits):
            pm = rng.permutation(n); fit, cal, te = pm[: n // 4], pm[n // 4: n // 2], pm[n // 2:]
            calnt = cal[y[cal] != 2]
            Prf = head(F_full, y, fit); S3, E3 = graded_scores(Prf, y)
            Pr1 = head(F_one, y, fit); S31, E31 = graded_scores(Pr1, y)
            S0, E0 = graded_scores(Q, y)
            u, u1 = H(p), H(f1)
            ones = np.zeros(len(te), int)
            for al in AL:
                acc.add("P1", al, u[te] <= thr(u[calnt], eb[calnt], al), ones, et[te])
                acc.add("P1s", al, u1[te] <= thr(u1[calnt], (fc1 != y)[calnt].astype(float), al), ones, et1[te])
                acc.add("P2", al, u[te] <= thr(u[cal], et[cal], al), ones, et[te])
                lam = ltt(u[cal][:, None], et[cal][:, None], al, a.delta, u[fit][:, None], et[fit][:, None])
                acc.add("P3", al, u[te] <= lam, ones, et[te])
                lam = ltt(S3[cal][:, [0]], E3[cal][:, [0]], al, a.delta, S3[fit][:, [0]], E3[fit][:, [0]])
                acc.add("P4", al, S3[te, 0] <= lam, ones, E3[te, 0])
                if nsa >= 200:
                    ct, tt, ft = cal[has[cal]], te[has[te]], fit[has[fit]]
                    ctn = ct[y[ct] != 2]
                    for key, cc in (("T1", ctn), ("T2", ct)):
                        ec = np.where(y[cc] == 2, 1.0, (sa_fc[cc] != y[cc]).astype(float))
                        ef = np.where(y[ft] == 2, 1.0, (sa_fc[ft] != y[ft]).astype(float)) if key == "T2" else (sa_fc[ft] != y[ft]).astype(float)
                        lam = ltt(sa_s[cc][:, None], ec[:, None], al, a.delta, sa_s[ft][:, None], ef[:, None])
                        acc.add(key, al, sa_s[tt] <= lam, np.zeros(len(tt), int), sa_et[tt])
                for key, S, E in (("G0", S0, E0), ("G1", S31, E31), ("G2", S3, E3)):
                    ok, ch = _claims(S[te], ltt(S[cal], E[cal], al, a.delta, S[fit], E[fit]))
                    acc.add(key, al, ok, ch, E[te][np.arange(len(te)), np.maximum(ch, 0)])
                ok, ch = _claims(S3[te], ltt_joint(S3[cal], E3[cal], S3[fit], E3[fit], al))
                acc.add("G3", al, ok, ch, E3[te][np.arange(len(te)), np.maximum(ch, 0)])
        for key, name in ROWS:
            if key.startswith("T") and nsa < 200:
                continue
            L.append(f"| {spec} | {j} | {n}{f' (persona n={nsa})' if key.startswith('T') else ''} | {(y == 2).mean():.2f} | "
                     f"{key} {name} | " + " | ".join(acc.cell(key, al) for al in AL) + " |")
            print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
