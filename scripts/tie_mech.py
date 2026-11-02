"""What kind of 'tie' does P(T) detect?  Shared loaders + analyses for the tie-mechanism follow-ups (2026-11-02).
  types   GenAI-Arena: human ties are mostly 'both bad' (bothbad_vote) and a few 'equally good' (tievote).
          AUROC of P(T) and of forced-choice U for each tie type vs. decided pairs, and bothbad vs tievote,
          per judge, with paired bootstrap CIs.
  sim     Is P(T) just a candidate-similarity detector?  Similarity covariates from pair_sim.py.  Per dataset/judge:
          tie AUROC of each similarity alone; Spearman of P(T) with them; P(T) AUROC within similarity quintiles;
          5-fold CV logistic AUROC of [sims], [sims + P(T)], [sims + U], [sims + P(T) + U].
usage (from $EAGLEVLM_DATA): python tie_mech.py types|sim [--out paper_tables/tie_<what>]"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import _sm, auroc

DIRS_DEF = "results,results_A,/media/shawn/C/papers/VLM/judgebench_local/runs/*/raw"
ITEMS_DEF = "items,/media/shawn/C/papers/VLM/judgebench_local/items"
JUDGES = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "r1reward", "llavaov7b", "llavacritic7b", "mmrlhfrm7b"]
H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))


def items(ds, dirs=ITEMS_DEF):
    for d in dirs.split(","):
        p = f"{d}/{ds}.jsonl"
        if os.path.exists(p):
            return {r["id"]: r for r in map(json.loads, open(p))}
    return {}


def load(ds, m, dirs=DIRS_DEF, conds=None):
    R = {}
    for d in [d for pat in dirs.split(",") for d in sorted(glob.glob(pat))]:
        for suf in ("", "_edit", "_full", "_s32"):
            for f in glob.glob(f"{d}/{ds}{suf}__{m}.jsonl") + glob.glob(f"{d}/{ds}{suf}__{m}.s*.jsonl"):
                for l in open(f):
                    x = json.loads(l)
                    if "lp" in x and (conds is None or x["cond"] in conds):
                        R.setdefault(x["id"], {})[(x["cond"], x["order"])] = _sm(x["lp"])
    return R


def signals(r, tri="tri", bin_="bin"):
    """(P(T) mean over orders, forced-choice U from bin, p(cand0) from bin) or None"""
    try:
        ab, ba = r[(tri, "AB")], r[(tri, "BA")]
        b1, b2 = r[(bin_, "AB")], r[(bin_, "BA")]
    except KeyError:
        return None
    pt = (ab.get("T", 0) + ba.get("T", 0)) / 2
    p = (b1["A"] / (b1["A"] + b1["B"]) + b2["B"] / (b2["A"] + b2["B"])) / 2
    return pt, H(p), p


def boot_auc(s, y, B=1000, seed=0):
    rng = np.random.default_rng(seed); s = np.asarray(s); y = np.asarray(y); n = len(y)
    out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        out.append(auroc(s[i], y[i]))
    return np.nanpercentile(out, [2.5, 97.5])


def boot_diff(s1, s2, y, B=1000, seed=0):
    rng = np.random.default_rng(seed); s1, s2, y = map(np.asarray, (s1, s2, y)); n = len(y)
    d = [auroc(s1[i], y[i]) - auroc(s2[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.nanpercentile(d, [2.5, 97.5])


def types(a):
    I = items("genai_arena")
    L = ["# GenAI-Arena: which human ties does P(T) detect?", "",
         "Tie types: bothbad = voter clicked 'both are bad'; tievote = 'tie' (equally good). Decided = left/right vote.",
         "Cells: AUROC [95% CI].  Δ = AUROC(P(T)) - AUROC(U), paired bootstrap.", ""]
    for sub in ("all", "generation", "edition"):
        L += [f"## subset: {sub}", "",
              "| judge | n bb/tie/dec | bothbad vs dec: P(T) | U | Δ | tievote vs dec: P(T) | U | Δ | bothbad vs tievote: P(T) | U |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for j in JUDGES:
            R = load("genai_arena", j)
            rows = []
            for i, r in R.items():
                if i not in I or (sub != "all" and I[i]["subset"] != sub):
                    continue
                s = signals(r)
                if s:
                    rows.append((I[i]["meta"]["vote_type"], s[0], s[1]))
            if not rows:
                continue
            vt = np.array([x[0] for x in rows]); pt = np.array([x[1] for x in rows]); u = np.array([x[2] for x in rows])
            dec = np.isin(vt, ["leftvote", "rightvote"])
            cells = [f"{(vt == 'bothbad_vote').sum()}/{(vt == 'tievote').sum()}/{dec.sum()}"]
            for t in ("bothbad_vote", "tievote"):
                m = dec | (vt == t); y = (vt[m] == t).astype(int)
                for s in (pt[m], u[m]):
                    lo, hi = boot_auc(s, y, a.boot)
                    cells.append(f"{auroc(s, y):.3f} [{lo:.2f},{hi:.2f}]")
                lo, hi = boot_diff(pt[m], u[m], y, a.boot)
                d = auroc(pt[m], y) - auroc(u[m], y)
                cells.append(f"{d:+.3f} [{lo:+.2f},{hi:+.2f}]" + (" *" if lo > 0 or hi < 0 else ""))
            m = np.isin(vt, ["bothbad_vote", "tievote"]); y = (vt[m] == "bothbad_vote").astype(int)
            cells += [f"{auroc(pt[m], y):.3f}", f"{auroc(u[m], y):.3f}"]
            L.append(f"| {j} | " + " | ".join(cells) + " |")
        L.append("")
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def sim(a):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    S = {}
    for l in open(a.sim):
        x = json.loads(l); S[(x["ds"], x["id"])] = x

    def feats(x):
        if "dino" in x:
            f = {"dino": x["dino"], "clip": x["clip"], "-pix": -x["pix"]}
            if "dino_src" in x:      # edit pairs: how unchanged is the less-changed / more-changed candidate
                f.update({"src_dino_max": max(x["dino_src"]), "src_dino_min": min(x["dino_src"]), "-src_pix_min": -min(x["pix_src"])})
            return f
        return {"bge": x["bge"], "lex": x["lex"], "lenratio": x["lenratio"]}

    def cv_auc(X, y, seed=0):
        if len(set(y)) < 2:
            return float("nan")
        out = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
            out[te] = m.predict_proba(X[te])[:, 1]
        return auroc(out, y)

    L = ["# Is P(T) a candidate-similarity detector?", "",
         "y = human tie (GenAI-Arena also split by tie type).  CV = 5-fold logistic regression AUROC.", ""]
    for dss in a.ds.split(","):
        ds, _, subset = dss.partition(":")          # "genai_arena:edition" restricts to one subset
        I = {k: v for k, v in items(ds).items() if not subset or v["subset"] == subset}
        tgt = {"all ties": lambda it: it["label"] == "tie"}
        if ds == "genai_arena":
            tgt["bothbad"] = lambda it: it["meta"]["vote_type"] == "bothbad_vote"
            tgt["tievote"] = lambda it: it["meta"]["vote_type"] == "tievote"
        if ds == "arena_tt" or ds == "synth_tie":
            for sb in sorted({it["subset"] for it in I.values() if it["label"] == "tie"}):
                tgt[sb] = (lambda sb: lambda it: it["subset"] == sb)(sb)
        for tn, tf in tgt.items():
            L += [f"## {dss} — {tn}", ""]
            first = True
            for j in a.judges.split(","):
                R = load(ds, j)
                rows = []
                for i, r in R.items():
                    if i not in I or (ds, i) not in S:
                        continue
                    it = I[i]
                    if tn != "all ties" and it["label"] == "tie" and not tf(it):
                        continue        # tie-type target: other ties are left out
                    sg = signals(r)
                    if sg:
                        rows.append((int(tf(it)), sg[0], sg[1], feats(S[(ds, i)])))
                if len(rows) < 50:
                    continue
                y = np.array([x[0] for x in rows]); pt = np.array([x[1] for x in rows]); u = np.array([x[2] for x in rows])
                fn = [k for k in rows[0][3] if all(k in x[3] for x in rows)]; F = np.array([[x[3][k] for k in fn] for x in rows])
                if first:
                    L += ["similarity alone (AUROC for tie): " + ", ".join(f"{k} {auroc(F[:, c], y):.3f}" for c, k in enumerate(fn)), "",
                          "| judge | n (ties) | P(T) | U | ρ(P(T), " + "/".join(fn) + ") | P(T) within " + fn[0] + " quintiles | CV sims | +P(T) | +U | +both |",
                          "|---|---|---|---|---|---|---|---|---|---|"]
                    first = False
                rho = "/".join(f"{spearmanr(pt, F[:, c]).statistic:+.2f}" for c in range(len(fn)))
                q = np.quantile(F[:, 0], [.2, .4, .6, .8]); b = np.searchsorted(q, F[:, 0])
                ws = [(auroc(pt[b == k], y[b == k]), (b == k).sum()) for k in range(5)]
                ws = [(v, w) for v, w in ws if not np.isnan(v)]
                strat = sum(v * w for v, w in ws) / sum(w for _, w in ws)
                lp, lu = np.log(pt + 1e-4), np.log(u + 1e-4)          # P(T) is heavy-tailed near 0: log scale for the linear model
                c0 = cv_auc(F, y); c1 = cv_auc(np.c_[F, lp], y); c2 = cv_auc(np.c_[F, lu], y); c3 = cv_auc(np.c_[F, lp, lu], y)
                L.append(f"| {j} | {len(y)} ({y.sum()}) | {auroc(pt, y):.3f} | {auroc(u, y):.3f} | {rho} | {strat:.3f} | {c0:.3f} | {c1:.3f} | {c2:.3f} | {c3:.3f} |")
            L.append("")
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["types", "sim"])
    ap.add_argument("--ds", default="genai_arena,editreward,pp")
    ap.add_argument("--sim", default="paper_tables/pair_sim.jsonl")
    ap.add_argument("--judges", default=",".join(JUDGES))
    ap.add_argument("--out", default="")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    a.out = a.out or f"paper_tables/tie_{a.what}"
    {"types": types, "sim": sim}[a.what](a)
