"""Standard comparison table: LLM-judge uncertainty / tie baselines vs. the tie channel P(T), same judges, same pairs.
Post-hoc on existing predictions (no GPU).

Every method m maps a pair to (verdict v, tie score t, escalation score e); larger t = "call it a tie", larger e =
"send to a human". Tasks:
  tie   - AUROC of t for human ties vs strict pairs (sets with human ties).
  route - three-way router at a fixed human budget (default 20%): escalate the top-budget pairs by e, call T where
          t >= tau, otherwise output v; a human answer is always right, T is right on human ties, v on strict pairs.
          tau (including "never T") and the order of the two steps are cross-fitted (2-fold, 5 random splits).
          Reported as agreement with people over all pairs; "never T" alone is the classic selective judge.
  err   - AUROC of e for judge errors on benchmarks with verified labels (no ties): selective judging.
Methods (LLM-judge origin in brackets):
  FC-1     forced choice, one order; e = t = 1 - max prob                 [vanilla LLM-as-a-judge]
  FC-swap  both orders averaged; e = t = H(p_bar)                         [swap debiasing, Zheng et al. 2023]
  Swap-tie tie when the two orders disagree (MT-Bench rule); e = H(p_bar) [Zheng et al. 2023]
  Tie-verbal answer A/B/tie, argmax per order; t = share of orders answering tie [MLLM-as-a-Judge, Chen et al. 2024]
  Conflict t = JS divergence between orders; e = H(p_bar)                 [position inconsistency, TrustJudge]
  Score-arg pointwise 1-9 argmax, t = e = -|s_A - s_B|                    [pointwise grading]
  Score-exp pointwise expected score, t = e = -|E s_A - E s_B|            [G-Eval; TrustJudge DSS]
  Likert-7 7-point comparison; t = P(about equal), e = -|E - 4|           [Arena-Hard / Prometheus-style]
  TJ-agg   bidirectional A/B/T likelihood aggregation, hard tie decision  [TrustJudge LAA]
  Panel    mean p_bar over the four VLM judges; e = t = H(mean)           [PoLL, Verga et al. 2024]
  PromptEns tri under four instructions; e = t = H(mean A/B)              [prompt ensembles]
  SimAnn   four annotator personas; e = t = 1 - majority agreement        [simulated annotators, Jung et al. 2025]
  Ours     t = P(T) (tie prompt, both orders), e = H(p_bar), v = forced choice
  Ours-2call same, but v and e from the tie prompt (2 calls, same cost as FC-swap)
  Ours+PE  t = mean P(T) over the four instructions
  Ours+SA  t = mean P(T) over the four personas
  Ours-Panel t = mean P(T) over the four VLM judges, e and v from Panel (compute-matched to Panel)
usage (from $EAGLEVLM_DATA): python baseline_table.py [--budget 0.2] [--out paper_tables/baseline_table]"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import _sm, auroc

ap = argparse.ArgumentParser()
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen35_9b,qwen25vl7b")
ap.add_argument("--text_judges", default="qwen3_8b,qwen25_7b")
ap.add_argument("--judge_models", default="r1reward,llavacritic7b,mmrlhfrm7b", help="dedicated multimodal judge / reward models")
ap.add_argument("--dirs", default="results,results_A,/media/shawn/C/papers/VLM/judgebench_local/runs/*/raw")
ap.add_argument("--items", default="items,/media/shawn/C/papers/VLM/judgebench_local/items")
ap.add_argument("--tie_ds", default="genai_arena,mllm_judge,editreward,pp,mtbench_human,arena55k")
ap.add_argument("--ver_ds", default="mmrb2,vlrb,mmrb,mjbench,imagerewarddb,qbench2,mmrlhf")
ap.add_argument("--native", default="/media/shawn/C/papers/VLM/judgebench_local/runs/N_native", help="native-protocol outputs (native_judge.py, mmrlhf_score.py)")
ap.add_argument("--budget", type=float, default=0.2)
ap.add_argument("--splits", type=int, default=5)
ap.add_argument("--boot", type=int, default=200, help="bootstrap reps for router differences (0 = skip)")
ap.add_argument("--out", default="paper_tables/baseline_table")
a = ap.parse_args()
DIRS = [d for pat in a.dirs.split(",") for d in sorted(glob.glob(pat))]
VLM = a.judges.split(","); TXT = a.text_judges.split(","); JM = [m for m in a.judge_models.split(",") if m]
H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))
PE = ["tri", "tri_p1", "tri_p2", "tri_p3"]; SA = ["tri_sa1", "tri_sa2", "tri_sa3", "tri_sa4"]


def items(ds):
    for d in a.items.split(","):
        p = f"{d}/{ds}.jsonl"
        if os.path.exists(p):
            return {r["id"]: r for r in map(json.loads, open(p))}
    return {}


def load(ds, m):
    R = {}
    for d in DIRS:
        for suf in ("", "_edit", "_full", "_s32"):
            for f in glob.glob(f"{d}/{ds}{suf}__{m}.jsonl") + glob.glob(f"{d}/{ds}{suf}__{m}.s*.jsonl"):
                for l in open(f):
                    x = json.loads(l)
                    if "lp" in x:
                        R.setdefault(x["id"], {})[(x["cond"], x["order"])] = _sm(x["lp"])
    return R


def pair(r, c):
    """(P(cand0 better) both orders, p_AB(cand0), p_BA(cand0), P(T) mean) for bin/tri-family; None if missing."""
    if (c, "AB") not in r or (c, "BA") not in r:
        return None
    ab, ba = r[(c, "AB")], r[(c, "BA")]
    pab = ab["A"] / (ab["A"] + ab["B"]); pba = ba["B"] / (ba["A"] + ba["B"])
    return (pab + pba) / 2, pab, pba, (ab.get("T", 0) + ba.get("T", 0)) / 2


def methods(r, panel):
    """dict name -> (v: cand0 better?, t, e) for one pair; methods with missing inputs are absent."""
    out = {}
    b = pair(r, "bin")
    if b:
        p, pab, pba, _ = b
        u = H(p); v = p > .5
        out["FC-1"] = (pab > .5, 1 - max(pab, 1 - pab), 1 - max(pab, 1 - pab))
        out["FC-swap"] = (v, u, u)
        out["Swap-tie"] = (v, float((pab > .5) != (pba > .5)) + 1e-3 * u, u)
        out["Conflict"] = (v, H(p) - (H(pab) + H(pba)) / 2, u)
        tr = pair(r, "tri")
        if tr:
            out["Ours"] = (v, tr[3], u)
        if panel is not None:
            out["Panel"] = (panel[0] > .5, H(panel[0]), H(panel[0]))
            if panel[1] is not None:
                out["Ours-Panel"] = (panel[0] > .5, panel[1], H(panel[0]))
    if ("score", "A") in r and ("score", "B") in r:
        qa, qb = r[("score", "A")], r[("score", "B")]
        ea = sum(int(k) * x for k, x in qa.items()); eb = sum(int(k) * x for k, x in qb.items())
        aa = int(max(qa, key=qa.get)); ab_ = int(max(qb, key=qb.get))
        out["Score-arg"] = (aa > ab_ if aa != ab_ else ea > eb, -abs(aa - ab_), -abs(aa - ab_))
        out["Score-exp"] = (ea > eb, -abs(ea - eb), -abs(ea - eb))
    if ("str7", "AB") in r and ("str7", "BA") in r:
        ab, ba = r[("str7", "AB")], r[("str7", "BA")]
        d = np.array([(ab[str(k)] + ba[str(8 - k)]) / 2 for k in range(1, 8)]); d = d / d.sum()
        E = float((np.arange(1, 8) * d).sum())
        out["Likert-7"] = (E < 4, d[3], -abs(E - 4))
    tr = pair(r, "tri")
    if tr and ("tri", "AB") in r:          # MLLM-as-a-Judge pair comparison: answer A, B or tie; read the argmax
        ab, ba = r[("tri", "AB")], r[("tri", "BA")]
        nT = sum(float(max(d, key=d.get) == "T") for d in (ab, ba)) / 2
        out["Tie-verbal"] = (tr[0] > .5, nT + 1e-3 * H(tr[0]), H(tr[0]))
    if tr:                                  # 2-call variant: verdict and U from the tie prompt renormalized over A/B
        out["Ours-2call"] = (tr[0] > .5, tr[3], H(tr[0]))
    if ("tri", "AB") in r and ("tri", "BA") in r:
        ab, ba = r[("tri", "AB")], r[("tri", "BA")]
        mA, mB, mT = ab["A"] + ba["B"], ab["B"] + ba["A"], ab["T"] + ba["T"]
        tie = float(mT > max(mA, mB))
        out["TJ-agg"] = (mA > mB, tie + 1e-3 * H(mA / (mA + mB)), H(mA / (mA + mB)))
    for name, cs, ours in (("PromptEns", PE, "Ours+PE"), ("SimAnn", SA, "Ours+SA")):
        ps = [pair(r, c) for c in cs]
        if all(ps):
            pm = np.mean([x[0] for x in ps])
            if name == "SimAnn":    # Jung et al.: confidence = agreement of simulated annotators with the majority
                maj = pm > .5; agree = np.mean([(x[0] > .5) == maj for x in ps]) + 1e-3 * abs(pm - .5)
                out[name] = (maj, 1 - agree, 1 - agree)
            else:
                out[name] = (pm > .5, H(pm), H(pm))
            if b:
                out[ours] = (b[0] > .5, np.mean([x[3] for x in ps]), H(b[0]))
    return out


def route(v, t, e, y_tie, y0, budget, rng, splits):
    """Cross-fitted three-way router; returns mean agreement with people (all pairs) and mean T rate."""
    n = len(t); k = int(round(budget * n))
    taus = np.r_[np.inf, np.quantile(t, np.linspace(.4, .995, 60))]

    def acc(idx, tau, first):
        tt, ee = t[idx], e[idx]; kk = int(round(budget * len(idx)))
        esc = np.zeros(len(idx), bool); T = np.zeros(len(idx), bool)
        if first == "esc":
            esc[np.argsort(-ee)[:kk]] = True; T = (~esc) & (tt >= tau)
        else:
            T = tt >= tau; rest = np.where(~T)[0]; esc[rest[np.argsort(-ee[rest])[:kk]]] = True
        right = esc | (T & y_tie[idx]) | (~esc & ~T & ~y_tie[idx] & (v[idx] == y0[idx]))
        return right.mean(), T.mean()
    res = []
    for _ in range(splits):
        perm = rng.permutation(n); halves = [perm[: n // 2], perm[n // 2:]]
        for fit, ev in (halves, halves[::-1]):
            best = max(((acc(fit, tau, f)[0], tau, f) for tau in taus for f in ("esc", "tie")), key=lambda z: z[0])
            res.append(acc(ev, best[1], best[2]))
    return float(np.mean([x[0] for x in res])), float(np.mean([x[1] for x in res]))


ORDER = ["FC-1", "FC-swap", "Tie-verbal", "Swap-tie", "Conflict", "Score-arg", "Score-exp", "Likert-7", "TJ-agg", "Panel",
         "PromptEns", "SimAnn", "Ours-2call", "Ours", "Ours+PE", "Ours+SA", "Ours-Panel"]
rows = []; ARR = {}
rng = np.random.default_rng(0)
for ds in a.tie_ds.split(",") + a.ver_ds.split(","):
    I = items(ds)
    if not I:
        continue
    is_tie = ds in a.tie_ds.split(",")
    judges = VLM + (TXT if ds in ("mtbench_human", "arena55k") else JM)
    R = {m: load(ds, m) for m in judges}
    pan = {}
    for i in I:
        ps = [pair(R[m][i], "bin") for m in VLM if i in R[m]]
        ts = [pair(R[m][i], "tri") for m in VLM if i in R[m]]
        if len(ps) == len(VLM) and all(ps):
            pan[i] = (float(np.mean([x[0] for x in ps])),
                      float(np.mean([x[3] for x in ts])) if len(ts) == len(VLM) and all(ts) else None)
    for m in judges:
        M = {i: methods(R[m][i], pan.get(i) if m in VLM else None) for i in R[m] if i in I}
        for name in ORDER:
            ids = [i for i in M if name in M[i] and str(I[i]["label"]) in ("0", "1", "tie")]
            if len(ids) < 50:
                continue
            v = np.array([M[i][name][0] for i in ids]); t = np.array([M[i][name][1] for i in ids], float)
            e = np.array([M[i][name][2] for i in ids], float)
            lab = np.array([str(I[i]["label"]) for i in ids]); y_tie = lab == "tie"; y0 = lab == "0"
            row = {"ds": ds, "judge": m, "method": name, "n": len(ids), "n_tie": int(y_tie.sum())}
            s = ~y_tie
            row["acc"] = float((v[s] == y0[s]).mean())
            ARR[(ds, m, name)] = dict(zip(ids, zip(v, t, e, y_tie, y0)))
            if is_tie and y_tie.sum() >= 15:
                row["tie_auroc"] = float(auroc(t, y_tie))
                row["route"], row["T_rate"] = route(v, t, e, y_tie, y0, a.budget, rng, a.splits)
            if not y_tie.any():
                err = v != y0
                row["err_auroc"] = float(auroc(e, err))
                kk = int(round(a.budget * len(ids))); keep = np.argsort(-e)[kk:]
                row["sel_acc"] = float((~err[keep]).mean())
            rows.append(row)
    print("done", ds, flush=True)

# ---- native protocols of the dedicated judges (response sets only): pseudo-judges "<model>@native", method "Native"
def native_methods():
    out = {}                                   # (model, id) -> (v, t, e)
    for m in ("r1reward", "llavacritic7b"):
        p = os.path.join(a.native, f"{m}.jsonl")
        if not os.path.exists(p):
            continue
        V = {}
        for l in open(p):
            x = json.loads(l); flip = x["order"] == "BA"
            for k, z in enumerate([x["parsed"]["greedy"]] + x["parsed"]["samples"]):
                if z is None:
                    continue
                c = "T" if z == "T" else ("0" if (z == "1") != flip else "1")
                V.setdefault(x["id"], []).append((c, k == 0))
        for i, vs in V.items():
            n = len(vs); n0 = sum(c == "0" for c, _ in vs); n1 = sum(c == "1" for c, _ in vs); nT = n - n0 - n1
            g = [c for c, gr in vs if gr and c != "T"]
            v = n0 > n1 if n0 != n1 else (g[0] == "0" if g else True)
            top = max(n0, n1, nT) / n
            t = (nT / n + 1e-3 * (1 - top)) if m == "llavacritic7b" else 1 - top
            out[(m, i)] = (v, t, 1 - top)
    p = os.path.join(a.native, "mmrlhf_score.jsonl")
    if os.path.exists(p):
        Rw = {}
        for l in open(p):
            x = json.loads(l); Rw.setdefault(x["id"], {})[x["cand"]] = x["reward"]
        for i, d in Rw.items():
            if len(d) == 2:
                dl = d[0] - d[1]; out[("mmrlhfrm7b", i)] = (dl > 0, -abs(dl), -abs(dl))
    return out


NAT = native_methods()
for ds in ["mllm_judge", "vlrb", "mmrb", "mmrlhf"]:
    I = items(ds)
    for m in ("r1reward", "llavacritic7b", "mmrlhfrm7b"):
        ids = [i for i in I if (m, i) in NAT and str(I[i]["label"]) in ("0", "1", "tie")]
        if len(ids) < 50:
            continue
        v = np.array([NAT[(m, i)][0] for i in ids]); t = np.array([NAT[(m, i)][1] for i in ids], float)
        e = np.array([NAT[(m, i)][2] for i in ids], float)
        lab = np.array([str(I[i]["label"]) for i in ids]); y_tie = lab == "tie"; y0 = lab == "0"; s_ = ~y_tie
        row = {"ds": ds, "judge": m + "@native", "method": "Native", "n": len(ids), "n_tie": int(y_tie.sum()),
               "acc": float((v[s_] == y0[s_]).mean())}
        if y_tie.sum() >= 15:
            row["tie_auroc"] = float(auroc(t, y_tie))
            row["route"], row["T_rate"] = route(v, t, e, y_tie, y0, a.budget, rng, a.splits)
        else:
            err = v != y0; row["err_auroc"] = float(auroc(e, err))
            kk = int(round(a.budget * len(ids))); row["sel_acc"] = float((~err[np.argsort(-e)[kk:]]).mean())
        rows.append(row)
print("native rows", sum(r["method"] == "Native" for r in rows), flush=True)

CMP = [("Ours", "FC-swap"), ("Ours", "TJ-agg"), ("Ours-2call", "FC-swap"), ("Ours-Panel", "Panel"),
       ("Ours+SA", "SimAnn"), ("Ours+PE", "PromptEns")]
ci = []
if a.boot:
    for ds in a.tie_ds.split(","):
        for m in VLM + TXT:
            for x, y in CMP:
                if (ds, m, x) not in ARR or (ds, m, y) not in ARR:
                    continue
                A, B = ARR[(ds, m, x)], ARR[(ds, m, y)]; ids = sorted(set(A) & set(B))
                if sum(A[i][3] for i in ids) < 15:
                    continue
                arr = lambda D, idx: [np.array([D[ids[j]][k] for j in idx]) for k in range(5)]
                full = np.arange(len(ids))
                d0 = route(*[arr(A, full)[k] for k in (0, 1, 2, 3, 4)], a.budget, rng, a.splits)[0] - \
                     route(*[arr(B, full)[k] for k in (0, 1, 2, 3, 4)], a.budget, rng, a.splits)[0]
                bs = []
                for _ in range(a.boot):
                    j = rng.integers(0, len(ids), len(ids))
                    bs.append(route(*arr(A, j), a.budget, rng, 1)[0] - route(*arr(B, j), a.budget, rng, 1)[0])
                lo, hi = np.percentile(bs, [2.5, 97.5])
                ci.append({"ds": ds, "judge": m, "cmp": f"{x} - {y}", "diff": float(d0), "lo": float(lo), "hi": float(hi)})
        print("ci", ds, flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(ci, open(a.out + "_ci.json", "w"), indent=1)
json.dump(rows, open(a.out + ".json", "w"), indent=1)


def table(metric, dss, judges, title):
    L = [f"### {title}", "", "| method | " + " | ".join(dss) + " | mean |", "|---" * (len(dss) + 2) + "|"]
    for name in ORDER:
        vals = []
        for ds in dss:
            x = [r[metric] for r in rows if r["ds"] == ds and r["method"] == name and r["judge"] in judges and metric in r]
            vals.append(np.mean(x) if len(x) == len(judges) else np.nan)
        if all(np.isnan(vals)):
            continue
        cell = lambda z: "-" if np.isnan(z) else f"{z:.3f}"
        mean = np.nanmean(vals) if not any(np.isnan(vals)) else np.nan
        L.append(f"| {name} | " + " | ".join(cell(z) for z in vals) + f" | {cell(mean)} |")
    return "\n".join(L) + "\n"


img_tie = [d for d in ["genai_arena", "mllm_judge", "editreward", "pp"]]
txt_tie = ["mtbench_human", "arena55k"]
ver = a.ver_ds.split(",")
md = [f"# Baseline table (budget {a.budget:.0%}); cells = mean over judges; '-' = method missing for some judge\n"]
for js, lab in ((VLM, "4 VLM judges"), (["qwen3vl8b", "internvl35_8b"], "Qwen3-VL-8B + InternVL3.5-8B (all methods)")):
    md.append(table("tie_auroc", img_tie + txt_tie, js, f"Tie detection AUROC - {lab}"))
    md.append(table("route", img_tie + txt_tie, js, f"Router agreement with people at {a.budget:.0%} human budget - {lab}"))
    md.append(table("err_auroc", ver, js, f"Error detection AUROC (verified labels) - {lab}"))
    md.append(table("sel_acc", ver, js, f"Accuracy after escalating {a.budget:.0%} - {lab}"))
md.append(table("tie_auroc", txt_tie, TXT, "Tie detection AUROC - text LLM judges (Qwen3-8B, Qwen2.5-7B)"))
md.append(table("route", txt_tie, TXT, "Router agreement - text LLM judges"))
for m in VLM + TXT:
    md.append(table("tie_auroc", img_tie + txt_tie if m in VLM else txt_tie, [m], f"Tie AUROC - {m}"))
    md.append(table("route", img_tie + txt_tie if m in VLM else txt_tie, [m], f"Router - {m}"))
md.append("### Router differences, paired bootstrap 95% CI (cross-fitted thresholds refit in each replicate)\n")
md.append("| set | judge | comparison | diff | CI |\n|---|---|---|---|---|")
for c in ci:
    md.append(f"| {c['ds']} | {c['judge']} | {c['cmp']} | {c['diff']:+.3f} | [{c['lo']:+.3f}, {c['hi']:+.3f}]{' *' if c['lo'] > 0 else ''} |")
open(a.out + ".md", "w").write("\n".join(md))
print("\n".join(md[:4]))
