"""Write the LaTeX tables of the 'close or can't tell' paper from saved results. Re-runnable: every cell whose result
does not exist yet is \\PH{...} (red placeholder), so the paper always compiles and fills in as results arrive.
usage (from $EAGLEVLM_DATA): python make_overleaf_tables.py --out /media/shawn/C/papers/VLM/overleaf/tables
         [--full /media/shawn/C/papers/VLM/judgebench_full] [--h100 h100]   (h100 = dir with collected server runs)"""
import argparse, collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import _sm, auroc

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
ap.add_argument("--full", default="/media/shawn/C/papers/VLM/judgebench_full"); ap.add_argument("--h100", default="h100")
ap.add_argument("--boot", type=int, default=1000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
J4 = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"]
JN = {"qwen3vl8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B", "qwen35_9b": "Qwen3.5-9B", "qwen25vl7b": "Qwen2.5-VL-7B"}
DN = {"genai_arena": "GenAI-Arena", "mllm_judge": "MLLM-as-a-Judge", "pp": "Place Pulse", "editreward": "EditReward",
      "mmrb2": "MMRB2", "vlrb": "VL-RewardBench", "mmrb": "Multimodal RB", "mjbench": "MJ-Bench", "imagerewarddb": "ImageRewardDB",
      "qbench2": "Q-Bench2", "mmrlhf": "MM-RLHF", "pp_full": "Place Pulse (tie-rich)", "mtbench_human": "MT-Bench human",
      "arena55k": "LMArena-55k"}
PH = lambda s="": "\\PH{" + s + "}"
f3 = lambda x: PH() if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}".lstrip("0").replace("-0.", "-.")
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
pp_ok = all(os.path.exists(f"results/pp__{m}.jsonl") for m in ("qwen35_9b", "qwen25vl7b"))   # PP re-run with current harness


def pp_fresh(js):
    """PP numbers in a summary JSON are usable only if it was written after the PP re-run (pilot outputs differed)."""
    return pp_ok and os.path.exists(js) and os.path.getmtime(js) > max(os.path.getmtime(f"results/pp__{m}.jsonl") for m in ("qwen35_9b", "qwen25vl7b"))


def write(name, body):
    open(os.path.join(a.out, name), "w").write(body); print("wrote", name)


# ---------------------------------------------------------------- T1 datasets
def count(path):
    if not os.path.exists(path):
        return None, None, None
    L = [json.loads(l) for l in open(path)]
    return len(L), sum(r["label"] == "tie" for r in L), L[0]["kind"]


rows = []
for d in ["genai_arena", "mllm_judge", "imagerewarddb", "editreward", "mmrb2", "pp", "vlrb", "mmrb", "mjbench", "qbench2", "mmrlhf"]:
    n, t, k = count(f"items/{d}.jsonl"); nf, tf, _ = count(f"{a.full}/items/{d}.jsonl")
    if d == "pp":
        nf, tf, _ = count(f"{a.full}/items/pp_full.jsonl")
    tgt = {"mmrb2": "3 annotators (1--6)", "genai_arena": "crowd tie / both-bad", "mllm_judge": "expert tie",
           "imagerewarddb": "equal expert rank", "editreward": "expert tie", "pp": "crowd equal vote"}.get(d, "verified (no ties)")
    rows.append(f"{DN[d]} & {'image' if k == 'img' else 'response'} & {n} & {t} & {nf if nf else '--'} & {tf if nf else '--'} & {tgt} \\\\")
for d in ["mtbench_human", "arena55k"]:
    nf, tf, _ = count(f"{a.full}/items/{d}.jsonl")
    rows.append(f"{DN[d]} & text & -- & -- & {nf if nf else PH()} & {tf if nf else PH()} & "
                f"{'experts, 1--5 votes per pair' if d == 'mtbench_human' else 'crowd tie'} \\\\")
write("datasets.tex", "\\begin{tabular}{llrrrrl}\\toprule\nDataset & Candidates & \\multicolumn{2}{c}{Capped set} & "
      "\\multicolumn{2}{c}{Full set} & Aleatoric target \\\\\n & & pairs & ties & pairs & ties & \\\\\\midrule\n"
      + "\n".join(rows[:6]) + "\n\\midrule\n" + "\n".join(rows[6:11]) + "\n\\midrule\n" + "\n".join(rows[11:]) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T2 forced choice conflates (tie_vs_unsure.json)
T = J("paper_tables/tie_vs_unsure.json") or {}
r2 = []
for d in ["genai_arena", "mllm_judge", "editreward", "pp"]:
    x = T.get(d, {}); ta = x.get("tie_auroc", {}); lc = x.get("lowconf_tie_vs_err", {}); stale = d == "pp" and not pp_fresh("paper_tables/tie_vs_unsure.json")
    g = lambda k, D: PH("rerun") if stale else f3(D.get(k))
    r2.append(f"{DN[d]} & {x.get('n_tie', PH())} & {g('U', ta)} & {g('ens:U', ta)} & {g('A', ta)} & {g('E', ta)} & {g('-pwgap', ta)} & "
              f"{g('E', lc)} & {g('A', lc)} \\\\")
m2 = T.get("mmrb2", {})
r2.append(f"\\midrule\nMMRB2 split vs.\\ unanimous & {round(m2.get('mmrb2_split_share', 0) * 1500) if m2 else PH()} & "
          f"{f3(m2.get('mmrb2_auroc_split_vs_unanimous', {}).get('U'))} & {f3(m2.get('mmrb2_auroc_split_vs_unanimous', {}).get('ens:U'))} & "
          f"{f3(m2.get('mmrb2_auroc_split_vs_unanimous', {}).get('A'))} & {f3(m2.get('mmrb2_auroc_split_vs_unanimous', {}).get('E'))} & "
          f"{f3(m2.get('mmrb2_auroc_split_vs_unanimous', {}).get('-pwgap'))} & -- & -- \\\\")
write("conflation.tex", "\\begin{tabular}{lrccccccc}\\toprule\n & & \\multicolumn{5}{c}{Tie (or split) AUROC} & "
      "\\multicolumn{2}{c}{Low-conf.\\ tie vs.\\ error} \\\\\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}\n"
      "Dataset & ties & $U$ & $U_\\text{ens}$ & $A$ & $E$ & $-|\\Delta s|$ & $E$ & $A$ \\\\\\midrule\n" + "\n".join(r2)
      + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T3 shared errors
M = J("paper_tables/mechanism_4j.json") or {}; X = J("paper_tables/mechanism_cross.json") or {}
r3 = []
for d in ["vlrb", "mmrb", "mllm_judge", "genai_arena", "mjbench", "mmrb2", "imagerewarddb", "qbench2", "pp"]:
    m, x = M.get(d, {}), X.get(d, {}); stale = d == "pp" and not (pp_fresh("paper_tables/mechanism_4j.json") and pp_fresh("paper_tables/mechanism_cross.json"))
    if stale:
        r3.append(f"{DN[d]} & " + " & ".join([PH("rerun")] * 7) + " \\\\"); continue
    au = x.get("auroc_all", {}); ex = x.get("auroc_excl_shared", {})
    r3.append(f"{DN[d]} & {f3(m.get('share_k_wrong', [None] * 5)[-1])} & {f3(m.get('unanimous_share_of_errors'))} & "
              f"{f3(m.get('conf_correct'))} & {f3(m.get('conf_unanimous_err'))} & {f3(m.get('conf_idiosyncratic_err'))} & "
              f"{f3(x.get('ivl_wrong_given_q', {}).get('3'))} & "
              f"{f3(np.mean(list(au.values())) if au else None)}$\\to${f3(np.mean(list(ex.values())) if ex else None)} \\\\")
write("shared_errors.tex", "\\begin{tabular}{lccccccc}\\toprule\n & all 4 & shared / & \\multicolumn{3}{c}{mean confidence $|2\\bar p-1|$} & "
      "InternVL wrong & AUROC all \\\\\\cmidrule(lr){4-6}\nDataset & wrong & errors & correct & shared err. & own err. & "
      "$\\mid$ 3 Qwen wrong & $\\to$ excl.\\ shared \\\\\\midrule\n" + "\n".join(r3) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T4 explicit channel (phase A), with paired bootstrap
def load(ds, m, dirs=None):
    """local results first, then any collected server run (h100/<phase>/merged); later rows of the same key win."""
    import glob as _g
    dirs = dirs or ["results", "results_A"] + sorted(_g.glob(f"{a.h100}/*/merged"))
    R = collections.defaultdict(dict)
    for dd in dirs:
        f = f"{dd}/{ds}__{m}.jsonl"
        if os.path.exists(f):
            for l in open(f):
                x = json.loads(l)
                if "lp" in x:
                    R[x["id"]][(x["cond"], x["order"])] = x["lp"]
    return R


H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))


def channel(ds, m):
    items = {json.loads(l)["id"]: json.loads(l) for l in open(f"items/{ds}.jsonl")}
    R = load(ds, m); U, PT, tie, y, pb = [], [], [], [], []
    for i, r in R.items():
        if i not in items or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            continue
        p = (_sm(r[("bin", "AB")])["A"] + _sm(r[("bin", "BA")])["B"]) / 2
        U.append(H(p)); pb.append(p); PT.append((_sm(r[("tri", "AB")])["T"] + _sm(r[("tri", "BA")])["T"]) / 2)
        tie.append(items[i]["label"] == "tie"); y.append(items[i]["label"] == 0)
    if len(U) < 0.9 * len(items):
        return None
    return np.array(U), np.array(PT), np.array(tie), np.array(y), np.array(pb)


def boot_delta(s1, s2, lab, B):
    rng = np.random.default_rng(0); n = len(lab); d = []
    for _ in range(B):
        k = rng.integers(0, n, n)
        if lab[k].min() == lab[k].max():
            continue
        d.append(auroc(s1[k], lab[k]) - auroc(s2[k], lab[k]))
    return np.percentile(d, [2.5, 97.5])


r4, r5 = [], []
for d in ["genai_arena", "mllm_judge", "editreward", "pp"]:
    cells = []
    for m in J4:
        c = channel(d, m) if (d != "pp" or m == "qwen3vl8b" or pp_ok) else None
        if c is None:
            cells += [PH(), PH(), PH()]; continue
        U, PT, tie, y, pb = c
        lo, hi = boot_delta(PT, U, tie, a.boot)
        cells += [f3(auroc(U, tie)), f3(auroc(PT, tie)), f"[{lo:+.2f},{hi:+.2f}]"]
    r4.append(f"{DN[d]} & " + " & ".join(cells) + " \\\\")
for d in ["genai_arena", "mllm_judge", "editreward", "pp"]:
    cells = []
    for m in J4:
        c = channel(d, m) if (d != "pp" or m == "qwen3vl8b" or pp_ok) else None
        if c is None:
            cells += [PH(), PH()]; continue
        U, PT, tie, y, pb = c
        err = ~tie & ((pb > .5) != y); low = U >= np.quantile(U, .7); sel = low & (tie | err)
        cells += [f3(auroc(U[sel], tie[sel])), f3(auroc(PT[sel], tie[sel]))]
    r5.append(f"{DN[d]} & " + " & ".join(cells) + " \\\\")
hdr = " & ".join(f"\\multicolumn{{3}}{{c}}{{{JN[m]}}}" for m in J4)
write("channel_tie.tex", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "ccc" * 4 + "}\\toprule\n & " + hdr + " \\\\\n"
      + "".join(f"\\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(4)) + "\nDataset" + " & $U$ & $P(T)$ & 95\\% CI $\\Delta$" * 4
      + " \\\\\\midrule\n" + "\n".join(r4) + "\n\\bottomrule\\end{tabular}}\n")
write("channel_lowconf.tex", "\\begin{tabular}{l" + "cc" * 4 + "}\\toprule\n & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{JN[m]}}}" for m in J4)
      + " \\\\\n" + "".join(f"\\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(4)) + "\nDataset" + " & $U$ & $P(T)$" * 4
      + " \\\\\\midrule\n" + "\n".join(r5) + "\n\\bottomrule\\end{tabular}\n")

# error detection on verified sets: U vs P(T) (dissociation)
r6 = []
for d in ["vlrb", "mmrb", "mjbench", "qbench2", "mmrlhf"]:
    cells = []
    for m in J4:
        c = channel(d, m)
        if c is None:
            cells += [PH(), PH()]; continue
        U, PT, tie, y, pb = c; err = (pb > .5) != y
        cells += [f3(auroc(U, err)), f3(auroc(PT, err))]
    r6.append(f"{DN[d]} & " + " & ".join(cells) + " \\\\")
write("channel_err.tex", "\\begin{tabular}{l" + "cc" * 4 + "}\\toprule\n & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{JN[m]}}}" for m in J4)
      + " \\\\\n" + "".join(f"\\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(4)) + "\nDataset" + " & $U$ & $P(T)$" * 4
      + " \\\\\\midrule\n" + "\n".join(r6) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- pre-registered skeletons for the server phases
SC = ["Qwen3-VL 2B", "4B", "8B", "32B", "Qwen2.5-VL 3B", "7B", "32B", "72B", "InternVL3.5 1B", "2B", "4B", "8B", "14B", "38B"]
write("scale.tex", "\\begin{tabular}{lccccc}\\toprule\nJudge & acc. & tie AUROC $P(T)$ & tie AUROC $U$ & err.\\ AUROC $U$ & shared-error rate \\\\\\midrule\n"
      + "\n".join(f"{s} & " + " & ".join([PH()] * 5) + " \\\\" for s in SC) + "\n\\bottomrule\\end{tabular}\n")
write("text_gap.tex", "\\begin{tabular}{llcccc}\\toprule\nJudge & Items & acc. & tie AUROC $P(T)$ & tie AUROC $U$ & low-conf.\\ tie vs.\\ err.\\ $P(T)$ \\\\\\midrule\n"
      + "\n".join(f"{j} & {it} & " + " & ".join([PH()] * 4) + " \\\\" for j, it in
                  [("Qwen3-8B (text LLM)", "text"), ("Qwen2.5-7B (text LLM)", "text"), ("Qwen3-32B (text LLM)", "text"),
                   ("Qwen2.5-72B (text LLM)", "text"), ("Qwen3-VL-8B", "text"), ("Qwen3-VL-8B", "image sets"),
                   ("InternVL3.5-8B", "text"), ("InternVL3.5-8B", "image sets")]) + "\n\\bottomrule\\end{tabular}\n")
write("nuisance.tex", "\\begin{tabular}{lcccc}\\toprule\nSignal & tie AUROC & err.\\ AUROC (verified) & low-conf.\\ tie vs.\\ err. & adds to $U$ ($\\Delta$AUROC) \\\\\\midrule\n"
      + "\n".join(f"{s} & " + " & ".join([PH()] * 4) + " \\\\" for s in
                  ["$U$ (forced choice)", "$P(T)$", "$\\overline{P(T)}$ over 6 variants", "prompt SD (3 paraphrases)",
                   "image SD (JPEG, crop)", "all variants SD"]) + "\n\\bottomrule\\end{tabular}\n")
write("reason.tex", "\\begin{tabular}{lcccccc}\\toprule\nJudge & acc. & T-rate & tie AUROC $r_T$ & tie AUROC $r_\\text{ent}$ & err.\\ AUROC $r_\\text{flip}$ & forced \\% \\\\\\midrule\n"
      + "\n".join(f"{s} & " + " & ".join([PH()] * 6) + " \\\\" for s in ["Qwen3-VL-8B-Thinking", "Qwen3.5-9B (thinking)", "Qwen3-8B (text, thinking)"])
      + "\n\\bottomrule\\end{tabular}\n")
write("lora.tex", "\\begin{tabular}{lcccccc}\\toprule\nJudge & training & acc. (strict) & T-rate & tie AUROC & MMRB2 $\\rho$ & err.\\ AUROC $U$ \\\\\\midrule\n"
      + "\n".join(f"{j} & {t} & " + " & ".join([PH()] * 5) + " \\\\" for j in ["Qwen3-VL-8B", "InternVL3.5-8B"]
                  for t in ["zero-shot", "LoRA hard labels", "LoRA soft labels"]) + "\n\\bottomrule\\end{tabular}\n")
write("evidence.tex", "\\begin{tabular}{lcccc}\\toprule\nCondition & acc. & err.\\ AUROC $U$ & tie AUROC $U$ & $\\Delta U$ on errors vs.\\ ties \\\\\\midrule\n"
      + "\n".join(f"{s} & " + " & ".join([PH()] * 4) + " \\\\" for s in ["768 px (default)", "112 px (low-res probe)", "1536 px + tiling", "describe-then-judge"])
      + "\n\\bottomrule\\end{tabular}\n")
# one-table summary of the pre-registered interventions for the main text (details in the supplement)
LEV = [("Prompt paraphrases (3)", "tri, capped aleatoric sets", "$\\pT$ tie AUROC stable; verdict SD tracks errors"),
       ("Image nuisance (JPEG, crop)", "tri, capped aleatoric sets", "SD tracks errors, not ties"),
       ("1536\\,px + tiling", "bin, capped sets", "fewer errors; ties unchanged"),
       ("Describe-then-judge", "bin, capped sets", "fewer perception errors"),
       ("Reasoning, $N{=}4$ samples", "aleatoric sets + text", "acc.\\ $+{\\ge}2$; $r_T$ tracks ties"),
       ("Scale 1B$\\to$72B (3 families)", "bin/tri", "error AUROC $\\uparrow$, shared errors $\\downarrow$"),
       ("Text LLM judges", "MT-Bench, LMArena", "cleaner $\\pT$ if the gap is visual"),
       ("LoRA on preference distributions", "full sets, 2 folds", "tie AUROC ${\\ge}.75$, T chosen on ties")]
write("levers_summary.tex", "\\begin{tabular}{lllcc}\\toprule\nIntervention & Setting & Pre-registered prediction & "
      "$\\Delta$ tie AUROC & $\\Delta$ err.\\ AUROC \\\\\\midrule\n"
      + "\n".join(f"{n} & {st} & {pr} & {PH()} & {PH()} \\\\" for n, st, pr in LEV) + "\n\\bottomrule\\end{tabular}\n")
print("pp rerun complete:", pp_ok)
