"""LaTeX tables for the ladder and PPI experiments + win/loss summary of escalation configs.
usage (from $EAGLEVLM_DATA): python make_extra_tables.py [--out ../../EAGLE-VLM/paper/tables] [--alpha .2]
Reads paper_tables/ladder_<small>_qwen3vl32b.json and paper_tables/ppi_genai_arena_<model>.json."""
import argparse, json, os, collections

NAMES = {"vlrb": "VL-RewardBench", "mmrb": "Multimodal RB", "mmrlhf": "MM-RLHF-RB", "mllm_judge": "MLLM-as-a-Judge",
         "genai_arena": "GenAI-Arena", "mjbench": "MJ-Bench", "mmrb2": "MMRB2", "imagerewarddb": "ImageRewardDB",
         "qbench2": "Q-Bench2", "editreward": "EditReward-Bench", "pp": "Place Pulse 2.0"}
ORDER = ["vlrb", "mmrb", "mmrlhf", "mllm_judge", "genai_arena", "mjbench", "mmrb2", "imagerewarddb", "qbench2", "editreward", "pp"]
SMALL = ["qwen3vl8b", "qwen35_9b", "qwen25vl7b"]
SNAME = {"qwen3vl8b": "Qwen3-VL-8B", "qwen35_9b": "Qwen3.5-9B", "qwen25vl7b": "Qwen2.5-VL-7B"}

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper_tables"); ap.add_argument("--alphas", default=".2,.3")
ap.add_argument("--large", default="qwen3vl32b")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
f2 = lambda x: f"{100 * x:.0f}"

# ---------- ladder: automation per config, one table per small judge ----------
summary = collections.Counter(); lines_sum = []
for sm in SMALL:
    p = f"paper_tables/ladder_{sm}_{a.large}.json"
    if not os.path.exists(p):
        continue
    L = json.load(open(p))
    cfgs = [c for c in ["P", "P>S fix", "P>S", "L", "P>L", "P>S>L"] if any(f"{c}@0.2" in L[d] for d in L)]
    al = [float(x) for x in a.alphas.split(",")]
    head = " & ".join(c.replace(">", "$\\to$").replace(" fix", " (fixed)") for c in cfgs)
    rows = []
    for d in ORDER:
        if d not in L:
            continue
        cells = []
        for x in al:
            vals = {c: L[d].get(f"{c}@{x}", {}).get("auto") for c in cfgs}
            best = max(v for v in vals.values() if v is not None)
            cells.append(" & ".join("--" if vals[c] is None else (f"\\textbf{{{f2(vals[c])}}}" if vals[c] == best and best > 0.005 else f2(vals[c])) for c in cfgs))
            # win/loss of protocol rung vs pairwise only (adaptive), counted where either certifies >= 2%
            pa, ps = vals["P"], vals.get("P>S")
            if ps is not None and max(pa, ps) >= .02:
                summary[(sm, "P>S vs P", "win" if ps > pa + .01 else ("loss" if ps < pa - .01 else "tie"))] += 1
            pf = vals.get("P>S fix")
            if pf is not None and max(pa, pf) >= .02:
                summary[(sm, "P>S fix vs P", "win" if pf > pa + .01 else ("loss" if pf < pa - .01 else "tie"))] += 1
            if vals.get("P>L") is not None and max(pa, vals["P>L"]) >= .02:
                summary[(sm, "P>L vs P", "win" if vals["P>L"] > pa + .01 else ("loss" if vals["P>L"] < pa - .01 else "tie"))] += 1
            if vals.get("P>S>L") is not None and vals.get("P>L") is not None and max(vals["P>L"], vals["P>S>L"]) >= .02:
                summary[(sm, "P>S>L vs P>L", "win" if vals["P>S>L"] > vals["P>L"] + .01 else ("loss" if vals["P>S>L"] < vals["P>L"] - .01 else "tie"))] += 1
        rows.append(f"{NAMES[d]} ({L[d]['n']}) & " + " & ".join([cells[0]] + cells[1:]) + r"\\")
    ncol = len(cfgs)
    tex = [r"\begin{tabular}{l" + ("|" + "c" * ncol) * len(al) + "}", r"\toprule",
           " & " + " & ".join(f"\\multicolumn{{{ncol}}}{{c}}{{$\\alpha={x:.2f}$}}" for x in al) + r"\\",
           "Dataset ($n$) & " + " & ".join([head] * len(al)) + r"\\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    open(f"{a.out}/ladder_{sm}.tex", "w").write("\n".join(tex) + "\n")
for k in sorted({(s, c) for s, c, _ in summary}):
    lines_sum.append(f"{k[0]:11s} {k[1]:14s} win={summary[k + ('win',)]} tie={summary[k + ('tie',)]} loss={summary[k + ('loss',)]}")
print("\n".join(lines_sum))
open(f"{a.out}/ladder_winloss.txt", "w").write("\n".join(lines_sum) + "\n")

# ---------- PPI on GenAI-Arena ----------
rows = []
for sm in SMALL:
    p = f"paper_tables/ppi_genai_arena_{sm}.json"
    if not os.path.exists(p):
        continue
    P = json.load(open(p))
    for sub in sorted(P):
        r = P[sub]
        cells = [f"{r['judge_only']['kendall']:.2f}"]
        for b in ["0.05", "0.1", "0.2"]:
            x = r[f"budget_{b}"]
            vals = [x["human_only_kendall"], x["ppi_uniform_kendall"], x["ppi_routed_kendall"]]
            best = max(vals)
            cells += [f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}" for v in vals]
        rows.append(f"{SNAME[sm]} & {sub} ({r['n_models']}) & " + " & ".join(cells) + r"\\")
tex = [r"\begin{tabular}{llc|ccc|ccc|ccc}", r"\toprule",
       r" & & Judge & \multicolumn{3}{c|}{5\% human} & \multicolumn{3}{c|}{10\% human} & \multicolumn{3}{c}{20\% human}\\",
       r"Judge & Subset ($K$) & only & Hum. & PPI & PPI-r & Hum. & PPI & PPI-r & Hum. & PPI & PPI-r\\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
open(f"{a.out}/ppi_genai_arena.tex", "w").write("\n".join(tex) + "\n")
print("wrote", a.out)
