"""LaTeX tables for the baseline comparison from paper_tables/baseline_table.json (see baseline_table.py).
Main table (multimodal tasks only), three blocks:
  A  MLLM-as-a-Judge line: general VLMs under its protocols, and dedicated multimodal judge / reward models
  B  LLM-judge uncertainty methods reproduced on VLM judges (reference)
  C  ours: tie channel P(T), on general VLMs and on the dedicated judges
Second table: text-only control (MT-Bench, Chatbot Arena), VLM vs text LLM judges.
--standalone F also writes a compilable document with citations as "(Author et al., year)".
usage: python make_baseline_tex.py --inp paper_tables/baseline_table.json --out ../overleaf/tables/baselines.tex [--standalone F]"""
import argparse, json, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="paper_tables/baseline_table.json")
ap.add_argument("--out", required=True)
ap.add_argument("--standalone", default="")
a = ap.parse_args()
R = json.load(open(a.inp))
G4 = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b"]; G2 = ["qwen3vl8b", "internvl35_8b"]
TIE = ["genai_arena", "mllm_judge", "editreward", "pp"]
HEAD = ["GenAI-Arena", "MLLM-Judge", "EditReward", "Place Pulse"]
VER = ["mmrb2", "vlrb", "mmrb", "mjbench", "imagerewarddb", "qbench2", "mmrlhf"]
TXT_DS = ["mtbench_human", "arena55k"]; TXT_J = ["qwen3_8b", "qwen25_7b"]
JM = {"llavacritic7b": ("LLaVA-Critic-7B", "xiong2025llavacritic"), "r1reward": ("R1-Reward-7B", "zhang2025r1reward"),
      "mmrlhfrm7b": ("MM-RLHF-Reward-7B", "zhang2025mmrlhf")}
CITE_TXT = {"zheng2023judging": "Zheng et al., 2023", "wang2025trustjudge": "Wang et al., 2025", "liu2023geval": "Liu et al., 2023",
            "verga2024replacing": "Verga et al., 2024", "jung2024trust": "Jung et al., 2024",
            "chen2024mllmasajudge": "Chen et al., 2024", "xiong2025llavacritic": "Xiong et al., 2025",
            "zhang2025r1reward": "Zhang et al., 2025", "zhang2025mmrlhf": "Zhang et al., 2025"}

# (label, calls, judges, method); method None -> block header
ROWS = [("\\emph{A. MLLM-as-a-Judge line: general VLMs (mean of four) under its protocols}", None, None, None),
        ("Pairwise, forced choice, both orders~\\cite{chen2024mllmasajudge}", "2", G4, "FC-swap"),
        ("Pairwise with tie option, argmax~\\cite{chen2024mllmasajudge}", "2", G4, "Tie-verbal"),
        ("Pointwise scoring~\\cite{chen2024mllmasajudge}", "2", G4, "Score-arg"),
        ("\\emph{A. Dedicated multimodal judge / reward models}", None, None, None)]
NATIVE = {"llavacritic7b": ("native: critique, own tie verdict, 6 samples/order", "12$\\times$G"),
          "r1reward": ("native: reason then 1/2, vote of 6 samples/order", "12$\\times$G"),
          "mmrlhfrm7b": ("native: critique + reward head, margin", "2$\\times$G+2")}
for m, (nm, ck) in JM.items():
    ROWS.append((f"{nm}, forced choice~\\cite{{{ck}}}", "2", [m], "FC-swap"))
    ROWS.append((f"{nm}, tie option, argmax", "2", [m], "Tie-verbal"))
    ROWS.append((f"{nm}, {NATIVE[m][0]}$^\\ddagger$", NATIVE[m][1], [m + "@native"], "Native"))
ROWS += [("\\emph{B. LLM-judge uncertainty methods reproduced on VLM judges (reference)}", None, None, None),
         ("Swap inconsistency $\\to$ tie~\\cite{zheng2023judging}", "2", G4, "Swap-tie"),
         ("Expected score~\\cite{liu2023geval,wang2025trustjudge}", "2", G4, "Score-exp"),
         ("Bidirectional A/B/T aggregation~\\cite{wang2025trustjudge}", "2", G4, "TJ-agg"),
         ("7-point comparison, P(equal)", "2", G4, "Likert-7"),
         ("Panel of four judges~\\cite{verga2024replacing}", "8", G4, "Panel"),
         ("Prompt ensemble$^\\dagger$", "8", G2, "PromptEns"),
         ("Simulated annotators$^\\dagger$~\\cite{jung2024trust}", "8", G2, "SimAnn"),
         ("\\emph{C. Ours: tie channel $\\pT$ (tie option, both orders)}", None, None, None),
         ("\\textbf{Ours}, general VLMs", "2", G4, "Ours-2call"),
         ("\\textbf{Ours}, general VLMs$^\\dagger$", "2", G2, "Ours-2call"),
         ("\\textbf{Ours} + prompt ensemble$^\\dagger$", "10", G2, "Ours+PE"),
         ("\\textbf{Ours}, panel of four judges", "16", G4, "Ours-Panel")]
for m, (nm, ck) in JM.items():
    ROWS.append((f"\\textbf{{Ours}} on {nm}", "2", [m], "Ours-2call"))


def cell(metric, ds, m, judges):
    x = [r[metric] for r in R if r["ds"] == ds and r["method"] == m and r["judge"] in judges and metric in r]
    return np.mean(x) if len(x) == len(judges) else np.nan


def vals(m, judges):
    er = [cell("err_auroc", d, m, judges) for d in VER]
    if m == "Native":                 # response sets only: mean over the verified response-judging sets present
        er = [z for z in er if not np.isnan(z)] or [np.nan]
    return [cell("tie_auroc", d, m, judges) for d in TIE] + [cell("route", d, m, judges) for d in TIE] + \
           [np.nan if any(np.isnan(er)) else np.mean(er)]


data = [(r, vals(r[3], r[2]) if r[3] else None) for r in ROWS]
data = [(r, v) for r, v in data if v is None or not all(np.isnan(v))]
best = []
for k in range(9):
    col = [v[k] for _, v in data if v is not None and not np.isnan(v[k])]
    best.append(max(col) if col else np.nan)
fmt = lambda z, k: "--" if np.isnan(z) else (f"\\textbf{{{z:.3f}}}" if abs(z - best[k]) < 5e-4 else f"{z:.3f}").replace("0.", ".", 1)
L = ["\\begin{table*}[t]\\centering\\small",
     "\\caption{\\textbf{Baselines on multimodal judging tasks.} Left: AUROC for detecting human ties. Middle: three-way "
     "router (tie / verdict / human) at a 20\\% human budget, agreement with people over all pairs (a tie counts as correct "
     "only on a human tie; thresholds cross-fitted, 2-fold $\\times$ 5). Right: AUROC for judge errors, mean over seven "
     "benchmarks with verified labels (MMRB2, VL-RewardBench, Multimodal RewardBench, MJ-Bench, ImageRewardDB, Q-Bench2, "
     "MM-RLHF; 700-pair caps for the dedicated judges). General VLMs: Qwen3-VL-8B, InternVL3.5-8B, Qwen3.5-9B, "
     "Qwen2.5-VL-7B; $^\\dagger$ Qwen3-VL-8B and InternVL3.5-8B only. $^\\ddagger$ Native protocol of the judge (official "
     "prompt and decoding, generation): defined only for text responses about an image, so it covers MLLM-Judge and the "
     "verified response sets (VL-RewardBench, Multimodal RewardBench, MM-RLHF). calls: forward passes per pair "
     "(G: a generated critique or reasoning trace). Bold: best in "
     "column.}\\label{tab:baselines}",
     "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llcccc|cccc|c}\\toprule",
     " & & \\multicolumn{4}{c|}{Tie detection AUROC} & \\multicolumn{4}{c|}{Router agreement, 20\\% human budget} & Err.\\ AUROC \\\\",
     "Method & calls & " + " & ".join(HEAD) + " & " + " & ".join(HEAD) + " & 7 verified \\\\"]
for r, v in data:
    if v is None:
        L.append("\\midrule\\multicolumn{11}{l}{" + r[0] + "} \\\\")
    else:
        L.append(f"{r[0]} & {r[1]} & " + " & ".join(fmt(z, k) for k, z in enumerate(v)) + " \\\\")
L += ["\\bottomrule\\end{tabular}}", "\\end{table*}"]
main = "\n".join(L) + "\n"

NAME = {"FC-swap": "Pairwise, forced choice, both orders~\\cite{chen2024mllmasajudge}",
        "Tie-verbal": "Pairwise with tie option, argmax~\\cite{chen2024mllmasajudge}",
        "Swap-tie": "Swap inconsistency $\\to$ tie~\\cite{zheng2023judging}", "Likert-7": "7-point comparison, P(equal)",
        "TJ-agg": "Bidirectional A/B/T aggregation~\\cite{wang2025trustjudge}", "Ours-2call": "\\textbf{Ours}: $\\pT$"}
T = ["\\begin{table}[t]\\centering\\small",
     "\\caption{\\textbf{Text-only control.} The same protocol on text pairs (no images): mean over the four VLM judges vs.\\ "
     "two text LLM judges (Qwen3-8B, Qwen2.5-7B). Cells: tie AUROC / router agreement at a 20\\% human budget.}\\label{tab:text_control}",
     "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lcccc}\\toprule",
     " & \\multicolumn{2}{c}{MT-Bench (human)} & \\multicolumn{2}{c}{Chatbot Arena} \\\\",
     "Method & VLMs & LLMs & VLMs & LLMs \\\\\\midrule"]
for m in ["FC-swap", "Tie-verbal", "Swap-tie", "Likert-7", "TJ-agg", "Ours-2call"]:
    cs = []
    for d in TXT_DS:
        for js in (G4, TXT_J):
            ta, ro = cell("tie_auroc", d, m, js), cell("route", d, m, js)
            cs.append("--" if np.isnan(ta) else f"{ta:.3f} / {ro:.3f}".replace("0.", "."))
    T.append(NAME[m] + " & " + " & ".join(cs) + " \\\\")
T += ["\\bottomrule\\end{tabular}}", "\\end{table}"]
text = "\n".join(T) + "\n"

open(a.out, "w").write(main)
open(a.out.replace(".tex", "_text.tex"), "w").write(text)
if a.standalone:
    sub = lambda s: re.sub(r"~\\cite\{([^}]*)\}", lambda m: " (" + "; ".join(CITE_TXT[k.strip()] for k in m.group(1).split(",")) + ")", s)
    body = sub(main).replace("\\begin{table*}[t]", "\\begin{table}[h]").replace("\\end{table*}", "\\end{table}") + \
        "\\clearpage\n" + sub(text).replace("\\begin{table}[t]", "\\begin{table}[h]").replace("\\columnwidth", "0.7\\linewidth")
    open(a.standalone, "w").write(
        "\\documentclass[10pt]{article}\n\\usepackage[landscape,margin=1.5cm]{geometry}\n\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n\\usepackage{amsmath}\n\\newcommand{\\pT}{P(\\mathrm{T})}\n\\pagestyle{empty}\n"
        "\\begin{document}\n" + body + "\\end{document}\n")
print(main)
