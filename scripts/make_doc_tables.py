"""Paper-style result tables for paper_logic.md, generated from the result files (no hand-copied numbers).
Modelled on SCOPE (arXiv 2602.13110: Table 1 accuracy / ECE / AUROC / AUPRC of the confidence; Table 3 coverage and empirical
risk per judge, failures marked) and Trust-or-Escalate (arXiv 2407.18370: Table 1 Acc / ECE / AUROC / AUPRC; Table 2 empirical
human agreement / coverage / guarantee success rate).  Every table reports every judge and the mean; '·' = method produced no
output (all splits abstained), '—' = not run.
Inputs (judgebench/paper_tables): conv_metrics_v2.md, chain*.md (later files override earlier ones for the same set/judge/step),
router*.jsonl, c3_diag__*.md, c3_g5_all.md.
usage (judgebench dir): python make_doc_tables.py --out paper_tables/doc_tables.md"""
import argparse, collections, glob, json, math, os, re
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper_tables/doc_tables.md")
ap.add_argument("--compact", action="store_true", help="one row per dataset x method: mean over general VLMs (min-max)")
a = ap.parse_args()
T = "paper_tables"
JUDGES = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b", "internvl35_38b", "r1reward", "llavacritic7b"]
JN = {"qwen3vl8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B", "qwen35_9b": "Qwen3.5-9B", "qwen25vl7b": "Qwen2.5-VL-7B",
      "qwen3vl32b": "Qwen3-VL-32B", "internvl35_38b": "InternVL3.5-38B", "r1reward": "R1-Reward*", "llavacritic7b": "LLaVA-Critic-7B*"}
GENERAL = JUDGES[:6]
SETS = ["mllm_judge", "mllm_judge_sp", "genaibench", "genai_arena/generation", "genai_arena/edition", "pickapic_u", "imagerewarddb",
        "mtbench_human", "arena55k", "polaris", "agiqa3k"]
SN = {"mllm_judge": "MLLM-as-a-Judge (pairs)", "mllm_judge_sp": "MLLM-as-a-Judge (score pairs)", "genaibench": "GenAI-Bench",
      "genai_arena/generation": "GenAI-Arena generation", "genai_arena/edition": "GenAI-Arena edition", "pickapic_u": "Pick-a-Pic",
      "imagerewarddb": "ImageRewardDB", "mtbench_human": "MT-Bench (text)", "arena55k": "Chatbot Arena 55k (text)",
      "polaris": "Polaris", "agiqa3k": "AGIQA-3K"}
META = {  # task, source, what our set is
    "mllm_judge": ("image + question -> two answers", "MLLM-as-a-Judge, ICML 2024", "full pair split (7,756)"),
    "mllm_judge_sp": ("image + question -> two answers, human 1-5 per answer", "MLLM-as-a-Judge score split, paired within question", "all within-question pairs (9,862)"),
    "genaibench": ("text -> two images, 3 raters x 1-5 per image", "GenAI-Bench, CVPR 2024 SynData / VQAScore", "600 of 1,600 prompts (9,000 pairs); full 24,000 running"),
    "genai_arena/generation": ("text -> two images, arena votes", "GenAI-Arena, NeurIPS 2024 D&B", "full (3,192)"),
    "genai_arena/edition": ("source image + instruction -> two edits", "GenAI-Arena, NeurIPS 2024 D&B", "full (983)"),
    "pickapic_u": ("text -> two images, user votes", "Pick-a-Pic, NeurIPS 2023", "standard test_unique + validation_unique (1,000)"),
    "imagerewarddb": ("text -> images, expert ranks / 1-7 scores", "ImageReward, NeurIPS 2023", "full pairwise expansion (7,230)"),
    "mtbench_human": ("text, multi-turn chat", "MT-Bench, NeurIPS 2023 D&B", "all expert-judged pairs (910)"),
    "arena55k": ("text, chat", "lmarena arena-human-preference-55k", "random sample at the natural tie rate (4,802); SCOPE samples 2,000 non-tied, ToE 5.2k"),
    "polaris": ("image -> two captions, human 0-1", "Polaris, CVPR 2024", "2 of 11 test shards (1,299 pairs); full pending"),
    "agiqa3k": ("text -> images, MOS quality + alignment", "AGIQA-3K, IEEE TCSVT 2023", "100 of 300 prompts (4,456 pairs); full pending"),
}
f2 = lambda v: "—" if v is None else ("·" if isinstance(v, float) and math.isnan(v) else f"{v:.2f}")
f3 = lambda v: "—" if v is None else ("·" if isinstance(v, float) and math.isnan(v) else f"{v:.3f}")


def mean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v) if v else float("nan")


def mm(vals, fmt=f2, inv=False):
    """mean (min-max) over judges; inv: report 1 - v"""
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if inv:
        v = [1 - x for x in v]
    if not v:
        return "·"
    if len(v) == 1 or max(v) - min(v) < 0.005:
        return fmt(sum(v) / len(v))
    return f"{fmt(sum(v) / len(v))} ({fmt(min(v))}–{fmt(max(v))})"


def rows(path):
    for line in open(path):
        if line.startswith("| ") and not line.startswith("| set") and "---" not in line:
            yield [c.strip() for c in line.strip().strip("|").split("|")]


# ---------------- conventional metrics
CONV = {}
p = f"{T}/conv_metrics_v2.md"
if os.path.exists(p):
    for c in rows(p):
        CONV[(c[0], c[1])] = dict(n=int(c[2]), tie=float(c[3]), acc_d=float(c[4]), acc_w=float(c[5]), acc_h=float(c[6]), acc3=float(c[7]),
                                  tshare=float(c[8]), tief1=float(c[9]), auroc_d=float(c[10]), aurc_d=float(c[11]), aurc_a=float(c[12]),
                                  sa_d=float(c[13]), sa_a=float(c[14]), ece_d=float(c[15]), ece_a=float(c[16]), auprc_d=float(c[17]),
                                  auroc_a=float(c[18]), auprc_a=float(c[19]))
# ---------------- chain (certification)
CH = {}
files = ([f"{T}/chain.md", f"{T}/chain_sp.md"] + sorted(glob.glob(f"{T}/chain_38b__*.md")) + sorted(glob.glob(f"{T}/chain_genaibench__*.md"))
         + sorted(glob.glob(f"{T}/chain_sp__*.md")) + sorted(glob.glob(f"{T}/chain_v2__*.md")))
for fp in files:
    if not os.path.exists(fp):
        continue
    for c in rows(fp):
        if len(c) < 8:
            continue
        step = c[4].split()[0]
        for k, al in zip(c[5:8], (0.1, 0.2, 0.3)):
            m = re.match(r"([\d.]+) / ([\d.]+) / ([\d.]+) / ([\d.]+|-) / ([\d.]+|-) / ([\d.]+|-)(?: \((\d+)% empty\))?", k)
            if not m:
                continue
            g = lambda x: float("nan") if x == "-" else float(x)
            CH[(c[0], c[1], step, al)] = dict(acc=g(m[1]), size=g(m[2]), dec=g(m[3]), serr=g(m[4]), risk=g(m[5]), viol=g(m[6]),
                                              empty=(int(m[7]) / 100 if m[7] else 0.0), n=c[2], src=os.path.basename(fp))
# ---------------- significance-corrected violation (c3_diag, 25/25/50, G2 and G5)
SIG = {}
for fp in glob.glob(f"{T}/c3_diag__*.md") + [f"{T}/c3_g5_all.md"]:
    if not os.path.exists(fp):
        continue
    for c in rows(fp):
        if len(c) < 7 or c[4] != "25/25/50":
            continue
        cells = c[6:]
        als = (0.1, 0.2, 0.3) if len(cells) >= 3 else (0.2,)
        for k, al in zip(cells, als):
            m = re.match(r"([\d.]+) / ([\d.]+) / ([\d.]+|-) / ([\d.]+) \(([\d.]+)\) / ([\d.]+|-) / ([\d.]+|-)", k)
            if m:
                g = lambda x: float("nan") if x == "-" else float(x)
                SIG[(c[0], c[1], c[5], al)] = dict(acc=g(m[1]), dec=g(m[2]), serr=g(m[3]), risk=g(m[4]), sd=g(m[5]), viol=g(m[6]), vsig=g(m[7]))
# ---------------- router
RT = {}
for fp in ["router_v3", "router_v4_38b", "router_genaibench_v3", "router_abs34_v1"]:
    q = f"{T}/{fp}.jsonl"
    if os.path.exists(q):
        for l in open(q):
            d = json.loads(l)
            if not d["judge"].endswith("_hr"):
                RT[(d["set"], d["judge"], d["variant"], d["rule"], d["alpha"])] = d

L = ["# 论文式结果表（自动生成）", "",
     "生成：`scripts/make_doc_tables.py`。数值全部来自结果文件，不手抄。`·` = 方法在所有划分上都没有输出（全部弃权）；`—` = 未运行。"
     "所有认证实验：25% 训练 / 25% 校准 / 50% 测试，100 次随机划分，δ = 0.1；带 * 的是专门训练的 judge。", ""]
if a.compact:
    L += ["**紧凑版**：每格 = 通用 VLM（Qwen3-VL-8B、InternVL3.5-8B、Qwen3.5-9B、Qwen2.5-VL-7B、Qwen3-VL-32B、InternVL3.5-38B 中已运行的）的均值，括号内为最小–最大；"
          "\"×n\" = 参与平均的 judge 数。逐 judge 的完整表见 `paper_logic_tables.md`。", ""]

# ===== Table 0: datasets
L += ["## 表 0　数据集", "", "| 数据集 | 任务 | 来源 | 本文使用的数据 | 对数 | 人类平局率 | 已评测的 judge |", "|---|---|---|---|---|---|---|"]
for s in SETS:
    js = [j for j in JUDGES if (s, j) in CONV]
    n = CONV[(s, js[0])]["n"] if js else None; tie = CONV[(s, js[0])]["tie"] if js else None
    L.append(f"| {SN[s]} | {META[s][0]} | {META[s][1]} | {META[s][2]} | {n if n else '—'} | {f2(tie)} | {', '.join(JN[j] for j in js) or '—'} |")
L += [""]

# ===== Table 1: judge accuracy and confidence quality (SCOPE Table 1 / ToE Table 1), published setting vs ties kept
L += ["## 表 1　Judge 准确率与置信度质量（对应 SCOPE 表 1、Trust or Escalate 表 1）", "",
      "置信度 = 双序平均二选一概率 p 的 BPE（SCOPE 的置信分数）。\"删平局\" = 已发表工作的设定（只在人类有胜负的样本上评估）；"
      "\"含平局\" = 全部样本，人类平局对任何二选一都算错。ECE 用 max(p, 1−p)；AUROC / AUPRC 衡量 BPE 识别判错的能力；AURC 为风险-覆盖曲线下面积（越低越好）。", "",
      "| 数据集 | judge | 人类平局 | judge 选平局 | 准确率 删平局 | 准确率 含平局 | 三选一准确率 | ECE 删平局 | ECE 含平局 | AUROC 删平局 | AUROC 含平局 | AUPRC 删平局 | AUPRC 含平局 | AURC 删平局 | AURC 含平局 |",
      "|" + "---|" * 15]
for s in (SETS if a.compact else []):
    src = [CONV[(s, x)] for x in GENERAL if (s, x) in CONV]
    if src:
        g = lambda k: mm([d[k] for d in src], f3)
        L.append(f"| {SN[s]} | 通用 VLM ×{len(src)} | {f2(src[0]['tie'])} | {g('tshare')} | {g('acc_d')} | {g('acc_w')} | {g('acc3')} | {g('ece_d')} | {g('ece_a')} | "
                 f"{g('auroc_d')} | {g('auroc_a')} | {g('auprc_d')} | {g('auprc_a')} | {g('aurc_d')} | {g('aurc_a')} |")
for s in ([] if a.compact else SETS):
    js = [j for j in JUDGES if (s, j) in CONV]
    for j in js + (["mean"] if js else []):
        src = [CONV[(s, x)] for x in (js if j == "mean" else [j]) if (x in GENERAL or j != "mean")]
        if not src:
            continue
        g = lambda k: mean([d[k] for d in src])
        name = "**均值（通用 VLM）**" if j == "mean" else JN[j]
        L.append(f"| {SN[s] if j == js[0] else ''} | {name} | {f2(g('tie'))} | {f3(g('tshare'))} | {f3(g('acc_d'))} | {f3(g('acc_w'))} | {f3(g('acc3'))} | "
                 f"{f3(g('ece_d'))} | {f3(g('ece_a'))} | {f3(g('auroc_d'))} | {f3(g('auroc_a'))} | {f3(g('auprc_d'))} | {f3(g('auprc_a'))} | {f3(g('aurc_d'))} | {f3(g('aurc_a'))} |")
L += [""]

# ===== Table 2: risk control (SCOPE Table 3 / ToE Table 2)
STEPS = [("P1", "SCOPE 原样（删平局校准）"), ("P1s", "SCOPE 原样，单序"), ("P2", "SCOPE，校准保留平局"), ("P3", "严格断言 + LTT（BPE）"),
         ("P4", "严格断言 + LTT + 全部组件 + 校准器"), ("T1", "Trust or Escalate 原样（删平局校准）"), ("T2", "Trust or Escalate，校准保留平局"),
         ("G0", "分级断言，无校准器"), ("G1", "分级断言 + 校准器，单序特征"), ("G2", "**分级断言 + 校准器（ours）**"), ("G3", "分级断言，每一级都 ≤ α")]
for al in ((0.2, 0.1) if a.compact else (0.2, 0.1, 0.3)):
    L += [f"## 表 2{'' if al == 0.2 else ('b' if al == 0.1 else 'c')}　风险控制，α = {al}（对应 SCOPE 表 3、Trust or Escalate 表 2）", "",
          "所有方法在含平局的测试集上部署。输出率 = SCOPE / ToE 的 coverage；实际错误率 = SCOPE 的 empirical FDR；人类一致率 = ToE 的 empirical human agreement（= 1 − 实际错误率）；"
          "保证成功率 = ToE 的 guarantee success rate（= 1 − 违约率，应 ≥ 1 − δ = 0.9）；判决率 = 明确\"A 更好 / B 更好\"的比例；集合大小 1 = 明确、2 = 弱断言、3 = 弃权；"
          "空划分 = 所有测试样本都弃权的划分比例。每格为各 judge 的值，最后一行为通用 VLM 均值。", "",
          "| 数据集 | 方法 | judge | 输出率 | 判决率 | 平均集合大小 | 实际错误率 | 人类一致率 | 保证成功率 | 严格断言错误率 | 空划分 |", "|" + "---|" * 11]
    for s in (SETS if a.compact else []):
        first = True
        for st, nm in STEPS:
            src = [CH[(s, x, st, al)] for x in GENERAL if (s, x, st, al) in CH]
            if not src:
                continue
            ga = lambda k, fm=f2, inv=False: mm([d[k] for d in src], fm, inv)
            L.append(f"| {SN[s] if first else ''} | {nm} | ×{len(src)} | {ga('acc')} | {ga('dec')} | {ga('size')} | {ga('risk', f3)} | {ga('risk', f3, True)} | "
                     f"{ga('viol', f2, True)} | {ga('serr', f3)} | {ga('empty')} |")
            first = False
    for s in ([] if a.compact else SETS):
        first = True
        for st, nm in STEPS:
            js = [j for j in JUDGES if (s, j, st, al) in CH]
            if not js:
                continue
            for j in js + ["mean"]:
                src = [CH[(s, x, st, al)] for x in js if x in GENERAL] if j == "mean" else [CH[(s, j, st, al)]]
                if not src:
                    continue
                g = lambda k: mean([d[k] for d in src])
                r = g("risk")
                L.append(f"| {SN[s] if first else ''} | {nm if j == js[0] else ''} | {'均值（通用 VLM）' if j == 'mean' else JN[j]} | {f2(g('acc'))} | {f2(g('dec'))} | {f2(g('size'))} | "
                         f"{f3(r)} | {f3(1 - r if r == r else float('nan'))} | {f2(1 - g('viol') if g('viol') == g('viol') else float('nan'))} | {f3(g('serr'))} | {f2(g('empty'))} |")
                first = False
    L += [""]

# ===== Table 3: violation vs test noise (G2) and separate strict threshold (G5)
L += ["## 表 3　违约率与测试集抽样波动；严格断言单独约束（α = 0.2）", "",
      "违约率按测试集错误率 > α 计算（与 ToE 的 guarantee success rate 相同口径）；显著违约 = 测试集错误率显著高于 α（二项检验 p < .05）。"
      "G5 = 严格断言与平局 / 弱断言分别用阈值，同时认证\"总体 ≤ α\"和\"严格断言 ≤ α\"。", "",
      "| 数据集 | judge | 方法 | 输出率 | 判决率 | 严格断言错误率 | 测试错误率 均值 (标准差) | 违约率 | 显著违约率 |", "|" + "---|" * 9]
for s in (SETS if a.compact else []):
    for meth in ("G2", "G5"):
        src = [SIG[(s, j, meth, 0.2)] for j in JUDGES if (s, j, meth, 0.2) in SIG]
        if src:
            g = lambda k, fm=f2: mm([d[k] for d in src], fm)
            L.append(f"| {SN[s]} | ×{len(src)} | {meth} | {g('acc')} | {g('dec')} | {g('serr', f3)} | {g('risk', f3)} ({g('sd', f3)}) | {g('viol')} | {g('vsig')} |")
for s in ([] if a.compact else SETS):
    for meth in ("G2", "G5"):
        for j in JUDGES:
            d = SIG.get((s, j, meth, 0.2))
            if d:
                L.append(f"| {SN[s]} | {JN[j]} | {meth} | {f2(d['acc'])} | {f2(d['dec'])} | {f3(d['serr'])} | {f3(d['risk'])} ({f3(d['sd'])}) | {f2(d['viol'])} | {f2(d['vsig'])} |")
L += [""]

# ===== Table 4: router
VAR = [("pairwise graded (G2)", "只用分级断言"), ("control: routed, no pointwise", "对照：路由但不看单项分数"), ("router: cells", "**路由：平局地图**"),
       ("router: logistic", "路由：逻辑回归"), ("router: jury pointwise", "路由：多 judge 单项分数平均")]
for al in ((0.2, 0.1) if a.compact else (0.2, 0.1, 0.3)):
    L += [f"## 表 4{'' if al == 0.2 else ('b' if al == 0.1 else 'c')}　成对 → 单项打分路由，α = {al}", "",
          "有信息量 = 严格断言 + 两种绝对质量结论（\"两者都不合格\"、\"两者没有明显差距\"）；绝对质量结论占比与其正确率单列；"
          "规则\"总体\" = 总错误率 ≤ α，\"每一级\" = 每种结论各自 ≤ α。", "",
          "| 数据集 | 规则 | 方法 | judge | 输出率 | 有信息量 | 判决率 | 绝对质量结论 | 其正确率 | 弱断言 | 实际错误率 | 人类一致率 | 保证成功率 |", "|" + "---|" * 13]
    for s in ["mllm_judge_sp", "genaibench", "imagerewarddb", "genai_arena/generation", "genai_arena/edition", "polaris", "agiqa3k"]:
        for rule, rn in (("overall", "总体"), ("every level", "每一级")):
            first = True
            for v, vn in VAR:
                if a.compact:
                    src = [RT[(s, x, v, rule, al)] for x in GENERAL if (s, x, v, rule, al) in RT]
                    if src:
                        g = lambda k, fm=f2, inv=False: mm([d.get(k) for d in src], fm, inv)
                        L.append(f"| {SN[s] if first else ''} | {rn if first else ''} | {vn} | ×{len(src)} | {g('accept')} | {g('informative')} | {g('decisive')} | "
                                 f"{g('level')} | {g('level_prec')} | {g('weak')} | {g('risk', f3)} | {g('risk', f3, True)} | {g('violation', f2, True)} |")
                        first = False
                    continue
                js = [j for j in JUDGES if (s, j, v, rule, al) in RT]
                for j in js + (["mean"] if js else []):
                    src = [RT[(s, x, v, rule, al)] for x in js if x in GENERAL] if j == "mean" else [RT[(s, j, v, rule, al)]]
                    if not src:
                        continue
                    g = lambda k: mean([d.get(k) for d in src])
                    r = g("risk")
                    L.append(f"| {SN[s] if first else ''} | {rn if first else ''} | {vn if j == js[0] else ''} | {'均值（通用 VLM）' if j == 'mean' else JN[j]} | "
                             f"{f2(g('accept'))} | {f2(g('informative'))} | {f2(g('decisive'))} | {f2(g('level'))} | {f2(g('level_prec'))} | {f2(g('weak'))} | "
                             f"{f3(r)} | {f3(1 - r if r == r else float('nan'))} | {f2(1 - g('violation') if g('violation') == g('violation') else float('nan'))} |")
                    first = False
    L += [""]
open(a.out, "w").write("\n".join(L) + "\n"); print(a.out, len(L), "lines")
