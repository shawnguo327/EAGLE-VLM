"""Paper tables from the pipeline's summary CSVs only (no hand-copied numbers).

    python -m eaglevlm.pipeline.tables configs/tables.yaml

Reads <results>/<experiment>/latest/summary.csv for the experiments named in the config and writes
<results>/tables/<timestamp>/{tables_full.md, tables_compact.md, sources.csv}: full = every judge plus the mean;
compact = one row per dataset x method, mean over the general VLMs with (min-max).  Table layout follows SCOPE
(arXiv 2602.13110, Tables 1 and 3) and Trust-or-Escalate (arXiv 2407.18370, Tables 1 and 2).
'·' = no output in any split (all abstained), '—' = not run."""
import csv, datetime, math, os, sys
import yaml

f2 = lambda v: "—" if v is None else ("·" if v != v else f"{v:.2f}")
f3 = lambda v: "—" if v is None else ("·" if v != v else f"{v:.3f}")


def num(x):
    if x in ("", None):
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return x


def read(path):
    return [{k: num(v) if k not in ("dataset", "judge", "method", "variant", "rule") else v for k, v in r.items()} for r in csv.DictReader(open(path))]


def mm(vals, fmt=f2):
    v = [x for x in vals if isinstance(x, float) and x == x]
    if not v:
        return "·"
    m = sum(v) / len(v)
    return fmt(m) if len(v) == 1 or max(v) - min(v) < 0.005 else f"{fmt(m)} ({fmt(min(v))}–{fmt(max(v))})"


def main():
    cfg = yaml.safe_load(open(sys.argv[1])); res = cfg["results"]; JN = cfg["judge_names"]; SN = cfg["dataset_names"]
    GEN = cfg["general_judges"]; META = cfg.get("datasets", {})
    out = os.path.join(res, "tables", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")); os.makedirs(out)
    src = {}
    for k, v in cfg["experiments"].items():                 # a value may list several experiments (rows are merged)
        for kk, name in enumerate(v if isinstance(v, list) else [v]):
            pth = os.path.realpath(os.path.join(res, name, "latest", "summary.csv"))
            if os.path.exists(pth):
                src[k if kk == 0 else f"{k}_{kk}"] = pth
            else:
                print(f"skip {name}: no summary.csv yet")
    with open(f"{out}/sources.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["table_input", "summary_csv"]); [w.writerow([k, v]) for k, v in src.items()]
    CONV = {(r["dataset"], r["judge"]): r for r in read(src["conventional"])} if "conventional" in src else {}
    CERT = {(r["dataset"], r["judge"], r["method"], r["alpha"]): r for r in read(src["certify"])} if "certify" in src else {}
    ROUT = {}
    for k in [k for k in src if k == "router" or k.startswith("router_")]:
        ROUT.update({(r["dataset"], r["judge"], r["variant"], r["rule"], r["alpha"]): r for r in read(src[k])})
    judges = cfg["judges"]; sets = cfg["dataset_order"]
    for compact in (False, True):
        L = ["# 论文式结果表（自动生成）", "",
             f"由 `python -m eaglevlm.pipeline.tables` 从以下 summary.csv 生成（{datetime.datetime.now():%Y-%m-%d %H:%M}）："] + [f"- {k}: `{v}`" for k, v in src.items()] + [
             "", "`·` = 所有划分都没有输出（全部弃权）；`—` = 未运行。认证实验：25% 训练 / 25% 校准 / 50% 测试，100 次随机划分，δ = 0.1。带 * 的是专门训练的 judge。", ""]
        if compact:
            L += ["**紧凑版**：每格 = 通用 VLM 的均值，括号内为最小–最大；×n = 参与平均的 judge 数。逐 judge 数字见 tables_full.md。", ""]
        # ---- Table 0
        L += ["## 表 0　数据集", "", "| 数据集 | 任务 | 来源 | 本文使用的数据 | 对数 | 人类平局率 | 已评测的 judge |", "|---|---|---|---|---|---|---|"]
        for s in sets:
            js = [j for j in judges if (s, j) in CONV]; m = META.get(s, ["", "", ""])
            n = int(CONV[(s, js[0])]["n"]) if js else None; tie = CONV[(s, js[0])]["human_tie"] if js else None
            L.append(f"| {SN[s]} | {m[0]} | {m[1]} | {m[2]} | {n if n else '—'} | {f2(tie)} | {', '.join(JN[j] for j in js) or '—'} |")
        # ---- Table 1
        K1 = [("judge_tie_share", "judge 选平局", f3), ("acc_ties_dropped", "准确率 删平局", f3), ("acc_ties_wrong", "准确率 含平局", f3), ("acc_3way", "三选一准确率", f3),
              ("ece_ties_dropped", "ECE 删平局", f3), ("ece_ties_kept", "ECE 含平局", f3), ("err_auroc_ties_dropped", "AUROC 删平局", f3),
              ("err_auroc_ties_kept", "AUROC 含平局", f3), ("err_auprc_ties_dropped", "AUPRC 删平局", f3), ("err_auprc_ties_kept", "AUPRC 含平局", f3),
              ("aurc_ties_dropped", "AURC 删平局", f3), ("aurc_ties_kept", "AURC 含平局", f3)]
        L += ["", "## 表 1　Judge 准确率与置信度质量（对应 SCOPE 表 1、Trust or Escalate 表 1）", "",
              "置信度 = 双序平均二选一概率 p 的 BPE。\"删平局\" = 已发表工作的设定；\"含平局\" = 全部样本，人类平局对任何二选一都算错。ECE 用 max(p, 1−p)；"
              "AUROC / AUPRC：BPE 识别判错的能力；AURC：风险-覆盖曲线下面积（越低越好）。", "",
              "| 数据集 | judge | 人类平局 | " + " | ".join(h for _, h, _ in K1) + " |", "|" + "---|" * (3 + len(K1))]
        for s in sets:
            js = [j for j in judges if (s, j) in CONV]
            if not js:
                continue
            if compact:
                rs = [CONV[(s, j)] for j in js if j in GEN]
                L.append(f"| {SN[s]} | 通用 VLM ×{len(rs)} | {f2(rs[0]['human_tie'])} | " + " | ".join(mm([r[k] for r in rs], fm) for k, _, fm in K1) + " |")
                continue
            for k_, j in enumerate(js + ["mean"]):
                rs = [CONV[(s, x)] for x in js if x in GEN] if j == "mean" else [CONV[(s, j)]]
                L.append(f"| {SN[s] if k_ == 0 else ''} | {'**均值（通用 VLM）**' if j == 'mean' else JN[j]} | {f2(rs[0]['human_tie'])} | "
                         + " | ".join(fm(sum(r[k] for r in rs) / len(rs)) for k, _, fm in K1) + " |")
        # ---- Table 2
        STEPS = cfg["methods"]
        K2 = [("acceptance", "输出率", f2), ("decisive", "判决率", f2), ("set_size", "平均集合大小", f2), ("risk", "实际错误率", f3),
              ("human_agreement", "人类一致率", f3), ("guarantee_success", "保证成功率", f2), ("significant_violation", "显著违约率", f2),
              ("strict_error", "严格断言错误率", f3), ("empty_splits", "空划分", f2),
              ("tie_share_accepted", "被接受样本中的平局占比", f3), ("tie_share_strict", "严格断言中的平局占比", f3)]
        for al in cfg["alphas_certify"]:
            L += ["", f"## 表 2 α = {al}　风险控制（对应 SCOPE 表 3、Trust or Escalate 表 2）", "",
                  "部署在含平局的测试集上。输出率 = coverage；实际错误率 = SCOPE 的 empirical FDR；人类一致率 = ToE 的 empirical human agreement；"
                  "保证成功率 = ToE 的 guarantee success rate（测试错误率 ≤ α 的划分比例，目标 ≥ 0.9）；显著违约率 = 测试错误率显著高于 α（二项检验 p < .05）的划分比例；"
                  "判决率 = 明确\"A 更好 / B 更好\"的比例；集合大小 1 = 明确、2 = 弱断言、3 = 弃权；空划分 = 全部弃权的划分比例；"
                  "被接受样本中的平局占比、严格断言中的平局占比：与表 0 的人类平局占比对比（命题 1：严格断言落在平局上必错；空 = 该结果生成时还没有这一列）。", "",
                  "| 数据集 | 方法 | judge | " + " | ".join(h for _, h, _ in K2) + " |", "|" + "---|" * (3 + len(K2))]
            for s in sets:
                first = True
                for st, nm in STEPS:
                    js = [j for j in judges if (s, j, st, al) in CERT]
                    if not js:
                        continue
                    if compact:
                        rs = [CERT[(s, j, st, al)] for j in js if j in GEN]
                        if rs:
                            L.append(f"| {SN[s] if first else ''} | {nm} | ×{len(rs)} | " + " | ".join(mm([r.get(k, float("nan")) for r in rs], fm) for k, _, fm in K2) + " |"); first = False
                        continue
                    for k_, j in enumerate(js + ["mean"]):
                        rs = [CERT[(s, x, st, al)] for x in js if x in GEN] if j == "mean" else [CERT[(s, j, st, al)]]
                        if not rs:
                            continue
                        vals = [mm([r.get(k, float("nan")) for r in rs], fm) if j == "mean" else fm(rs[0].get(k, float("nan"))) for k, _, fm in K2]
                        if j == "mean":
                            vals = [v.split(" (")[0] for v in vals]
                        L.append(f"| {SN[s] if first else ''} | {nm if k_ == 0 else ''} | {'均值（通用 VLM）' if j == 'mean' else JN[j]} | " + " | ".join(vals) + " |"); first = False
        # ---- Table 4 router
        VAR = cfg["router_variants"]
        K4 = [("acceptance", "输出率", f2), ("informative", "有信息量", f2), ("decisive", "判决率", f2), ("level", "绝对质量结论", f2),
              ("level_precision", "其正确率", f2), ("weak", "弱断言", f2), ("risk", "实际错误率", f3), ("human_agreement", "人类一致率", f3),
              ("guarantee_success", "保证成功率", f2), ("significant_violation", "显著违约率", f2)]
        for al in cfg["alphas_router"]:
            L += ["", f"## 表 4 α = {al}　成对 → 单项打分路由", "",
                  "有信息量 = 严格断言 + 两种绝对质量结论（\"两者都不合格\"、\"两者没有明显差距\"）；规则\"总体\" = 总错误率 ≤ α，\"每一级\" = 每种结论各自 ≤ α。"
                  "对照 = 同样路由、同样结论，但不看单项分数（用训练集基础比例估计错误率）。", "",
                  "| 数据集 | 规则 | 方法 | judge | " + " | ".join(h for _, h, _ in K4) + " |", "|" + "---|" * (4 + len(K4))]
            for s in cfg["router_datasets"]:
                for rule, rn in (("overall", "总体"), ("every level", "每一级")):
                    first = True
                    for v, vn in VAR:
                        js = [j for j in judges if (s, j, v, rule, al) in ROUT]
                        if not js:
                            continue
                        if compact:
                            rs = [ROUT[(s, j, v, rule, al)] for j in js if j in GEN]
                            L.append(f"| {SN[s] if first else ''} | {rn if first else ''} | {vn} | ×{len(rs)} | " + " | ".join(mm([r[k] for r in rs], fm) for k, _, fm in K4) + " |")
                            first = False; continue
                        for k_, j in enumerate(js + ["mean"]):
                            rs = [ROUT[(s, x, v, rule, al)] for x in js if x in GEN] if j == "mean" else [ROUT[(s, j, v, rule, al)]]
                            vals = [mm([r[k] for r in rs], fm).split(" (")[0] if j == "mean" else fm(rs[0][k]) for k, _, fm in K4]
                            L.append(f"| {SN[s] if first else ''} | {rn if first else ''} | {vn if k_ == 0 else ''} | {'均值（通用 VLM）' if j == 'mean' else JN[j]} | " + " | ".join(vals) + " |")
                            first = False
        open(f"{out}/tables_{'compact' if compact else 'full'}.md", "w").write("\n".join(L) + "\n")
    if cfg.get("latex_dir"):
        from eaglevlm.pipeline import latex
        cfg["_stamp"] = os.path.basename(out)
        print("latex:", latex.write(cfg, CONV, CERT, ROUT, cfg["latex_dir"]))
        import shutil
        for f in os.listdir(cfg["latex_dir"]):
            shutil.copy(os.path.join(cfg["latex_dir"], f), out)
    link = os.path.join(res, "tables", "latest")
    if os.path.islink(link):
        os.remove(link)
    os.symlink(os.path.basename(out), link); print(out)


if __name__ == "__main__":
    main()
