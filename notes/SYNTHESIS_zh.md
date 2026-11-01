# 扩展文献综述总览（2026-10-31）：通用 VLM、各领域的 VLM 评审，以及 EAGLE-V 能为别人做什么

本目录下四份英文详稿（约 280 篇文献，arXiv 编号均已对照 abs 页或 arXiv API 核实；未核实的条目在原文中标了 † / [M] / U）：

| 文件 | 范围 |
|---|---|
| `A_general_vlm_2025_2026.md` | 通用 VLM 2025-01 → 2026-10：模型家族、多模态推理与 RL、评审/奖励模型如何进入训练、评测危机、VLM 校准与弃权 |
| `B_medicine_science_judges.md` | 医学（放射报告评测、医学 VQA、医学奖励模型、临床转诊）、科学图表/文档、教育与无障碍、农业生态等小众领域 |
| `C_urban_geo_embodied_judges.md` | 城市与地理、遥感、自动驾驶、具身机器人、GUI/网页智能体、视频与生成评测 |
| `D_eaglev_novelty_applications.md` | 新颖性核查（最近邻 10 篇）+ 应用排序 + 三种论文定位 |

---

## 一、最重要的一条：EAGLE 的 Δp 与 SCOPE 的 BPE 是同一个信号

**SCOPE**（Badshah, Emami, Sajjad，arXiv 2602.13110，2026-02-13，**ICML 2026**，已亲自核对摘要）提出 Bidirectional Preference Entropy。它把 judge 在两种顺序下的偏好概率取平均得到 p̄，再取熵 H(p̄)，然后做共形阈值，使接受集合上的错误率 ≤ α。

H(p̄) 是 |2p̄−1| = Δp 的严格单调函数，因此**两者排序完全相同：任何阈值下接受同一批样本，风险—覆盖曲线也完全一样**。

- **对 EAGLE 文本版的影响（需要你判断）。** 如果 EAGLE 第一次公开（arXiv 或投稿）早于 2026-02，就是同期独立工作，引用并说明等价即可。如果晚于它，修改稿就不能把 Δp 本身当贡献。贡献应放在 SCOPE 没有的部分：pointwise 回退、BT/Davidson 排序，以及 E2/E3 中“Δp 能排风险但排不准干预价值”的发现。SCOPE 正文也把 pointwise 列为 future work。**IJCAI 投稿前这一点必须处理。**
- **对 EAGLE-V 的影响。** 不能声称“提出了新的置信度”，要写成“沿用 SCOPE/EAGLE 的双向置信度，研究它在视觉上的表现”。

---

## 二、通用 VLM 两年来的变化（A 文件要点）

1. **架构已收敛。** 主流组合是原生分辨率 ViT + MLP 合并器 + MoE LLM + 文本—视觉联合预训练，差异主要来自数据和 RL。
2. **“思考”成了一个模式，而不是单独的模型。** Qwen3-VL、GLM-4.5V、Kimi-VL、MiMo-VL 都同时提供思考和非思考两种模式。
3. **RL 改善的主要是“用证据”，不是“看清”。** 只奖励最终答案的 RLVR 留下了感知错误和 reward hacking 的空间（2607.09492）。2026 年的主流做法是增加针对感知的奖励。
4. **评审已经成为奖励通道。**
   - Pref-GRPO（2508.20751）直接用组内两两胜率作奖励。
   - 也有用 rubric 打分的评审直接接入 GRPO。
   - 在 MMRB2（CVPR 2026）上，GPT-5 和 Gemini 2.5 Pro 与人类一致率只有 66–75%，而人类之间 > 90%。
5. **成对评审比单点打分更可靠，但有偏差。** GenArena 上成对与排行榜的 ρ=0.86，单点只有 0.36。偏差包括：
   - **forced-choice**：在两个一样好的候选里硬选一个（2604.21523）；
   - **informativeness bias**：30–50% 的情况下不看图就选信息更多的回答（2604.17768，ACL 2026）；
   - 位置偏差：图像编辑评测中换序会翻转多达 60.9% 的判决（2610.01670）。
6. **VLM 校准在 2025–26 成了独立子领域。**
   - 口头报告的置信度差，token 概率好（2607.22034）。
   - CoT 让 token 概率过度自信，但基于一致性的信号依然稳健（2603.16728）。
   - 两点都支持“首 token + 双顺序”的做法。
7. **你原综述的主要缺口：** 没有校准/弃权一节，没有 arena 式评测，评审/奖励模型只写到 R1-Reward，GPT-5.4 的数字引自新闻网站。

---

## 三、各领域现状（B、C 文件要点）

| 领域 | 现在怎么评 | 可用的人类成对 / 一致性数据 | 有没有“带保证的评审” |
|---|---|---|---|
| 医学放射报告 | GREEN、RaTEScore、CheXprompt、RadFact 等，**全是纯文本、参照式、从不弃权、不看图** | ReXVal（200 对，6 位放射科医生，需 PhysioNet 认证）、RadEvalX（100 份，开放）、Med-RewardBench（1,026 例，需向作者申请） | 无。医学的选择性预测（ReXTrust、RadFlag、共形对齐）针对的都是**生成器**，不是评审 |
| 科学图表 / 文档 | 图表评审之间分歧大（2505.08468） | SciCap-Eval（3,600 条）、VisualProcessBench | Consensus Entropy（CVPR 2026）用多个 VLM 的一致性路由 OCR，无保证。这是 EAGLE-V 最直接的基线 |
| 教育 | GPT-5 批改手写微积分时加置信过滤，**约 70% 仍退回人工**（2510.05162） | FERMAT（ACL 2025） | 只有启发式过滤 |
| 城市感知 | UrbanAlign、Place Pulse 上的 LLM 研究 | Place Pulse 2.0 | 无。GPT-4o 零样本 56.7%（κ=0.15），且有欧美文化偏差（2604.20048）。ICML 2026 有文章呼吁“城市感知基准要报告可靠性”（2606.00871） |
| 遥感 | 多用 BLEU/CIDEr 或未经验证的 GPT 评审 | VRSBench；**尚无人类成对偏好集** | 无 |
| 自动驾驶 | DriveCritic（ICRA 2026）做轨迹对的人类偏好 | DriveCritic（约 5.7k 对）、DriveJudge（3.3 万） | 只有一篇 5 页扩展摘要做单点共形（2610.02765） |
| 机器人 | VLM 做成功检测 / 奖励（RoboReward、TOPReward） | FailBench（最好 0.77 平衡准确率，模糊时偏向判“成功”）、ROBORMBENCH | 无（只读单点 token 概率） |
| GUI 智能体 | VLM 判轨迹 | AgentRewardBench、Online-Mind2Web | CORA 做的是**动作**的风险控制，不是判决 |
| 长视频 | 前沿评审成对准确率 75.4%（PlaylistEval） | SLVMEval、PlaylistEval | 无 |
| T2I 生成 | VQAScore、ImageReward、HPSv3、UnifiedReward | GenAI-Arena、MJ-Bench、MMRB2 | SHIP（2608.21748）控制的是**放出图像**的风险，不是评审与人类一致 |

**共同结论：每个领域都已经有人类成对数据可以测评审，但没有一个领域的评审能“知道自己什么时候不可靠”并带保证地升级。** 最近的两篇多模态工作都是本月出的：

- 2610.01243：换序一致性过滤，作者明说“无校准、无保证”；
- 2604.25235：给 VLM 评审打分做共形区间，单点、不做选择。

窗口还开着，但在收窄。SCOPE 作者已经出了后续（Judge, Retrieve, or Abstain，COLM 2026），下一步很可能就是视觉。

---

## 四、如果 EAGLE-V 做成了，能帮到别人什么（按 影响 × 单人可行性 排序）

| # | 应用 | 谁会用 | 现在怎么做 | EAGLE-V 带来什么 | 演示数据 |
|---|---|---|---|---|---|
| 1 | **少标注的基准 / 排行榜评测（PPI）** | 做 benchmark 和评测的人 | 全人工，或评审数字不带置信区间 | 用置信度决定哪些样本给人标，PPI 给出**对任意评审都有效**的 BT 排名置信区间 | VL-RewardBench、GenAI-Arena |
| 2 | **城市感知问卷分流** | 城市科学、地理、HCI 做感知调查的人 | 随机或 TrueSkill 主动采样，全靠众包 | 有把握的对交给 VLM，人只看没把握的；顺便按 ICML 2026 那篇的要求报告可靠性 | Place Pulse 2.0（你的主场） |
| 3 | **少投票的生成模型竞技场** | GenAI-Arena、K-Sort Arena | 全人工投票，或 K-Sort Eval 的启发式修正（无置信区间） | 只把低置信对局交给人；用 Davidson 处理“都不好”的平票 | GenAI-Arena 公开投票 |
| 4 | **图像生成 best-of-N 重排** | 做推理时扩展的人 | 单点奖励，或不设停止规则的 VLM 锦标赛 | 有把握的对局便宜地定下来，没把握的升级；与 SHIP 互补 | GenEval + FLUX-schnell |
| 5 | **多模态 DPO 偏好数据筛选** | 做 MLLM 对齐的人 | RLAIF-V 等用分差启发式过滤 | 保留集合的标签错误率**有保证** | RLAIF-V、VLFeedback |
| 6 | 智能体 / 机器人成功判定 | 做 agent RL 的人 | 单次 VLM 判决 | 不确定的轨迹升级给人，减少假“成功”样本 | AgentRewardBench、FailBench |
| 7 | 会弃权的多模态 RL 奖励 | 做 GRPO 的人 | 评审始终给奖励 | 评审没把握时给零或中性奖励，减少 reward hacking | UnifiedReward + 3B 模型 |
| 8 | 医学报告审阅 / 遥感描述审阅 | 临床 AI、遥感 | 纯文本指标 | 能看图的评审，把专家时间花在没把握的样本上 | ReXVal（数据获取慢）、VRSBench |

注意：MIMIC 数据按 DUA 不能发给 API 评审，只能用本地模型（如 MedGemma）。

**对单人 3 个月最划算的是 1–3。** 置信度弱（AUROC≈.62）只损失效率，不损失 PPI 的有效性，而且计算主要是 CPU 统计，可以复用已经跑完的评审输出。

---

## 五、今天在 Place Pulse 上发现的数据问题（影响之前的结论）

用本地 annotations 做了核查：

1. **“多票对”大多是重复提交，不是独立的人。**
   - 同一对图的相邻两票中，有 15,697 次间隔 < 1 分钟。这些票**左右摆放 100% 相同**，一致率 90.1%。
   - 间隔 ≥ 1 天的只有 83 次，一致率 0.614。
   - 所以之前用来衡量“人类分歧”的多票对（2,284 / 713 对）主要反映重复点击。**“Δp 与人类分歧无关（ρ≈0）”这一条需要去重后重算。**
   - 共去掉 16,790 张一天内的重复票。
2. **人类投票本身的可预测性很低。** 去重后每个类别各用 90% 的投票拟合图像级 Bradley–Terry 模型，在 10% 留出投票上的结果：

   | 类别 | 留出准确率 | 可解释方差比例 |
   |---|---|---|
   | beautiful | .628 | .098 |
   | boring | .560 | .022 |
   | depressing | .602 | .062 |
   | lively | .603 | .058 |
   | safety | .632 | .092 |
   | wealthy | .631 | .097 |

   每张图只有约 2–4 票，BT 欠拟合。但这仍说明**单票几乎是抛硬币**，VLM 的 .645 已经和“用人类票拟合的模型”相当，甚至更好。
   这正好对应论文里的命题 2（PPI 收益上限 = Var(η)/Var(y)）：**在 Place Pulse 的单票层面，任何评审都省不了多少人工标注；收益要在图像排名层面谈。**这是论文里一个干净、可报告的负结果。

---

## 六、论文草稿

位置：`/home/shawn/bjserver-papers/VLM/eaglev-paper/`。`main.tex` 加 `sections/`，`refs.bib` 有 59 条，全部来自 arXiv API 元数据。

- **题目（暂定）：** *Seeing Is Not Certainty: Risk-Controlled Escalation for Pairwise Vision-Language Judges*
- **定位：** D 文件的方案 A 加方案 B：诊断 + 带保证的升级阶梯 + PPI 排名。
- **贡献：**
  1. 双向置信度在 VLM 上的诊断：会饱和，对人类分歧盲；
  2. 成对 → 单点 → 强模型 → 人的阶梯，用 fixed-sequence 联合校准，给出有限样本保证；
  3. 按置信度路由的 PPI-BT/Davidson 排名，加上“收益上限”命题。
- **还缺什么：** 所有服务器数字都用红色 `\TBD{}` 占位，`\todo{}` 标出待办。跑完 judgebench 后，运行 `make_paper_tables.py` 会自动生成 `tables/*.tex`。
