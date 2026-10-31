# VLM 与专用视觉任务：2026 / CVPR 2025 小规模对照

本轮完成了城市主观感知和带标注的客观视觉任务。最值得扩大验证的现象，是小模型的单图事实提取与双图相对判断之间存在明显差距；目前没有证据把所有这种差距都归因于视觉压缩，也不能泛化到所有 VLM。

## 实际模型和协议

| 模型 | 年份/会议依据 | 本轮用途 |
|---|---|---|
| MiniCPM-V 4.6，约1.3B | 2026 官方发布 | 同一权重4x/16x；城市评分、客观计数和比较 |
| dino.txt，约0.868B | CVPR 2025 | 无LLM解码器的视觉文本相似度排序；2模板族×2预处理 |
| DEIM-D-FINE-S，约10.2M | CVPR 2025 | COCO监督训练的专用检测器；bbox AP、存在性、计数 |
| Qwen3-VL-8B / Qwen3.5-9B | 官方已下载模型 | 沿用城市初评；新增相同COCO图对任务 |

城市侧固定300图对、600张不重复图，六维度各50对，299对只有一次人类投票。所有VLM使用BF16、贪心短输出、不启用思考；每对做原序和交换顺序比较，以及两次独立单图评分。

客观侧从COCO val2017固定随机抽取200图，再仅依据GT构造60个不复用图片的比较对：120图、近差30对/远差30对，目标类别为person/car/bus/truck/bicycle/motorcycle。含目标类crowd标注的图不用于该类计数。48对是0-vs-positive，只有12对是both-positive；单图72个GT>0、48个GT=0。常数零计数基线exact为40%。这些结果不是全量COCO官方榜单。

## 城市感知结果

| 配置 | 直接二选一，AB/BA平均 | 单图1–10评分导出排序 | 换序仍选择同图 |
|---|---:|---:|---:|
| Qwen3-VL-8B-Instruct | 63.0% | 64.0% | 78.0% |
| Qwen3.5-9B | 66.7% | 62.2% | 82.7% |
| MiniCPM-V-4.6-4x | 64.3% | 54.5% | 79.3% |
| MiniCPM-V-4.6-16x | 61.7% | 53.0% | 72.7% |

评分同分计0.5；比较标签反映已有投票，不代表可靠的人类共识。MiniCPM在原400×300街景上每图视觉token为4x模式252、16x模式63；两模式也改变原生merger路径，不能称纯token数量的因果实验。

dino.txt采用预先固定正负描述的余弦相似度差，没有拟合测试标签，也没有1–10分校准。四个变体分别为：

| 变体 | 排序一致率 |
|---|---:|
| center_crop224__caption | 58.7% |
| center_crop224__perception | 59.0% |
| full_letterbox224__caption | 58.3% |
| full_letterbox224__perception | 58.0% |

仅换描述模板族，约22.3%–23.0%的图对预测翻转；仅换center crop/保留全图的letterbox，约12.3%翻转。总体准确率差的置信区间跨零。这说明平均分可以掩盖个体判断变化；模板族保留相同形容词，但不是严格语义等价的完美干预。dino.txt按单图效用排序，换序不变是计算结构的性质，不能当作额外实测优势。

## 客观计数与比较

| 配置 | 单图计数exact | 正计数侧exact（72侧） | 直接双图比较 | 单图计数导出排序 | 换序一致 |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | 84.2% | 75.0% | 91.7% | 95.8% | 96.7% |
| Qwen3.5-9B | 83.3% | 72.2% | 96.7% | 95.8% | 100.0% |
| MiniCPM-V-4.6-4x | 74.2% | 56.9% | 70.8% | 94.2% | 45.0% |
| MiniCPM-V-4.6-16x | 75.0% | 58.3% | 70.0% | 95.0% | 43.3% |
| DEIM-S，固定阈值0.4 | 79.2% | 66.7% | — | 94.2% | 单图计数构造 |

DEIM在随机200图的bbox AP/AP50为54.61/70.34，AP保留全部top300检测并按标准crowd规则计算；0.4阈值仅用于计数和存在性，未按本轮成绩调参。它接受过COCO监督训练，与通用VLM零样本提示的差异不能解释为同训练条件下的架构优劣。

MiniCPM 4x在32对中能把两张单图都数对，但这些图对的直接比较准确率只有71.9%，换序一致率43.8%；16x相应为34对、67.6%、35.3%。这种按模型输出筛出的子集只能作诊断，不能用于因果推断或直接跨模型比较。Qwen3.5在41对两侧计数都正确的图对中，比较和换序均100%；本轮不支持“所有VLM感知正确后仍无法比较”。

同60对中，DEIM严格排序正确的55对，Qwen3.5两个顺序均全对；DEIM两侧计数均正确的38对也没有Qwen3.5比较失败。因此不能把这批数据描述为专用CV普遍优于Qwen。

## 压缩模式的配对结果

以下差值均为16x减4x。城市的直接比较准确率变化−2.67个百分点（95% CI −5.50至+0.33）；换序一致率变化−6.67个百分点（−11.67至−1.33）。客观计数exact变化+0.83个百分点（−2.50至+4.17），直接比较变化−0.83个百分点（−4.17至+3.33），换序一致率变化−1.67个百分点（−8.33至+6.67）。

城市稳定性下降值得复查，但客观任务没有明确的模式差异。MiniCPM的大部分客观比较失稳在4x时已存在，当前结果不支持把它主要归因于压缩。小样本、多项探索和不同任务难度都限制了结论。

## 值得继续验证的方向

优先研究单图视觉证据如何进入多图相对判断。先在客观任务中区分图像身份绑定、事实提取和比较规则三类误差，再将有效的约束迁移到城市偏好。下一步可固定图对比较：直接双图输入、各图独立结构化事实后作符号比较、把同一份模型自提取事实提供给文本比较器，以及专用CV事实辅助。事实来源和计算预算需分开控制，不能用GT偷偷代替模型证据。

这条线的潜在增量应是可识别的错误分解和对应修复，而不是再报告一组模型分数。单图计数后作大小比较是简单基线，本身不是新方法；MiniCPM只有约1.3B，不能直接把其困难泛化成8B/9B模型的共同缺陷。

第二优先是任务依赖的稳定性约束：在相同预算下同时考察正确率和对表述/顺序的敏感性。先扩大both-positive计数、空间关系和定位样本，再验证是否存在跨模型规律。对于城市任务，应补重复人类判断或显式建模标注不确定性，避免优化单次投票噪声。

不建议直接把城市评分、顺序偏差或按后果分配视觉预算当成新颖点：UrbanAlign、PairBench/CompareBench、Visual Persuasion及Consequence-Sensitive Visual Token Compression分别已有直接相关工作。本轮是寻找假设的诊断，不是新颖性证明。

## 运行兼容性与验证

当前Transformers 5.16.1的MiniCPM 16x ViT merger假设同批网格尺寸一致，原生COCO混合尺寸推理报reshape错误，且没有产生有效预测。本实验只包装该模块实例：异形图片按各自H×W拆分后调用原生merger，保持权重、像素、顺序不变；同形输入继续走原生路径。原始失败记录保存在results_count_failed，修复源码和SHA记录在run_config，逐图参考与同形原生验证误差均为0。没有修改共享环境。

额外固定前16对分别作batch_size=1复核。每个MiniCPM模式64个输出中63个与batch_size=8一致，32个单图计数全部一致，差异均为同一图对的一次BA选择。批处理数值敏感性应保留记录；该复核不支持大幅比较差距完全由批处理错误造成。

全部主要VLM输出均有效；每配置每个图对、每个任务只保存一条预测。指标方向、分母及原始输出另经独立复算。所有GPU任务已结束后可重跑 scripts/run_all_vlms.sh；DEIM和dino.txt各自目录有独立复现入口。

## 来源与后续候选

- [MiniCPM-V 4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) — 2026 official model release；本轮已运行。
- [DINOv2 Meets Text / dino.txt](https://openaccess.thecvf.com/content/CVPR2025/html/Jose_DINOv2_Meets_Text_A_Unified_Framework_for_Image-_and_Pixel-Level_CVPR_2025_paper.html) — CVPR 2025 main conference；本轮已运行。
- [DEIM](https://github.com/Intellindust-AI-Lab/DEIM) — CVPR 2025 main conference；本轮已运行。
- [Mono-InternVL](https://github.com/OpenGVLab/Mono-InternVL) — CVPR 2025 main conference; researched, not run；仅文献核查，未复现其论文数值。
- [LLaVA-OneVision-2](https://arxiv.org/abs/2605.25979) — 2026-05 preprint; researched, not run；仅文献核查，未复现其论文数值。
- [UrbanAlign](https://arxiv.org/abs/2602.19442) — 2026 preprint; prior urban preference calibration；仅文献核查，未复现其论文数值。
- [PairBench](https://arxiv.org/abs/2502.15210) — 2025 preprint; prior comparison consistency evaluation；仅文献核查，未复现其论文数值。
- [CompareBench](https://arxiv.org/abs/2509.22737) — 2025 preprint; prior visual comparison benchmark；仅文献核查，未复现其论文数值。
- [Visual Persuasion](https://arxiv.org/abs/2602.15278) — 2026 preprint; prior controlled visual choices；仅文献核查，未复现其论文数值。
- [Consequence-Sensitive Visual Token Compression](https://arxiv.org/abs/2608.09176) — 2026-08 preprint; prior consequence-weighted visual budgets；仅文献核查，未复现其论文数值。

服务器根目录：`/media/shawn/C/place-pulse-2.0/experiments/vlm-cv-scout-20261030`。
主要数据文件：placepulse/summary.json、count_summary.json、dinotxt/summary.json、coco_deim/summary.json、coco_deim/count_pair_summary.json。逐条预测、提示词、源码、样本哈希和权重来源均已保存。
