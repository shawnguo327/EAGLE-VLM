# EAGLE-V 多卡运行包（8×H100）

研究问题：VLM judge 置信度低时，到底是**两个候选质量相近**（aleatoric，人类也分歧/平局），还是**模型无法确认**（epistemic）？
本包在多卡机器上跑本地之外的所有阶段，一条命令起停、看进度、打包带回。

```
eaglev_pack/
  run.sh            唯一入口（check / models / plan / start / status / stop / collect / logs）
  config.yaml       全部设置：解释器、GPU、模型（HF repo、每进程卡数、batch）、数据集、各阶段
  setup_env.sh      建 conda 环境（版本与 bjserver 一致）；requirements.txt
  code/             EAGLE-VLM 代码快照（VERSION 记录 commit）
  data/items/       11 个数据集（12,753 对）；data/images/ 全部图片（含 Place Pulse，路径已改为相对）
  models/           ./run.sh models 下载到这里（不随包传输）
  runs/<phase>/     raw/ 分片输出  logs/ 每任务日志  state/ 进度  tasks.json 任务状态
  return/           ./run.sh collect 生成的回传包
```

## 1. 传输（本机 → 服务器）

```bash
# 在 bjserver 上打包（只含代码+数据，约 3.5 GB；图片用硬链接，不额外占空间）
bash /media/shawn/C/papers/EAGLE-VLM/pack/build_pack.sh /media/shawn/C/papers/VLM/eaglev_pack
# 传到 H100（rsync 可断点续传；再次运行只传变化部分）
rsync -avP --exclude models --exclude runs --exclude return \
      /media/shawn/C/papers/VLM/eaglev_pack/ USER@H100:/data/eaglev_pack/
```
服务器不能联网时，把模型一并传过去：在 `config.yaml` 里给模型写 `path:`，或 `rsync` 到 `eaglev_pack/models/<name>/`。

## 2. 服务器上一次性准备

```bash
cd /data/eaglev_pack
bash setup_env.sh eaglev               # 末行打印解释器路径 -> 填到 config.yaml 的 python:
./run.sh check                         # 环境/数据/图片/模型逐项检查，最后一行 CHECK PASSED
export HF_ENDPOINT=https://hf-mirror.com   # 如需镜像
./run.sh models                     # 下载 config 中全部模型（约 760 GB）；也可只给阶段名，如 ./run.sh models core text
```

## 3. 运行

```bash
./run.sh start smoke smoke_reason D_smoke        # 冒烟：小模型 + 文本 LLM + 推理 + LoRA（约 20 分钟）
./run.sh status                                  # 全部 done 后：
./run.sh start smoke_big                         # 大模型加载（32B 单卡 / 38B 2 卡 / 72B 4 卡）
./run.sh plan core text text_big nuisance scale_small scale_big reason B B_big D D_hard C_hires C_desc C_judge
# 正式：一个优先级队列（命令行顺序 = 优先级），空闲 GPU 自动回填后面的任务；backlog 放最后，用来填满剩余时间
./run.sh start core text text_big nuisance scale_small scale_big reason B B_big D D_hard C_hires C_desc C_judge \
               scale_big_more scale_big_full reason_big
```
- `start` 立即返回，调度器在后台（断开 ssh 不受影响）。
- 单卡模型每张 GPU 跑 2 个进程（`slots_per_gpu: 2`；单进程受 CPU 图像预处理限制，实测两进程吞吐翻倍）；多卡模型、32B 和训练独占 GPU。
- 任何时候停下都可以：已完成的行全部保留，下次 `start` 续跑；按优先级排序保证先拿到最重要的结果。
- 任务写出错误行（OOM 等）时以退出码 3 结束 → 自动重试（只重做错误行），用完重试次数后标记 `partial` 但不阻塞后续阶段。
- `./run.sh plan ...` 打印每阶段/模型的调用量与 GPU 时估算（速度取自 config，冒烟后可按 status 的实测速率修正 `speed`）。

## 4. 监视

```bash
./run.sh status            # 一次性
./run.sh status -w 30      # 每 30 秒刷新：每任务 done/total、%、速率、ETA、GPU 利用率/显存
./run.sh logs scale/qwen25vl72b/s0 -f    # 跟踪单个任务日志
cat runs/scheduler.log     # 调度器日志
```

## 5. 停止

```bash
./run.sh stop              # 只按记录的 PID 结束调度器及其任务（不会误杀他人进程）
```

## 6. 回传

```bash
./run.sh collect scale C_hires C_desc C_judge D          # 合并分片 -> runs/<phase>/merged/，打包 return/eaglev_*.tar.gz
./run.sh collect B --hidden                              # B 的隐状态（约 1–2 GB/模型）
./run.sh collect D --lora                                # 附带 LoRA 权重
# 本机取回并解包到数据目录
rsync -avP USER@H100:/data/eaglev_pack/return/eaglev_scale-*.tar.gz /media/shawn/C/papers/VLM/judgebench/h100/
cd /media/shawn/C/papers/VLM/judgebench/h100 && tar xzf eaglev_scale-*.tar.gz
```
合并后的结果与本地 `results/` 同格式（`{dataset}__{model}.jsonl`），本地分析：
```bash
cd /media/shawn/C/papers/VLM/judgebench; export EAGLEVLM_DATA=$PWD; S=/media/shawn/C/papers/EAGLE-VLM/scripts
python $S/analyze_A.py --dirs h100/scale/merged --judges qwen3vl2b,qwen3vl4b,qwen3vl32b,... --tag scale
python $S/probe_hidden.py --model qwen3vl8b --dirs h100/B/hidden --results results
python $S/analyze_A.py --dirs h100/D/merged ...          # tri_ft 结果
```

## 阶段与预算（8×H100；`./run.sh plan` 的估算，速度按 bjserver 实测 ×1.3 标定，误差约 ±30–50%）

数据：48k 对（含 6 个“全量”集，人类平局/分歧样本是现在的 4–5 倍；PP 平局集 6000 对；文本 judge 集 MT-Bench 人类判断 910 对 + LMArena 3000 对）。

| 优先级 | 阶段 | 内容 | GPU 时 |
|---|---|---|---|
| 1 | core | 4 个基础 judge × 全量 × bin/tri/str7/score | 13.5 |
| 2 | text, text_big | 文本域对照：文本 LLM（8B/7B/32B/72B）与 VLM 在同一协议下判文本对 | 14.1 |
| 3 | nuisance | 3 种改写指令 + JPEG + 裁剪：epistemic = 对无关扰动的敏感度 | 16.9 |
| 4 | scale_small, scale_big | 3 家族 1B→72B 尺寸阶梯（大模型 bin/tri） | 87.5（72B 约 47） |
| 5 | reason | 推理 + 自洽采样（N=4，思考模式，未完成样本 budget forcing） | 41.7 |
| 6 | B, B_big | 4 层隐状态（线性探针） | 11.5 |
| 7 | D, D_hard | LoRA 拟合人类偏好分布（两折）+ 硬标签对照 | 23.2 |
| 8 | C_hires, C_desc, C_judge | 更多视觉证据：高分辨率/切片、先描述再判断 | 13.5 |
| | **主计划合计** | | **≈222 GPU 时 ≈ 28 小时** |
| 9 | backlog | 大模型 str7/score、32B 全量、32B 推理 | 109（填满剩余时间） |

租 24 小时：主计划前 7 个优先级约 200 GPU 时，基本跑满；C 阶段与 backlog 能跑多少算多少。

## 条件说明（code/eaglevlm/judge_bench.py）

| 条件 | 读出 | 作用 |
|---|---|---|
| bin | A/B | 基线（强制二选一） |
| tri | A/B/T | 显式“质量相近”通道 P(T) |
| str7 | 1..7 | 偏好强度分布（4 = 相近） |
| score | 1..9 | 逐点评分 |
| bin_hires | A/B | 1536 px；InternVL 打开动态切片：更多视觉证据能否消除“无法确认” |
| bin_desc | A/B | 先描述再判断（描述来自 C_desc） |
| tri_p1..3 | A/B/T | 三种改写指令（指令扰动） |
| tri_jpeg / tri_crop | A/B/T | JPEG q30 / 中心裁剪 90%（图像扰动；文本项跳过） |
| reason | 判决样本 | 推理后输出 Verdict，N 次采样（reason_judge.py） |
| tri_ft / tri_hard | A/B/T | D / D_hard 阶段 LoRA 后的折外预测 |
