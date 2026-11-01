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
./run.sh models smoke scale B C_hires C_desc D   # 只下载这些阶段用到的模型（共约 490 GB），可重复运行补齐
```

## 3. 运行

```bash
./run.sh start smoke                   # 先冒烟：3 个小模型 × 每数据集 2 对（约 5 分钟）
./run.sh status                        # 全部 done 后再继续
./run.sh start smoke_big D_smoke       # 多卡模型加载 + LoRA 训练冒烟
./run.sh start scale B C_hires C_desc C_judge D   # 正式：按顺序执行，全程后台
```
- `start` 立即返回，调度器在后台（`nohup` 等价，断开 ssh 不受影响）。
- 每个阶段按 `GPU数 / 每模型卡数` 自动切分片，同一模型的多个分片并行，大模型（32B/38B 2 卡、72B 4 卡）用 `device_map=auto`。
- 任务失败自动重试 `retries` 次；所有任务可续跑（已完成的行跳过），`stop` 后再 `start` 同一阶段不会重复计算。
- 改参数：编辑 `config.yaml`（例如 `gpus: [0,1,2,3]` 只用 4 张卡；某模型 `bs: 4` 防 OOM）。分片数在阶段第一次启动时固定在 `runs/<phase>/manifest.json`，改卡数后要删掉它（及该阶段 raw/）重跑。
- `./run.sh plan scale` 打印将要执行的全部命令（不运行）。

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

## 阶段与预算（8×H100，HF Transformers 推理；12,753 对）

| 阶段 | 内容 | 调用量 | 估计墙钟 |
|---|---|---|---|
| smoke / smoke_big / D_smoke | 冒烟 | 极少 | 共约 20 分钟（含模型加载） |
| scale | 11 个新尺寸 × bin,score,tri,str7（每对 8 次调用） | 约 1.1M | 小模型 10–20 分钟/个；32B/38B 约 1 小时/个；72B 约 4–6 小时；合计约 10–12 小时 |
| B | 7 个模型 × bin,tri + 4 层隐状态 | 约 0.36M | 约 3–4 小时（72B 占一半） |
| C_hires | 1536 px（InternVL 切片）× bin_hires,tri | 约 0.2M | 约 1–2 小时 |
| C_desc → C_judge | 自描述 256 token 生成 → 带描述判断 | 约 18k 生成 + 0.1M | 约 2–3 小时 |
| D | 8B LoRA 两折（每折 4 卡 DDP，约 12k 对 × 2 顺序）+ 折外预测 | — | 约 2–3 小时/模型 |

全部约 20–25 小时墙钟（约 8 GPU·天）。最先回传 `scale`，它决定论文主线（epistemic 随规模下降、aleatoric 不降？）。

## 条件说明（code/eaglevlm/judge_bench.py）

| 条件 | 读出 | 作用 |
|---|---|---|
| bin | A/B | 基线（强制二选一） |
| tri | A/B/T | 显式“质量相近”通道 P(T) |
| str7 | 1..7 | 偏好强度分布（4 = 相近） |
| score | 1..9 | 逐点评分 |
| bin_hires | A/B | 1536 px；InternVL 打开动态切片：更多视觉证据能否消除“无法确认” |
| bin_desc | A/B | 先描述再判断（描述来自 C_desc） |
| tri_ft | A/B/T | D 阶段 LoRA 后的折外预测 |
