#!/bin/bash
# Best-of-N queue on GPU3: download -> generate (Qwen2.5-VL-7B, N=4) -> items -> judge (qwen3vl8b: bin,tri,score)
cd /media/vipuser/Data/eaglevlm/judgebench_local; export EAGLEVLM_DATA=$PWD
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; B=/media/vipuser/Data/eaglevlm/bon_pipeline.py; O=/media/vipuser/Data/eaglevlm/out_bon; mkdir -p $O
$P $B download > $O/download.log 2>&1 && \
CUDA_VISIBLE_DEVICES=3 $P $B generate --n 4 --bs 8 > $O/generate.log 2>&1 && \
$P $B items > $O/items.log 2>&1 && \
CUDA_VISIBLE_DEVICES=3 $P /media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py --model qwen3vl8b --model-path /media/vipuser/Data/models/Qwen3-VL-8B-Instruct \
   --items items/bon.jsonl --out $O/bon__qwen3vl8b.s0of1.jsonl --conds bin,tri,score --bs 8 --progress $O/judge.progress.json > $O/judge.log 2>&1
echo "BON_DONE $(date)" >> $O/done.txt
