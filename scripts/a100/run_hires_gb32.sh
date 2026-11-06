#!/bin/bash
# resolution ablation: Qwen3-VL-32B at full resolution (long side <= 1536) on the >768-px GenAI-Bench pairs, A100 GPU3
cd /media/vipuser/Data/eaglevlm/judgebench_local; export EAGLEVLM_DATA=$PWD
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models; O=/media/vipuser/Data/eaglevlm/out_hires
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P $J --model qwen3vl32b --model-path $M/Qwen3-VL-32B-Instruct \
  --items items/genaibench_big.jsonl --out $O/genaibench_big__qwen3vl32b.s0of1.jsonl --conds bin_hires,tri_hires,score_hires --bs 4 \
  --progress $O/logs/gb_32b.progress.json > $O/logs/gb_32b.log 2>&1
echo "genaibench_big qwen3vl32b done $(date)" >> $O/logs/done.txt
