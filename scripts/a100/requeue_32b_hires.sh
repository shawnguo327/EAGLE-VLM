#!/bin/bash
# 32B full-resolution run (OOM'd while the 38B pointwise shard held GPU0/1): start after s38 shard 1 finishes
cd /media/vipuser/Data/eaglevlm/judgebench_local; export EAGLEVLM_DATA=$PWD
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models; O=/media/vipuser/Data/eaglevlm/out_hires
until grep -q "s38 internvl35_38b shard 1/2 done" /media/vipuser/Data/eaglevlm/out_q3/logs/done.txt 2>/dev/null; do sleep 120; done
CUDA_VISIBLE_DEVICES=1 $P $J --model qwen3vl32b --model-path $M/Qwen3-VL-32B-Instruct --plan $O/plan_32b.json \
  --conds bin_hires,tri_hires,score_hires --bs 4 --progress $O/logs/mj_32b.progress.json >> $O/logs/mj_32b.log 2>&1
echo "mllm_judge_big 32b done $(date)" >> $O/logs/done.txt
