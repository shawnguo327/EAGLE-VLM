#!/bin/bash
# AGIQA-3K + Polaris (human absolute scores; router datasets 3 and 4): bin,tri,score.
# GPU3 (free after the Best-of-N smoke test failed): Qwen3-VL-8B, InternVL3.5-8B, Qwen3-VL-32B.
cd /media/vipuser/Data/eaglevlm/judgebench_local; export EAGLEVLM_DATA=$PWD
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models; O=/media/vipuser/Data/eaglevlm/out_abs; mkdir -p $O/logs
# (moved to GPU3, which freed first)
run() { g=$1; m=$2; mp=$3; bs=$4
  for d in polaris agiqa3k; do
    CUDA_VISIBLE_DEVICES=$g $P $J --model $m --model-path $mp --items items/$d.jsonl --out $O/${d}__$m.s0of1.jsonl \
      --conds bin,tri,score --bs $bs --progress $O/logs/${d}_$m.progress.json > $O/logs/${d}_$m.log 2>&1
    echo "$d $m done $(date)" >> $O/logs/done.txt
  done; }
run 3 qwen3vl8b $M/Qwen3-VL-8B-Instruct 8; run 3 internvl35_8b $M/InternVL3_5-8B-HF 8; run 3 qwen3vl32b $M/Qwen3-VL-32B-Instruct 4
