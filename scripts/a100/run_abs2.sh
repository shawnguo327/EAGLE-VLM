#!/bin/bash
# AGIQA-3K + Polaris (human absolute scores; router datasets 3 and 4): bin,tri,score.
# GPU0 after the full-resolution GenAI-Bench run (qwen3vl8b, pid 300951): Qwen3-VL-8B then InternVL3.5-8B, then Qwen3-VL-32B.
cd /media/vipuser/Data/eaglevlm/judgebench_local; export EAGLEVLM_DATA=$PWD
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models; O=/media/vipuser/Data/eaglevlm/out_abs; mkdir -p $O/logs
while kill -0 300951 2>/dev/null; do sleep 60; done
run() { g=$1; m=$2; mp=$3; bs=$4
  for d in polaris agiqa3k; do
    CUDA_VISIBLE_DEVICES=$g $P $J --model $m --model-path $mp --items items/$d.jsonl --out $O/${d}__$m.s0of1.jsonl \
      --conds bin,tri,score --bs $bs --progress $O/logs/${d}_$m.progress.json > $O/logs/${d}_$m.log 2>&1
    echo "$d $m done $(date)" >> $O/logs/done.txt
  done; }
run 0 qwen3vl8b $M/Qwen3-VL-8B-Instruct 8; run 0 internvl35_8b $M/InternVL3_5-8B-HF 8; run 0 qwen3vl32b $M/Qwen3-VL-32B-Instruct 4
