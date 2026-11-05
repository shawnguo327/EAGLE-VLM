#!/bin/bash
# GenAI-Bench scored pairs: bin,tri,score for 5 judges.  GPU3 now; GPU2 after the hard-label control re-run finishes.
cd /media/vipuser/Data/eaglevlm/judgebench_local; export EAGLEVLM_DATA=$PWD
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models; O=/media/vipuser/Data/eaglevlm/out_abs; mkdir -p $O/logs
run() { g=$1; m=$2; mp=$3
  CUDA_VISIBLE_DEVICES=$g $P $J --model $m --model-path $mp --items items/genaibench.jsonl --out $O/genaibench__$m.s0of1.jsonl \
    --conds bin,tri,score --bs 8 --progress $O/logs/genaibench_$m.progress.json > $O/logs/genaibench_$m.log 2>&1
  echo "genaibench $m done $(date)" >> $O/logs/done.txt; }
( run 3 qwen3vl8b $M/Qwen3-VL-8B-Instruct; run 3 internvl35_8b $M/InternVL3_5-8B-HF; run 3 qwen35_9b $M/Qwen3.5-9B ) &
( until grep -q "DONE" /media/vipuser/Data/eaglevlm/out_lora/logs/lm_hard_all_f1.log 2>/dev/null; do sleep 60; done
  run 2 qwen3vl32b $M/Qwen3-VL-32B-Instruct; run 2 qwen25vl7b $M/Qwen2.5-VL-7B-Instruct ) &
wait
