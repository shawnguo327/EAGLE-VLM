#!/bin/bash
# GenAI-Bench extended from 600 to all 1,600 prompts (24,000 pairs; old ids/labels unchanged). judge_bench resumes into
# the existing output files, so only the 15,000 new pairs are judged.  GPU given as $1, models as the remaining args.
cd /media/vipuser/Data/eaglevlm/judgebench_local; export EAGLEVLM_DATA=$PWD
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models; O=/media/vipuser/Data/eaglevlm/out_abs
declare -A MP=([qwen3vl8b]=Qwen3-VL-8B-Instruct [internvl35_8b]=InternVL3_5-8B-HF [qwen35_9b]=Qwen3.5-9B [qwen25vl7b]=Qwen2.5-VL-7B-Instruct [qwen3vl32b]=Qwen3-VL-32B-Instruct)
g=$1; shift
for m in "$@"; do
  bs=8; [ $m = qwen3vl32b ] && bs=4
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P $J --model $m --model-path $M/${MP[$m]} --items items/genaibench.jsonl \
    --out $O/genaibench__$m.s0of1.jsonl --conds bin,tri,score --bs $bs --progress $O/logs/genaibench_full_$m.progress.json > $O/logs/genaibench_full_$m.log 2>&1
  echo "genaibench FULL $m done $(date)" >> $O/logs/done.txt
done
