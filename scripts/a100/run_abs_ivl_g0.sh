#!/bin/bash
# InternVL3.5-8B on AGIQA-3K + Polaris on GPU0 (in parallel with the GPU3 queue; own output shard name so the two never
# write the same file; the loader keeps the first source for duplicate ids)
cd /media/vipuser/Data/eaglevlm/judgebench_local; export EAGLEVLM_DATA=$PWD
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models; O=/media/vipuser/Data/eaglevlm/out_abs
for d in polaris agiqa3k; do
  CUDA_VISIBLE_DEVICES=0 $P $J --model internvl35_8b --model-path $M/InternVL3_5-8B-HF --items items/$d.jsonl --out $O/${d}__internvl35_8b.s0of1g0.jsonl \
    --conds bin,tri,score --bs 8 --progress $O/logs/${d}_internvl35_8b_g0.progress.json > $O/logs/${d}_internvl35_8b_g0.log 2>&1
  echo "$d internvl35_8b (gpu0) done $(date)" >> $O/logs/done.txt
done
