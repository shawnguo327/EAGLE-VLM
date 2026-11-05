#!/bin/bash
# resolution ablation on the A100 after InternVL3.5-38B shard 1 frees GPU0/1:
#   GPU0: qwen3vl8b full-res on GenAI-Bench (the 768-px reference runs on GPU3);  GPU1: qwen3vl32b full-res on MLLM-J sets
cd /media/vipuser/Data/eaglevlm/judgebench_local; export EAGLEVLM_DATA=$PWD
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models; O=/media/vipuser/Data/eaglevlm/out_hires; mkdir -p $O/logs
until grep -q "b38 internvl35_38b shard 1/2 done" /media/vipuser/Data/eaglevlm/out_q3/logs/done.txt 2>/dev/null; do sleep 120; done
( CUDA_VISIBLE_DEVICES=0 $P $J --model qwen3vl8b --model-path $M/Qwen3-VL-8B-Instruct --items items/genaibench_big.jsonl \
    --out $O/genaibench_big__qwen3vl8b.s0of1.jsonl --conds bin_hires,tri_hires,score_hires --bs 4 --progress $O/logs/gb_8b.progress.json > $O/logs/gb_8b.log 2>&1
  echo "genaibench_big qwen3vl8b done $(date)" >> $O/logs/done.txt ) &
( python3 -c "import json; print(json.dumps([{'items': f'items/{d}.jsonl', 'out': f'$O/{d}__qwen3vl32b.s0of1.jsonl'} for d in ('mllm_judge_sp_big', 'mllm_judge_big')]))" > $O/plan_32b.json
  CUDA_VISIBLE_DEVICES=1 $P $J --model qwen3vl32b --model-path $M/Qwen3-VL-32B-Instruct --plan $O/plan_32b.json \
    --conds bin_hires,tri_hires,score_hires --bs 4 --progress $O/logs/mj_32b.progress.json > $O/logs/mj_32b.log 2>&1
  echo "mllm_judge_big 32b done $(date)" >> $O/logs/done.txt ) &
wait
