#!/usr/bin/env bash
# after the fill pass and the 32B download: stronger-judge rung, pairwise only, all datasets + Place Pulse
until grep -q FILL_DONE logs/run_all.log 2>/dev/null && grep -q DL_DONE logs/dl_32b.log 2>/dev/null; do sleep 120; done
export EAGLEVLM_DATA=$PWD
for d in pp mmrlhf editreward qbench2 vlrb mmrb mllm_judge genai_arena mjbench mmrb2 imagerewarddb; do
  echo "=== $(date) qwen3vl32b $d" >> logs/run_qwen3vl32b.log
  /home/shawn/anaconda3/envs/lis-omni/bin/python /media/shawn/C/papers/EAGLE-VLM/eaglevlm/judge_bench.py --model qwen3vl32b --items items/$d.jsonl --out results/${d}__qwen3vl32b.jsonl --conds bin --bs 4 >> logs/run_qwen3vl32b.log 2>&1
done
echo DONE_32B >> logs/run_all.log
