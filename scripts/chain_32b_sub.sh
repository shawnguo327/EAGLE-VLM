#!/usr/bin/env bash
# 32B rung on <=700-pair stratified subsets (items32/), after the full Place Pulse run (PID 41751) finishes
while kill -0 41751 2>/dev/null; do sleep 60; done
export EAGLEVLM_DATA=$PWD
for d in mmrlhf editreward qbench2 vlrb mmrb mllm_judge genai_arena mjbench mmrb2 imagerewarddb; do
  echo "=== $(date) qwen3vl32b $d" >> logs/run_qwen3vl32b.log
  /home/shawn/anaconda3/envs/lis-omni/bin/python /media/shawn/C/papers/EAGLE-VLM/eaglevlm/judge_bench.py --model qwen3vl32b --items items32/$d.jsonl --out results/${d}__qwen3vl32b.jsonl --conds bin --bs 8 >> logs/run_qwen3vl32b.log 2>&1
done
echo DONE_32B >> logs/run_all.log
