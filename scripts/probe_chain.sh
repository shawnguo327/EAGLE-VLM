#!/usr/bin/env bash
# vision-specific uncertainty probes (bin_blind for response judging, bin_lowres everywhere); one judge per invocation
m=$1; export EAGLEVLM_DATA=$PWD
for d in vlrb mmrb mllm_judge mmrlhf genai_arena mjbench mmrb2 imagerewarddb qbench2 editreward pp; do
  echo "=== $(date) $m $d" >> logs/probe_$m.log
  /home/shawn/anaconda3/envs/lis-omni/bin/python /media/shawn/C/papers/EAGLE-VLM/eaglevlm/judge_bench.py --model $m --items items/$d.jsonl --out results_probe/${d}__$m.jsonl --conds bin_blind,bin_lowres --bs 8 >> logs/probe_$m.log 2>&1
done
echo PROBE_DONE_$m >> logs/run_all.log
