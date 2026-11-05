#!/usr/bin/env bash
# InternVL3.5-8B: $1 = main (bin,score -> results/) or probe (bin_blind,bin_lowres -> results_probe/)
m=internvl35_8b; export EAGLEVLM_DATA=$PWD
if [ "$1" = main ]; then conds=bin,score; out=results; else conds=bin_blind,bin_lowres; out=results_probe; fi
for d in pp vlrb mmrb mllm_judge mmrlhf genai_arena mjbench mmrb2 imagerewarddb qbench2 editreward; do
  echo "=== $(date) $m $1 $d" >> logs/ivl_$1.log
  /home/shawn/anaconda3/envs/lis-omni/bin/python /media/shawn/C/papers/EAGLE-VLM/eaglevlm/judge_bench.py --model $m --items items/$d.jsonl --out $out/${d}__$m.jsonl --conds $conds --bs 8 >> logs/ivl_$1.log 2>&1
done
echo IVL_DONE_$1 >> logs/run_all.log
