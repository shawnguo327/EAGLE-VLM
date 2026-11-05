#!/usr/bin/env bash
# Phase A (local, 1 GPU): tri (A/B/T) + str7 (7-point) on all 11 datasets x 4 judges -> results_A/.
# First re-runs pp bin,score for the 3 Qwen judges with the current harness (old pilot outputs in results_pilot_pp/).
cd /media/shawn/C/papers/VLM/judgebench; export EAGLEVLM_DATA=$PWD
PY=/home/shawn/anaconda3/envs/lis-omni/bin/python; JB=/media/shawn/C/papers/EAGLE-VLM/eaglevlm/judge_bench.py
DS="pp mmrb2 genai_arena mllm_judge editreward vlrb mmrb mjbench imagerewarddb qbench2 mmrlhf"
for m in qwen3vl8b internvl35_8b qwen35_9b qwen25vl7b; do
  if [ $m != internvl35_8b ]; then
    echo "=== $(date) $m pp rerun" >> logs/A.log
    $PY $JB --model $m --items items/pp.jsonl --out results/pp__$m.jsonl --conds bin,score --bs 8 --progress logs/progress_A_pp_$m.json >> logs/A.log 2>&1
  fi
  python3 -c "import json,sys; print(json.dumps([{\"items\":f\"items/{d}.jsonl\",\"out\":f\"results_A/{d}__$m.jsonl\"} for d in sys.argv[1:]]))" $DS > logs/plan_A_$m.json
  echo "=== $(date) $m tri,str7" >> logs/A.log
  $PY $JB --model $m --plan logs/plan_A_$m.json --conds tri,str7 --bs 8 --progress logs/progress_A_$m.json >> logs/A.log 2>&1
done
echo A_DONE >> logs/run_all.log
