#!/usr/bin/env bash
# wait for first-pass runs, then resume each judge (retries rows that failed on not-yet-uploaded images)
for p in 16180 16181 16182; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
for m in qwen3vl8b qwen35_9b qwen25vl7b; do MODELS=$m bash /media/shawn/C/papers/EAGLE-VLM/scripts/run_all.sh > logs/fill_$m.out 2>&1 & done
wait
echo FILL_DONE >> logs/run_all.log
