#!/usr/bin/env bash
# Run the EAGLE-protocol judge on every converted benchmark. Code lives in the repo; data and results in $EAGLEVLM_DATA.
# Pass 1: Qwen3-VL-8B, pairwise + pointwise. Pass 2: Qwen3.5-9B and Qwen2.5-VL-7B, pairwise + pointwise.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export EAGLEVLM_DATA="${EAGLEVLM_DATA:-/media/shawn/C/papers/VLM/judgebench}"
PY="${PY:-/home/shawn/anaconda3/envs/lis-omni/bin/python}"
cd "$EAGLEVLM_DATA"; mkdir -p results logs
DS="${DS:-mmrlhf editreward qbench2 vlrb mmrb mllm_judge genai_arena mjbench mmrb2 imagerewarddb}"
MODELS="${MODELS:-qwen3vl8b qwen35_9b qwen25vl7b}"
for m in $MODELS; do
  for d in $DS; do
    echo "=== $(date) $m $d" >> logs/run_$m.log
    $PY "$REPO/eaglevlm/judge_bench.py" --model $m --items items/$d.jsonl --out results/${d}__${m}.jsonl --conds bin,score --bs 8 >> logs/run_$m.log 2>&1
  done
done
$PY "$REPO/eaglevlm/analyze_bench.py" results > results/table.md 2> logs/analyze.err
$PY "$REPO/eaglevlm/make_paper_tables.py" results paper_tables >> logs/analyze.err 2>&1
echo ALL_RUNS_DONE >> logs/run_all.log
