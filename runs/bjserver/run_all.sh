#!/usr/bin/env bash
# Run the EAGLE-protocol judge on every converted benchmark.
# Pass 1: Qwen3-VL-8B, pairwise + pointwise.  Pass 2: Qwen3.5-9B and Qwen2.5-VL-7B, pairwise only.
cd "$(dirname "$0")"
PY=/home/shawn/anaconda3/envs/lis-omni/bin/python
mkdir -p results logs
DS="vlrb mmrb mmrlhf mllm_judge genai_arena mjbench editreward mmrb2 qbench2 imagerewarddb"
for d in $DS; do
  $PY judge_bench.py --model qwen3vl8b --items items/$d.jsonl --out results/${d}__qwen3vl8b.jsonl --conds bin,score --bs 8 >> logs/run_qwen3vl8b.log 2>&1
done
for m in qwen35_9b qwen25vl7b; do
  for d in $DS; do
    $PY judge_bench.py --model $m --items items/$d.jsonl --out results/${d}__${m}.jsonl --conds bin --bs 8 >> logs/run_$m.log 2>&1
  done
done
$PY analyze_bench.py results > results/table.md 2> logs/analyze.err

# paper tables (diagnostic + selective/ladder) for eaglev-paper/tables/
$PY make_paper_tables.py results paper_tables >> logs/analyze.err 2>&1
echo ALL_RUNS_DONE >> logs/run_qwen3vl8b.log
