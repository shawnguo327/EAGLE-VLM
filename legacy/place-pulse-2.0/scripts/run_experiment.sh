#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/shawn/C/place-pulse-2.0
EXPERIMENT=${1:-$ROOT/experiments/vlm-pilot-20261030}
PER_CATEGORY=${2:-50}
PYTHON=/home/shawn/anaconda3/envs/lis-omni/bin/python

# Use a new experiment path when changing samples, prompts or runtime settings.
if [[ ! -f "$EXPERIMENT/pilot.jsonl" ]]; then
  python3 "$ROOT/scripts/prepare_pilot.py" --root "$ROOT" --out "$EXPERIMENT" --per-category "$PER_CATEGORY"
fi
for MODEL in Qwen3-VL-8B-Instruct Qwen3.5-9B; do
  "$PYTHON" -u "$ROOT/scripts/run_pilot.py" --root "$ROOT" --experiment "$EXPERIMENT" --model "$MODEL" --batch-size 8
done
python3 "$ROOT/scripts/summarize_pilot.py" --experiment "$EXPERIMENT"
