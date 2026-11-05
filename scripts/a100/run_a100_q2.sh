#!/bin/bash
# A100 queue 2 (2026-11-04): simulated-annotator personas (Trust-or-Escalate baseline) + blind judging, full MLLM-Judge / GenAI.
# GPU 0,1: qwen3vl8b then internvl35_8b (2 shards each); GPU 2,3: qwen3vl32b (2 shards).  judge_bench resumes.
cd /media/vipuser/Data/eaglevlm/judgebench_local
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models; O=/media/vipuser/Data/eaglevlm/out_q2; mkdir -p $O/sa $O/blind $O/plans $O/logs
SETS_SA="mllm_judge mllm_judge_rest mllm_judge_rest2 genai_arena genai_arena_rest"
SETS_BL="mllm_judge mllm_judge_rest mllm_judge_rest2"
plan() { python3 -c "import json,sys; print(json.dumps([{'items': f'items/{d}.jsonl', 'out': f'$O/{sys.argv[1]}/{d}__{sys.argv[2]}.s{sys.argv[3]}of2.jsonl'} for d in sys.argv[4:]]))" "$@"; }
run() {  # gpu model path shard
  g=$1; m=$2; mp=$3; s=$4
  plan sa $m $s $SETS_SA > $O/plans/sa_${m}_$s.json; plan blind $m $s $SETS_BL > $O/plans/blind_${m}_$s.json
  CUDA_VISIBLE_DEVICES=$g $P $J --model $m --model-path $mp --plan $O/plans/sa_${m}_$s.json --conds tri_sa1,tri_sa2,tri_sa3,tri_sa4 \
     --bs 8 --shard $s --nshard 2 --progress $O/logs/sa_${m}_$s.progress.json > $O/logs/sa_${m}_$s.log 2>&1
  CUDA_VISIBLE_DEVICES=$g $P $J --model $m --model-path $mp --plan $O/plans/blind_${m}_$s.json --conds bin_blind,tri_blind \
     --bs 8 --shard $s --nshard 2 --progress $O/logs/blind_${m}_$s.progress.json > $O/logs/blind_${m}_$s.log 2>&1
  echo "$m shard $s done $(date)" >> $O/logs/done.txt
}
( run 0 qwen3vl8b $M/Qwen3-VL-8B-Instruct 0; run 0 internvl35_8b $M/InternVL3_5-8B-HF 0 ) &
( run 1 qwen3vl8b $M/Qwen3-VL-8B-Instruct 1; run 1 internvl35_8b $M/InternVL3_5-8B-HF 1 ) &
( run 2 qwen3vl32b $M/Qwen3-VL-32B-Instruct 0 ) &
( run 3 qwen3vl32b $M/Qwen3-VL-32B-Instruct 1 ) &
wait
