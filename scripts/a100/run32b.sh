#!/bin/bash
# Qwen3-VL-32B on 4 x A100: shard i on GPU i.  Pass 1 = pairwise bin+tri, pass 2 = pointwise score.
# Outputs: out32b/{pair,score}/<ds>__qwen3vl32b.s<i>of4.jsonl  (judge_bench resumes from existing rows)
cd /media/vipuser/Data/eaglevlm/judgebench_local
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models/Qwen3-VL-32B-Instruct; O=/media/vipuser/Data/eaglevlm/out32b; mkdir -p $O/pair $O/score $O/plans $O/logs
PAIR="pickapic_u imagerewarddb imagerewarddb_rest imagerewarddb_rest2 mllm_judge_rest mllm_judge_rest2 pickapic"
SCORE="pickapic_u genai_arena genai_arena_rest imagerewarddb imagerewarddb_rest imagerewarddb_rest2 mllm_judge mllm_judge_rest mllm_judge_rest2 pickapic"
plan() { python3 -c "import json,sys; print(json.dumps([{'items': f'items/{d}.jsonl', 'out': f'$O/{sys.argv[1]}/{d}__qwen3vl32b.s{sys.argv[2]}of4.jsonl'} for d in sys.argv[3:]]))" "$@"; }
for i in 0 1 2 3; do
  plan pair $i $PAIR > $O/plans/pair_$i.json; plan score $i $SCORE > $O/plans/score_$i.json
  ( CUDA_VISIBLE_DEVICES=$i $P $J --model qwen3vl32b --model-path $M --plan $O/plans/pair_$i.json --conds bin,tri \
        --bs 8 --shard $i --nshard 4 --progress $O/logs/pair_$i.progress.json > $O/logs/pair_$i.log 2>&1
    CUDA_VISIBLE_DEVICES=$i $P $J --model qwen3vl32b --model-path $M --plan $O/plans/score_$i.json --conds score \
        --bs 16 --shard $i --nshard 4 --progress $O/logs/score_$i.progress.json > $O/logs/score_$i.log 2>&1
    echo "shard $i done $(date)" >> $O/logs/done.txt ) &
done
wait
