#!/bin/bash
# A100 queue 3 (2026-11-05): fills the GPUs for ~10 h after queue 2.
#  GPU0: qwen35_9b personas (MLLM-Judge)  -> qwen3vl8b tie-prompt paraphrases -> InternVL3.5-38B (with GPU1, shard 1/2)
#  GPU1: qwen25vl7b personas              -> internvl35_8b paraphrases
#  GPU2,3: wait for 32B personas -> qwen3vl32b paraphrases (2 shards) -> InternVL3.5-38B on GPU2+3 (shard 0/2)
cd /media/vipuser/Data/eaglevlm/judgebench_local
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models; O=/media/vipuser/Data/eaglevlm/out_q3; mkdir -p $O/sa $O/para $O/b38 $O/s38 $O/plans $O/logs
Q2=/media/vipuser/Data/eaglevlm/out_q2/logs/done.txt
MJ="mllm_judge mllm_judge_rest mllm_judge_rest2"; GA="genai_arena genai_arena_rest"
MAIN="$MJ $GA pickapic_u imagerewarddb imagerewarddb_rest imagerewarddb_rest2"
plan() { python3 -c "import json,sys; print(json.dumps([{'items': f'items/{d}.jsonl', 'out': f'$O/{sys.argv[1]}/{d}__{sys.argv[2]}.s{sys.argv[3]}of{sys.argv[4]}.jsonl'} for d in sys.argv[5:]]))" "$@"; }
run() {  # tag gpus model path conds shard nshard bs datasets...
  t=$1; g=$2; m=$3; mp=$4; c=$5; s=$6; ns=$7; bs=$8; shift 8
  plan $t $m $s $ns "$@" > $O/plans/${t}_${m}_$s.json
  dm=cuda:0; [[ $g == *,* ]] && dm=auto
  CUDA_VISIBLE_DEVICES=$g $P $J --model $m --model-path $mp --plan $O/plans/${t}_${m}_$s.json --conds $c --bs $bs \
     --shard $s --nshard $ns --device-map $dm --progress $O/logs/${t}_${m}_$s.progress.json > $O/logs/${t}_${m}_$s.log 2>&1
  echo "$t $m shard $s/$ns done $(date)" >> $O/logs/done.txt
}
waitfor() { until grep -q "$2" "$1" 2>/dev/null; do sleep 60; done; }
SA=tri_sa1,tri_sa2,tri_sa3,tri_sa4; PARA=tri_p1,tri_p2,tri_p3
(
  waitfor $Q2 "internvl35_8b shard 0 done"
  run sa 0 qwen35_9b $M/Qwen3.5-9B $SA 0 1 8 $MJ
  run para 0 qwen3vl8b $M/Qwen3-VL-8B-Instruct $PARA 0 1 8 $MJ $GA
  echo "gpu0 free $(date)" >> $O/logs/done.txt
) &
(
  waitfor $Q2 "internvl35_8b shard 1 done"
  run sa 1 qwen25vl7b $M/Qwen2.5-VL-7B-Instruct $SA 0 1 8 $MJ
  run para 1 internvl35_8b $M/InternVL3_5-8B-HF $PARA 0 1 8 $MJ $GA
  echo "gpu1 free $(date)" >> $O/logs/done.txt
) &
(
  waitfor $Q2 "qwen3vl32b shard 0 done"
  run para 2 qwen3vl32b $M/Qwen3-VL-32B-Instruct $PARA 0 2 8 $MJ $GA
  echo "gpu2 free $(date)" >> $O/logs/done.txt
) &
(
  waitfor $Q2 "qwen3vl32b shard 1 done"
  run para 3 qwen3vl32b $M/Qwen3-VL-32B-Instruct $PARA 1 2 8 $MJ $GA
  echo "gpu3 free $(date)" >> $O/logs/done.txt
) &
(  # InternVL3.5-38B once downloaded: two 2-GPU instances
  waitfor /media/vipuser/Data/raw/internvl38b.log "DL_OK"
  (
    waitfor $O/logs/done.txt "gpu2 free"; waitfor $O/logs/done.txt "gpu3 free"
    run b38 2,3 internvl35_38b $M/InternVL3_5-38B-HF bin,tri 0 2 8 $MAIN mtbench_human arena55k
    run s38 2,3 internvl35_38b $M/InternVL3_5-38B-HF score 0 2 16 $MAIN
  ) &
  (
    waitfor $O/logs/done.txt "gpu0 free"; waitfor $O/logs/done.txt "gpu1 free"
    run b38 0,1 internvl35_38b $M/InternVL3_5-38B-HF bin,tri 1 2 8 $MAIN mtbench_human arena55k
    run s38 0,1 internvl35_38b $M/InternVL3_5-38B-HF score 1 2 16 $MAIN
  ) &
  wait
) &
wait
