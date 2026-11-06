#!/bin/bash
# Judges on Best-of-N candidate pairs (ScienceQA_TEST, VLMEvalKit InternVL2_5-8B-BoN-8 candidates; items from
# make_bon_items.py): pairwise bin/tri and pointwise scores on all 16,136 cyclic pairs (every candidate is scored in both of
# its pairs; the score prompt depends only on the question and the candidate, so the generic pipeline needs no special case).
# A100 GPU given as $1, starts after the process $2 exits (none = now); models as the remaining args.
cd /media/vipuser/Data/eaglevlm/judgebench_local; export EAGLEVLM_DATA=$PWD
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models; O=/media/vipuser/Data/eaglevlm/out_bonjudge; mkdir -p $O/logs
declare -A MP=([qwen3vl8b]=Qwen3-VL-8B-Instruct [internvl35_8b]=InternVL3_5-8B-HF [qwen35_9b]=Qwen3.5-9B [qwen25vl7b]=Qwen2.5-VL-7B-Instruct [qwen3vl32b]=Qwen3-VL-32B-Instruct)
g=$1; w=$2; shift 2
[ "$w" != none ] && while kill -0 $w 2>/dev/null; do sleep 60; done
for m in "$@"; do
  bs=8; [ $m = qwen3vl32b ] && bs=4
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P $J --model $m --model-path $M/${MP[$m]} --items items/bon_scienceqa.jsonl \
    --out $O/bon_scienceqa__$m.s0of1.jsonl --conds bin,tri,score --bs $bs --progress $O/logs/bon_scienceqa_$m.progress.json > $O/logs/bon_scienceqa_$m.log 2>&1
  echo "bon_scienceqa $m (gpu $g) exit $? $(date)" >> $O/logs/done.txt
done
