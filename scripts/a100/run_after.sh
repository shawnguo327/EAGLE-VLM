#!/bin/bash
# run_after.sh <wait-for: pid | "file:<path>:<text>" | none> <gpu> <model> <items file> <out file> [conds]
# waits, then judges <items> with <model> on <gpu>, resuming into <out>
W=$1; g=$2; m=$3; it=$4; out=$5; conds=${6:-bin,tri,score}
cd /media/vipuser/Data/eaglevlm/judgebench_local; export EAGLEVLM_DATA=$PWD
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
declare -A MP=([qwen3vl8b]=Qwen3-VL-8B-Instruct [internvl35_8b]=InternVL3_5-8B-HF [qwen35_9b]=Qwen3.5-9B [qwen25vl7b]=Qwen2.5-VL-7B-Instruct [qwen3vl32b]=Qwen3-VL-32B-Instruct)
case $W in
  none) ;;
  file:*) f=$(echo $W | cut -d: -f2); t=$(echo $W | cut -d: -f3-); until grep -q "$t" $f 2>/dev/null; do sleep 120; done ;;
  *) while kill -0 $W 2>/dev/null; do sleep 120; done ;;
esac
bs=8; [ $m = qwen3vl32b ] && bs=4
tag=$(basename $out .jsonl)
CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P $J --model $m --model-path /media/vipuser/Data/models/${MP[$m]} \
  --items $it --out $out --conds $conds --bs $bs --progress /media/vipuser/Data/eaglevlm/out_abs/logs/$tag.progress.json > /media/vipuser/Data/eaglevlm/out_abs/logs/$tag.log 2>&1
echo "$tag ($m, gpu $g) done $(date)" >> /media/vipuser/Data/eaglevlm/out_abs/logs/done.txt
