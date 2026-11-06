#!/bin/bash
# MMMU_DEV_VAL Best-of-N (same official VLMEvalKit InternVL2_5-8B-BoN-8 protocol), candidate dump on, A100 GPU2.
W=/media/vipuser/Data/eaglevlm; C=/media/vipuser/Data/miniconda3
cd $W/VLMEvalKit
export CUDA_VISIBLE_DEVICES=2 BON_DUMP=$W/out_bon/mmmu_bon8_dump.jsonl TMPDIR=/media/vipuser/Data/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "mmmu start (gpu2) $(date)" >> $W/out_bon/mmmu.status
$C/envs/vlmeval/bin/python run.py --config $W/bon_local_mmmu.json --mode infer --work-dir $W/out_bon/vlmeval > $W/out_bon/mmmu.log 2>&1
echo "mmmu exit $? $(date)" >> $W/out_bon/mmmu.status
