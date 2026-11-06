#!/bin/bash
# Full ScienceQA_TEST Best-of-N (official VLMEvalKit InternVL2_5-8B-BoN-8: InternVL2.5-8B x 8 CoT samples + VisualPRM-8B),
# candidate dump on, alone on A100 GPU0 (the smoke test OOM'd while sharing GPU0 with a judge job).
W=/media/vipuser/Data/eaglevlm; C=/media/vipuser/Data/miniconda3
cd $W/VLMEvalKit
export CUDA_VISIBLE_DEVICES=0 BON_DUMP=$W/out_bon/scienceqa_bon8_dump.jsonl TMPDIR=/media/vipuser/Data/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "full start (gpu0) $(date)" >> $W/out_bon/full.status
$C/envs/vlmeval/bin/python run.py --config $W/bon_local.json --mode infer --work-dir $W/out_bon/vlmeval > $W/out_bon/full.log 2>&1
echo "full exit $? $(date)" >> $W/out_bon/full.status
