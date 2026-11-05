#!/bin/bash
# Full ScienceQA_TEST Best-of-N (official VLMEvalKit InternVL2_5-8B-BoN-8 = InternVL2.5-8B x 8 CoT samples + VisualPRM-8B),
# candidate dump on. Starts when GPU2 (GenAI-Bench qwen25vl7b, pid 314549) and GPU3 (AGIQA/Polaris queue, pid 317827) are
# free; data-parallel over the two GPUs with torchrun (VLMEvalKit splits the questions by rank). Reuses the smoke-test
# work dir, so finished questions are not redone.
W=/media/vipuser/Data/eaglevlm; C=/media/vipuser/Data/miniconda3
while kill -0 314549 2>/dev/null || kill -0 317827 2>/dev/null; do sleep 120; done
while pgrep -f "run_bon_smoke2.sh" >/dev/null; do sleep 60; done
cd $W/VLMEvalKit
export CUDA_VISIBLE_DEVICES=2,3 BON_DUMP=$W/out_bon/scienceqa_bon8_dump.jsonl TMPDIR=/media/vipuser/Data/tmp
echo "full start $(date)" >> $W/out_bon/full.status
$C/envs/vlmeval/bin/torchrun --nproc-per-node=2 run.py --config $W/bon_local.json --mode infer --work-dir $W/out_bon/vlmeval > $W/out_bon/full.log 2>&1
echo "full exit $? $(date)" >> $W/out_bon/full.status
