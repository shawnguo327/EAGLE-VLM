#!/bin/bash
# Best-of-N smoke test on A100 GPU0 (next to the 8B full-res job) with the official VLMEvalKit protocol (InternVL2_5-8B-BoN-8 = InternVL2.5-8B policy,
# 8 CoT samples, VisualPRM-8B selection), local weights, candidate dump enabled. Waits for the GPU3 GenAI-Bench job.
W=/media/vipuser/Data/eaglevlm; C=/media/vipuser/Data/miniconda3
while kill -0 300871 2>/dev/null; do sleep 60; done   # GPU0 frees ~48 GB when InternVL3.5-38B finishes
cd $W/VLMEvalKit
export CUDA_VISIBLE_DEVICES=0 BON_DUMP=$W/out_bon/scienceqa_bon8_dump.jsonl HF_HUB_OFFLINE=0 TMPDIR=/media/vipuser/Data/tmp
mkdir -p $W/out_bon
echo "start $(date)" > $W/out_bon/smoke.status
timeout 2400 $C/envs/vlmeval/bin/python run.py --config $W/bon_local.json --mode infer --work-dir $W/out_bon/vlmeval > $W/out_bon/smoke.log 2>&1
echo "exit $? $(date)" >> $W/out_bon/smoke.status
