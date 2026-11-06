#!/bin/bash
# LoRA dose-response (tie-channel diagnostic): same settings as run_lora.sh but 3 epochs instead of 1, A100 GPU1,
# both folds (cross-fitting), language-model target first, then connector-only.  Question: does the released tie
# channel keep growing with more training (under-training) and does the connector-only variant catch up?
cd /media/vipuser/Data/eaglevlm/judgebench_local
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; T=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/train_pref_v2.py
M=/media/vipuser/Data/models/Qwen3-VL-8B-Instruct; O=/media/vipuser/Data/eaglevlm/out_lora; mkdir -p $O/logs
DS=mllm_judge_sp,mllm_judge,mllm_judge_rest,mllm_judge_rest2,genai_arena,genai_arena_rest
export EAGLEVLM_DATA=/media/vipuser/Data/eaglevlm/judgebench_local
for spec in "lm_e3 lm 0" "lm_e3 lm 1" "connector_e3 connector 0" "connector_e3 connector 1"; do
  set -- $spec; tag=$1; tgt=$2; f=$3
  CUDA_VISIBLE_DEVICES=1 $P $T --model qwen3vl8b --model-path $M --datasets $DS --out $O --fold $f --tag $tag --target $tgt \
    --hard 0 --epochs 3 --lr 1e-4 --lora-r 16 --bs 2 --grad-accum 8 --max-train 6000 --eval-bs 8 \
    --progress $O/logs/${tag}_f$f.progress.json > $O/logs/${tag}_f$f.log 2>&1
  echo "$tag fold $f done $(date)" >> $O/logs/done.txt
done
