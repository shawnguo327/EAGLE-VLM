#!/bin/bash
# Tie-channel LoRA diagnostic (2026-11-05): Qwen3-VL-8B, 2-fold cross-fitting, GPU0 = fold 0, GPU1 = fold 1.
# runs per GPU in order: lm (soft labels, ties = T) -> connector (same targets, LoRA on the merger only) -> lm_hard (no ties)
cd /media/vipuser/Data/eaglevlm/judgebench_local
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; T=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/train_pref_v2.py
M=/media/vipuser/Data/models/Qwen3-VL-8B-Instruct; O=/media/vipuser/Data/eaglevlm/out_lora; mkdir -p $O/logs
DS=mllm_judge_sp,mllm_judge,mllm_judge_rest,mllm_judge_rest2,genai_arena,genai_arena_rest
export EAGLEVLM_DATA=/media/vipuser/Data/eaglevlm/judgebench_local
for f in 0 1; do
  (
    for spec in "lm lm 0" "connector connector 0" "lm_hard lm 1"; do
      set -- $spec; tag=$1; tgt=$2; hard=$3
      CUDA_VISIBLE_DEVICES=$f $P $T --model qwen3vl8b --model-path $M --datasets $DS --out $O --fold $f --tag $tag --target $tgt \
        --hard $hard --epochs 1 --lr 1e-4 --lora-r 16 --bs 2 --grad-accum 8 --max-train 6000 --eval-bs 8 \
        --progress $O/logs/${tag}_f$f.progress.json > $O/logs/${tag}_f$f.log 2>&1
      echo "$tag fold $f done $(date)" >> $O/logs/done.txt
    done
  ) &
done
wait
