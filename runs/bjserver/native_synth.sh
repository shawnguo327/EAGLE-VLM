#!/bin/bash
# decisive test (2026-11-02 22:40): do fine-tuned judges still separate paraphrase ties from decided pairs under their NATIVE protocol?
cd /media/shawn/C/papers/VLM/judgebench_local
export EAGLEVLM_DATA=/media/shawn/C/papers/VLM/judgebench_local PYTHONWARNINGS=ignore
IT=items/synth_tie.jsonl; O=/media/shawn/C/papers/VLM/judgebench_local/runs/N_native_synth; mkdir -p $O
JM=/media/shawn/C/papers/VLM/judge_models; VP=/home/shawn/anaconda3/envs/vllm_g4/bin/python; C=/media/shawn/C/papers/EAGLE-VLM/eaglevlm
case "$1" in
 r1) PYTHONPATH=$JM/np_shim $VP $C/native_judge.py --model r1reward --model-path $JM/R1-Reward --items $IT --out $O --gpu-mem 0.45 ;;
 llava)
  PYTHONPATH=$JM/np_shim $VP $C/native_judge.py --model llavacritic7b --model-path $JM/llava-critic-7b-vllm --items $IT --out $O --gpu-mem 0.42
  PYTHONPATH=$JM/np_shim $VP $C/native_judge.py --model mmrlhf_critic --model-path $JM/MM-RLHF-Reward-7B-vllm --items $IT --out $O --gpu-mem 0.42
  /home/shawn/anaconda3/envs/lis-omni/bin/python $C/mmrlhf_score.py --hf-model $JM/MM-RLHF-Reward-7B-hf --orig $JM/MM-RLHF-Reward-7B --critiques $O/mmrlhf_critic.jsonl --items $IT --out $O/mmrlhf_score.jsonl ;;
esac
echo DONE_$1
