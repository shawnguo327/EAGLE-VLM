export HF_ENDPOINT=https://hf-mirror.com
B=/home/shawn/anaconda3/envs/lis-omni/bin
$B/hf download yifanzhang114/R1-Reward --local-dir R1-Reward
$B/hf download lmms-lab/llava-critic-7b --local-dir llava-critic-7b
$B/hf download llava-hf/llava-onevision-qwen2-7b-ov-hf --local-dir llava-ov-7b-hf-template --exclude "*.safetensors"
$B/hf download yifanzhang114/MM-RLHF-Reward-7B-llava-ov-qwen --local-dir MM-RLHF-Reward-7B
echo DL_DONE
