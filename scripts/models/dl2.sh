export HF_ENDPOINT=https://hf-mirror.com
B=/home/shawn/anaconda3/envs/lis-omni/bin
$B/hf download llava-hf/llava-onevision-qwen2-7b-ov-hf --local-dir llava-ov-7b-hf
$B/hf download BAAI/bge-large-en-v1.5 --local-dir bge-large-en-v1.5
echo DL2_DONE
