#!/bin/bash
# VLMEvalKit (open-compass, official; vetted 2026-11-05) in its own env; litellm pinned <= 1.82.6 (1.82.7/1.82.8 were
# malicious uploads); the judge env is not touched.
set -e
C=/media/vipuser/Data/miniconda3; W=/media/vipuser/Data/eaglevlm; cd $W
[ -d VLMEvalKit ] || git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit && git rev-parse HEAD > ../VLMEvalKit.commit && echo "VLMEvalKit $(cat ../VLMEvalKit.commit)" && cd ..
source $C/etc/profile.d/conda.sh
[ -d $C/envs/vlmeval ] || conda create -y -p $C/envs/vlmeval python=3.11 --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
P=$C/envs/vlmeval/bin/pip; T=https://pypi.tuna.tsinghua.edu.cn/simple; export PIP_NO_CACHE_DIR=1 TMPDIR=/media/vipuser/Data/tmp
# same CUDA wheels as the judge env (tuna for the nvidia deps, pytorch index for torch itself)
$P install -i $T nvidia-cudnn-cu12==9.19.0.56 nvidia-cublas-cu12==12.8.4.1 nvidia-cusparselt-cu12==0.7.1 nvidia-nccl-cu12==2.28.9 \
   nvidia-cuda-runtime-cu12==12.8.90 nvidia-cuda-nvrtc-cu12==12.8.93 nvidia-cufft-cu12==11.3.3.83 nvidia-curand-cu12==10.3.9.90 \
   nvidia-cusolver-cu12==11.7.3.90 nvidia-cusparse-cu12==12.5.8.93 nvidia-nvjitlink-cu12==12.8.93 nvidia-nvtx-cu12==12.8.90 \
   nvidia-cuda-cupti-cu12==12.8.90 nvidia-cufile-cu12==1.13.1.3 nvidia-nvshmem-cu12==3.4.5 triton==3.6.0
$P install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
echo "litellm<=1.82.6" > $W/vlmeval_constraints.txt
$P install -i $T -c $W/vlmeval_constraints.txt -e VLMEvalKit
$P install -i $T -c $W/vlmeval_constraints.txt "transformers==5.16.1" qwen-vl-utils==0.0.14 accelerate
$P list 2>/dev/null | grep -i -E "^(litellm|transformers|torch |vlmeval)" 
$C/envs/vlmeval/bin/python -c "import vlmeval, torch, transformers; print('VLMEVAL_OK', torch.__version__, transformers.__version__)"
