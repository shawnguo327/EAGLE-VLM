#!/bin/bash
# retry of env.sh: torch's CUDA deps from the Tsinghua PyPI mirror (download.pytorch.org stalled), versions = bjserver
set -e
C=/media/vipuser/Data/miniconda3; P=$C/envs/judge/bin/pip; T=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_NO_CACHE_DIR=1 TMPDIR=/media/vipuser/Data/tmp
$P install -i $T cuda-bindings==12.9.7 cuda-pathfinder==1.6.0 cuda-toolkit==12.8.1 filelock==3.32.3 fsspec==2026.7.0 \
  networkx==3.6.1 nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-cupti-cu12==12.8.90 nvidia-cuda-nvrtc-cu12==12.8.93 \
  nvidia-cuda-runtime-cu12==12.8.90 nvidia-cudnn-cu12==9.19.0.56 nvidia-cufft-cu12==11.3.3.83 nvidia-cufile-cu12==1.13.1.3 \
  nvidia-curand-cu12==10.3.9.90 nvidia-cusolver-cu12==11.7.3.90 nvidia-cusparse-cu12==12.5.8.93 nvidia-cusparselt-cu12==0.7.1 \
  nvidia-nccl-cu12==2.28.9 nvidia-nvjitlink-cu12==12.8.93 nvidia-nvshmem-cu12==3.4.5 nvidia-nvtx-cu12==12.8.90 \
  sympy==1.14.0 triton==3.6.0 typing_extensions==4.16.0 jinja2
$P install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
$P install -i $T transformers==5.16.1 accelerate==1.14.0 safetensors==0.8.0 \
   tokenizers==0.23.2 huggingface_hub==1.33.0 qwen-vl-utils==0.0.14 numpy==2.5.2 pillow==12.3.0 PyYAML==6.0.3 \
   psutil==7.2.2 tqdm==4.70.0 sentencepiece==0.2.2 scipy==1.18.1 scikit-learn==1.9.0 av==18.1.0
$C/envs/judge/bin/python -c "import torch,transformers;print('ENV_OK',torch.__version__,torch.version.cuda,transformers.__version__)"
