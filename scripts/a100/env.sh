#!/bin/bash
# EAGLE-VLM judge env on the A100 box, pinned to bjserver's lis-omni versions (2026-11-04)
set -e
C=/media/vipuser/Data/miniconda3
until grep -q MINICONDA_OK /media/vipuser/Data/eaglevlm/install_conda.log; do sleep 5; done
source $C/etc/profile.d/conda.sh
conda config --set show_channel_urls yes
[ -d $C/envs/judge ] || conda create -y -p $C/envs/judge python=3.12 --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
P=$C/envs/judge/bin/pip
export PIP_NO_CACHE_DIR=1 TMPDIR=/media/vipuser/Data/tmp; mkdir -p $TMPDIR
$P install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
$P install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==5.16.1 accelerate==1.14.0 safetensors==0.8.0 \
   tokenizers==0.23.2 huggingface_hub==1.33.0 qwen-vl-utils==0.0.14 numpy==2.5.2 pillow==12.3.0 PyYAML==6.0.3 \
   psutil==7.2.2 tqdm==4.70.0 sentencepiece==0.2.2 scipy==1.18.1 scikit-learn==1.9.0 av==18.1.0
$C/envs/judge/bin/python -c "import torch,transformers;print('ENV_OK',torch.__version__,torch.version.cuda,transformers.__version__)"
