#!/usr/bin/env bash
# Create the conda env used on the server (versions match the bjserver run). Then set `python:` in config.yaml.
set -e
ENV=${1:-eaglev}
conda create -y -n $ENV python=3.12
PY=$(conda run -n $ENV which python)
$PY -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
$PY -m pip install -r "$(dirname "$0")/requirements.txt"
$PY -c "import torch,transformers,peft;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count(),'transformers',transformers.__version__,'peft',peft.__version__)"
echo "set in config.yaml ->  python: $PY"
