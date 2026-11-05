#!/bin/bash
# download the 5 judge models onto the Data disk, pinned to the revisions used on bjserver (2026-11-04)
B=/media/vipuser/Data/miniconda3/bin; M=/media/vipuser/Data/models; mkdir -p $M
export HF_HOME=/media/vipuser/Data/hf TMPDIR=/media/vipuser/Data/tmp PIP_NO_CACHE_DIR=1; mkdir -p $TMPDIR
$B/pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple "huggingface_hub>=0.34" modelscope && echo TOOLS_OK
dl_hf() {   # repo revision dir
  for ep in https://hf-mirror.com https://huggingface.co; do
    HF_ENDPOINT=$ep $B/python - "$@" <<'PY' && return 0
import sys
from huggingface_hub import snapshot_download
repo, rev, d = sys.argv[1:4]
snapshot_download(repo, revision=rev, local_dir=d, max_workers=4)
print("DL_OK", repo, rev)
PY
    echo "DL_FAIL $1 via $ep"
  done; return 1
}
dl_ms() {   # repo dir   (bjserver copies came from ModelScope; sha256 checked against their manifest afterwards)
  $B/python - "$@" <<'PY'
import sys
from modelscope import snapshot_download
repo, d = sys.argv[1:3]
snapshot_download(repo, local_dir=d)
print("DL_OK", repo)
PY
}
dl_ms Qwen/Qwen3-VL-8B-Instruct $M/Qwen3-VL-8B-Instruct || echo "DL_FAIL Qwen3-VL-8B"
dl_hf OpenGVLab/InternVL3_5-8B-HF 741a7d03020411e666c6109218ab71e08151ef86 $M/InternVL3_5-8B-HF
dl_ms Qwen/Qwen3.5-9B $M/Qwen3.5-9B || echo "DL_FAIL Qwen3.5-9B"
dl_hf Qwen/Qwen2.5-VL-7B-Instruct cc594898137f460bfe9f0759e9844b3ce807cfb5 $M/Qwen2.5-VL-7B-Instruct
dl_hf Qwen/Qwen3-VL-32B-Instruct 0cfaf48183f594c314753d30a4c4974bc75f3ccb $M/Qwen3-VL-32B-Instruct
echo ALL_DONE; du -sh $M/*
