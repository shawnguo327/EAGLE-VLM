#!/bin/bash
# VisionArena-Battle download; HF_TOKEN comes from the environment of this process only
/media/vipuser/Data/miniconda3/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
assert os.environ.get("HF_TOKEN"), "no token"
d = snapshot_download("lmarena-ai/VisionArena-Battle", repo_type="dataset", local_dir="/media/vipuser/Data/raw/VisionArena-Battle", max_workers=8)
print("VA_OK", d)
PY
