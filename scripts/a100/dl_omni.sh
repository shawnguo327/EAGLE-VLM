#!/bin/bash
/media/vipuser/Data/miniconda3/bin/python - <<'PY'
from huggingface_hub import snapshot_download
d = snapshot_download("HongbangYuan/OmniRewardBench", repo_type="dataset", local_dir="/media/vipuser/Data/raw/OmniRewardBench", max_workers=8)
print("OMNI_OK", d)
PY
