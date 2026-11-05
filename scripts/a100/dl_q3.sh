#!/bin/bash
# downloads for queue 3; the HF token is read from stdin into this process's environment only (never written to disk)
read -r HF_TOKEN; export HF_TOKEN
B=/media/vipuser/Data/miniconda3/bin; export TMPDIR=/media/vipuser/Data/tmp
R=/media/vipuser/Data/raw; mkdir -p $R
( $B/python - <<'PY'
from huggingface_hub import snapshot_download
d = snapshot_download("lmarena-ai/VisionArena-Battle", repo_type="dataset", local_dir="/media/vipuser/Data/raw/VisionArena-Battle",
                      max_workers=8)
print("VA_OK", d)
PY
) > $R/visionarena.log 2>&1 &
( $B/python -c "from modelscope import snapshot_download as s; s('OpenGVLab/InternVL3_5-38B-HF', local_dir='/media/vipuser/Data/models/InternVL3_5-38B-HF'); print('DL_OK InternVL3_5-38B-HF')" ) > $R/internvl38b.log 2>&1 &
wait
