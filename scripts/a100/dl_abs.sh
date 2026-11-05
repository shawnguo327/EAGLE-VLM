#!/bin/bash
# absolute-score datasets for routing (public, official HF endpoint); judge env python (urllib3/brotli bug in base)
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; R=/media/vipuser/Data1/raw; mkdir -p $R
export TMPDIR=/media/vipuser/Data1/tmp; mkdir -p $TMPDIR
$P - <<'PY' > $R/dl_abs.log 2>&1
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
R = "/media/vipuser/Data1/raw"
api = HfApi()
for f in api.dataset_info("chaofengc/IQA-PyTorch-Datasets", files_metadata=True).siblings:
    if "AGIQA" in f.rfilename: print("AGIQA file", f.rfilename, f.size, flush=True)
d = snapshot_download("BaiqiL/GenAI-Bench", repo_type="dataset", local_dir=f"{R}/GenAI-Bench", max_workers=8); print("OK GenAI-Bench", d, flush=True)
d = hf_hub_download("chaofengc/IQA-PyTorch-Datasets", "AGIQA-3K.zip", repo_type="dataset", local_dir=f"{R}/IQA"); print("OK AGIQA", d, flush=True)
fs = sorted(f for f in api.list_repo_files("yuwd/Polaris", repo_type="dataset") if f.startswith("data/test-"))[:2]
d = snapshot_download("yuwd/Polaris", repo_type="dataset", local_dir=f"{R}/Polaris", allow_patterns=fs + ["README.md"], max_workers=4); print("OK Polaris", fs, flush=True)
PY
