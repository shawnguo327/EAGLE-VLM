#!/bin/bash
# download remaining judge models from ModelScope (fast here); verify every safetensors sha256 against bjserver's copy
B=/media/vipuser/Data/miniconda3/bin; M=/media/vipuser/Data/models; L=/media/vipuser/Data/eaglevlm
export TMPDIR=/media/vipuser/Data/tmp
for spec in OpenGVLab/InternVL3_5-8B-HF:InternVL3_5-8B-HF Qwen/Qwen3.5-9B:Qwen3.5-9B Qwen/Qwen2.5-VL-7B-Instruct:Qwen2.5-VL-7B-Instruct Qwen/Qwen3-VL-32B-Instruct:Qwen3-VL-32B-Instruct; do
  repo=${spec%%:*}; d=${spec##*:}
  for t in 1 2 3; do
    $B/python -c "from modelscope import snapshot_download as s; s('$repo', local_dir='$M/$d'); print('DL_OK $repo')" && break
    echo "DL_RETRY $repo $t"
  done
done
cd $M && grep -E "safetensors$" $L/model_sha256.txt > $L/sha_expected.txt && sha256sum -c $L/sha_expected.txt > $L/sha_check.txt 2>&1
grep -c ": OK" $L/sha_check.txt; grep -v ": OK" $L/sha_check.txt | head; echo ALL_DONE
