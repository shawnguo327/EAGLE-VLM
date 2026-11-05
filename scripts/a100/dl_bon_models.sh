#!/bin/bash
# official OpenGVLab weights from ModelScope (same repos as HF); record sha256 of every file incl. remote-code .py
M=/media/vipuser/Data1/models; mkdir -p $M; export TMPDIR=/media/vipuser/Data1/tmp
set -e
P=/media/vipuser/Data/miniconda3/bin/python
$P -c "from modelscope import snapshot_download as s; s('OpenGVLab/InternVL2_5-8B', local_dir='$M/InternVL2_5-8B'); print('DL_OK InternVL2_5-8B')"
$P -c "from modelscope import snapshot_download as s; s('OpenGVLab/VisualPRM-8B', local_dir='$M/VisualPRM-8B'); print('DL_OK VisualPRM-8B')"
for d in InternVL2_5-8B VisualPRM-8B; do (cd $M/$d && sha256sum $(ls | grep -v -E "^\.") > ../$d.sha256); done
echo ALL_DL_OK
