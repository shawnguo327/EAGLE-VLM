#!/bin/bash
# Reproduce every analysis result and paper table from the saved judge outputs (CPU only; no model is run).
# Usage: bash reproduce.sh [workers]
set -e
cd "$(dirname "$0")"
W=${1:-32}; P=${PYTHON:-python}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for c in conventional certify_main certify_calsize router_main \
         section2_phenomena section2_decompose section2_tie_anatomy section2_edit_bothbad section2_base_vs_judge section2_lora \
         section3_mondrian section3_tie_map setup_resolution; do
  $P -m eaglevlm.pipeline.run configs/exp/$c.yaml --workers $W
done
$P -m eaglevlm.pipeline.tables configs/tables.yaml
