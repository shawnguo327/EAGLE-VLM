#!/bin/bash
# Experiment 5 control (A100 GPU2, alongside the queue's next job, after the image-removed run): wrong-image forced-choice
# and tie prompts on the MLLM-as-a-Judge score pairs.  Raw outputs go to shuf_out/ (not synced: their conditions are
# named bin / tri); the finished file is copied to out_abs/ with the conditions renamed bin_shuf / tri_shuf.
cd /media/vipuser/Data/eaglevlm; S=/media/vipuser/Data/eaglevlm/shuf_out; mkdir -p $S
while kill -0 527105 2>/dev/null; do sleep 60; done
for m in qwen3vl8b qwen35_9b; do
  ./run_after.sh none 2 $m items/mllm_judge_sp_shufimg.jsonl $S/mllm_judge_sp__$m.shufimg_raw.jsonl bin,tri
  python3 - "$S/mllm_judge_sp__$m.shufimg_raw.jsonl" "out_abs/mllm_judge_sp__$m.sshufimg.jsonl" <<'PY'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]; tmp = dst + ".tmp"; n = 0
with open(tmp, "w") as f:
    for l in open(src):
        try:
            x = json.loads(l)
        except ValueError:
            continue
        x["cond"] = {"bin": "bin_shuf", "tri": "tri_shuf"}[x["cond"]]; f.write(json.dumps(x) + "\n"); n += 1
os.replace(tmp, dst); print(dst, n, "records")
PY
done
echo "shufimg done $(date)"
