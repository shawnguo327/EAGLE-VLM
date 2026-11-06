#!/bin/bash
cd /media/vipuser/Data/eaglevlm
P8=$(ps aux | grep judge_bench | grep "model qwen3vl8b" | grep "items/genaibench.jsonl" | grep -v grep | awk '{print $2}' | head -1)
BON=$(pgrep -f run_bon_full_g0.sh | head -1); HR=$(pgrep -f run_hires_gb32.sh | head -1); LO=$(pgrep -f run_lora_dose.sh | head -1)
echo "8B=$P8 BON=$BON HR=$HR LORA=$LO" > out_abs/logs/dispatch_gb.txt
cd judgebench_local
/media/vipuser/Data/miniconda3/envs/judge/bin/python - <<'PY' >> ../out_abs/logs/dispatch_gb.txt
import json
old = {json.loads(l)["id"] for l in open("items/genaibench_600prompts.jsonl.bak")}
new = [l for l in open("items/genaibench.jsonl") if json.loads(l)["id"] not in old]
h = len(new) // 2
open("items/genaibench_new_a.jsonl", "w").writelines(new[:h]); open("items/genaibench_new_b.jsonl", "w").writelines(new[h:]); print("new", len(new), "half", h)
PY
cd ..
O=/media/vipuser/Data/eaglevlm/out_abs; I=items/genaibench.jsonl
nohup ./run_after.sh ${P8:-none} 2 internvl35_8b $I $O/genaibench__internvl35_8b.s0of1.jsonl > /dev/null 2>&1 < /dev/null &
nohup ./run_after.sh ${BON:-none} 0 qwen35_9b $I $O/genaibench__qwen35_9b.s0of1.jsonl > /dev/null 2>&1 < /dev/null &
nohup ./run_after.sh "file:$O/logs/done.txt:genaibench__qwen35_9b.s0of1 (qwen35_9b, gpu 0) done" 0 qwen25vl7b $I $O/genaibench__qwen25vl7b.s0of1.jsonl > /dev/null 2>&1 < /dev/null &
nohup ./run_after.sh ${HR:-none} 3 qwen3vl32b items/genaibench_new_a.jsonl $O/genaibench__qwen3vl32b.s1of3.jsonl > /dev/null 2>&1 < /dev/null &
nohup ./run_after.sh ${LO:-none} 1 qwen3vl32b items/genaibench_new_b.jsonl $O/genaibench__qwen3vl32b.s2of3.jsonl > /dev/null 2>&1 < /dev/null &
sleep 1; pgrep -af run_after.sh | grep -v pgrep >> out_abs/logs/dispatch_gb.txt
