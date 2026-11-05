#!/bin/bash
# Resume the paused InternVL3.5-38B shard 1 on GPU0+1 after the LoRA diagnostic (judge_bench skips finished rows).
cd /media/vipuser/Data/eaglevlm/judgebench_local
P=/media/vipuser/Data/miniconda3/envs/judge/bin/python; J=/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm/judge_bench.py
M=/media/vipuser/Data/models/InternVL3_5-38B-HF; O=/media/vipuser/Data/eaglevlm/out_q3
L=/media/vipuser/Data/eaglevlm/out_lora/logs/done.txt
until grep -q "lm_hard fold 0 done" $L 2>/dev/null && grep -q "lm_hard fold 1 done" $L 2>/dev/null; do sleep 120; done
CUDA_VISIBLE_DEVICES=0,1 $P $J --model internvl35_38b --model-path $M --plan $O/plans/b38_internvl35_38b_1.json --conds bin,tri --bs 8 \
  --shard 1 --nshard 2 --device-map auto --progress $O/logs/b38_internvl35_38b_1.progress.json >> $O/logs/b38_internvl35_38b_1.log 2>&1
echo "b38 internvl35_38b shard 1/2 done $(date)" >> $O/logs/done.txt
python3 -c "import json; d=json.load(open('$O/plans/b38_internvl35_38b_1.json')); print(json.dumps([{'items': e['items'], 'out': e['out'].replace('/b38/', '/s38/')} for e in d if not any(x in e['items'] for x in ('mtbench_human', 'arena55k'))]))" > $O/plans/s38_internvl35_38b_1.json
CUDA_VISIBLE_DEVICES=0,1 $P $J --model internvl35_38b --model-path $M --plan $O/plans/s38_internvl35_38b_1.json --conds score --bs 16 \
  --shard 1 --nshard 2 --device-map auto --progress $O/logs/s38_internvl35_38b_1.progress.json >> $O/logs/s38_internvl35_38b_1.log 2>&1
echo "s38 internvl35_38b shard 1/2 done $(date)" >> $O/logs/done.txt
