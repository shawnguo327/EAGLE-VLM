#!/bin/bash
# when the overnight queue is done, rerun the main table on all sets (32B included) and cap_compare
cd /media/shawn/C/papers/EAGLE-VLM/pack
P=/home/shawn/anaconda3/envs/lis-omni/bin/python; C=/media/shawn/C/papers/VLM/judgebench_local/config.yaml
until $P ctl.py --config $C status 2>&1 | grep -A2 "^\[N3_rest2_32b\]" | grep -q "done=1 running=0 pending=0"; do sleep 120; done
cd /media/shawn/C/papers/VLM/judgebench
$P /media/shawn/C/papers/EAGLE-VLM/scripts/cert_main.py --out paper_tables/cert_main_v2.md > logs/cert_main_v2.log 2>&1 &
$P /media/shawn/C/papers/EAGLE-VLM/scripts/cap_compare.py --out paper_tables/cap_compare_v2.md > logs/cap_compare_v2.log 2>&1 &
wait; echo "chain_final done $(date)"
