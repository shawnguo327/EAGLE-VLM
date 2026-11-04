#!/bin/bash
# after the current queue drains, run the natural-rate mainstream extensions
cd /media/shawn/C/papers/EAGLE-VLM/pack
P=/home/shawn/anaconda3/envs/lis-omni/bin/python; C=/media/shawn/C/papers/VLM/judgebench_local/config.yaml
until $P ctl.py --config $C status 2>&1 | grep -A3 "^\[Z3_ablate_rest\]" | grep -q "running=0 pending=0"; do sleep 30; done
$P ctl.py --config $C stop
$P ctl.py --config $C start N2_ext_vlm N2_ext_32b
echo "chain_n2 started at $(date)"
