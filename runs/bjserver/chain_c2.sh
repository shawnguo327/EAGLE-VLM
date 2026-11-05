#!/bin/bash
# wait for caption_cands.py to write the text twins, then put the caption-judge phases first and the 32B run right after
LOG=/media/shawn/C/papers/VLM/judgebench/logs/caption_full.log
until grep -q "wrote pp_cap" $LOG; do sleep 60; done
cd /media/shawn/C/papers/EAGLE-VLM/pack
P=/home/shawn/anaconda3/envs/lis-omni/bin/python; C=/media/shawn/C/papers/VLM/judgebench_local/config.yaml
$P ctl.py --config $C stop
$P ctl.py --config $C start N1_full_vlm N1_full_llm C2_capjudge_vlm C2_capjudge_llm T2_32b_tie P4_posbias_ctrl P5_posbias_32b P2_posbias_txt Z3_ablate_rest
echo "chain_c2 restarted scheduler at $(date)"
