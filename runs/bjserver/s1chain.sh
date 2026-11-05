#!/bin/bash
cd /media/shawn/C/papers/VLM/judgebench_local
PYTHONWARNINGS=ignore PYTHONPATH=/media/shawn/C/papers/VLM/judge_models/np_shim /home/shawn/anaconda3/envs/vllm_g4/bin/python /media/shawn/C/papers/EAGLE-VLM/scripts/make_synth_ties.py --llm /home/shawn/code/Jev-Test/models/Qwen3-8B --out items/synth_tie.jsonl --n 400 --gpu-mem 0.5
/home/shawn/anaconda3/envs/lis-omni/bin/python /media/shawn/C/papers/EAGLE-VLM/pack/ctl.py --config config.yaml start S1_synth T1_arena_tt J4_llavaov S1_synth_ov L4_scale_3b
echo CHAIN_STARTED
