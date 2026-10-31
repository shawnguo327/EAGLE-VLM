while kill -0 18256
32270 2>/dev/null; do sleep 30; done
/home/shawn/anaconda3/envs/lis-omni/bin/python judge_pp.py --model qwen25vl7b --input sample.jsonl --out raw_sample_qwen25vl7b.jsonl --conds bin,tri,score
/home/shawn/anaconda3/envs/lis-omni/bin/python judge_pp.py --model qwen25vl7b --input multi.jsonl --out raw_multi_qwen25vl7b.jsonl --conds bin,tri,score
echo CHAIN3_DONE
