PY=/home/shawn/anaconda3/envs/lis-omni/bin/python
while kill -0 15970 2>/dev/null; do sleep 20; done
$PY judge_pp.py --input multi.jsonl --out raw_multi_qwen3vl8b.jsonl
$PY judge_pp.py --input rank_pairs.jsonl --out raw_rankpairs_qwen3vl8b.jsonl --conds bin
$PY judge_pp.py --input rank_images.jsonl --out raw_rankimages_qwen3vl8b.jsonl --conds score
$PY judge_pp.py --model qwen35_9b --input sample.jsonl --out raw_sample_qwen35_9b.jsonl --conds bin,tri,low,score
$PY judge_pp.py --model qwen35_9b --input multi.jsonl --out raw_multi_qwen35_9b.jsonl --conds bin,tri,low,score
echo CHAIN2_DONE
