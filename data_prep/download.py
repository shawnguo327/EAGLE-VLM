import sys,time
from huggingface_hub import snapshot_download as s
J=[("MMInstruction/VL-RewardBench",["data/*","README.md"]),
   ("syhuggingface/multimodal_rewardbench",None),
   ("yifanzhang114/MM-RLHF-RewardBench",None),
   ("TIGER-Lab/GenAI-Arena-Bench",["image_generation/test-*","image_edition/test-*","README.md"]),
   ("MJ-Bench/MJ-Bench",["data/alignment.parquet","data/safety.parquet","data/bias.parquet","README.md"]),
   ("TIGER-Lab/EditReward-Bench",["2pair/*","README.md"]),
   ("rl-research/multimodal-rewardbench-2",["t2i/*","edit/*","README.md","LICENSE.md"]),
   ("q-future/Q-Bench2-HF",["data/dev-*","README.md"]),
   ("ONE-Lab/MLLM-as-a-Judge",["Benchmark/*","Hard & HQ/*","README.md","image/*"]),
   ("THUDM/ImageRewardDB",["images/test/*","metadata-test.parquet","metadata-validation.parquet","README.md","ImageRewardDB.py"]),
   ("Winston-Yuan/VLRMBench",["benchmark_data/*","README.md"]),
]
for rid,pat in J:
    t=time.time()
    try:
        p=s(rid,repo_type="dataset",local_dir="raw/"+rid.replace("/","__"),allow_patterns=pat,max_workers=8)
        print("OK",rid,round(time.time()-t),"s",flush=True)
    except Exception as e: print("FAIL",rid,e,flush=True)
print("ALL_DONE",flush=True)
