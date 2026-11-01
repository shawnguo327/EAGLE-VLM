"""Download the raw judge benchmarks for the full (uncapped) item sets + text-domain judge sets.
usage: HF_ENDPOINT=https://hf-mirror.com python download_full.py <prep_root>   (raw/ is created under prep_root)"""
import sys, time
from huggingface_hub import snapshot_download as s
root = sys.argv[1]
J = [("MMInstruction/VL-RewardBench", ["data/*", "README.md"]),
     ("syhuggingface/multimodal_rewardbench", None),
     ("yifanzhang114/MM-RLHF-RewardBench", None),
     ("TIGER-Lab/GenAI-Arena-Bench", ["image_generation/test-*", "image_edition/test-*", "README.md"]),
     ("MJ-Bench/MJ-Bench", ["data/alignment.parquet", "data/safety.parquet", "data/bias.parquet", "README.md"]),
     ("TIGER-Lab/EditReward-Bench", ["2pair/*", "3pair/*", "4pair/*", "README.md"]),
     ("rl-research/multimodal-rewardbench-2", None),
     ("q-future/Q-Bench2-HF", ["data/dev-*", "README.md"]),
     ("ONE-Lab/MLLM-as-a-Judge", ["Benchmark/*", "Hard & HQ/*", "README.md", "image/*", "raw_data/*"]),
     ("THUDM/ImageRewardDB", ["images/test/*", "metadata-test.parquet", "metadata-validation.parquet", "README.md", "ImageRewardDB.py"]),
     ("lmsys/mt_bench_human_judgments", None),
     ("lmsys/chatbot_arena_conversations", None)]
for rid, pat in J:
    t = time.time()
    for attempt in range(3):
        try:
            s(rid, repo_type="dataset", local_dir=f"{root}/raw/" + rid.replace("/", "__"), allow_patterns=pat, max_workers=8)
            print("OK", rid, round(time.time() - t), "s", flush=True); break
        except Exception as e:
            print("FAIL", attempt, rid, repr(e)[:300], flush=True); time.sleep(10)
print("ALL_DONE", flush=True)
