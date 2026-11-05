from huggingface_hub import HfApi, snapshot_download
api = HfApi(); fs = sorted(f for f in api.list_repo_files("yuwd/Polaris", repo_type="dataset") if f.startswith("data/test-"))
print(len(fs), "test shards")
snapshot_download("yuwd/Polaris", repo_type="dataset", local_dir="/media/vipuser/Data1/raw/Polaris", allow_patterns=fs, max_workers=4)
print("POLARIS_ALL_OK", flush=True)
