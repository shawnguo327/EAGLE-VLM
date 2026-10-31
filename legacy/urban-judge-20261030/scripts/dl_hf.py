import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from huggingface_hub import snapshot_download as s, HfApi
a=HfApi()
for q in ["YJMob100K","LYMob","humob","yjmob"]:
    try: print(q,[d.id for d in a.list_datasets(search=q,limit=10)],flush=True)
    except Exception as e: print(q,"ERR",e)
print(s("Davidham3/FourSquare_NYC_TKY", repo_type="dataset", local_dir="/media/shawn/C/papers/VLM/urban-judge-20261030/data/fsq"))
