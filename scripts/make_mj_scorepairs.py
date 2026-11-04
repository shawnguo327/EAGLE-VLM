"""MLLM-as-a-Judge SCORE split -> pairwise items with human absolute scores (the setting of Kumar et al. 2026, made
pairwise).  Every question was answered by 4-5 models, each answer scored 1-5 by a human annotator.  All answer pairs
to the same question: label = higher human score better, equal score = tie.  meta.scores keeps both human scores, so
pair types (both good / both bad / equal-mid / decided by gap) come from the human absolute reference.
A/B orientation by a hash of the id.  Writes items/mllm_judge_sp.jsonl and copies the images.
usage: python make_mj_scorepairs.py"""
import itertools, json, os, shutil, zlib, collections
RAW = "/media/shawn/C/papers/VLM/judgebench_full/raw/ONE-Lab__MLLM-as-a-Judge"
LOC = "/media/shawn/C/papers/VLM/judgebench_local"
CRIT = json.loads(open(f"{LOC}/items/mllm_judge.jsonl").readline())["criterion"]
S = [json.loads(l) for l in open(f"{RAW}/Benchmark/score.jsonl")]
by = collections.defaultdict(list)
for s in S:
    try:
        h = int(float(s["human"]))
    except Exception:
        continue
    by[s["image_path"]].append((s, h))
os.makedirs(f"{LOC}/images/mllm_judge_sp", exist_ok=True)
out = []
for img, L in by.items():
    src = f"{RAW}/image/{img}"
    if not os.path.exists(src):
        continue
    dst = f"images/mllm_judge_sp/{img}"
    if not os.path.exists(f"{LOC}/{dst}"):
        shutil.copy(src, f"{LOC}/{dst}")
    for (s0, h0), (s1, h1) in itertools.combinations(L, 2):
        if s0["answer"].strip() == s1["answer"].strip():
            continue
        iid = f"mllm_judge_sp/{img.split('.')[0]}_{s0['score_id']}_{s1['score_id']}"
        if zlib.crc32(iid.encode()) % 2:
            (s0, h0), (s1, h1) = (s1, h1), (s0, h0)
        lab = "tie" if h0 == h1 else ("0" if h0 > h1 else "1")
        out.append(dict(id=iid, dataset="mllm_judge_sp", subset=s0["original_dataset"], kind="resp",
                        question=s0["instruction"], context=[dst], cand=[s0["answer"], s1["answer"]], criterion=CRIT,
                        label=lab, meta=dict(scores=[h0, h1], models=[s0["name"], s1["name"]], score_ids=[s0["score_id"], s1["score_id"]])))
with open(f"{LOC}/items/mllm_judge_sp.jsonl", "w") as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
c = collections.Counter(x["label"] for x in out)
sc = [x["meta"]["scores"] for x in out]
print(len(out), "pairs from", len(by), "questions;", dict(c), "tie share", round(c["tie"] / len(out), 3))
print("both >= 4:", sum(min(s) >= 4 for s in sc), " both <= 2:", sum(max(s) <= 2 for s in sc),
      " ties both>=4:", sum(min(s) >= 4 and s[0] == s[1] for s in sc), " ties both<=2:", sum(max(s) <= 2 and s[0] == s[1] for s in sc))
