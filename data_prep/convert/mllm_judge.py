"""MLLM-as-a-Judge (ONE-Lab/MLLM-as-a-Judge), Benchmark/pair.jsonl -> items/mllm_judge.jsonl
human_answer / human: A, B, or C (= tie, explicit human label -> "tie"). Subset = original_dataset.
Hard & HQ/ pair files are not converted: their (id, pair_id) keys do not line up with Benchmark/pair.jsonl
(0/116 Hard and 19/133 HQ keys match), so membership cannot be flagged reliably."""
import sys, pathlib, json, ast; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import *

DS = "mllm_judge"
SRC = RAW / "ONE-Lab__MLLM-as-a-Judge"
CRIT = "Prefer the response that better answers the instruction about the image: accurate, helpful, and free of hallucination."


def ans(x):
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return None, x
    return x.get("name"), x.get("answer")


def keys(path):
    out = set()
    for l in open(path):
        r = json.loads(l)
        out.add((str(r["id"]), str(r["pair_id"])))
    return out


recs, full, skipped = [], collections.Counter(), collections.Counter()
for ix, l in enumerate(open(SRC / "Benchmark/pair.jsonl")):
    r = json.loads(l)
    h = (r.get("human_answer") or r.get("human") or "").strip().upper()
    full[r["original_dataset"]] += 1
    img = SRC / "image" / r["image_path"]
    (n1, a1), (n2, a2) = ans(r["answer1"]), ans(r["answer2"])
    if h not in ("A", "B", "C") or not img.exists() or not a1 or not a2:
        skipped["bad label" if h not in ("A", "B", "C") else "missing image" if not img.exists() else "empty answer"] += 1
        continue
    c = [(a1, n1), (a2, n2)]
    w = 1 if h == "B" else 0
    q = r["instruction"]
    recs.append(dict(id=f"{DS}/{ix:05d}_{r['id']}_{r['pair_id']}", dataset=DS, subset=r["original_dataset"], kind="resp",
                     question=q, criterion=CRIT, win=c[w], lose=c[1 - w], tie=h == "C", _img=img,
                     meta=dict(orig_id=r["id"], pair_id=r["pair_id"], row=ix, human=h, true_answers=r.get("true_answers"))))
full = dict(full) | {"total": sum(full.values()), "skipped": dict(skipped)}
recs = orient(stratified_cap(recs))


def mat(r):
    r["context"] = [save_img(r["_img"], DS, r["_img"].stem)]
    r["meta"]["models"] = [c[1] for c in r["cand"]]
    r["cand"] = [c[0] for c in r["cand"]]
    return r


write(DS, recs, mat, full, "Explicit human ties ('C') kept as 'tie'. Hard & HQ/ files not converted (keys don't align with Benchmark/).")
