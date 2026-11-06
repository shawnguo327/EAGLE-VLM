"""Judge items from a VLMEvalKit Best-of-N candidate file (output of bon_candidates.py).

For every question the 8 candidates (sampling order) are paired cyclically, (k, k+1 mod 8): every candidate appears in
exactly two pairs, and the pair types (both correct / both wrong / one correct) keep their natural rates.  The label comes
from VLMEvalKit's correctness of each candidate (boxed answer, exact matching): the correct one is better; both correct
or both wrong -> tie.  The criterion is the one already used for MM-RLHF reasoning items.  The question is the prompt the
policy saw (VLMEvalKit's CoT prompt) with the <image> placeholders removed; its images are shown as context.
Writes  <data>/items/<name>.jsonl   all cyclic pairs
        <data>/images/<name>/       copies of the question images
usage: python make_bon_items.py --cands .../boxed/candidates.jsonl.gz --name bon_scienceqa --data <judgebench_local>"""
import argparse, gzip, json, os, re, shutil
ap = argparse.ArgumentParser()
ap.add_argument("--cands", required=True); ap.add_argument("--name", required=True); ap.add_argument("--data", required=True)
a = ap.parse_args()
CRIT = "Prefer the response whose reasoning and final answer are correct for the image."
os.makedirs(f"{a.data}/images/{a.name}", exist_ok=True)
rows = [json.loads(l) for l in gzip.open(a.cands, "rt")]
P, types = [], {}
for r in rows:
    ctx = []
    for p in r["images"]:
        dst = f"images/{a.name}/{os.path.basename(p)}"
        if not os.path.exists(f"{a.data}/{dst}"):
            shutil.copyfile(p, f"{a.data}/{dst}")
        ctx.append(dst)
    q = re.sub(r"(Image-\d+:\s*)?<image>\s*", "", r["prompt"]).strip()
    n = len(r["candidates"])
    for k in range(n):
        k2 = (k + 1) % n; h = (r["hit"][k], r["hit"][k2])
        lab = "tie" if h[0] == h[1] else (0 if h[0] else 1)
        pt = "decided" if lab != "tie" else ("both_correct" if h[0] else "both_wrong")
        x = dict(id=f"{a.name}/{r['index']}_{k}{k2}", dataset=a.name, subset=r.get("category") or a.name, kind="resp",
                 question=q, context=ctx, cand=[r["candidates"][k], r["candidates"][k2]], criterion=CRIT, label=lab,
                 meta=dict(question_index=r["index"], slots=[k, k2], hit=list(h), option=[r["option"][k], r["option"][k2]],
                           prm=[r["prm"][k], r["prm"][k2]], prm_slot=r["prm_slot"], answer=r["answer"], pair_type=pt))
        P.append(x); types[pt] = types.get(pt, 0) + 1
with open(f"{a.data}/items/{a.name}.jsonl", "w") as f:
    for x in P:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"{a.name}: {len(rows)} questions, {len(P)} pairs {types}, tie share {1 - types.get('decided', 0) / len(P):.3f}")
