"""Review improvement plan, experiment 5 (visual dependence): wrong-image control for the MLLM-as-a-Judge score pairs.
Every pair keeps its id, question and answers, but its image is replaced by the image of ANOTHER source of the same
subset (a derangement over the subset's unique images, fixed seed), so no pair keeps its own image and the substitute is
plausible for the task type.  Usage: make_shufimg_items.py <items in> <items out>"""
import collections, json, random, sys
src, dst = sys.argv[1], sys.argv[2]
L = [json.loads(l) for l in open(src)]
imgs = collections.defaultdict(list)
for x in L:
    k = tuple(x["context"])
    if k not in imgs[x["subset"]]:
        imgs[x["subset"]].append(k)
rng = random.Random(20261106); mp = {}
for sub, ks in imgs.items():
    if len(ks) < 2:
        raise SystemExit(f"subset {sub} has a single image; cannot derange")
    order = list(range(len(ks))); rng.shuffle(order)
    for a, b in zip(order, order[1:] + order[:1]):
        mp[(sub, ks[a])] = ks[b]
n_same = 0
with open(dst, "w") as f:
    for x in L:
        y = dict(x); old = tuple(x["context"]); new = mp[(x["subset"], old)]
        n_same += new == old
        y["context"] = list(new); y["meta"] = dict(x.get("meta", {}), shufimg_from=list(old))
        f.write(json.dumps(y, ensure_ascii=False) + "\n")
print(f"{len(L)} pairs, {sum(len(v) for v in imgs.values())} images in {len(imgs)} subsets; pairs keeping their own image: {n_same}")
