import json, hashlib, zlib, collections, sys
sys.path.insert(0, "/media/shawn/C/papers/EAGLE-VLM")
I = [json.loads(l) for l in open("items/vlrb.jsonl")]
print(I[0].keys()); print({k: (str(v)[:150]) for k, v in I[0].items()})
h = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
H = collections.Counter(h(r["context"][0]) for r in I if r["context"])
print("items", len(I), "distinct image hashes", len(H), "max dup", H.most_common(1)[0][1])
sub = collections.Counter(r.get("subset") for r in I); print(sub)
# shorter-response baseline
def txt(c): return c if isinstance(c, str) else json.dumps(c)
acc = [ (len(txt(r["cand"][0])) < len(txt(r["cand"][1]))) == (r["label"] == 0) for r in I if r["label"] in (0, 1)]
print("shorter-wins acc", sum(acc) / len(acc))
for s in sub:
    a = [ (len(txt(r["cand"][0])) < len(txt(r["cand"][1]))) == (r["label"] == 0) for r in I if r["label"] in (0, 1) and r.get("subset") == s]
    print(" ", s, len(a), round(sum(a) / len(a), 3))
