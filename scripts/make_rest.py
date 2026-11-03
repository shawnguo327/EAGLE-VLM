"""Complement the capped item sets so the main tie analyses run on full / natural-rate data (2026-11-03).
  genai_arena_rest  every GenAI-Arena-Bench pair (genai_arena_full.jsonl, 4175) not already in genai_arena.jsonl (1500)
                    -> the union is the complete image_generation + image_edition benchmark.
  arena55k_rest     arena55k.jsonl was water-filled to 50% ties (1500 tie + 1500 win).  Add random usable non-tie battles
                    (same filter as the converter) until the union has the natural tie share of the usable pool; the
                    union is then a proportionally stratified random sample (tie / win), i.e. natural-rate.
Same ids / dataset field as the base sets; tie_mech.items() and load() merge the '_rest' files by id.
usage: python make_rest.py  (writes into judgebench_local/items)"""
import json, random, csv, sys, types
csv.field_size_limit(sys.maxsize)

IT = "/media/shawn/C/papers/VLM/judgebench_local/items"
SRC = "/media/shawn/C/papers/VLM/judgebench_full/raw/lmarena-ai__arena-human-preference-55k/train.csv"
CRIT = "Prefer the response that is more helpful, relevant, accurate and appropriate for the user's request."
rd = lambda f: [json.loads(l) for l in open(f"{IT}/{f}.jsonl")]

# GenAI-Arena: complement
base = {x["id"] for x in rd("genai_arena")}
rest = [x for x in rd("genai_arena_full") if x["id"] not in base]
with open(f"{IT}/genai_arena_rest.jsonl", "w") as f:
    for x in rest:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("genai_arena_rest", len(rest), "tie share", round(sum(str(x["label"]) == "tie" for x in rest) / len(rest), 3))


# Arena 55k: natural-rate complement
def one(x):
    try:
        v = json.loads(x)
    except Exception:
        return None
    return v if isinstance(v, list) else None


asc = lambda t: sum(c.isascii() for c in t) / max(1, len(t))
usable = []
for row in csv.DictReader(open(SRC)):
    r = types.SimpleNamespace(id=row['id'], model_a=row['model_a'], model_b=row['model_b'], prompt=row['prompt'],
                              response_a=row['response_a'], response_b=row['response_b'],
                              winner_model_a=int(row['winner_model_a']), winner_model_b=int(row['winner_model_b']),
                              winner_tie=int(row['winner_tie']))
    P, A, B = one(r.prompt), one(r.response_a), one(r.response_b)
    if not P or not A or not B or len(P) != 1 or len(A) != 1 or len(B) != 1:
        continue
    q, a, b = P[0], A[0], B[0]
    if not all(isinstance(x, str) and x.strip() for x in (q, a, b)):
        continue
    if max(len(q), len(a), len(b)) > 6000 or min(asc(q), asc(a), asc(b)) < .97:
        continue
    usable.append((r, q, a, b))
t_nat = sum(bool(r.winner_tie) for r, *_ in usable) / len(usable)
b55 = rd("arena55k"); have = {x["id"] for x in b55}; n_tie = sum(str(x["label"]) == "tie" for x in b55)
need = int(round(n_tie / t_nat)) - len(b55)
pool = [u for u in usable if not u[0].winner_tie and f"arena55k/{u[0].id}" not in have]
rng = random.Random(1); pick = rng.sample(pool, need); out = []
for r, q, a, b in pick:
    win, lose = ((a, r.model_a), (b, r.model_b)) if r.winner_model_a else ((b, r.model_b), (a, r.model_a))
    g = rng.randint(0, 1); cand = [win, lose] if g == 0 else [lose, win]
    out.append(dict(id=f"arena55k/{r.id}", dataset="arena55k", subset="win", kind="text", question=q, context=[],
                    cand=[c[0] for c in cand], criterion=CRIT, label=str(g),
                    meta=dict(orig_id=int(r.id), models=[c[1] for c in cand])))
with open(f"{IT}/arena55k_rest.jsonl", "w") as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"arena55k usable {len(usable)}, natural tie share {t_nat:.3f}; base {len(b55)} ({n_tie} ties) + rest {len(out)} "
      f"-> union {len(b55) + len(out)}, tie share {n_tie / (len(b55) + len(out)):.3f}")
