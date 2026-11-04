"""Natural-rate extensions of two mainstream multimodal judge benchmarks (2026-11-03).
  imagerewarddb_rest  imagerewarddb.jsonl (1500) holds NON-TIE pairs only, while the full pairwise expansion of
                      ImageRewardDB (imagerewarddb_full.jsonl, 7230) has 11.5% ties (equal human rank).  Add random
                      ties + random non-ties (not already in the base) so that the union has N_IR items at the natural
                      tie share: a proportionally stratified random sample of the full set.
  mllm_judge_rest     mllm_judge.jsonl (1500) is a random subset of MLLM-as-a-Judge pairs (7756, 20.5% ties).  Add a
                      simple random sample of the remaining pairs so that the union has N_MJ items (stays random).
NOTE (review): the 1500-item base files were built round-robin over subset x label, so base + rest is NOT a
proportional sample; analyses must use label_noise_lib.eval_items() (exact subset x tie proportions).
The rest files take items from *_full.jsonl, whose ids are disjoint from the base (the shared ids may differ in A/B
orientation, which is why the _full outputs are never merged).  tie_mech.items()/load() merge '<ds>_rest' by id.
usage: python make_rest2.py [N_IR] [N_MJ]"""
import json, random, sys

IT = "/media/shawn/C/papers/VLM/judgebench_local/items"
N_IR = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
N_MJ = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
rd = lambda f: [json.loads(l) for l in open(f"{IT}/{f}.jsonl")]
istie = lambda x: str(x["label"]) == "tie"


def write(name, xs):
    with open(f"{IT}/{name}.jsonl", "w") as f:
        for x in xs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


# ImageRewardDB: proportionally stratified
rng = random.Random(2)
full, base = rd("imagerewarddb_full"), rd("imagerewarddb")
have = {x["id"] for x in base}
assert not any(istie(x) for x in base)
t_nat = sum(map(istie, full)) / len(full)
n_tie = int(round(t_nat * N_IR)); n_dec = N_IR - n_tie - len(base)
ties = [x for x in full if istie(x) and x["id"] not in have]
decs = [x for x in full if not istie(x) and x["id"] not in have]
rest = rng.sample(ties, n_tie) + rng.sample(decs, n_dec)
rng.shuffle(rest); write("imagerewarddb_rest", rest)
print(f"imagerewarddb_rest {len(rest)} (ties {n_tie}, decided {n_dec}); union {len(rest) + len(base)}, "
      f"tie share {n_tie / (len(rest) + len(base)):.3f} (natural {t_nat:.3f})")

# MLLM-as-a-Judge: simple random complement
full, base = rd("mllm_judge_full"), rd("mllm_judge")
have = {x["id"] for x in base}
pool = [x for x in full if x["id"] not in have]
rest = rng.sample(pool, N_MJ - len(base)); write("mllm_judge_rest", rest)
u = rest + base
print(f"mllm_judge_rest {len(rest)}; union {len(u)}, tie share {sum(map(istie, u)) / len(u):.3f} "
      f"(full {sum(map(istie, full)) / len(full):.3f})")
