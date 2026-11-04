"""Input-parity smoke test (CPU only, no weights): build the exact model inputs judge_bench.py would feed, for a fixed
sample of items x conds x orders, and print a fingerprint per model (token ids hash + pixel tensor stats).
Run on two machines; identical fingerprints = identical env, paths, processors, prompts.
usage (cwd = judgebench_local): python input_parity.py <judge_bench dir> <model_name=model_path> [...]"""
import hashlib, json, sys
sys.argv, args = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, args[0])
import judge_bench as jb
from transformers import AutoProcessor

SAMPLE = [("genai_arena", 0), ("genai_arena", 1600), ("genai_arena_edit", 0), ("pickapic_u", 3), ("imagerewarddb_rest2", 5),
          ("mllm_judge_rest2", 7), ("mllm_judge", 11)]
rows = []
for ds, k in SAMPLE:
    L = open(f"items/{ds}.jsonl").readlines()
    rows.append(json.loads(L[k % len(L)]))
for spec in args[1:]:
    name, path = spec.split("=")
    internvl = "internvl" in (name + path).lower()
    proc = AutoProcessor.from_pretrained(path, **({"crop_to_patches": False} if internvl else {}))
    proc.tokenizer.padding_side = "left"
    h = hashlib.sha256(); n = 0
    for r in rows:
        for c, orders in (("bin", "AB"), ("tri", "AB"), ("tri", "BA"), ("score", "AB")):
            for o in orders if c != "score" else ("A", "B"):
                msgs = [jb.build(r, c, o)]
                inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt", padding=True, enable_thinking=False)
                h.update(inp["input_ids"].numpy().tobytes())
                if "pixel_values" in inp:
                    pv = inp["pixel_values"].float()
                    h.update(f"{tuple(pv.shape)}{pv.sum().item():.3f}{pv.abs().mean().item():.5f}".encode())
                n += 1
    print(f"{name}: {n} inputs, fingerprint {h.hexdigest()[:16]}", flush=True)
