"""Modality-controlled test (2026-11-03): is judging IMAGES the bottleneck for certification, or the task / labels?
Step 1 (this script, vLLM env): describe every candidate image of the image-candidate generation sets with one fixed
captioner (Qwen3-VL-8B, HF transformers, images downscaled to <= 768 px), greedy, prompt-agnostic (the captioner never sees the generation prompt or the other candidate).
Step 2: write text-candidate twins of the same pairs (same ids + '#cap', same human labels): question / criterion as in
the image item, cand = the two descriptions.  The judges then run bin + tri on the twins (judge_bench.py, kind=text).
Sets: GenAI-Arena generation (full), ImageRewardDB, Place Pulse.
usage: <lis-omni python> caption_cands.py [--limit N]"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/media/shawn/C/papers/VLM/judgebench_local")
ap.add_argument("--model", default="/media/shawn/C/papers/VLM/place-pulse-2.0/models/Qwen3-VL-8B-Instruct")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max_tokens", type=int, default=320)
a = ap.parse_args()
IT = f"{a.root}/items"; CAP = f"{IT}/captions_qwen3vl8b.jsonl"
SETS = [("genai_arena", "generation"), ("genai_arena_rest", "generation"), ("imagerewarddb", None), ("pp", None)]
PROMPT = ("Describe this image in detail and objectively, for someone who cannot see it. Cover: the main subjects and "
          "their attributes (colors, counts, poses), the setting and spatial layout, any visible text, the visual style, "
          "and the image quality, including any visible flaws or artifacts (distorted or malformed parts, blur, noise, "
          "odd lighting). Do not guess what the image was meant to show. Write one paragraph of at most 180 words.")


def cands(x):
    return x["cand"] if isinstance(x["cand"], list) else eval(x["cand"])


rows = []
for ds, sub in SETS:
    for l in open(f"{IT}/{ds}.jsonl"):
        x = json.loads(l)
        if sub is None or x["subset"] == sub:
            rows.append(x)
imgs = sorted({c for x in rows for c in cands(x)})
done = {}
if os.path.exists(CAP):
    done = {r["img"]: r["text"] for r in map(json.loads, open(CAP))}
todo = [p for p in imgs if p not in done]
if a.limit:
    todo = todo[: a.limit]
print(f"{len(rows)} pairs, {len(imgs)} images, {len(done)} captioned, {len(todo)} to do", flush=True)

if todo:   # HF transformers (as judge_bench.py): vLLM envs here either lack Qwen3-VL or crash on this Blackwell GPU
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.model); proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0",
                                                        attn_implementation="sdpa").eval()
    def small(p):
        im = Image.open(f"{a.root}/{p}").convert("RGB"); im.thumbnail((768, 768)); return im
    with open(CAP, "a") as f:
        for s in range(0, len(todo), a.bs):
            chunk = todo[s: s + a.bs]
            msgs = [[{"role": "user", "content": [{"type": "image", "image": small(p)}, {"type": "text", "text": PROMPT}]}]
                    for p in chunk]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=a.max_tokens, do_sample=False)
            txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            for p, t in zip(chunk, txt):
                done[p] = t.strip(); f.write(json.dumps({"img": p, "text": done[p]}) + "\n")
            f.flush()
            if (s // a.bs) % 20 == 0:
                print(f"{s + len(chunk)}/{len(todo)}", flush=True)

# step 2: text twins
if all(c in done for x in rows for c in cands(x)):
    out = {}
    for x in rows:
        base = x["dataset"]
        crit = (x.get("criterion") or "") + " The candidates are written descriptions of two images; judge the images they describe."
        y = dict(id=x["id"] + "#cap", dataset=base + "_cap", subset=x["subset"], kind="text", question=x["question"],
                 context=[], cand=[f"[Description of image]\n{done[c]}" for c in cands(x)], criterion=crit.strip(),
                 label=x["label"], meta=dict(src_id=x["id"]))
        out.setdefault(base + "_cap", []).append(y)
    for name, L in out.items():
        with open(f"{IT}/{name}.jsonl", "w") as f:
            for y in L:
                f.write(json.dumps(y, ensure_ascii=False) + "\n")
        print("wrote", name, len(L))
