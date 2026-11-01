"""Phase C1: generate detailed image descriptions for describe-then-judge (judge_bench.py --conds bin_desc).
resp items: one description of the question image(s) -> "ctx": [text]
img items : one description per candidate image w.r.t. the generation prompt / edit instruction -> "cand": [d0, d1]
Output JSONL rows {"id", "ctx": [...], "cand": [...]}; resumable by id; --shard/--nshard and --plan as in judge_bench.
usage: python describe.py --model qwen3vl8b --plan plan.json --max-new-tokens 256 --progress p.json
"""
import argparse, json, os, time, torch
from judge_bench import load_img, load_model, in_shard, Progress, MAXSIDE

DESC_CTX = ("Describe the image(s) in detail. Transcribe all visible text and numbers exactly, list the objects with "
            "their counts, attributes and spatial relations, and describe any chart or diagram precisely.")
DESC_CAND = ("This image was produced for the request: \"{q}\". Describe the image in detail: objects and their counts, "
             "attributes, visible text, spatial layout, visual quality and defects, and every way it matches or deviates "
             "from the request. Do not say which image is better.")


def requests(r):
    """list of (slot, messages); slot = ('ctx', 0) or ('cand', k)."""
    if r["kind"] == "text":
        return []
    if r["kind"] == "resp":
        if not r["context"]:
            return []
        c = [{"type": "image", "image": load_img(p, MAXSIDE)} for p in r["context"]] + [{"type": "text", "text": DESC_CTX}]
        return [(("ctx", 0), [{"role": "user", "content": c}])]
    out = []
    for k in (0, 1):
        c = []
        if r["context"]:
            c.append({"type": "text", "text": "Source image:"})
            c += [{"type": "image", "image": load_img(p, MAXSIDE)} for p in r["context"]]
            c.append({"type": "text", "text": "Result image:"})
        c += [{"type": "image", "image": load_img(r["cand"][k], MAXSIDE)}, {"type": "text", "text": DESC_CAND.format(q=r["question"][:1500])}]
        out.append((("cand", k), [{"role": "user", "content": c}]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl8b"); ap.add_argument("--model-path", default=None)
    ap.add_argument("--items"); ap.add_argument("--out"); ap.add_argument("--plan")
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--device-map", default="cuda:0"); ap.add_argument("--progress", default=None)
    a = ap.parse_args()
    plan = json.load(open(a.plan)) if a.plan else [{"items": a.items, "out": a.out}]
    todo = []
    for k, e in enumerate(plan):
        rows = [json.loads(l) for l in open(e["items"])]
        rows = [r for r in rows if in_shard(r["id"], a.shard, a.nshard)][: a.limit or None]
        done = {json.loads(l)["id"] for l in open(e["out"])} if os.path.exists(e["out"]) else set()
        todo += [(k, r) for r in rows if r["id"] not in done]
    print("items", len(todo), flush=True)
    prog = Progress(a.progress, len(todo), a.model); prog.write(0, force=True)
    if not todo:
        prog.write(0, "done", force=True); print("DONE 0"); return
    proc, model = load_model(a.model, a.model_path, a.device_map)
    files = {k: open(e["out"], "a", buffering=1) for k, e in enumerate(plan)}
    t0 = time.time(); s = 0; bs = a.bs
    while s < len(todo):
        B = todo[s:s + bs]
        try:
            reqs = [(n, slot, m) for n, (_, r) in enumerate(B) for slot, m in requests(r)]
            res = {n: {"ctx": [], "cand": ["", ""]} for n in range(len(B))}
            for i in range(0, len(reqs), bs):
                R = reqs[i:i + bs]
                inp = proc.apply_chat_template([m for _, _, m in R], tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt", padding=True,
                                               enable_thinking=False).to("cuda:0")
                with torch.inference_mode():
                    g = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
                txt = proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
                for (n, (kind, k), _), t in zip(R, txt):
                    if kind == "ctx":
                        res[n]["ctx"].append(t.strip())
                    else:
                        res[n]["cand"][k] = t.strip()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                s += 1; bs = a.bs
            else:
                bs = max(1, bs // 2)
            continue
        except Exception as e:
            print("error", repr(e)[:200], flush=True); s += len(B); continue
        for n, (k, r) in enumerate(B):
            files[k].write(json.dumps({"id": r["id"], **res[n]}) + "\n")
        s += len(B); prog.write(s, current=plan[B[-1][0]]["items"])
        bs = min(a.bs, bs * 2) if bs < a.bs else bs
    prog.write(len(todo), "done", force=True); print("DONE", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
