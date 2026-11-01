"""Phase H: reasoning judges with self-consistency. Each pair (both orders) is judged N times with sampling; the model
reasons (thinking models: in <think>; instruct models: step by step) and ends with "Verdict: A|B|T". The empirical
verdict distribution over N samples is a sampling-based uncertainty estimate, and reasoning is the LLM-judge
practice whose transfer to VLM judges we test: does it remove "cannot tell" but leave "close" as T?
Output rows: {"id","cond":"reason","order","verdicts":[...],"lens":[...],"trunc":k,"sample":first text (clipped)}.
Resumable, --plan/--shard/--progress as in judge_bench.py.
usage: python reason_judge.py --model qwen3vl8b_thinking --plan p.json --n 4 --max-new-tokens 1536 [--thinking]"""
import argparse, json, os, re, time, torch
from judge_bench import build, load_model, in_shard, Progress, PAIR

SUFFIX = ("\n\nThink it through, then finish with a final line of the form 'Verdict: A', 'Verdict: B' or 'Verdict: T' "
          "(T if the two are of essentially equal quality).")
VERD = re.compile(r"verdict\s*[:：]\s*\**\s*([ABT])\b", re.I)


def parse(t):
    t = t.split("</think>")[-1] if "</think>" in t else t
    m = VERD.findall(t)
    return m[-1].upper() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--model-path", default=None)
    ap.add_argument("--items"); ap.add_argument("--out"); ap.add_argument("--plan")
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--thinking", action="store_true", help="enable_thinking=True in the chat template (Qwen3/3.5)")
    ap.add_argument("--text-llm", action="store_true"); ap.add_argument("--kinds", default="resp,img,text")
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--device-map", default="cuda:0"); ap.add_argument("--progress", default=None)
    a = ap.parse_args()
    plan = json.load(open(a.plan)) if a.plan else [{"items": a.items, "out": a.out}]
    kinds = set(a.kinds.split(","))
    jobs = []
    for k, e in enumerate(plan):
        rows = [json.loads(l) for l in open(e["items"])]
        rows = [r for r in rows if in_shard(r["id"], a.shard, a.nshard) and r["kind"] in kinds
                and (not a.text_llm or r["kind"] == "text")][: a.limit or None]
        done = set()
        if os.path.exists(e["out"]):
            for l in open(e["out"]):
                d = json.loads(l)
                if "verdicts" in d:
                    done.add((d["id"], d["order"]))
        jobs += [(k, r, o) for r in rows for o in ("AB", "BA") if (r["id"], o) not in done]
    jobs.sort(key=lambda j: len(json.dumps(j[1]["cand"])))
    print("jobs", len(jobs), "samples", len(jobs) * a.n, flush=True)
    prog = Progress(a.progress, len(jobs), a.model); prog.write(0, force=True)
    if not jobs:
        prog.write(0, "done", force=True); print("DONE 0"); return
    proc, model = load_model(a.model, a.model_path, a.device_map, text_llm=a.text_llm)
    files = {k: open(e["out"], "a", buffering=1) for k, e in enumerate(plan)}
    t0 = time.time(); s = 0; bs = a.bs
    while s < len(jobs):
        B = jobs[s:s + bs]
        try:
            msgs = []
            for _, r, o in B:
                m = build(r, "tri", o)
                m[0]["content"][-1]["text"] = m[0]["content"][-1]["text"].rsplit(" Answer with exactly one letter", 1)[0] + SUFFIX
                msgs.append(m)
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True, enable_thinking=a.thinking).to("cuda:0")
            with torch.inference_mode():
                g = model.generate(**inp, do_sample=True, temperature=a.temperature, top_p=a.top_p,
                                   num_return_sequences=a.n, max_new_tokens=a.max_new_tokens)
            gen = g[:, inp["input_ids"].shape[1]:]
            txt = proc.batch_decode(gen, skip_special_tokens=True)
            eos = proc.tokenizer.eos_token_id; pad = proc.tokenizer.pad_token_id
            lens = [int(((x != pad) & (x != eos)).sum()) for x in gen]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                for k, r, o in B:
                    files[k].write(json.dumps({"id": r["id"], "cond": "reason", "order": o, "error": "oom"}) + "\n")
                s += 1; bs = a.bs
            else:
                bs = max(1, bs // 2)
            continue
        except Exception as e:
            for k, r, o in B:
                files[k].write(json.dumps({"id": r["id"], "cond": "reason", "order": o, "error": repr(e)[:200]}) + "\n")
            s += len(B); continue
        for j, (k, r, o) in enumerate(B):
            T = txt[j * a.n:(j + 1) * a.n]; L = lens[j * a.n:(j + 1) * a.n]
            V = [parse(t) for t in T]
            files[k].write(json.dumps({"id": r["id"], "cond": "reason", "order": o, "verdicts": V, "lens": L,
                                       "trunc": sum(l >= a.max_new_tokens for l in L), "sample": T[0][-3000:]},
                                      ensure_ascii=False) + "\n")
        s += len(B); prog.write(s, current=plan[B[-1][0]]["items"])
        bs = min(a.bs, bs * 2) if bs < a.bs else bs
    prog.write(len(jobs), "done", force=True); print("DONE", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
