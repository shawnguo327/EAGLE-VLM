"""Phase H: reasoning judges with self-consistency. Each pair (both orders) is judged N times with sampling; the model
reasons (thinking models: in <think>; instruct models: step by step) and ends with "Verdict: A|B|T". The empirical
verdict distribution over N samples is a sampling-based uncertainty estimate, and reasoning is the LLM-judge
practice whose transfer to VLM judges we test: does it remove "cannot tell" but leave "close" as T?
Budget forcing: a sample that ends without a parsable verdict (usually truncated at --max-new-tokens) is closed with
"</think>\n\nVerdict:" (or "\nVerdict:") appended to its own reasoning and one forward pass reads the A/B/T logits.
Output rows: {"id","cond":"reason","order","verdicts":[...],"forced":[0/1...],"forced_lp":[{A,B,T}|null...],
"lens":[...],"trunc":k,"sample":first text (clipped)}.
Resumable, --plan/--shard/--progress as in judge_bench.py.
usage: python reason_judge.py --model qwen3vl8b_thinking --plan p.json --n 4 --max-new-tokens 1536 [--thinking]"""
import argparse, json, os, re, sys, time, torch
from judge_bench import build, load_model, in_shard, Progress, PAIR

SUFFIX = ("\n\nThink it through, then finish with a final line of the form 'Verdict: A', 'Verdict: B' or 'Verdict: T' "
          "(T if the two are of essentially equal quality).")
VERD = re.compile(r"verdict\s*[:：]\s*\**\s*([ABT])\b", re.I)


def force_verdicts(proc, model, msg, texts, thinking, ids):
    """One forward pass per unfinished sample: prompt + its reasoning + forced verdict prefix -> log-probs of A/B/T."""
    prompt = proc.apply_chat_template([msg], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    prompt = prompt[0] if isinstance(prompt, list) else prompt
    imgs = [x["image"] for x in msg[0]["content"] if x["type"] == "image"]
    out = []
    for t in texts:
        close = "\n</think>\n\nVerdict:" if (thinking or "<think>" in t) and "</think>" not in t else "\nVerdict:"
        kw = dict(text=[prompt + t + close], return_tensors="pt", padding=True)
        if imgs:
            kw["images"] = imgs
        inp = proc(**kw).to("cuda:0")
        with torch.inference_mode():
            lp = torch.log_softmax(model(**inp).logits[0, -1, :].float(), -1)
        out.append({k: torch.logsumexp(lp[ids[k]], 0).item() for k in "ABT"})     # "A" and " A" variants
    return out


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
    tok = proc.tokenizer
    kid = {c: torch.tensor(sorted({tok.encode(c, add_special_tokens=False)[0], tok.encode(" " + c, add_special_tokens=False)[0]}),
                           device="cuda:0") for c in "ABT"}
    nerr = 0
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
            V = [parse(t) for t in txt]; FL = [None] * len(txt)
            for j in range(len(B)):
                miss = [i for i in range(j * a.n, (j + 1) * a.n) if V[i] is None]
                if miss:
                    lps = force_verdicts(proc, model, msgs[j], [txt[i] for i in miss], a.thinking, kid)
                    for i, l in zip(miss, lps):
                        FL[i] = l; V[i] = max(l, key=l.get)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                for k, r, o in B:
                    files[k].write(json.dumps({"id": r["id"], "cond": "reason", "order": o, "error": "oom"}) + "\n")
                nerr += len(B); s += 1; bs = a.bs
            else:
                bs = max(1, bs // 2)
            continue
        except Exception as e:
            for k, r, o in B:
                files[k].write(json.dumps({"id": r["id"], "cond": "reason", "order": o, "error": repr(e)[:200]}) + "\n")
            nerr += len(B); s += len(B); continue
        for j, (k, r, o) in enumerate(B):
            sl = slice(j * a.n, (j + 1) * a.n); T = txt[sl]; L = lens[sl]
            files[k].write(json.dumps({"id": r["id"], "cond": "reason", "order": o, "verdicts": V[sl],
                                       "forced": [int(x is not None) for x in FL[sl]], "forced_lp": FL[sl], "lens": L,
                                       "trunc": sum(l >= a.max_new_tokens for l in L), "sample": T[0][-3000:]},
                                      ensure_ascii=False) + "\n")
        s += len(B); prog.write(s, current=plan[B[-1][0]]["items"])
        bs = min(a.bs, bs * 2) if bs < a.bs else bs
    prog.write(len(jobs), "done", force=True); print("DONE", round(time.time() - t0), "errors", nerr, flush=True)
    if nerr:
        sys.exit(3)          # the scheduler retries the task; the resume step redoes only the error rows


if __name__ == "__main__":
    main()
