"""Closed-source / hosted judges under exactly the same protocol as eaglevlm/judge_bench.py: the same prompts and image
resizing (build() is imported from judge_bench), the same answer keys and the same output rows, so the outputs drop into
the analysis pipeline unchanged (configs/paths.*.yaml: add the output directory to paths.outputs).

Works with any OpenAI-compatible chat-completions endpoint that returns token log-probabilities for the first output
token (OpenAI gpt-4o / gpt-4.1 family; Alibaba Model Studio qwen3-vl-plus; OpenRouter with a provider that supports
logprobs).  For APIs without logprobs use --samples N (N samples at temperature 1; key frequencies, add-0.5 smoothing).

Modes
  dry-run        count calls and estimate input tokens and cost; nothing is sent
  online         call the API (--workers concurrent requests; retries with back-off; resumable)
  batch-prepare  write OpenAI Batch API request files (<= --batch-mb each; 50% cheaper); submit them with the OpenAI CLI/SDK
  batch-parse    convert Batch API output files (--batch-out glob) into result rows

Readout: top_logprobs of the first output token.  Answer keys absent from the returned top-k get a floor log-prob
(smallest returned log-prob minus log 10), the same order of magnitude a local model would give; 'mass' = probability
on the answer keys among the returned tokens.  Pointwise ('score') calls depend only on the candidate, so identical
candidates in different pairs are scored once and reused ('cached': true).

Examples (EAGLEVLM_DATA = the judgebench_local directory with items/ and images/; key in an environment variable):
  python eaglevlm/api_judge.py --mode dry-run --items items/mllm_judge.jsonl --conds bin,tri,score --model gpt-4o-2024-11-20 \
      --price-in 1.25 --image-tokens 765
  OPENAI_API_KEY=... python eaglevlm/api_judge.py --mode online --items items/mllm_judge.jsonl --conds bin,tri,score \
      --model gpt-4o-2024-11-20 --out runs_api/raw/mllm_judge__gpt4o.jsonl --workers 8
  DASHSCOPE_API_KEY=... python eaglevlm/api_judge.py --mode online --base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
      --api-key-env DASHSCOPE_API_KEY --model qwen3-vl-plus --top-logprobs 5 --items ... --out ...
"""
import argparse, base64, concurrent.futures as cf, glob, io, json, math, os, random, sys, threading, time, types, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import torch  # noqa: F401  (judge_bench imports torch at module level but build() does not use it)
except ImportError:
    sys.modules["torch"] = types.ModuleType("torch")
import judge_bench as JB  # noqa: E402


def to_openai(msgs, detail):
    out = []
    for m in msgs:
        content = []
        for c in m["content"]:
            if c["type"] == "text":
                content.append({"type": "text", "text": c["text"]})
            else:
                b = io.BytesIO(); c["image"].save(b, "JPEG", quality=92)
                url = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
                content.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
        out.append({"role": m["role"], "content": content})
    return out


def count_text(msgs):
    t = 0; n_img = 0
    for m in msgs:
        for c in m["content"]:
            if c["type"] == "text":
                t += len(c["text"])
            else:
                n_img += 1
    return t, n_img


def score_key(r, order):
    x = r["cand"][0] if order == "A" else r["cand"][1]
    return json.dumps([r["kind"], r["question"], r.get("context", []), x], ensure_ascii=False)


def jobs_for(items, conds, done):
    J = []
    for r in items:
        for c in conds:
            if not JB.applies(c, r):
                continue
            for o in (["AB", "BA"] if c in JB.PAIR else ["A", "B"]):
                if (r["id"], c, o) not in done:
                    J.append((r, c, o))
    return J


def parse_top(top, keys, samples=None):
    """top: list of {token, logprob}; returns (lp dict over keys, mass)"""
    if samples is not None:
        cnt = {k: 0.5 for k in keys}
        for s in samples:
            k = s.strip()[:1].upper() if s else ""
            if k in cnt:
                cnt[k] += 1
        z = sum(cnt.values()); return {k: math.log(v / z) for k, v in cnt.items()}, 1.0
    acc = {}
    for t in top:
        k = t["token"].strip()
        k = k.upper() if len(k) == 1 and k.isalpha() else k
        if k in keys:                         # variants of one key (e.g. "A" and " A") are summed
            acc[k] = math.log(math.exp(acc[k]) + math.exp(t["logprob"])) if k in acc else t["logprob"]
    floor = (min(t["logprob"] for t in top) - math.log(10)) if top else -30.0
    lp = {k: acc.get(k, floor) for k in keys}
    return lp, float(sum(math.exp(v) for k, v in acc.items()))


def request_body(a, msgs):
    body = {"model": a.model, "messages": msgs, a.max_tokens_param: 1}
    if a.samples:
        body.update(temperature=1.0, n=a.samples)
    else:
        body.update(temperature=0, logprobs=True, top_logprobs=a.top_logprobs)
        if a.seed is not None:
            body["seed"] = a.seed
    if a.extra_json:
        body.update(json.loads(a.extra_json))
    return body


def post(a, body):
    key = os.environ.get(a.api_key_env, "")
    req = urllib.request.Request(a.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(a.retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as f:
                return json.loads(f.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < a.retries - 1:
                time.sleep(min(60, 2 ** attempt + random.random())); continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < a.retries - 1:
                time.sleep(min(60, 2 ** attempt + random.random())); continue
            raise RuntimeError(repr(e)[:300])


def row_from_response(r, c, o, resp, a):
    keys = JB.keys_of(c)
    ch = resp["choices"]
    if a.samples:
        lp, mass = parse_top(None, keys, [x["message"]["content"] or "" for x in ch])
    else:
        lpc = (ch[0].get("logprobs") or {}).get("content") or []
        if not lpc or not lpc[0].get("top_logprobs"):
            raise RuntimeError("no top_logprobs returned (provider ignored logprobs?)")
        lp, mass = parse_top(lpc[0]["top_logprobs"], keys)
    return {"id": r["id"], "cond": c, "order": o, "lp": lp, "mass": mass,
            "api": {"model": resp.get("model", a.model), "answer": ch[0]["message"]["content"]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["dry-run", "online", "batch-prepare", "batch-parse"])
    ap.add_argument("--items", required=True); ap.add_argument("--conds", default="bin,tri,score"); ap.add_argument("--out")
    ap.add_argument("--model", required=True); ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY"); ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--samples", type=int, default=0, help="APIs without logprobs: N samples at temperature 1")
    ap.add_argument("--max-tokens-param", default="max_completion_tokens", help="max_tokens for older OpenAI-compatible servers")
    ap.add_argument("--detail", default="high"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-json", default=None, help='extra body fields, e.g. {"provider": {"order": ["Parasail"], "require_parameters": true}}')
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only the first N calls (pilot)")
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens (dry-run)")
    ap.add_argument("--image-tokens", type=int, default=765, help="input tokens per image (dry-run)")
    ap.add_argument("--batch-dir", default=None); ap.add_argument("--batch-mb", type=float, default=180)
    ap.add_argument("--batch-out", default=None, help="glob of Batch API output files (batch-parse)")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(JB.ROOT / a.items if not os.path.isabs(a.items) else a.items)]
    conds = a.conds.split(",")
    done = set()
    if a.out and os.path.exists(a.out):
        for l in open(a.out):
            try:
                d = json.loads(l)
            except ValueError:
                continue
            if "lp" in d:
                done.add((d["id"], d["cond"], d["order"]))
    J = jobs_for(items, conds, done)
    # pointwise de-duplication: one call per distinct (question, context, candidate)
    first, dup = {}, []
    JJ = []
    for r, c, o in J:
        if c.startswith("score"):
            k = (c, score_key(r, o))
            if k in first:
                dup.append(((r, c, o), k)); continue
            first[k] = (r, c, o)
        JJ.append((r, c, o))
    if a.limit:
        JJ = JJ[: a.limit]
    print(f"{len(items)} items, {len(J)} result rows to produce, {len(JJ)} API calls ({len(dup)} pointwise rows reused)", flush=True)

    if a.mode == "dry-run":
        chars = imgs = 0                      # analytic count from the item records (no image is loaded)
        for r, c, o in JJ:
            pair = c in JB.PAIR
            if r["kind"] in ("resp", "text"):
                imgs += len(r.get("context") or []) if r["kind"] == "resp" else 0
                chars += len(r["question"]) + (len(r["cand"][0]) + len(r["cand"][1]) if pair else len(r["cand"][0 if o == "A" else 1])) + 400
            else:
                imgs += len(r.get("context") or []) + (2 if pair else 1); chars += len(r["question"]) + 300
        tok = chars / 4 + imgs * a.image_tokens + 10 * len(JJ)
        print(f"text chars {chars:,}  images {imgs:,}  est. input tokens {tok/1e6:.1f}M  est. cost ${tok/1e6*a.price_in:,.0f} at ${a.price_in}/M")
        return

    if a.mode == "batch-prepare":
        os.makedirs(a.batch_dir, exist_ok=True); part, size, f = 0, 0, None
        for r, c, o in JJ:
            body = request_body(a, to_openai(JB.build(r, c, o), a.detail))
            line = json.dumps({"custom_id": json.dumps([r["id"], c, o]), "method": "POST", "url": "/v1/chat/completions", "body": body}) + "\n"
            if f is None or size + len(line) > a.batch_mb * 1e6:
                if f:
                    f.close()
                part += 1; size = 0; f = open(f"{a.batch_dir}/requests_{part:03d}.jsonl", "w")
            f.write(line); size += len(line)
        if f:
            f.close()
        json.dump({"dup": [[[r["id"], c, o], k] for (r, c, o), k in dup], "items": a.items, "model": a.model},
                  open(f"{a.batch_dir}/dedup_map.json", "w"))
        print(f"wrote {part} request files to {a.batch_dir}"); return

    out = open(a.out, "a", buffering=1); lock = threading.Lock(); results = {}
    byid = {r["id"]: r for r in items}

    def emit(row):
        with lock:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    if a.mode == "batch-parse":
        dm = json.load(open(f"{a.batch_dir}/dedup_map.json")) if a.batch_dir and os.path.exists(f"{a.batch_dir}/dedup_map.json") else {"dup": []}
        n = 0
        for fp in sorted(glob.glob(a.batch_out)):
            for l in open(fp):
                d = json.loads(l); i, c, o = json.loads(d["custom_id"])
                try:
                    row = row_from_response(byid[i], c, o, d["response"]["body"], a)
                except Exception as e:
                    emit({"id": i, "cond": c, "order": o, "error": repr(e)[:200]}); continue
                emit(row); n += 1
                if c.startswith("score"):
                    results[(c, score_key(byid[i], o))] = row
        for x in dm["dup"]:
            (i, c, o), k = x[0], tuple(x[1])
            src = results.get(k)
            if src:
                emit(dict(src, id=i, order=o, cached=True))
        print(f"parsed {n} responses into {a.out}"); return

    def work(job):
        r, c, o = job
        try:
            resp = post(a, request_body(a, to_openai(JB.build(r, c, o), a.detail)))
            row = row_from_response(r, c, o, resp, a)
        except Exception as e:
            emit({"id": r["id"], "cond": c, "order": o, "error": repr(e)[:300]}); return None
        emit(row)
        if c.startswith("score"):
            results[(c, score_key(r, o))] = row
        return row

    t0 = time.time(); n = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for _ in ex.map(work, JJ):
            n += 1
            if n % 200 == 0:
                print(f"{n}/{len(JJ)} calls, {time.time() - t0:.0f}s", flush=True)
    for (r, c, o), k in dup:
        src = results.get(k)
        if src:
            emit(dict(src, id=r["id"], order=o, cached=True))
    print(f"done: {n} calls -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
