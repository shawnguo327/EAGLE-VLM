"""Async client to local vLLM OpenAI servers. Judging calls read first-token
logprobs over option letters; solving calls run thinking mode."""
import asyncio, math
from openai import AsyncOpenAI

PORTS = {"qwen3-14b": 8014, "qwen3-8b": 8008, "qwen3-4b": 8004, "mistral-24b": 8024, "glm-32b": 8032, "llama-8b": 8038, "llama-70b": 8070}
_clients = {}
usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def client(model):
    if model not in _clients:
        _clients[model] = AsyncOpenAI(base_url=f"http://localhost:{PORTS[model]}/v1", api_key="x", timeout=3600)
    return _clients[model]


async def first_token_dist(model, messages, letters, sem):
    """Return dict letter->prob (renormalized over `letters`) from the first
    generated token's top-20 logprobs, thinking disabled."""
    async with sem:
        for attempt in range(3):
            try:
                r = await client(model).chat.completions.create(
                    model=model, messages=messages, max_tokens=2, temperature=0,
                    logprobs=True, top_logprobs=20,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                break
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2)
    usage["calls"] += 1
    usage["prompt_tokens"] += r.usage.prompt_tokens
    usage["completion_tokens"] += r.usage.completion_tokens
    lp = r.choices[0].logprobs.content[0].top_logprobs
    raw = {}
    for t in lp:
        tok = t.token.strip().strip("*").strip()
        if tok in letters:
            raw[tok] = max(raw.get(tok, -1e9), t.logprob)
    probs = {l: math.exp(raw[l]) if l in raw else 0.0 for l in letters}
    z = sum(probs.values())
    mass = z  # how much of the first-token mass fell on valid letters
    probs = {l: (p / z if z > 0 else 1.0 / len(letters)) for l, p in probs.items()}
    return probs, mass, r.choices[0].message.content


async def solve(model, messages, sem, max_tokens=6000):
    async with sem:
        r = await client(model).chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.6, top_p=0.95,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    usage["calls"] += 1
    usage["prompt_tokens"] += r.usage.prompt_tokens
    usage["completion_tokens"] += r.usage.completion_tokens
    txt = r.choices[0].message.content or ""
    if "</think>" in txt:
        txt = txt.split("</think>")[-1]
    return txt.strip(), r.choices[0].finish_reason


async def plain(model, messages, sem, max_tokens=512, temperature=0.0):
    async with sem:
        r = await client(model).chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    usage["calls"] += 1
    usage["prompt_tokens"] += r.usage.prompt_tokens
    usage["completion_tokens"] += r.usage.completion_tokens
    return (r.choices[0].message.content or "").strip()
