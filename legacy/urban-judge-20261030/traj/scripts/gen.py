import asyncio, pickle, json, sys
from common import *
import llm
P = pickle.load(open(f"{WS}/out/prep.pkl", "rb")); poi = P["poi"]
N = 6
INTRO = ("The city is a 200x200 grid of 500 m cells, cell (x,y). Times are 30-minute slots (HH:MM). "
         "Each line is a stay: time or time range, cell, the top POI categories of that cell, and the distance moved from the previous stay. "
         "Records are sparse (phone location pings), so gaps are normal.")


def ctx_text(t):
    return "\n\n".join(f"Day -{k}:\n" + (render(r, poi) if r else "(no records)") for k, r in t["ctx"])


def gen_msgs(t):
    return [{"role": "user", "content": f"{INTRO}\n\nHere are one person's location records for the past 7 days (Day -1 is yesterday; Day -7 is the same weekday as tomorrow):\n\n{ctx_text(t)}\n\n"
             "Predict this person's records for tomorrow (Day 0). Output ONLY lines of the form `HH:MM-HH:MM (x,y)` or `HH:MM (x,y)`, one stay per line, in chronological order, nothing else."}]


async def one(i, t, sem, f):
    async with sem:
        for a in range(3):
            try:
                r = await llm.client("qwen3-8b").chat.completions.create(
                    model="qwen3-8b", messages=gen_msgs(t), n=N, temperature=0.8, top_p=0.95, max_tokens=900,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                break
            except Exception as e:
                print("err", i, e, flush=True)
                if a == 2: return
                await asyncio.sleep(3)
    llm.usage["calls"] += 1; llm.usage["prompt_tokens"] += r.usage.prompt_tokens; llm.usage["completion_tokens"] += r.usage.completion_tokens
    outs = [c.message.content for c in r.choices]
    f.write(json.dumps({"i": i, "uid": t["uid"], "d": t["d"], "texts": outs, "recs": [parse_gen(o) for o in outs],
                        "prompt_tokens": r.usage.prompt_tokens}) + "\n"); f.flush()


async def main():
    sem = asyncio.Semaphore(24)
    with open(f"{WS}/out/gens.jsonl", "w") as f:
        await asyncio.gather(*[one(i, t, sem, f) for i, t in enumerate(P["targets"])])
    json.dump(llm.usage, open(f"{WS}/out/usage_gen.json", "w")); print(llm.usage)

if __name__ == "__main__":
    asyncio.run(main())
