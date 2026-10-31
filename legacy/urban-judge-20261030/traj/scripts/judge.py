"""Pairwise LLM judge, both orders, first-token logprobs over A/B.
usage: python judge.py cf|cc|bon"""
import asyncio, pickle, json, sys, itertools, random
from common import *
import llm
from gen import INTRO, ctx_text  # noqa (gen.py guarded below)

JOB = sys.argv[1]
import os
MODEL = os.environ.get("MODEL", "qwen3-8b")
TAG = "" if MODEL == "qwen3-8b" else "_" + MODEL.split("-")[-1]
EV = JOB in ("cfe", "cfe2", "cce", "bone", "cfs")


def stats_line(recs):
    f = feats(recs)
    return (f"Stats: {int(f[0])} pings, {int(f[1])} stays, {int(f[2])} unique cells; most-visited cell share {100*f[8]:.0f}% of pings; "
            f"radius of gyration {f[3]:.1f} km; jumps between stays mean {f[4]:.1f} / median {f[6]:.1f} / max {f[5]:.1f} km; "
            f"{100*f[7]:.0f}% of jumps over 5 km.")


def rend(recs, poi):
    if JOB == "cfs":
        return stats_line(recs)
    return render(recs, poi) + ("\n" + stats_line(recs) if EV else "")
P = pickle.load(open(f"{WS}/out/prep{SUF}.pkl", "rb")); poi = P["poi"]; L = P["ladder"]; T = P["targets"]
G = {}
for l in (open(f"{WS}/out/gens.jsonl") if CITY == "D" else []):
    r = json.loads(l); G[r["i"]] = [[tuple(x) for x in recs] for recs in r["recs"]]
gidx = sorted(i for i in G if G[i] and len(G[i][0]) >= 2)

CF_Q = ("Below are two records of one day of movement in a city. Which of these two is more likely to be a real person's actual, "
        "unaltered day of movement (as opposed to synthetic or perturbed data)? Answer with a single letter: A or B.")
CC_Q = ("Below are two candidate records for this person's next day (Day 0). Which one is more likely to be this person's actual next day? "
        "Answer with a single letter: A or B.")


if JOB == "cfh":
    CF_Q = CF_Q + (" Hint: synthetic/perturbed records often have random spatial jitter (many short-lived stays scattered around, frequent "
                   "back-and-forth jumps of several km between neighbouring 30-minute slots, few repeated cells), while real people "
                   "dwell at a few places (home, work) for long periods and make purposeful trips.")
if JOB == "cfs":
    CF_Q = CF_Q.replace("two records of one day", "summary statistics of two records of one day")
if JOB in ("cfe2", "cfs"):
    CF_Q = CF_Q + (" Background on real data: in real phone-ping days at this resolution the same cells recur a lot - typically about half as many "
                   "unique cells as pings (median 9 unique cells for 20 pings), the most-visited cell (usually home) holds a large share of pings "
                   "(median ~27%), and most jumps between consecutive stays are short (median ~1.8 km), with occasional longer trips.")
if JOB in ("cfe", "cfe2"):
    CF_Q = CF_Q + " Each record is followed by a computed Stats line summarising it."
if JOB in ("cce", "bone"):
    CC_Q = CC_Q + " Each candidate is followed by a computed Stats line summarising it."
if JOB == "ccd":
    CC_Q = ("Below are two candidate records for this person's next day (Day 0). Note that people's days vary: the actual next day is often NOT a "
            "copy of any past day, and a candidate that merely repeats or averages the history is not necessarily the real one. "
            "Which one is more likely to be this person's actual next day? Answer with a single letter: A or B.")


def cf_msgs(ta, tb):
    return [{"role": "user", "content": f"{INTRO}\n\n{CF_Q}\n\nRecord A:\n{ta}\n\nRecord B:\n{tb}\n\nRespond with only the single letter A or B, nothing else."}]


def cc_msgs(t, ta, tb):
    return [{"role": "user", "content": f"{INTRO}\n\nHere are one person's location records for the past 7 days (Day -1 is yesterday; Day -7 is the same weekday as Day 0):\n\n{ctx_text(t)}\n\n"
             f"{CC_Q}\n\nCandidate A:\n{ta}\n\nCandidate B:\n{tb}\n\nRespond with only the single letter A or B, nothing else."}]


def item(rung, k):
    if rung == "llmgen":
        i = gidx[k % len(gidx)]; return G[i][0]
    return L[rung][k]


pairs = []  # (meta, ctx_target or None, recs_a, recs_b)
if JOB == "cf":
    rungs = ["real", "gauss1", "gauss2", "gauss4", "gauss8", "tshuffle", "markov", "uniform", "llmgen"]
    K = 150
    for a, b in itertools.combinations(rungs, 2):
        for k in range(K):
            ka, kb = k, 300 + k
            pairs.append(({"a": a, "b": b, "k": k}, None, item(a, ka), item(b, kb if b != "llmgen" else k + 7)))
    for k in range(K):  # null pairs real vs real
        pairs.append(({"a": "real", "b": "real2", "k": k}, None, L["real"][k], L["real"][450 + k % 150]))
elif JOB == "cfB":  # city-B replication of real-vs-rung and adjacent ladder pairs
    for a, b in [("real", r) for r in ["gauss1", "gauss2", "gauss4", "gauss8", "tshuffle", "markov", "uniform"]] + [("gauss1", "gauss2"), ("gauss2", "gauss4"), ("gauss4", "gauss8"), ("gauss8", "uniform")]:
        for k in range(150):
            pairs.append(({"a": a, "b": b, "k": k}, None, item(a, k), item(b, 300 + k)))
    for rung in ["gauss1", "gauss2", "gauss4", "gauss8"]:
        for k in range(150):
            pairs.append(({"a": "real", "b": rung + "_matched", "k": k}, None, L["real"][k], L[rung][k]))
elif JOB == "cfm":  # matched: real day vs its own perturbation
    for rung in ["gauss1", "gauss2", "gauss4", "gauss8", "tshuffle"]:
        for k in range(150):
            pairs.append(({"a": "real", "b": rung, "k": k}, None, L["real"][k], L[rung][k]))
elif JOB in ("cfe", "cfe2", "cfs"):
    for rung in os.environ.get("RUNGS", "gauss1,gauss2,gauss4,gauss8,tshuffle,markov,llmgen,real2").split(","):
        for k in range(150):
            if rung == "real2":
                pairs.append(({"a": "real", "b": "real2", "k": k}, None, L["real"][k], L["real"][450 + k]))
            else:
                pairs.append(({"a": "real", "b": rung, "k": k}, None, L["real"][k], item(rung, 300 + k if rung != "llmgen" else k + 7)))
elif JOB == "cce":
    for i in gidx:
        t = T[i]
        for name, recs in {"llmgen": G[i][0], "copy7": t["copy7"], "other_day": t["other_day"]}.items():
            pairs.append(({"a": "real", "b": name, "i": i}, t, t["real"], recs))
elif JOB == "cfh":
    for rung in ["gauss1", "gauss2", "gauss4", "gauss8", "tshuffle", "llmgen"]:
        for k in range(150):
            pairs.append(({"a": "real", "b": rung, "k": k}, None, L["real"][k], item(rung, 300 + k if rung != "llmgen" else k + 7)))
elif JOB == "cfe2":
    CF_Q = CF_Q + (" Background on real data: in real phone-ping days at this resolution the same cells recur a lot - typically about half as many "
                   "unique cells as pings (median 9 unique cells for 20 pings), the most-visited cell (usually home) holds a large share of pings "
                   "(median ~27%), and most jumps between consecutive stays are short (median ~1.8 km), with occasional longer trips.")
if JOB in ("cfe", "cfe2"):
    CF_Q = CF_Q + " Each record is followed by a computed Stats line summarising it."
if JOB in ("cce", "bone"):
    CC_Q = CC_Q + " Each candidate is followed by a computed Stats line summarising it."
if JOB == "ccd":
    for i in gidx:
        t = T[i]
        for name, recs in {"llmgen": G[i][0], "other_day": t["other_day"], "copy7": t["copy7"], "markov": t["markov"]}.items():
            pairs.append(({"a": "real", "b": name, "i": i}, t, t["real"], recs))
elif JOB == "cft":  # thinking-mode judge on a subset (unmatched, like cf)
    for rung in ["gauss2", "gauss8", "tshuffle", "markov", "llmgen"]:
        for k in range(40):
            pairs.append(({"a": "real", "b": rung, "k": k}, None, L["real"][k], item(rung, 300 + k if rung != "llmgen" else k + 7)))
elif JOB == "cc":
    for i in gidx:
        t = T[i]
        cands = {"llmgen": G[i][0], "other_day": t["other_day"], "markov": t["markov"], "copy7": t["copy7"],
                 "other_user": T[gidx[(gidx.index(i) + 1) % len(gidx)]]["real"], "stayhome": t["stayhome"]}
        for name, recs in cands.items():
            pairs.append(({"a": "real", "b": name, "i": i}, t, t["real"], recs))
elif JOB in ("bon", "bone"):
    for i in gidx:
        t = T[i]
        for x, y in itertools.combinations(range(len(G[i])), 2):
            if not G[i][x] or not G[i][y]: continue
            pairs.append(({"a": x, "b": y, "i": i}, t, G[i][x], G[i][y]))
print(JOB, "pairs", len(pairs), flush=True)


import re
THINK_TAIL = "Respond with only the single letter A or B, nothing else."
async def think_pick(msgs, sem):
    m = [dict(msgs[0])]
    m[0]["content"] = m[0]["content"].replace(THINK_TAIL, "Think it through, then end your response with exactly 'Final answer: A' or 'Final answer: B'.")
    txt, fr = await llm.solve("qwen3-8b", m, sem, max_tokens=4000)
    mm = re.findall(r"Final answer:\s*\**\s*([AB])", txt)
    if mm: return {"A": 1.0 if mm[-1] == "A" else 0.0, "B": 0.0 if mm[-1] == "A" else 1.0}, 1.0, fr
    return {"A": 0.5, "B": 0.5}, 0.0, fr


async def run(meta, t, ra, rb, sem, f):
    ta, tb = rend(ra, poi), rend(rb, poi)
    if JOB == "cft":
        try:
            (p1, m1, _), (p2, m2, _) = await asyncio.gather(think_pick(cf_msgs(ta, tb), sem), think_pick(cf_msgs(tb, ta), sem))
        except Exception as e:
            print("err", meta, e, flush=True); return
        pa = 0.5 * (p1["A"] + p2["B"])
        f.write(json.dumps({**meta, "pA_ab": p1["A"], "pA_ba": p2["A"], "p_a": pa, "dp": abs(2 * pa - 1), "mass1": m1, "mass2": m2}) + "\n"); f.flush()
        return
    mk = (lambda x, y: cf_msgs(x, y)) if t is None else (lambda x, y: cc_msgs(t, x, y))
    try:
        (p1, m1, _), (p2, m2, _) = await asyncio.gather(
            llm.first_token_dist(MODEL, mk(ta, tb), ["A", "B"], sem),
            llm.first_token_dist(MODEL, mk(tb, ta), ["A", "B"], sem))
    except Exception as e:
        print("err", meta, e, flush=True); return
    pa = 0.5 * (p1["A"] + p2["B"])  # prob that item a is preferred, order-averaged
    f.write(json.dumps({**meta, "pA_ab": p1["A"], "pA_ba": p2["A"], "p_a": pa, "dp": abs(2 * pa - 1),
                        "mass1": m1, "mass2": m2}) + "\n"); f.flush()


async def main():
    sem = asyncio.Semaphore(32)
    with open(f"{WS}/out/judge_{JOB}{TAG}.jsonl", "w") as f:
        await asyncio.gather(*[run(m, t, a, b, sem, f) for m, t, a, b in pairs])
    json.dump(llm.usage, open(f"{WS}/out/usage_{JOB}{TAG}.json", "w")); print(llm.usage)

asyncio.run(main())
