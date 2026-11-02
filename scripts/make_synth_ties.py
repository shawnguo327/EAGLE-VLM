"""Controlled tie set (tie-mechanism follow-up 4, 2026-11-02).  From a verified-correct response R about an image
(chosen response of VL-RewardBench hallucination/general and Multimodal RewardBench vqa/correctness items), a text LLM
(Qwen3-8B, no thinking) writes
  P  paraphrase of R (same claims),
  W1, W2  R with ONE visual-detail phrase replaced (the LLM names phrase + replacement as JSON; the substitution is
          done here, so exactly one span differs; W1 and W2 change different phrases),
  F  R plus one sentence contradicting an earlier statement (visible without the image).
Pairs (label):  TG = R vs P (tie, both good)     TB = W1 vs W2 (tie, both bad)
                DV = R vs W1 (R; needs the image) DT = R vs F (R; visible in the text)
Checks by the same LLM: paraphrase keeps the claims; each replacement changes what is claimed to be visible; bases failing any check are
dropped.  Candidate order is fixed by crc32 of the id.
usage (vllm env): python make_synth_ties.py --llm QWEN3_8B --out items/synth_tie.jsonl [--n 320]"""
import argparse, json, os, random, re, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--llm", required=True)
ap.add_argument("--root", default="/media/shawn/C/papers/VLM/judgebench_local")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=320)
ap.add_argument("--gpu-mem", type=float, default=0.35)
a = ap.parse_args()

src = []
for ds, subs in (("vlrb", {"hallucination", "general"}), ("mmrb", {"vqa", "correctness"})):
    for l in open(f"{a.root}/items/{ds}.jsonl"):
        r = json.loads(l)
        if r["subset"] in subs and r["kind"] == "resp" and len(r.get("context", [])) == 1 and r["label"] in (0, 1):
            R = r["cand"][r["label"]] or ""
            if 150 <= len(R) <= 1800:
                src.append((ds, r, R))
random.Random(0).shuffle(src)
vl = [s for s in src if s[0] == "vlrb"][: int(a.n * 0.65)]; mm = [s for s in src if s[0] == "mmrb"][: a.n - len(vl)]
src = vl + mm
print(len(src), "bases", flush=True)

import numpy as _np
if not hasattr(_np, "long"):
    _np.long = _np.int64; _np.ulong = _np.uint64
from vllm import LLM, SamplingParams
llm = LLM(model=a.llm, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, seed=0)
sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1500, seed=0)
sp0 = SamplingParams(temperature=0, max_tokens=8)
KW = {"chat_template_kwargs": {"enable_thinking": False}}


def gen(prompts, params=sp):
    out = llm.chat([[{"role": "user", "content": p}] for p in prompts], params, use_tqdm=False, **KW)
    return [o.outputs[0].text.strip() for o in out]


def clean(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    return re.sub(r"^(here is|modified response|rewritten response)[^\n]*:\s*\n", "", t, flags=re.I).strip()


Q = lambda s: s[1]["question"]
PARA = ("Rewrite the following answer about an image using different wording and sentence structure. Keep every factual "
        "claim exactly the same (no claim added, removed or changed) and keep roughly the same length. Output only the "
        "rewritten answer.\n\nQuestion: {q}\n\nAnswer:\n{r}")
VIS = ("Here is an answer about an image. Pick THREE different short phrases in it (1-6 words each, copied exactly, not "
       "overlapping) that each state a concrete visual detail (a colour, a count, a position, an object's identity, or text "
       "visible in the image), and for each give a replacement that states a different but plausible value. Reply with "
       "JSON only: [{{\"original\": \"...\", \"replacement\": \"...\"}}, ...]\n\nQuestion: {q}\n\nAnswer:\n{r}")
FLAW = ("Here is an answer about an image. Write ONE extra sentence that directly contradicts a specific earlier statement "
        "of the answer, so that any reader notices the inconsistency without seeing the image. Reply with the sentence only."
        "\n\nAnswer:\n{r}")
SAME = ("Do text 1 and text 2 make exactly the same factual claims (wording may differ)? Answer yes or no.\n\n"
        "Text 1:\n{x}\n\nText 2:\n{y}")
CHG = ("In a description of an image, the phrase \"{o}\" is replaced by \"{n}\". Does this change what is claimed to be "
       "visible in the image (not just the wording)? Answer yes or no.")


def edits(text, raw):
    """two valid single-span substitutions with disjoint spans, or (None, None)"""
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        L = json.loads(m.group(0))
    except Exception:
        return None, None
    ok = []
    for d in L:
        try:
            o, n = d["original"].strip(), d["replacement"].strip()
        except Exception:
            continue
        if o and n and o != n and text.count(o) == 1 and all(o not in p and p not in o for p, _ in ok):
            ok.append((o, n))
    return (ok[0], ok[1]) if len(ok) >= 2 else (None, None)


P = [clean(t) for t in gen([PARA.format(q=Q(s), r=s[2]) for s in src])]
EE = [edits(s[2], t) for s, t in zip(src, gen([VIS.format(q=Q(s), r=s[2]) for s in src]))]
E1 = [(s[2].replace(*e[0]), e[0]) if e[0] else (None, None) for s, e in zip(src, EE)]
E2 = [(s[2].replace(*e[1]), e[1]) if e[1] else (None, None) for s, e in zip(src, EE)]
W1 = [e[0] or "" for e in E1]; W2 = [e[0] or "" for e in E2]
F = [s[2].rstrip() + " " + clean(t).split("\n")[0].strip() for s, t in zip(src, gen([FLAW.format(r=s[2]) for s in src]))]
yes = lambda ts: [t.strip().lower().startswith("yes") for t in ts]
c_p = yes(gen([SAME.format(x=s[2], y=p) for s, p in zip(src, P)], sp0))
c_1 = yes(gen([CHG.format(o=e[1][0], n=e[1][1]) if e[1] else "Answer no." for e in E1], sp0))
c_2 = yes(gen([CHG.format(o=e[1][0], n=e[1][1]) if e[1] else "Answer no." for e in E2], sp0))
c_12 = [bool(e1[1] and e2[1]) for e1, e2 in zip(E1, E2)]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out + ".gen.jsonl", "w") as g:      # every generation and check, for auditing
    for k, s in enumerate(src):
        g.write(json.dumps({"base": s[1]["id"], "R": s[2], "P": P[k], "W1": W1[k], "W2": W2[k], "F": F[k],
                            "E1": E1[k][1], "E2": E2[k][1], "ok": [c_p[k], c_1[k], c_2[k], c_12[k]]}) + "\n")
f = open(a.out, "w"); kept = 0
for k, s in enumerate(src):
    ok = c_p[k] and c_1[k] and c_2[k] and c_12[k] and s[2].strip() not in (P[k].strip(), W1[k].strip(), F[k].strip())
    if not ok:
        continue
    kept += 1
    base = s[1]
    for typ, x, y, lab in (("TG", s[2], P[k], "tie"), ("TB", W1[k], W2[k], "tie"), ("DV", s[2], W1[k], 0), ("DT", s[2], F[k], 0)):
        iid = f"synth_tie/{typ}/{base['id']}"
        flip = zlib.crc32(iid.encode()) % 2 == 1
        cand = [y, x] if flip else [x, y]
        label = lab if lab == "tie" else (1 if flip else 0)
        f.write(json.dumps({"id": iid, "dataset": "synth_tie", "subset": typ, "kind": "resp", "question": base["question"],
                            "context": base["context"], "cand": cand, "criterion": base.get("criterion", ""),
                            "label": label, "meta": {"base": base["id"], "src": s[0]}}) + "\n")
print(f"kept {kept}/{len(src)} bases; checks: para {sum(c_p)}, w1 {sum(c_1)}, w2 {sum(c_2)}, w1!=w2 {sum(c_12)}", flush=True)
print("SYNTH_DONE")
