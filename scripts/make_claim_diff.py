"""Real-data transfer of the level decomposition: where do two real responses disagree about the image?
A text LLM (Qwen3-8B, no thinking) lists up to 4 points on which response 1 and response 2 make different claims
about the image, each as two mutually exclusive statements (s1 from response 1, s2 from response 2; if one response is
silent, its side is the negation of the other's claim).  Each point becomes a perception item (claim_check): "which of
these two statements is accurate about the image?" that the VLM judges answer with the image (and blind as control).
Pairs with no differing visual claim are recorded with n_points = 0 (candidates for 'equivalent content').
usage (vllm env): python make_claim_diff.py --llm QWEN3_8B --ds mllm_judge,vlrb_s32,mmrb_s32 --root DATA --out items/claim_check.jsonl"""
import argparse, json, os, re, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--llm", required=True); ap.add_argument("--ds", required=True)
ap.add_argument("--root", default="/media/shawn/C/papers/VLM/judgebench_local"); ap.add_argument("--out", required=True)
ap.add_argument("--gpu-mem", type=float, default=0.38)
a = ap.parse_args()
P = ("Two AI responses answer the same question about an image (you cannot see the image).\n\nQuestion: {q}\n\n"
     "Response 1:\n{r1}\n\nResponse 2:\n{r2}\n\n"
     "List the points (at most 4, most important first) on which the two responses make DIFFERENT factual claims about "
     "what the image shows or what the answer is. Ignore wording, style, length and claims made by only one response "
     "unless the other response contradicts or omits something essential to the answer. For each point write two "
     "short, self-contained, mutually exclusive statements about the image: s1 = what response 1 claims, s2 = what "
     "response 2 claims (if one response is silent, use the negation of the other's claim). If the responses agree on "
     "all facts, return an empty list.\nReply with JSON only: [{{\"s1\": \"...\", \"s2\": \"...\"}}, ...]")
clip = lambda t, n=3000: (t or "")[:n]
items = []
for ds in a.ds.split(","):
    items += [json.loads(l) for l in open(f"{a.root}/items/{ds}.jsonl")]
items = [r for r in items if r["kind"] == "resp" and r.get("context")]
print(len(items), "pairs", flush=True)
import numpy as _np
if not hasattr(_np, "long"):
    _np.long = _np.int64; _np.ulong = _np.uint64
from vllm import LLM, SamplingParams
llm = LLM(model=a.llm, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, seed=0)
out = llm.chat([[{"role": "user", "content": P.format(q=clip(r["question"], 1500), r1=clip(r["cand"][0]), r2=clip(r["cand"][1]))}]
                for r in items], SamplingParams(temperature=0, max_tokens=700), use_tqdm=False,
               chat_template_kwargs={"enable_thinking": False})
f = open(a.out, "w"); g = open(a.out.replace(".jsonl", "_pairs.jsonl"), "w"); npts = 0
for r, o in zip(items, out):
    t = o.outputs[0].text
    m = re.search(r"\[.*\]", t, re.S)
    try:
        L = [d for d in json.loads(m.group(0)) if d.get("s1") and d.get("s2") and d["s1"].strip() != d["s2"].strip()][:4]
        ok = True
    except Exception:
        L, ok = [], False
    g.write(json.dumps({"id": r["id"], "ds": r["dataset"], "parsed": ok, "n_points": len(L), "points": L}) + "\n")
    for k, d in enumerate(L):
        iid = f"claim_check/{r['id']}/{k}"
        flip = zlib.crc32(iid.encode()) % 2 == 1
        s1, s2 = d["s1"].strip(), d["s2"].strip()
        f.write(json.dumps({"id": iid, "dataset": "claim_check", "subset": r["dataset"], "kind": "resp",
                            "question": "Which of these two statements is accurate about the image?", "context": r["context"],
                            "cand": [s2, s1] if flip else [s1, s2], "criterion": "Prefer the statement that is accurate about the image.",
                            "label": 1 if flip else 0,          # 'label' = index of s1 (the claim of the pair's response 1 = cand0)
                            "meta": {"pair": r["id"], "k": k, "s1_is_cand0_claim": True}}) + "\n")
        npts += 1
print(npts, "claim items", flush=True); print("CLAIMS_DONE")
