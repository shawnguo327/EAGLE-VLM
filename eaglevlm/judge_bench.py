"""Unified VLM-as-judge harness (EAGLE protocol) for pairwise judge benchmarks.

Item schema (one JSON per line in items/<dataset>.jsonl):
  id, dataset, subset, kind ("resp" | "img"),
  question   : user question (resp) or generation prompt / edit instruction (img)
  context    : list of image paths shown before the candidates (resp: the question images; img-edit: source image)
  cand       : [A, B]  -> resp: two response strings; img: two image paths
  criterion  : short phrase describing what "better" means
  label      : 0 (A better) | 1 (B better) | "tie"
Conditions: bin (A/B, both orders), score (1-9 pointwise per candidate). Readout = first-token logits.
Vision probes (pairwise, both orders): bin_blind = question/context images removed (resp items only; measures
reliance on the language prior), bin_lowres = every image downsampled to LOWRES px on its long side (measures
dependence on fine visual detail).
"""
import argparse, json, os, pathlib, time, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from PIL import Image

ROOT = pathlib.Path(os.environ.get("EAGLEVLM_DATA", os.getcwd()))  # data dir: items/, images/
PP = pathlib.Path("/media/shawn/C/papers/VLM/place-pulse-2.0/models")
MODELS = {"qwen3vl8b": PP / "Qwen3-VL-8B-Instruct", "qwen35_9b": PP / "Qwen3.5-9B",
          "qwen25vl7b": pathlib.Path("/media/shawn/C/nyserver-files/mntstorage/models/Qwen2.5-VL-7B-Instruct"),
          "qwen3vl32b": PP / "Qwen3-VL-32B-Instruct", "internvl35_8b": PP / "InternVL3_5-8B-HF"}
MAXSIDE = 768
LOWRES = 112
MAXCHARS = 4000

BIN_RESP = ("Question: {q}\n\n[Response A]\n{a}\n\n[Response B]\n{b}\n\n"
            "You are judging the two responses to the question about the image(s). Criterion: {crit} "
            "Which response is better? Answer with exactly one letter: A or B.")
SCORE_RESP = ("Question: {q}\n\n[Response]\n{a}\n\nCriterion: {crit} Rate this response on an integer scale "
              "from 1 (very poor) to 9 (excellent). Answer with exactly one digit.")
BIN_IMG = ("{q}\nCriterion: {crit} Which image is better, Image A or Image B? "
           "Answer with exactly one letter: A or B.")
SCORE_IMG = ("{q}\nCriterion: {crit} Rate this image on an integer scale from 1 (very poor) to 9 (excellent). "
             "Answer with exactly one digit.")


def load_img(p, maxside=MAXSIDE):
    im = Image.open(ROOT / p if not os.path.isabs(p) else p).convert("RGB")
    w, h = im.size
    s = maxside / max(w, h)
    if s < 1:
        im = im.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)
    return im


def clip(t):
    t = t or ""
    return t if len(t) <= MAXCHARS else t[:MAXCHARS] + " …[truncated]"


def build(r, cond, order):
    c = []
    ms = LOWRES if cond == "bin_lowres" else MAXSIDE
    load = lambda p: load_img(p, ms)
    if cond.startswith("bin"):
        cond = "bin" if cond != "bin_blind" else "blind"
    if r["kind"] == "resp":
        if cond == "blind":
            c.append({"type": "text", "text": "[The image(s) for this question are not shown.]"})
            cond = "bin"
        else:
            for p in r["context"]:
                c.append({"type": "image", "image": load(p)})
        a, b = (r["cand"][0], r["cand"][1]) if order in ("AB", "A") else (r["cand"][1], r["cand"][0])
        if cond == "bin":
            c.append({"type": "text", "text": BIN_RESP.format(q=clip(r["question"]), a=clip(a), b=clip(b), crit=r["criterion"])})
        else:
            x = r["cand"][0] if order == "A" else r["cand"][1]
            c.append({"type": "text", "text": SCORE_RESP.format(q=clip(r["question"]), a=clip(x), crit=r["criterion"])})
    else:
        q = r["question"]
        if r["context"]:
            c.append({"type": "text", "text": "Source image:"})
            for p in r["context"]:
                c.append({"type": "image", "image": load(p)})
        if cond == "bin":
            pair = r["cand"] if order == "AB" else r["cand"][::-1]
            for i, p in enumerate(pair):
                c += [{"type": "text", "text": f"Image {'AB'[i]}:"}, {"type": "image", "image": load(p)}]
            c.append({"type": "text", "text": BIN_IMG.format(q=q, crit=r["criterion"])})
        else:
            p = r["cand"][0] if order == "A" else r["cand"][1]
            c += [{"type": "text", "text": "Candidate image:"}, {"type": "image", "image": load_img(p)},
                  {"type": "text", "text": SCORE_IMG.format(q=q, crit=r["criterion"])}]
    return [{"role": "user", "content": c}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl8b")
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default="bin,score")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    mp = MODELS[a.model]
    # InternVL: one 448px tile per image (no dynamic tiling), keeping the visual token budget comparable to Qwen.
    # Must be set at construction: per-call processor_kwargs are ignored in batched apply_chat_template.
    proc = AutoProcessor.from_pretrained(mp, **({"crop_to_patches": False} if "internvl" in a.model else {}))
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(mp, dtype=torch.bfloat16, device_map={"": "cuda:0"},
                                                        attn_implementation="sdpa").eval()
    tok = proc.tokenizer
    ids = {c: tok.encode(c, add_special_tokens=False)[0] for c in ["A", "B"] + [str(i) for i in range(1, 10)]}
    rows = [json.loads(l) for l in open(a.items)]
    if a.limit:
        rows = rows[:a.limit]
    jobs = []
    for r in rows:
        for c in a.conds.split(","):
            if c == "bin_blind" and r["kind"] != "resp":
                continue
            for o in (("AB", "BA") if c.startswith("bin") else ("A", "B")):
                jobs.append((r, c, o))
    outp = pathlib.Path(a.out)
    done = set()
    if outp.exists():
        for l in open(outp):
            d = json.loads(l)
            if "lp" in d:  # error rows (OOM, missing image) are retried on the next run
                done.add((d["id"], d["cond"], d["order"]))
    jobs = [j for j in jobs if (j[0]["id"], j[1], j[2]) not in done]
    # sort by rough size so batches are homogeneous
    jobs.sort(key=lambda j: (j[1], len(json.dumps(j[0]["cand"]))))
    print("jobs", len(jobs), flush=True)
    t0 = time.time()
    with open(outp, "a", buffering=1) as f:
        s = 0
        bs = a.bs
        while s < len(jobs):
            B = jobs[s:s + bs]
            try:
                msgs = [build(*j) for j in B]
                inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt", padding=True, enable_thinking=False).to("cuda:0")
                with torch.inference_mode():
                    lp = torch.log_softmax(model(**inp).logits[:, -1, :].float(), -1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    for (r, c, o) in B:
                        f.write(json.dumps({"id": r["id"], "cond": c, "order": o, "error": "oom"}) + "\n")
                    s += 1
                    bs = a.bs
                else:
                    bs = max(1, bs // 2)
                continue
            except Exception as e:
                for (r, c, o) in B:
                    f.write(json.dumps({"id": r["id"], "cond": c, "order": o, "error": repr(e)[:200]}) + "\n")
                s += len(B)
                continue
            for (r, c, o), row in zip(B, lp):
                keys = ["A", "B"] if c.startswith("bin") else [str(i) for i in range(1, 10)]
                raw = {k: row[ids[k]].item() for k in keys}
                f.write(json.dumps({"id": r["id"], "cond": c, "order": o, "lp": raw,
                                    "mass": float(sum(torch.exp(torch.tensor(list(raw.values())))))}) + "\n")
            s += len(B)
            if (s // max(bs, 1)) % 25 == 0:
                print(s, len(jobs), round(time.time() - t0), flush=True)
            bs = min(a.bs, bs * 2) if bs < a.bs else bs
    print("DONE", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
