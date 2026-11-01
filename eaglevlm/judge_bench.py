"""Unified VLM-as-judge harness (EAGLE protocol) for pairwise judge benchmarks.

Item schema (one JSON per line in items/<dataset>.jsonl):
  id, dataset, subset, kind ("resp" | "img"),
  question   : user question (resp) or generation prompt / edit instruction (img)
  context    : list of image paths shown before the candidates (resp: the question images; img-edit: source image)
  cand       : [A, B]  -> resp: two response strings; img: two image paths
  criterion  : short phrase describing what "better" means
  label      : 0 (A better) | 1 (B better) | "tie"
Readout = first-token log-probs over the answer keys. Conditions:
  pairwise (both orders AB/BA)
    bin        A/B
    tri        A/B/T  (T = essentially equal quality; an explicit channel for "the two are close")
    str7       1..7   (1 = A much better ... 4 = about equal ... 7 = B much better)
    bin_blind  A/B without the question images (resp items only; reliance on the language prior)
    bin_lowres A/B with every image at LOWRES px (dependence on fine detail)
    bin_hires  A/B with images at HIRES px (more visual evidence; use --tiling for InternVL)
    bin_desc   A/B with generated detailed descriptions added to the images (--desc; see describe.py)
    tri_p1..3  A/B/T with three paraphrased instructions (prompt-nuisance ensemble)
    tri_jpeg   A/B/T with every image JPEG-recompressed at quality 30 (image-nuisance)
    tri_crop   A/B/T with every image center-cropped to 90% (image-nuisance)
    tri_blind  A/B/T without the question images (resp items; does the verdict use the image at all?)
    tri_wrong  A/B/T with the question images of another item of the same dataset (resp items; image-swap control)
    tri_zoom   A/B/T plus close-ups of one region of source/A/B (edit items with a source image). Region = the window
               the edits changed most (aligned edits) or where A and B differ most (label-free, fixed rule). Close-ups from the native image.
    tri_zoomr  same, region of the same size at a random position (seeded by the id): untargeted control
    tri_zoomlo same region as tri_zoom, but cropped from the 768-px image the judge already sees (no new pixels)
  pointwise (per candidate A/B)
    score      1..9
Item kinds: resp (question images + two text responses), img (two images), text (no images; text-domain judge sets,
used to compare VLM judges with text LLM judges under the same protocol; --text-llm loads a text-only causal LM).
Run modes: single (--items/--out) or a plan file (--plan, JSON list of {items, out}); --shard/--nshard split the
items by a hash of the id so several GPUs can share a dataset. Resumable: finished (id, cond, order) rows are skipped.
--hidden 0.25,0.5,0.75,1.0 also saves last-token hidden states at those layer fractions to <out>.hid/*.npz.
"""
import argparse, glob, json, os, pathlib, time, zlib, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from PIL import Image

ROOT = pathlib.Path(os.environ.get("EAGLEVLM_DATA", os.getcwd()))  # data dir: items/, images/
PP = pathlib.Path("/media/shawn/C/papers/VLM/place-pulse-2.0/models")
MODELS = {"qwen3vl8b": PP / "Qwen3-VL-8B-Instruct", "qwen35_9b": PP / "Qwen3.5-9B",
          "qwen25vl7b": pathlib.Path("/media/shawn/C/nyserver-files/mntstorage/models/Qwen2.5-VL-7B-Instruct"),
          "qwen3vl32b": PP / "Qwen3-VL-32B-Instruct", "internvl35_8b": PP / "InternVL3_5-8B-HF"}
MAXSIDE, LOWRES, HIRES = 768, 112, 1536
MAXCHARS = 4000
PAIR = {"bin", "tri", "str7", "bin_blind", "bin_lowres", "bin_hires", "bin_desc", "tri_p1", "tri_p2", "tri_p3",
        "tri_jpeg", "tri_crop", "tri_blind", "tri_wrong", "tri_zoom", "tri_zoomr", "tri_zoomlo"}
IMG_ONLY = {"tri_jpeg", "tri_crop", "bin_lowres", "bin_hires", "bin_desc"}      # no-ops on text items: skipped there
RESP_ONLY = {"bin_blind", "tri_blind", "tri_wrong"}                              # need question images + text responses
ZOOM = {"tri_zoom", "tri_zoomr", "tri_zoomlo"}                                   # need an edit source image
ZOOM_FRAC, ZOOM_SIDE, ALIGN_TOL = 0.4, 512, 40.0        # window = 40% of width and height; close-ups shown at 512 px max side


def applies(c, r):
    if c in RESP_ONLY:
        return r["kind"] == "resp" and bool(r["context"])
    if c in ZOOM:
        if c == "tri_zoomlo" and r["kind"] == "img" and r["context"]:   # identical to tri_zoom if nothing was downscaled
            return any(max(Image.open(ROOT / p).size) > MAXSIDE for p in [r["context"][0], *r["cand"]])
        return r["kind"] == "img" and bool(r["context"])
    if c in IMG_ONLY:
        return r["kind"] != "text"
    return True


_BOX = {}


def zoom_box(r, random_pos=False):
    """Normalized (x0, y0, x1, y1). Targeted: window maximizing |A - B| on a
    128x128 grid (all images resized to the grid, so boxes are in relative coordinates). Uses no label.
    If both candidates are pixel-aligned with the source (same aspect, mean |edit - source| < ALIGN_TOL): window of
    largest |A - src| + |B - src| (the edited region). Otherwise (free-form edits, e.g. MMRB2): largest |A - B|."""
    import numpy as np
    n = 128; w = int(round(ZOOM_FRAC * n))
    if random_pos:
        g = np.random.default_rng(zlib.crc32(r["id"].encode()))
        x, y = g.integers(0, n - w + 1, 2)
        return x / n, y / n, (x + w) / n, (y + w) / n
    if r["id"] not in _BOX:
        arr = lambda p: np.asarray(Image.open(ROOT / p).convert("RGB").resize((n, n), Image.BILINEAR), dtype=np.float32)
        A, B = arr(r["cand"][0]), arr(r["cand"][1])
        sp = r["context"][0]; S0 = arr(sp)
        ar = lambda p: (lambda z: z[0] / z[1])(Image.open(ROOT / p).size)
        aligned = all(abs(ar(c) / ar(sp) - 1) < .05 for c in r["cand"]) and \
            max(np.abs(A - S0).mean(), np.abs(B - S0).mean()) < ALIGN_TOL
        # aligned (local) edits: where the edits changed the source; otherwise where the two candidates differ
        d = (np.abs(A - S0) + np.abs(B - S0)).mean(-1) if aligned else np.abs(A - B).mean(-1)
        I = np.pad(d.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        S = I[w:, w:] - I[:-w, w:] - I[w:, :-w] + I[:-w, :-w]          # window sums, shape (n-w+1, n-w+1)
        y, x = np.unravel_index(int(S.argmax()), S.shape)
        _BOX[r["id"]] = (x / n, y / n, (x + w) / n, (y + w) / n)
    return _BOX[r["id"]]


def closeup(p, box, lo=False):
    im = Image.open(ROOT / p if not os.path.isabs(p) else p).convert("RGB")
    if lo:
        im = load_img(p)                                  # exactly the image the judge sees
    W, H = im.size
    c = im.crop((int(box[0] * W), int(box[1] * H), max(int(box[0] * W) + 1, int(box[2] * W)),
                 max(int(box[1] * H) + 1, int(box[3] * H))))
    s = ZOOM_SIDE / max(c.size)
    return c.resize((max(28, int(c.size[0] * s)), max(28, int(c.size[1] * s))), Image.BICUBIC)


def keys_of(c):
    if c.startswith("tri"):
        return ["A", "B", "T"]
    return {"str7": [str(i) for i in range(1, 8)], "score": [str(i) for i in range(1, 10)]}.get(c, ["A", "B"])

ASK_RESP = {
    "bin": "You are judging the two responses to the question about the image(s). Criterion: {crit} "
           "Which response is better? Answer with exactly one letter: A or B.",
    "tri": "You are judging the two responses to the question about the image(s). Criterion: {crit} "
           "Which response is better? If the two are of essentially equal quality, answer T. "
           "Answer with exactly one letter: A, B or T.",
    "str7": "You are judging the two responses to the question about the image(s). Criterion: {crit} "
            "Compare them on a 7-point scale: 1 = A is much better, 2 = A is better, 3 = A is slightly better, "
            "4 = they are about equal, 5 = B is slightly better, 6 = B is better, 7 = B is much better. "
            "Answer with exactly one digit."}
ASK_IMG = {
    "bin": "{q}\nCriterion: {crit} Which image is better, Image A or Image B? Answer with exactly one letter: A or B.",
    "tri": "{q}\nCriterion: {crit} Which image is better, Image A or Image B? If the two are of essentially equal "
           "quality, answer T. Answer with exactly one letter: A, B or T.",
    "str7": "{q}\nCriterion: {crit} Compare Image A and Image B on a 7-point scale: 1 = A is much better, "
            "2 = A is better, 3 = A is slightly better, 4 = they are about equal, 5 = B is slightly better, "
            "6 = B is better, 7 = B is much better. Answer with exactly one digit."}
# paraphrased tri instructions (same answer keys) for the prompt-nuisance ensemble
ASK_RESP |= {
    "tri_p1": "Two assistants answered the question about the image(s). Criterion: {crit} Decide which answer is better. "
              "Reply A if Response A is better, B if Response B is better, or T if neither is clearly better. Reply with one letter.",
    "tri_p2": "Evaluate both responses against this criterion: {crit} If one response is clearly superior, output its letter "
              "(A or B). If they are of comparable quality, output T. Output a single letter.",
    "tri_p3": "As an impartial judge, compare Response A and Response B for the question about the image(s). Criterion: {crit} "
              "Answer A, B, or T (tie). One letter only."}
ASK_IMG |= {
    "tri_p1": "{q}\nCriterion: {crit} Decide which image is better. Reply A if Image A is better, B if Image B is better, or T "
              "if neither is clearly better. Reply with one letter.",
    "tri_p2": "{q}\nEvaluate both images against this criterion: {crit} If one image is clearly superior, output its letter "
              "(A or B). If they are of comparable quality, output T. Output a single letter.",
    "tri_p3": "{q}\nAs an impartial judge, compare Image A and Image B. Criterion: {crit} Answer A, B, or T (tie). One letter only."}
SCORE_RESP = ("Question: {q}\n\n[Response]\n{a}\n\nCriterion: {crit} Rate this response on an integer scale "
              "from 1 (very poor) to 9 (excellent). Answer with exactly one digit.")
SCORE_IMG = ("{q}\nCriterion: {crit} Rate this image on an integer scale from 1 (very poor) to 9 (excellent). "
             "Answer with exactly one digit.")


def load_img(p, maxside=MAXSIDE, perturb=None):
    im = Image.open(ROOT / p if not os.path.isabs(p) else p).convert("RGB")
    if perturb == "crop":
        w, h = im.size; dw, dh = int(w * .05), int(h * .05); im = im.crop((dw, dh, w - dw, h - dh))
    w, h = im.size
    s = maxside / max(w, h)
    if s < 1:
        im = im.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)
    if perturb == "jpeg":
        import io
        b = io.BytesIO(); im.save(b, "JPEG", quality=30); b.seek(0); im = Image.open(b).convert("RGB")
    return im


def clip(t):
    t = t or ""
    return t if len(t) <= MAXCHARS else t[:MAXCHARS] + " …[truncated]"


def build(r, cond, order, desc=None):
    """Chat messages for one call. `desc` = {id: {"ctx": [str], "cand": [str, str]}} for bin_desc."""
    c = []
    ms = {"bin_lowres": LOWRES, "bin_hires": HIRES}.get(cond, MAXSIDE)
    pert = {"tri_jpeg": "jpeg", "tri_crop": "crop"}.get(cond)
    load = lambda p: load_img(p, ms, pert)
    ask = cond if cond in ("tri", "str7", "tri_p1", "tri_p2", "tri_p3") else "tri" if cond.startswith("tri") else "bin"
    D = desc.get(r["id"]) if (cond == "bin_desc" and desc is not None) else None
    if cond == "bin_desc" and D is None:
        raise KeyError(f"no description for {r['id']}")
    if r["kind"] in ("resp", "text"):
        if r["kind"] == "text":
            pass
        elif cond in ("bin_blind", "tri_blind"):
            c.append({"type": "text", "text": "[The image(s) for this question are not shown.]"})
        else:
            for p in (r["_wrong"] if cond == "tri_wrong" else r["context"]):
                c.append({"type": "image", "image": load(p)})
        if D is not None and D.get("ctx"):
            c.append({"type": "text", "text": "Detailed description of the image(s):\n" + clip(" ".join(D["ctx"]))})
        if cond in PAIR:
            a, b = (r["cand"][0], r["cand"][1]) if order == "AB" else (r["cand"][1], r["cand"][0])
            t = ASK_RESP[ask].format(crit=r["criterion"])
            if r["kind"] == "text":
                t = t.replace("to the question about the image(s)", "to the user's question").replace(
                    "the question about the image(s)", "the user's question").replace("Two assistants answered the question",
                                                                                     "Two assistants answered the user's question")
            c.append({"type": "text", "text": f"Question: {clip(r['question'])}\n\n[Response A]\n{clip(a)}\n\n"
                                              f"[Response B]\n{clip(b)}\n\n" + t})
        else:
            x = r["cand"][0] if order == "A" else r["cand"][1]
            c.append({"type": "text", "text": SCORE_RESP.format(q=clip(r["question"]), a=clip(x), crit=r["criterion"])})
    else:
        q = r["question"]
        if r["context"]:
            c.append({"type": "text", "text": "Source image:"})
            for p in r["context"]:
                c.append({"type": "image", "image": load(p)})
        if cond in PAIR:
            idx = (0, 1) if order == "AB" else (1, 0)
            for i, k in enumerate(idx):
                c += [{"type": "text", "text": f"Image {'AB'[i]}:"}, {"type": "image", "image": load(r["cand"][k])}]
                if D is not None:
                    c.append({"type": "text", "text": f"Detailed description of Image {'AB'[i]}: {clip(D['cand'][k])}"})
            if cond in ZOOM:
                box = zoom_box(r, random_pos=cond == "tri_zoomr")
                c.append({"type": "text", "text": "Close-ups of the same region in each image. Source (close-up):"})
                c.append({"type": "image", "image": closeup(r["context"][0], box, cond == "tri_zoomlo")})
                for i, k in enumerate(idx):
                    c += [{"type": "text", "text": f"Image {'AB'[i]} (close-up):"},
                          {"type": "image", "image": closeup(r["cand"][k], box, cond == "tri_zoomlo")}]
            c.append({"type": "text", "text": ASK_IMG[ask].format(q=q, crit=r["criterion"])})
        else:
            p = r["cand"][0] if order == "A" else r["cand"][1]
            c += [{"type": "text", "text": "Candidate image:"}, {"type": "image", "image": load_img(p)},
                  {"type": "text", "text": SCORE_IMG.format(q=q, crit=r["criterion"])}]
    return [{"role": "user", "content": c}]


def in_shard(i, shard, nshard):
    return nshard <= 1 or zlib.crc32(i.encode()) % nshard == shard


class TextProc:
    """Processor-like adapter for text-only causal LMs: flattens the text parts of each message."""
    def __init__(self, tok):
        self.tokenizer = tok

    def apply_chat_template(self, msgs, **kw):
        flat = [[{"role": m["role"], "content": "\n\n".join(x["text"] for x in m["content"] if x["type"] == "text")}
                 for m in ms] for ms in msgs]
        return self.tokenizer.apply_chat_template(flat, **kw)

    def batch_decode(self, *a, **kw):
        return self.tokenizer.batch_decode(*a, **kw)

    def __call__(self, text=None, images=None, **kw):
        return self.tokenizer(text, **kw)


def load_model(name, path, device_map="cuda:0", tiling=False, text_llm=False):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    mp = path or MODELS[name]
    dm = {"": "cuda:0"} if device_map == "cuda:0" else device_map
    if text_llm:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(mp); tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16, device_map=dm, attn_implementation="sdpa").eval()
        return TextProc(tok), model
    internvl = "internvl" in (name + str(mp)).lower()
    # InternVL: by default one 448px tile per image (no dynamic tiling), keeping the visual token budget comparable
    # to Qwen. Must be set at construction: per-call processor_kwargs are ignored in batched apply_chat_template.
    proc = AutoProcessor.from_pretrained(mp, **({"crop_to_patches": bool(tiling)} if internvl else {}))
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        mp, dtype=torch.bfloat16, device_map=dm, attn_implementation="sdpa").eval()
    return proc, model


class Progress:
    def __init__(self, path, total, model):
        self.path, self.total, self.model, self.t0, self.last = path, total, model, time.time(), 0

    def write(self, done, status="running", force=False, current=""):
        if not self.path or (not force and time.time() - self.last < 15):
            return
        self.last = el = time.time(); el -= self.t0
        rate = done / el if el > 0 else 0
        d = {"model": self.model, "done": done, "total": self.total, "elapsed_s": round(el), "rate": round(rate, 2),
             "eta_s": round((self.total - done) / rate) if rate > 0 else None, "status": status, "current": current,
             "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = self.path + ".tmp"; json.dump(d, open(tmp, "w")); os.replace(tmp, self.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl8b")
    ap.add_argument("--model-path", default=None, help="overrides MODELS[model]")
    ap.add_argument("--items"); ap.add_argument("--out")
    ap.add_argument("--plan", help="JSON file: list of {items, out}; overrides --items/--out")
    ap.add_argument("--conds", default="bin,score")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--device-map", default="cuda:0", help="cuda:0 or auto (spread a large model over visible GPUs)")
    ap.add_argument("--tiling", action="store_true", help="InternVL dynamic tiling (for bin_hires)")
    ap.add_argument("--text-llm", action="store_true", help="text-only causal LM (text items only)")
    ap.add_argument("--desc", default=None, help="glob of describe.py outputs (for bin_desc)")
    ap.add_argument("--hidden", default="", help="layer fractions, e.g. 0.25,0.5,0.75,1.0 -> <out>.hid/*.npz")
    ap.add_argument("--progress", default=None, help="write a progress JSON here every ~15 s")
    a = ap.parse_args()
    plan = json.load(open(a.plan)) if a.plan else [{"items": a.items, "out": a.out}]
    conds = a.conds.split(",")
    desc = None
    if "bin_desc" in conds:
        desc = {}
        for f in glob.glob(a.desc):
            for l in open(f):
                d = json.loads(l)
                if "ctx" in d:
                    desc[d["id"]] = d
        print("descriptions", len(desc), flush=True)

    # ---- jobs over all plan entries (resume: skip finished rows)
    jobs = []
    for k, e in enumerate(plan):
        rows = [json.loads(l) for l in open(e["items"])]
        if "tri_wrong" in conds:      # donor = next item (crc32 order) of the same file with different question images
            pool = sorted([r for r in rows if r["kind"] == "resp" and r["context"]], key=lambda r: zlib.crc32(r["id"].encode()))
            for j, r in enumerate(pool):
                for t in range(1, len(pool)):
                    dn = pool[(j + t) % len(pool)]
                    if dn["context"] != r["context"]:
                        r["_wrong"] = dn["context"]; break
        rows = [r for r in rows if in_shard(r["id"], a.shard, a.nshard)]
        if a.limit:
            rows = rows[:a.limit]
        done = set()
        if os.path.exists(e["out"]):
            for l in open(e["out"]):
                d = json.loads(l)
                if "lp" in d:  # error rows (OOM, missing image) are retried on the next run
                    done.add((d["id"], d["cond"], d["order"]))
        for r in rows:
            for c in conds:
                if not applies(c, r) or (c == "tri_wrong" and "_wrong" not in r):
                    continue
                if a.text_llm and r["kind"] != "text":
                    continue
                if c == "bin_desc" and r["id"] not in desc:
                    continue
                for o in (("AB", "BA") if c in PAIR else ("A", "B")):
                    if (r["id"], c, o) not in done:
                        jobs.append((k, r, c, o))
    jobs.sort(key=lambda j: (j[2], len(json.dumps(j[1]["cand"]))))  # homogeneous batches
    print("jobs", len(jobs), flush=True)
    prog = Progress(a.progress, len(jobs), a.model)
    prog.write(0, force=True)
    if not jobs:
        prog.write(0, "done", force=True); print("DONE 0", flush=True); return

    proc, model = load_model(a.model, a.model_path, a.device_map, a.tiling, a.text_llm)
    tok = proc.tokenizer
    ids = {c: tok.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "T"] + [str(i) for i in range(1, 10)]}
    fr = [float(x) for x in a.hidden.split(",") if x]
    hid_buf, hid_chunk = [], 0
    if fr:
        for e in plan:
            os.makedirs(e["out"] + ".hid", exist_ok=True)
            hid_chunk = max(hid_chunk, len(glob.glob(e["out"] + ".hid/*.npz")))

    def flush_hidden():
        nonlocal hid_buf, hid_chunk
        import numpy as np
        by = {}
        for k, i, c, o, h in hid_buf:
            by.setdefault(k, []).append((i, c, o, h))
        for k, L in by.items():
            np.savez(f"{plan[k]['out']}.hid/{hid_chunk:05d}.npz", id=np.array([x[0] for x in L]),
                     cond=np.array([x[1] for x in L]), order=np.array([x[2] for x in L]),
                     layers=np.array(fr), h=np.stack([x[3] for x in L]))
        hid_buf, hid_chunk = [], hid_chunk + 1

    files = {k: open(e["out"], "a", buffering=1) for k, e in enumerate(plan)}
    t0 = time.time(); s = 0; bs = a.bs; dev = "cuda:0"; nerr = 0
    while s < len(jobs):
        B = jobs[s:s + bs]
        try:
            msgs = [build(r, c, o, desc) for (_, r, c, o) in B]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True, enable_thinking=False).to(dev)
            with torch.inference_mode():
                out = model(**inp, output_hidden_states=bool(fr))
                lp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
                if fr:
                    hs = out.hidden_states; nl = len(hs) - 1
                    li = [max(1, min(nl, round(f * nl))) for f in fr]
                    H = torch.stack([hs[l][:, -1, :].float() for l in li], 1).half().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                for (k, r, c, o) in B:
                    files[k].write(json.dumps({"id": r["id"], "cond": c, "order": o, "error": "oom"}) + "\n")
                nerr += len(B); s += 1; bs = a.bs
            else:
                bs = max(1, bs // 2)
            continue
        except Exception as e:
            for (k, r, c, o) in B:
                files[k].write(json.dumps({"id": r["id"], "cond": c, "order": o, "error": repr(e)[:200]}) + "\n")
            nerr += len(B); s += len(B)
            continue
        for n, ((k, r, c, o), row) in enumerate(zip(B, lp)):
            raw = {key: row[ids[key]].item() for key in keys_of(c)}
            files[k].write(json.dumps({"id": r["id"], "cond": c, "order": o, "lp": raw,
                                       "mass": float(sum(torch.exp(torch.tensor(list(raw.values())))))}) + "\n")
            if fr:
                hid_buf.append((k, r["id"], c, o, H[n]))
        if fr and len(hid_buf) >= 2048:
            flush_hidden()
        s += len(B)
        if (s // max(bs, 1)) % 25 == 0:
            print(s, len(jobs), round(time.time() - t0), flush=True)
        prog.write(s, current=f"{B[-1][2]} {plan[B[-1][0]]['items']}")
        bs = min(a.bs, bs * 2) if bs < a.bs else bs
    if fr and hid_buf:
        flush_hidden()
    prog.write(len(jobs), "done", force=True)
    print("DONE", round(time.time() - t0), "errors", nerr, flush=True)
    if nerr:
        raise SystemExit(3)  # the scheduler retries the task; the resume step redoes only the error rows


if __name__ == "__main__":
    main()
