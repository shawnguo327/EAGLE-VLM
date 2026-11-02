"""Similarity of the two candidates of each pair (covariates for the tie-mechanism analysis, tie_mech.py sim).
Image pairs: DINOv2-L CLS cosine, CLIP-L image cosine, pixel L1 at 256x256 (grey), and for edit pairs the same
similarities of each candidate to the source image (did the edit change anything?).
Text pairs: BGE-large cosine, difflib character ratio, length ratio.
usage: python pair_sim.py --ds genai_arena,editreward,pp,mllm_judge,mtbench_human,arena55k --out paper_tables/pair_sim.jsonl"""
import argparse, difflib, json, os
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--roots", default="/media/shawn/C/papers/VLM/judgebench,/media/shawn/C/papers/VLM/judgebench_local")
ap.add_argument("--dino", default="facebook/dinov2-large")
ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
ap.add_argument("--bge", default="/media/shawn/C/papers/VLM/judge_models/bge-large-en-v1.5")
ap.add_argument("--out", required=True)
a = ap.parse_args()
ROOTS = a.roots.split(",")
dev = "cuda:0"


def find(p):
    for r in ROOTS:
        q = os.path.join(r, p)
        if os.path.exists(q):
            return q
    raise FileNotFoundError(p)


def load_items(ds):
    for r in ROOTS:
        p = f"{r}/items/{ds}.jsonl"
        if os.path.exists(p):
            return [json.loads(l) for l in open(p)]
    raise FileNotFoundError(ds)


done = set()
if os.path.exists(a.out):
    done = {(json.loads(l)["ds"], json.loads(l)["id"]) for l in open(a.out)}
f = open(a.out, "a")
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor, AutoTokenizer
M = {}


def img_models():
    if "dino" not in M:
        M["dino"] = (AutoImageProcessor.from_pretrained(a.dino), AutoModel.from_pretrained(a.dino, dtype=torch.float16).to(dev).eval())
        M["clip"] = (CLIPProcessor.from_pretrained(a.clip), CLIPModel.from_pretrained(a.clip, dtype=torch.float16).to(dev).eval())
    return M["dino"], M["clip"]


@torch.inference_mode()
def img_feats(paths):
    (dp, dm), (cp, cm) = img_models()
    ims = [Image.open(find(p)).convert("RGB") for p in paths]
    d = dm(**dp(images=ims, return_tensors="pt").to(dev, torch.float16)).last_hidden_state[:, 0]
    c = cm.get_image_features(**cp(images=ims, return_tensors="pt").to(dev, torch.float16))
    c = c if torch.is_tensor(c) else c.pooler_output
    g = [np.asarray(im.convert("L").resize((256, 256)), np.float32) / 255 for im in ims]
    return torch.nn.functional.normalize(d.float(), dim=-1), torch.nn.functional.normalize(c.float(), dim=-1), g


@torch.inference_mode()
def txt_emb(ts):
    if "bge" not in M:
        M["bge"] = (AutoTokenizer.from_pretrained(a.bge), AutoModel.from_pretrained(a.bge, dtype=torch.float16).to(dev).eval())
    tok, m = M["bge"]
    x = tok(ts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
    return torch.nn.functional.normalize(m(**x).last_hidden_state[:, 0].float(), dim=-1)


for ds in a.ds.split(","):
    n = 0
    for r in load_items(ds):
        if (ds, r["id"]) in done:
            continue
        row = {"ds": ds, "id": r["id"]}
        if r["kind"] == "img":
            paths = list(r["cand"]) + (list(r["context"][:1]) if r.get("context") else [])
            d, c, g = img_feats(paths)
            row.update(dino=float(d[0] @ d[1]), clip=float(c[0] @ c[1]), pix=float(np.abs(g[0] - g[1]).mean()))
            if len(paths) == 3:
                row.update(dino_src=[float(d[k] @ d[2]) for k in (0, 1)], clip_src=[float(c[k] @ c[2]) for k in (0, 1)],
                           pix_src=[float(np.abs(g[k] - g[2]).mean()) for k in (0, 1)])
        else:
            t0, t1 = r["cand"][0] or "", r["cand"][1] or ""
            e = txt_emb([t0, t1])
            row.update(bge=float(e[0] @ e[1]), lex=difflib.SequenceMatcher(None, t0[:4000], t1[:4000]).ratio(),
                       lenratio=min(len(t0), len(t1)) / max(1, len(t0), len(t1)))
        f.write(json.dumps(row) + "\n"); n += 1
        if n % 200 == 0:
            f.flush(); print(ds, n, flush=True)
    f.flush(); print(ds, "done", n, flush=True)
print("SIM_DONE")
