"""Judge-free control for the GenAI-Arena hidden-state probes: can off-the-shelf image/text features (CLIP-L, DINOv2-L)
of the two candidates and the prompt separate 'both bad' from 'equally good' (and from decided pairs) as well as the
VLM judge's verdict-position hidden state does?  If yes, the hidden-state gain is image content, not judgement.
Features per pair (order-symmetric): CLIP e0+e1, |e0-e1|, DINO d0+d1, |d0-d1|, cos(prompt, e_k) min/max, cos(e0,e1).
usage (lis-omni env, from judgebench dir): python feat_probe.py --out paper_tables/feat_probe_genai.md"""
import argparse, glob, json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import ttype
from eaglev_stats import auroc
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--roots", default="/media/shawn/C/papers/VLM/judgebench,/media/shawn/C/papers/VLM/judgebench_local")
ap.add_argument("--clip", default=glob.glob("/home/shawn/yinanshao/hf_cache/hub/models--openai--clip-vit-large-patch14/snapshots/*/")[0])
ap.add_argument("--dino", default="facebook/dinov2-large")
ap.add_argument("--cache", default="paper_tables/feat_genai.npz")
ap.add_argument("--out", required=True)
a = ap.parse_args()
ROOTS = a.roots.split(",")
find = lambda p: next(os.path.join(r, p) for r in ROOTS if os.path.exists(os.path.join(r, p)))
IT = [json.loads(l) for l in open(next(f"{r}/items/genai_arena.jsonl" for r in ROOTS if os.path.exists(f"{r}/items/genai_arena.jsonl")))]

if os.path.exists(a.cache):
    z = np.load(a.cache, allow_pickle=True); F = dict(zip(z["id"], z["f"]))
else:
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cp, cm = CLIPProcessor.from_pretrained(a.clip), CLIPModel.from_pretrained(a.clip, dtype=torch.float16).to(dev).eval()
    dp, dm = AutoImageProcessor.from_pretrained(a.dino), AutoModel.from_pretrained(a.dino, dtype=torch.float16).to(dev).eval()
    nz = lambda x: torch.nn.functional.normalize((x if torch.is_tensor(x) else x.pooler_output).float(), dim=-1)
    F = {}
    with torch.inference_mode():
        for k, it in enumerate(IT):
            ims = [Image.open(find(p)).convert("RGB") for p in it["cand"]]
            e = nz(cm.get_image_features(**cp(images=ims, return_tensors="pt").to(dev, torch.float16)))
            d = nz(dm(**dp(images=ims, return_tensors="pt").to(dev, torch.float16)).last_hidden_state[:, 0])
            t = nz(cm.get_text_features(**cp(text=[it["question"]], return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)))
            s = (e @ t.T).squeeze(-1).cpu().numpy()
            e, d = e.cpu().numpy(), d.cpu().numpy()
            F[it["id"]] = np.concatenate([e[0] + e[1], np.abs(e[0] - e[1]), d[0] + d[1], np.abs(d[0] - d[1]),
                                          [s.min(), s.max(), float(e[0] @ e[1]), float(d[0] @ d[1])]]).astype(np.float32)
            if k % 200 == 0:
                print(k, flush=True)
    np.savez(a.cache, id=np.array(list(F)), f=np.stack(list(F.values())))


def cv(X, y):
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        st = [StandardScaler()] + ([PCA(min(256, len(tr) - 1), random_state=0)] if X.shape[1] > 256 else [])
        m = make_pipeline(*st, LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1], max_iter=3000, cv=3)).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return auroc(out, y)


T = {"human tie": lambda t: 1 if t in ("bothbad_vote", "tievote") else 0,
     "both bad vs decided": lambda t: 1 if t == "bothbad_vote" else 0 if t in ("leftvote", "rightvote") else None,
     "equally good vs decided": lambda t: 1 if t == "tievote" else 0 if t in ("leftvote", "rightvote") else None,
     "both bad vs equally good": lambda t: 1 if t == "bothbad_vote" else 0 if t == "tievote" else None}
L = ["# Judge-free feature probe (CLIP-L + DINOv2-L of both candidates and the prompt), GenAI-Arena", "",
     "AUROC, 5-fold CV logistic (PCA-256).  Compare with the verdict-position hidden-state probes (probe_*_H2.md).", "",
     "| subset | target | n (pos) | all features | 4 scalars only (prompt match min/max, pair similarity) |", "|---|---|---|---|---|"]
for sub in ("", "edition", "generation"):
    for tn, tf in T.items():
        rows = [(tf(ttype(it)), F[it["id"]]) for it in IT if it["id"] in F and (not sub or it["subset"] == sub)]
        rows = [r for r in rows if r[0] is not None]
        y = np.array([r[0] for r in rows]); X = np.stack([r[1] for r in rows])
        if min(y.sum(), len(y) - y.sum()) < 10:
            continue
        L.append(f"| {sub or 'all'} | {tn} | {len(y)} ({y.sum()}) | {cv(X, y):.3f} | {cv(X[:, -4:], y):.3f} |")
        print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
