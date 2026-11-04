"""Gallery of typical GenAI-Arena pairs for manual inspection (edit and generation; both-bad / tie / decided).
For each selected pair a folder with the source image (edits), both candidates, a side-by-side sheet with the text
burned in, and info.json (instruction / prompt, human vote, models, judge signals).  Selection per subset x group:
  both-bad and tie: 12 'detected' (highest Qwen3-VL-32B P(T)) + 12 'missed' (lowest P(T)) + 12 random
  decided: 12 random + 12 'false alarm' (highest P(T) although the human picked a side)
index.md / index.html list everything with thumbnails.
usage (judgebench dir): python tie_gallery.py --out /media/shawn/C/papers/VLM/judgebench/tie_examples"""
import argparse, html, json, os, random, shutil, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load
from label_noise_lib import per_judge

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/media/shawn/C/papers/VLM/judgebench/tie_examples")
ap.add_argument("--root", default="/media/shawn/C/papers/VLM/judgebench_local")
ap.add_argument("--k", type=int, default=12)
a = ap.parse_args()
I = items("genai_arena")
J = ["qwen3vl32b", "qwen3vl8b"]
P = {j: per_judge("genai_arena", j, I) for j in J}
D = np.arange(1, 10)


def point(j):
    R = load("genai_arena", j, conds={"score"}); out = {}
    for i, r in R.items():
        if ("score", "A") in r and ("score", "B") in r:
            out[i] = [round(float(np.array([r[("score", o)].get(str(d), 0) for d in D]) @ D /
                                  sum(r[("score", o)].get(str(d), 0) for d in D)), 2) for o in "AB"]
    return out


W = {j: point(j) for j in J}
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
except Exception:
    font = ImageFont.load_default()


def wrap(t, n=70):
    w, line, out = t.split(), "", []
    for x in w:
        if len(line) + len(x) + 1 > n:
            out.append(line); line = x
        else:
            line = (line + " " + x).strip()
    return out + [line]


def sheet(ims, labels, text, path, h=384):
    ims = [im.convert("RGB") for im in ims]
    ims = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in ims]
    lines = sum((wrap(t) for t in text), [])
    W_ = sum(im.width for im in ims) + 10 * (len(ims) + 1); H_ = h + 40 + 24 * len(lines) + 20
    c = Image.new("RGB", (W_, H_), "white"); d = ImageDraw.Draw(c); x = 10
    for im, lb in zip(ims, labels):
        c.paste(im, (x, 10)); d.text((x, h + 14), lb, fill="black", font=font); x += im.width + 10
    for k, ln in enumerate(lines):
        d.text((10, h + 44 + 24 * k), ln, fill="black", font=font)
    c.save(path, quality=90)


os.makedirs(a.out, exist_ok=True); rows = []
rng = random.Random(0)
for sub in ("edition", "generation"):
    ids = [i for i in sorted(I) if I[i]["subset"] == sub and i in P["qwen3vl32b"]]
    vt = {i: I[i]["meta"]["vote_type"] for i in ids}
    pt = {i: P["qwen3vl32b"][i][1][2] for i in ids}
    for grp, vts in (("bothbad", {"bothbad_vote"}), ("tie", {"tievote"}), ("decided", {"leftvote", "rightvote"})):
        g = [i for i in ids if vt[i] in vts]
        srt = sorted(g, key=lambda i: -pt[i])
        if grp == "decided":
            sel = [("random", i) for i in rng.sample(g, min(a.k, len(g)))] + [("false-alarm", i) for i in srt[: a.k]]
        else:
            sel = [("detected", i) for i in srt[: a.k]] + [("missed", i) for i in srt[-a.k:]]
            rest = [i for i in g if i not in set(srt[: a.k] + srt[-a.k:])]
            sel += [("random", i) for i in rng.sample(rest, min(a.k, len(rest)))]
        for tag, i in sel:
            x = I[i]; d = os.path.join(a.out, sub, grp, f"{tag}__{i.split('/')[1]}"); os.makedirs(d, exist_ok=True)
            imgs, labels = [], []
            for k, p in enumerate(x["context"]):
                shutil.copy(os.path.join(a.root, p), os.path.join(d, "source" + os.path.splitext(p)[1])); imgs.append(Image.open(os.path.join(a.root, p))); labels.append("SOURCE")
            for k, p in enumerate(x["cand"]):
                nm = f"cand{'AB'[k]}_" + "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in x['meta']['models'][k])
                shutil.copy(os.path.join(a.root, p), os.path.join(d, nm + os.path.splitext(p)[1])); imgs.append(Image.open(os.path.join(a.root, p))); labels.append(f"{'AB'[k]}: {x['meta']['models'][k]}")
            sig = {j: dict(p_A_better=round(float(P[j][i][0]), 3), tri_P=[round(float(v), 3) for v in P[j][i][1]],
                           order_consistent=bool(P[j][i][2]), pointwise_E=W[j].get(i)) for j in J if i in P[j]}
            human = {"leftvote": "A better", "rightvote": "B better", "tievote": "TIE (equally good)", "bothbad_vote": "BOTH BAD"}[vt[i]]
            info = dict(id=i, subset=sub, group=grp, selection=tag, instruction_or_prompt=x["question"], human_vote=human,
                        models=x["meta"]["models"], source_prompt=x["meta"].get("source_prompt"),
                        target_prompt=x["meta"].get("target_prompt"), judges=sig)
            json.dump(info, open(os.path.join(d, "info.json"), "w"), indent=1, ensure_ascii=False)
            s32 = sig.get("qwen3vl32b", {})
            txt = [f"{'EDIT: ' if sub == 'edition' else 'PROMPT: '}{x['question']}", f"HUMAN: {human}",
                   f"32B P(A,B,T) = {s32.get('tri_P')}  pointwise E[s] A/B = {s32.get('pointwise_E')}  ({tag})"]
            sheet(imgs, labels, txt, os.path.join(d, "sheet.jpg"))
            rows.append((sub, grp, tag, i, x["question"], human, s32.get("tri_P"), s32.get("pointwise_E"), os.path.relpath(d, a.out)))
        print(sub, grp, len(sel), flush=True)
md = ["# GenAI-Arena tie gallery", "", "Groups: bothbad / tie / decided; selection: detected (high 32B P(T)), missed (low P(T)), random, "
      "false-alarm (decided but high P(T)).  Each folder: source (edits), candA_<model>, candB_<model>, sheet.jpg, info.json.", "",
      "| subset | group | selection | id | instruction / prompt | human | 32B P(A,B,T) | 32B pointwise A/B | folder |", "|---|---|---|---|---|---|---|---|---|"]
md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4][:80]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} |" for r in rows]
open(os.path.join(a.out, "index.md"), "w").write("\n".join(md) + "\n")
h = ["<html><meta charset='utf-8'><body style='font-family:sans-serif'><h2>GenAI-Arena tie gallery</h2>"]
for sub in ("edition", "generation"):
    for grp in ("bothbad", "tie", "decided"):
        h.append(f"<h3>{sub} / {grp}</h3>")
        for r in [r for r in rows if r[0] == sub and r[1] == grp]:
            h.append(f"<div style='margin:8px 0'><b>{html.escape(r[2])}</b> {html.escape(r[3])} — human: {html.escape(r[5])}; 32B P(A,B,T)={r[6]}; "
                     f"pointwise={r[7]}<br><img src='{r[8]}/sheet.jpg' style='max-width:1200px'></div>")
h.append("</body></html>")
open(os.path.join(a.out, "index.html"), "w").write("\n".join(h))
print("wrote", len(rows), "pairs to", a.out)
