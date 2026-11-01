"""Bring the 2026-10-30 Place Pulse pilot (urban-judge-20261030/pp) into the judgebench format.
Writes $EAGLEVLM_DATA/items/pp.jsonl and results/pp__<model>.jsonl (bin + score rows; A = left image).
The pilot sample contains strict votes only."""
import json, os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/media/shawn/C/papers/VLM/urban-judge-20261030/pp"
DATA = os.environ.get("EAGLEVLM_DATA", "/media/shawn/C/papers/VLM/judgebench")
IMG = "/media/shawn/C/papers/VLM/place-pulse-2.0/"

with open(os.path.join(DATA, "items/pp.jsonl"), "w") as f:
    for l in open(os.path.join(SRC, "sample.jsonl")):
        r = json.loads(l)
        lab = {"left": 0, "right": 1}.get(r["choice"], "tie")
        f.write(json.dumps(dict(id=f"pp/{r['id']}", dataset="pp", subset=r["category"], kind="img",
                                question=r["question"], context=[],
                                cand=[IMG + r["left_image"], IMG + r["right_image"]],
                                criterion="Judge only the visual appearance of the street view.", label=lab)) + "\n")

for m in ("qwen3vl8b", "qwen35_9b", "qwen25vl7b"):
    with open(os.path.join(DATA, f"results/pp__{m}.jsonl"), "w") as f:
        for l in open(os.path.join(SRC, f"raw_sample_{m}.jsonl")):
            d = json.loads(l)
            if d.get("cond") in ("bin", "score") and "lp" in d:
                d["id"] = f"pp/{d['id']}"
                f.write(json.dumps(d) + "\n")
print("ok")
