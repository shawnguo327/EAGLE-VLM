"""Out-of-fold predictions of a saved LoRA adapter on ALL eval items (ties included) - used for the hard-label control,
whose training run skipped tie items in its eval set.  Same fold rule as train_pref_v2 (crc32(id) % 2 == fold -> eval).
usage: python eval_adapter.py --adapter <dir> --fold f --tag lm_hard_all --datasets ... --out <dir>"""
import argparse, json, os, sys, zlib
import torch
sys.path.insert(0, "/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm")
from judge_bench import build, load_model, ROOT
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--model-path", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--datasets", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()
proc, model = load_model("qwen3vl8b", a.model_path, "cuda:0"); model = PeftModel.from_pretrained(model, a.adapter).eval()
kid = torch.tensor([proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABT"], device="cuda:0")
rows = []
for ds in a.datasets.split(","):
    for l in open(ROOT / "items" / f"{ds}.jsonl"):
        r = json.loads(l)
        if zlib.crc32(r["id"].encode()) % 2 == a.fold:
            rows.append(r)
files = {}
for i in range(0, len(rows), a.bs):
    B = [(r, o) for r in rows[i:i + a.bs] for o in ("AB", "BA")]
    inp = proc.apply_chat_template([build(r, "tri", o) for r, o in B], tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt", padding=True, enable_thinking=False).to("cuda:0")
    with torch.inference_mode():
        lp = torch.log_softmax(model(**inp).logits[:, -1, :].float(), -1)[:, kid]
    for (r, o), row in zip(B, lp):
        ds = r["dataset"]
        if ds not in files:
            files[ds] = open(os.path.join(a.out, f"{ds}__qwen3vl8b_{a.tag}.f{a.fold}.r0.jsonl"), "w", buffering=1)
        files[ds].write(json.dumps({"id": r["id"], "cond": f"tri_{a.tag}", "order": o, "lp": dict(zip("ABT", row.tolist()))}) + "\n")
print("DONE", len(rows))
