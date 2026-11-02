"""MM-RLHF-Reward step 2: scalar reward from (image, critic prompt, generated critique) with the model's own reward head.
Follows inference/Multimodal-Reward/get_mm-rlhf_reward.py: the conversation [user: <image> prompt, assistant: critique]
is passed through the model and the head (dense -> tanh -> out_proj, weights rm_head.* in the original checkpoint) is
applied to the final-layer hidden state of the last token of the assistant turn.
usage (lis-omni env): python mmrlhf_score.py --hf-model DIR_HF --orig DIR_ORIG --critiques mmrlhf_critic.jsonl --items a.jsonl,b.jsonl --out F"""
import argparse, glob, json, os
import torch
from PIL import Image
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--hf-model", required=True); ap.add_argument("--orig", required=True)
ap.add_argument("--critiques", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
ROOT = os.environ.get("EAGLEVLM_DATA", os.getcwd())
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
proc = AutoProcessor.from_pretrained(a.hf_model)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(a.hf_model, dtype=torch.bfloat16, device_map="cuda:0").eval()
W = {}
for f in glob.glob(os.path.join(a.orig, "*.safetensors")):
    with safe_open(f, "pt") as g:
        for k in g.keys():
            if k.startswith("rm_head."):
                W[k] = g.get_tensor(k).to("cuda:0", torch.float32)
assert len(W) == 4, W.keys()


def head(h):
    h = torch.tanh(h.float() @ W["rm_head.dense.weight"].T + W["rm_head.dense.bias"])
    return (h @ W["rm_head.out_proj.weight"].T + W["rm_head.out_proj.bias"]).squeeze(-1)


I = {json.loads(l)["id"]: json.loads(l) for f in a.items.split(",") for l in open(f)}
done = set()
if os.path.exists(a.out):
    done = {(json.loads(l)["id"], json.loads(l)["cand"]) for l in open(a.out)}
f = open(a.out, "a")
end_id = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
for l in open(a.critiques):
    x = json.loads(l)
    if (x["id"], x["cand"]) in done or x["id"] not in I:
        continue
    r = I[x["id"]]
    ims = [Image.open(os.path.join(ROOT, p)).convert("RGB") for p in r["context"]]
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in ims] + [{"type": "text", "text": x["prompt"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": x["critique"]}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=False)
    inp = proc(images=ims, text=text, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out = model(**inp, output_hidden_states=True)
    ids = inp["input_ids"][0]
    pos = (ids == end_id).nonzero().flatten()
    p = int(pos[-1]) if len(pos) else len(ids) - 1
    rew = float(head(out.hidden_states[-1][0, p]))
    f.write(json.dumps({"id": x["id"], "cand": x["cand"], "reward": rew}) + "\n"); f.flush()
print("DONE")
