"""Convert an original LLaVA-NeXT 'LlavaQwenForCausalLM' checkpoint (LLaVA-OneVision family: LLaVA-Critic,
MM-RLHF-Reward) to the transformers LlavaOnevisionForConditionalGeneration layout.
Config / processor / chat template come from llava-hf/llava-onevision-qwen2-7b-ov-hf (same architecture);
weights are renamed and every target tensor is checked for name and shape.
usage: python convert_llava_ov.py SRC_DIR TEMPLATE_DIR OUT_DIR [v4]
  v4: write transformers-v4 parameter names (what vLLM's LLaVA-OneVision loader expects)"""
import glob, json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

src, tpl, out = sys.argv[1:4]; V4 = len(sys.argv) > 4 and sys.argv[4] == "v4"
os.makedirs(out, exist_ok=True)
for f in os.listdir(tpl):
    if f.endswith((".json", ".txt", ".jinja")) and not f.endswith("index.json"):
        shutil.copy(os.path.join(tpl, f), out)

from transformers import AutoConfig, LlavaOnevisionForConditionalGeneration
cfg = AutoConfig.from_pretrained(out)
with torch.device("meta"):
    ref = LlavaOnevisionForConditionalGeneration(cfg)
want = {k: v.shape for k, v in ref.state_dict().items()}

sd = {}
for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
    sd.update(load_file(f))


def rename(k):
    if k.startswith("lm_head."):
        return k
    if k.startswith("model.vision_tower.vision_tower."):
        rest = k[len("model.vision_tower.vision_tower."):]
        return "model.vision_tower." + (rest[len("vision_model."):] if rest.startswith("vision_model.") else rest)
    if k.startswith("model.mm_projector.0."):
        return "model.multi_modal_projector.linear_1." + k.split(".")[-1]
    if k.startswith("model.mm_projector.2."):
        return "model.multi_modal_projector.linear_2." + k.split(".")[-1]
    if k == "model.image_newline":
        return "model.image_newline"
    if k.startswith("model."):
        return "model.language_model." + k[len("model."):]
    return None


new, skipped = {}, []
for k, v in sd.items():
    t = rename(k)
    if t is None or t not in want:
        skipped.append(k); continue
    if tuple(v.shape) != tuple(want[t]) and t.endswith(("embed_tokens.weight", "lm_head.weight")) \
            and v.shape[1:] == want[t][1:] and v.shape[0] < want[t][0]:   # template pads the vocab: zero rows
        v = torch.cat([v, v.new_zeros(want[t][0] - v.shape[0], *v.shape[1:])])
    if tuple(v.shape) != tuple(want[t]):
        sys.exit(f"shape mismatch {k} -> {t}: {tuple(v.shape)} vs {tuple(want[t])}")
    new[t] = v.to(torch.bfloat16).contiguous()
missing = [k for k in want if k not in new]
tied = getattr(cfg.get_text_config(), "tie_word_embeddings", False)
if missing == ["lm_head.weight"] and tied:
    missing = []
print(f"mapped {len(new)}/{len(want)}; skipped source keys: {skipped[:10]} ({len(skipped)}); missing: {missing[:10]}")
if missing:
    sys.exit("missing target tensors")
if V4:
    def v4(k):
        for a_, b_ in (("model.vision_tower.", "vision_tower.vision_model."), ("model.multi_modal_projector.", "multi_modal_projector."),
                       ("model.image_newline", "image_newline"), ("model.language_model.", "language_model.model."),
                       ("lm_head.", "language_model.lm_head.")):
            if k.startswith(a_):
                return b_ + k[len(a_):]
        return k
    new = {v4(k): v for k, v in new.items()}
# shard ~4 GB
shards, cur, size, idx = [], {}, 0, {}
for k in sorted(new):
    n = new[k].numel() * 2
    if cur and size + n > 4e9:
        shards.append(cur); cur, size = {}, 0
    cur[k] = new[k]; size += n
shards.append(cur)
for i, s in enumerate(shards):
    name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
    save_file(s, os.path.join(out, name), metadata={"format": "pt"})
    idx.update({k: name for k in s})
json.dump({"metadata": {"total_size": sum(v.numel() * 2 for v in new.values())}, "weight_map": idx},
          open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
print("wrote", out)
