"""Phase D: LoRA-tune a VLM judge so that its first-token distribution over {A, B, T} matches the human preference
distribution (soft labels), i.e. give the judge an explicit, trained channel for "the two are close".
Targets (per pair, order AB; order BA is the mirrored target):
  verified datasets            winner = 1
  human tie (MLLM-Judge, Place Pulse, EditReward, GenAI-Arena tievote [+ bothbad if --bothbad-tie])   T = 1
  MMRB2 (3 annotators, 1..6)   each annotator: side by v>=4, weight w = {|v-3.5|: .5->.5, 1.5->.75, 2.5->1};
                               side gets w, T gets 1-w; averaged over annotators (majority side = label)
Two-fold cross-fitting: fold f trains on items with crc32(id) % 2 != f and predicts the others, so every item gets an
out-of-fold prediction. Predictions are written in judge_bench format (cond "tri_ft") to
<out>/{dataset}__{model}_ft.f{fold}.jsonl.  Multi-GPU: torchrun --nproc_per_node N train_pref.py ...
"""
import argparse, ast, json, math, os, random, time, zlib
import torch, torch.nn.functional as F
from judge_bench import build, load_model, ROOT

W = {0.5: .5, 1.5: .75, 2.5: 1.0}


def target(r, bothbad_tie=True):
    """[pA, pB, pT] for order AB (cand0 = A); None to skip."""
    m = r.get("meta") or {}
    if isinstance(m, str):
        try:
            m = ast.literal_eval(m)
        except Exception:
            m = {}
    if r["label"] == "tie":
        if r["dataset"] == "genai_arena" and m.get("vote_type") == "bothbad_vote" and not bothbad_tie:
            return None
        return [0., 0., 1.]
    win = 0 if r["label"] == 0 else 1
    if r["dataset"] == "mmrb2" and m.get("human_annotations"):
        ha = m["human_annotations"]; ha = ast.literal_eval(ha) if isinstance(ha, str) else ha
        maj = (sum(ha) / len(ha)) >= 3.5
        t = [0., 0., 0.]
        for v in ha:
            w = W[abs(v - 3.5)]; side = win if (v >= 4) == maj else 1 - win
            t[side] += w / len(ha); t[2] += (1 - w) / len(ha)
        return t
    t = [0., 0., 0.]; t[win] = 1.; return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl8b"); ap.add_argument("--model-path", default=None)
    ap.add_argument("--datasets", default="mmrb2,genai_arena,mllm_judge,pp,editreward,vlrb,mmrb,mjbench,imagerewarddb,qbench2,mmrlhf")
    ap.add_argument("--out", required=True); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8); ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--bothbad-tie", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=0); ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--eval-bs", type=int, default=8); ap.add_argument("--progress", default=None)
    a = ap.parse_args()
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if ddp:
        torch.distributed.init_process_group("nccl"); torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    os.makedirs(a.out, exist_ok=True)

    train, evals = [], []
    for ds in a.datasets.split(","):
        for l in open(ROOT / "items" / f"{ds}.jsonl"):
            r = json.loads(l); t = target(r, bool(a.bothbad_tie))
            if t is None:
                continue
            (evals if zlib.crc32(r["id"].encode()) % 2 == a.fold else train).append((r, t))
    random.Random(0).shuffle(train)
    if a.max_train:
        train = train[:a.max_train]
    if a.max_eval:
        evals = evals[:a.max_eval]
    # both orders; mirrored target for BA
    ex = [(r, "AB", t) for r, t in train] + [(r, "BA", [t[1], t[0], t[2]]) for r, t in train]
    random.Random(1).shuffle(ex)
    ex = ex[rank::world]
    if rank == 0:
        print(f"train pairs {len(train)} (x2 orders), eval pairs {len(evals)}", flush=True)

    from peft import LoraConfig, get_peft_model
    proc, model = load_model(a.model, a.model_path, dev)
    tok = proc.tokenizer
    kid = torch.tensor([tok.encode(c, add_special_tokens=False)[0] for c in "ABT"], device=dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)")
    model = get_peft_model(model, cfg)
    if rank == 0:
        model.print_trainable_parameters()
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if ddp else model
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(ex) * a.epochs / (a.bs * a.grad_accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * max(0.0, 1 - s / max(1, steps)))
    t0 = time.time(); model.train(); step = 0; it = 0; run = 0.0
    n_micro = math.ceil(len(ex) * a.epochs / a.bs)
    for mi in range(n_micro):
        B = [ex[(mi * a.bs + j) % len(ex)] for j in range(a.bs)]
        inp = proc.apply_chat_template([build(r, "tri", o) for r, o, _ in B], tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt", padding=True, enable_thinking=False).to(dev)
        logits = net(**inp).logits[:, -1, :].float()[:, kid]
        tgt = torch.tensor([t for _, _, t in B], device=dev) * (1 - a.eps) + a.eps / 3
        loss = -(tgt * F.log_softmax(logits, -1)).sum(-1).mean() / a.grad_accum
        loss.backward(); run += loss.item(); it += 1
        if it % a.grad_accum == 0 or mi == n_micro - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if rank == 0 and step % 10 == 0:
                print(f"step {step}/{steps} loss {run / 10:.4f} {time.time() - t0:.0f}s", flush=True); run = 0.0
            if rank == 0 and a.progress:
                json.dump({"model": a.model, "phase": "train", "fold": a.fold, "done": step, "total": steps,
                           "elapsed_s": round(time.time() - t0), "status": "running"}, open(a.progress, "w"))
    if rank == 0:
        model.save_pretrained(os.path.join(a.out, f"lora_{a.model}_f{a.fold}"))

    # ---- out-of-fold predictions, both orders, judge_bench format
    model.eval(); mine = evals[rank::world]
    files = {}
    for i in range(0, len(mine), a.eval_bs):
        B = [(r, o) for r, _ in mine[i:i + a.eval_bs] for o in ("AB", "BA")]
        inp = proc.apply_chat_template([build(r, "tri", o) for r, o in B], tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt", padding=True, enable_thinking=False).to(dev)
        with torch.inference_mode():
            lp = torch.log_softmax(model(**inp).logits[:, -1, :].float(), -1)[:, kid]
        for (r, o), row in zip(B, lp):
            ds = r["dataset"]
            if ds not in files:
                files[ds] = open(os.path.join(a.out, f"{ds}__{a.model}_ft.f{a.fold}.r{rank}.jsonl"), "w", buffering=1)
            files[ds].write(json.dumps({"id": r["id"], "cond": "tri_ft", "order": o,
                                        "lp": dict(zip("ABT", row.tolist()))}) + "\n")
    if rank == 0 and a.progress:
        json.dump({"model": a.model, "phase": "train", "fold": a.fold, "done": steps, "total": steps,
                   "elapsed_s": round(time.time() - t0), "status": "done"}, open(a.progress, "w"))
    if ddp:
        torch.distributed.barrier(); torch.distributed.destroy_process_group()
    print("DONE", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
