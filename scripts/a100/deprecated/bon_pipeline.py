"""Best-of-N with verifiable answers (reward-model use of the judge).
Stages:  download  -> MathVista testmini (1000) + ScienceQA test questions with an image (random 1000)
         generate  -> policy Qwen2.5-VL-7B samples N answers per question (T=0.8, top-p .95), final line 'Answer: ...'
         items     -> all answer pairs within a question become judge items (kind resp, the question image as context);
                      label from correctness vs the gold answer: one right & one wrong -> the right one is better;
                      both right / both wrong -> tie (meta.pair_type = both-correct / both-wrong / decided)
usage: python bon_pipeline.py {download,generate,items} [--n 4] [--gpu 0]"""
import argparse, io, itertools, json, os, random, re, sys, zlib
ROOT = "/media/vipuser/Data/eaglevlm/judgebench_local"; RAW = "/media/vipuser/Data1/raw/bon"; os.makedirs(RAW, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--n", type=int, default=4); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--model-path", default="/media/vipuser/Data/models/Qwen2.5-VL-7B-Instruct")
a = ap.parse_args()
L = "ABCDEFGHIJ"


def questions():
    import pyarrow.parquet as pq
    out = []
    t = pq.read_table(f"{RAW}/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet").to_pylist()
    for r in t:
        img = r["decoded_image"]["bytes"] if isinstance(r.get("decoded_image"), dict) else None
        if not img:
            continue
        ch = r.get("choices") or []
        out.append(dict(qid=f"mathvista/{r['pid']}", src="mathvista", img=img, question=r["question"], choices=list(ch),
                        answer=str(r["answer"]), qtype=r["question_type"], atype=r.get("answer_type"), precision=r.get("precision"),
                        subset=(r.get("metadata") or {}).get("task", "mathvista") if isinstance(r.get("metadata"), dict) else "mathvista"))
    t = pq.read_table(f"{RAW}/ScienceQA/data/test-00000-of-00001-f0e719df791966ff.parquet").to_pylist()
    t = [r for r in t if isinstance(r.get("image"), dict) and r["image"].get("bytes")]
    random.Random(0).shuffle(t)
    for k, r in enumerate(t[:1000]):
        out.append(dict(qid=f"scienceqa/{k}", src="scienceqa", img=r["image"]["bytes"], question=r["question"] + (f"\nContext: {r['hint']}" if r.get("hint") else ""),
                        choices=list(r["choices"]), answer=r["choices"][int(r["answer"])], qtype="multi_choice", atype="text", precision=None,
                        subset=r.get("subject", "scienceqa")))
    return out


def prompt(q):
    s = q["question"]
    if q["choices"]:
        s += "\nOptions:\n" + "\n".join(f"({L[i]}) {c}" for i, c in enumerate(q["choices"]))
    return s + "\nSolve the problem step by step (briefly), then give the final answer on the last line as 'Answer: <answer>'" + \
        (" using the option letter." if q["choices"] else ".")


def correct(q, text):
    m = re.findall(r"Answer:\s*(.+)", text)
    if not m:
        return False, None
    ans = m[-1].strip().strip(".").strip()
    if q["choices"]:
        lm = re.match(r"\(?([A-J])\)?(\s|$|\.)", ans)
        pick = q["choices"][L.index(lm.group(1))] if lm and L.index(lm.group(1)) < len(q["choices"]) else ans
        return pick.strip().lower() == q["answer"].strip().lower(), ans
    try:
        x, g = float(re.findall(r"-?\d+\.?\d*", ans.replace(",", ""))[0]), float(q["answer"])
        tol = 10 ** -(int(q["precision"]) if q["precision"] else 0) / 2 + 1e-9 if q["atype"] == "float" else 1e-6
        return abs(x - g) <= max(tol, 1e-6), ans
    except Exception:
        return ans.strip().lower() == q["answer"].strip().lower(), ans


if a.stage == "download":
    from huggingface_hub import hf_hub_download
    hf_hub_download("AI4Math/MathVista", "data/testmini-00000-of-00001-725687bf7a18d64b.parquet", repo_type="dataset", local_dir=f"{RAW}/MathVista")
    hf_hub_download("derek-thomas/ScienceQA", "data/test-00000-of-00001-f0e719df791966ff.parquet", repo_type="dataset", local_dir=f"{RAW}/ScienceQA")
    print("DOWNLOAD_OK")
elif a.stage == "generate":
    import torch
    from PIL import Image
    sys.path.insert(0, "/media/vipuser/Data/eaglevlm/EAGLE-VLM/eaglevlm"); from judge_bench import load_model
    Q = questions(); os.makedirs(f"{ROOT}/images/bon", exist_ok=True)
    out_path = f"{RAW}/generations.jsonl"; done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["qid"] for l in open(out_path)}
    proc, model = load_model("qwen25vl7b", a.model_path, "cuda:0"); proc.tokenizer.padding_side = "left"
    f = open(out_path, "a", buffering=1)
    todo = [q for q in Q if q["qid"] not in done]
    for i in range(0, len(todo), a.bs):
        B = todo[i:i + a.bs]; msgs = []
        for q in B:
            p = f"{ROOT}/images/bon/{q['qid'].replace('/', '_')}.png"
            if not os.path.exists(p):
                Image.open(io.BytesIO(q["img"])).convert("RGB").save(p)
            im = Image.open(p); im.thumbnail((768, 768))
            msgs.append([{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt(q)}]}])
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            g = model.generate(**inp, do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=512, num_return_sequences=a.n)
        texts = proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        for k, q in enumerate(B):
            ans = texts[k * a.n:(k + 1) * a.n]
            f.write(json.dumps(dict(qid=q["qid"], answers=ans, correct=[correct(q, t)[0] for t in ans])) + "\n")
        print(f"{i + len(B)}/{len(todo)}", flush=True)
    print("GENERATE_OK")
elif a.stage == "items":
    Q = {q["qid"]: q for q in questions()}; G = [json.loads(l) for l in open(f"{RAW}/generations.jsonl")]
    CRIT = "Prefer the response that answers the question about the image correctly, with correct reasoning."
    out = []
    for g in G:
        q = Q[g["qid"]]
        for i, j in itertools.combinations(range(len(g["answers"])), 2):
            ci, cj = g["correct"][i], g["correct"][j]
            iid = f"bon/{g['qid'].replace('/', '_')}_{i}{j}"
            lab = "tie" if ci == cj else ("0" if ci else "1")
            out.append(dict(id=iid, dataset="bon", subset=q["src"], kind="resp", question=prompt(q),
                            context=[f"images/bon/{g['qid'].replace('/', '_')}.png"], cand=[g["answers"][i], g["answers"][j]], criterion=CRIT,
                            label=lab, meta=dict(qid=g["qid"], idx=[i, j], correct=[ci, cj],
                                                 pair_type="both-correct" if ci and cj else "both-wrong" if not (ci or cj) else "decided")))
    with open(f"{ROOT}/items/bon.jsonl", "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    import collections
    acc = sum(sum(g["correct"]) for g in G) / sum(len(g["correct"]) for g in G)
    print("ITEMS_OK", len(out), "pairs;", dict(collections.Counter(x["meta"]["pair_type"] for x in out)), f"policy accuracy {acc:.3f}")
