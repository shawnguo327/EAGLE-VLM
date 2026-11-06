"""Per-candidate correctness for VLMEvalKit Best-of-N runs, scored with VLMEvalKit's own MCQ evaluator.

The run (official VLMEvalKit `InternVL2_5-8B-BoN-8`: InternVL2.5-8B samples 8 CoT answers, VisualPRM-8B picks one) was made
with our env-gated BON_DUMP patch, which records all 8 candidates and the PRM scores without changing the selected answer.
For every candidate slot k (sampling order; slot 0 is the greedy sample, slots 1-7 use T=0.6, top-p=0.95) this script
writes a copy of VLMEvalKit's prediction sheet in which `prediction` is candidate k and runs
`dataset.evaluate(sheet, model='exact_matching')` on it.  Post-processing (--postprocess):
  none   the raw response, as the official config leaves it (use_mpo_prompt=False, use_postprocess=False).  VLMEvalKit
         then relies on an LLM answer extractor (GPT); without one, exact matching fails on most chain-of-thought answers.
  boxed  VLMEvalKit's own InternVL chain-of-thought post-processing (internvl_chat.extract_boxed_content, the code path of
         use_postprocess=True): the content of the last \boxed{...}, which the official CoT prompt asks for.  Main setting.  The extracted option per candidate comes from the same function the
evaluator calls (vlmeval.dataset.utils.multiple_choice.extract_answer_from_item with no LLM judge).

usage (VLMEvalKit dir, vlmeval env):
  python bon_candidates.py --dataset ScienceQA_TEST --dump .../scienceqa_bon8_dump.jsonl \
      --pred .../InternVL2_5-8B-BoN-8-local_ScienceQA_TEST.xlsx --out .../bon_eval/ScienceQA_TEST
writes <out>/cand{k}.xlsx (+ VLMEvalKit's *_exact_matching_result / _acc files), <out>/candidates.jsonl.gz and <out>/summary.csv"""
import argparse, collections, gzip, json, os, re
import pandas as pd
from vlmeval.dataset import build_dataset
from vlmeval.dataset.utils.multiple_choice import extract_answer_from_item

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--dump", required=True); ap.add_argument("--pred", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8); ap.add_argument("--nproc", type=int, default=8)
ap.add_argument("--postprocess", choices=["none", "boxed"], default="boxed")
a = ap.parse_args(); a.out = os.path.join(a.out, a.postprocess); os.makedirs(a.out, exist_ok=True)
if a.postprocess == "boxed":
    from vlmeval.vlm.internvl.internvl_chat import extract_boxed_content as PP
else:
    PP = lambda t: t

ds = build_dataset(a.dataset)
pred = pd.read_excel(a.pred)
sel = {int(i): str(p) for i, p in zip(pred["index"], pred["prediction"])}


def qindex(paths):
    b = os.path.basename(paths[0]); m = re.match(r"(\d+)", b)
    return int(m.group(1))


# ---- dump lines -> question index; keep the line whose PRM-top response is VLMEvalKit's recorded prediction
rows = collections.defaultdict(list)
for line in open(a.dump):
    d = json.loads(line)
    if d["dataset"] != a.dataset or len(d["candidates"]) != a.n:
        continue
    rows[qindex(d["images"])].append(d)
D, unmatched = {}, []
for idx, ds_rows in rows.items():
    ok = [d for d in ds_rows if d["prm_sorted"][0][0].strip() == sel.get(idx, "\0").strip()]
    if ok:
        D[idx] = ok[-1]
    else:
        unmatched.append(idx)
missing = sorted(set(sel) - set(D))
print(f"questions in sheet {len(sel)}, dump lines {sum(map(len, rows.values()))}, matched {len(D)}, "
      f"dump lines not matching the recorded prediction {len(unmatched)}, sheet rows without a dump line {len(missing)}")
pred = pred[pred["index"].isin(D)].reset_index(drop=True)

# ---- per-slot sheets scored by VLMEvalKit
hits = {}
for k in range(a.n):
    sh = pred.copy(); sh["prediction"] = [PP(D[int(i)]["candidates"][k]) for i in sh["index"]]
    f = os.path.join(a.out, f"cand{k}.xlsx"); sh.to_excel(f, index=False)
    acc = ds.evaluate(f, model="exact_matching", nproc=a.nproc)
    res = [x for x in os.listdir(a.out) if x.startswith(f"cand{k}_exact_matching_result") and x.endswith((".xlsx", ".csv", ".tsv"))]
    r = pd.read_excel(os.path.join(a.out, res[0])) if res[0].endswith(".xlsx") else pd.read_csv(os.path.join(a.out, res[0]), sep=None, engine="python")
    hits[k] = {int(i): int(h) for i, h in zip(r["index"], r["hit"])}
    print(f"slot {k}: VLMEvalKit acc table\n{acc}")
# the recorded (PRM-selected) prediction itself, as a check against VLMEvalKit's own number for the run
sh = pred.copy(); sh["prediction"] = [PP(str(x)) for x in sh["prediction"]]
f = os.path.join(a.out, "selected.xlsx"); sh.to_excel(f, index=False); acc_sel = ds.evaluate(f, model="exact_matching", nproc=a.nproc)
print(f"PRM-selected: VLMEvalKit acc table\n{acc_sel}")

# ---- per-question rows
meta = ds.data.set_index("index")
out = []
for _, it in pred.iterrows():
    i = int(it["index"]); d = D[i]; c = d["candidates"]
    score = {}
    for r, s in d["prm_sorted"]:
        score.setdefault(r, s)
    opts = []
    for k in range(a.n):
        x = it.copy(); x["prediction"] = PP(c[k])
        opts.append(extract_answer_from_item(None, x, a.dataset)["opt"])
    out.append(dict(index=i, answer=str(meta.loc[i, "answer"]), category=str(meta.loc[i].get("category", meta.loc[i].get("split", ""))),
                    hit=[hits[k][i] for k in range(a.n)], option=opts, prm=[float(score[t]) for t in c],
                    prm_slot=c.index(d["prm_sorted"][0][0]), candidates=c, prompt=d["prompt"], images=d["images"]))
with gzip.open(os.path.join(a.out, "candidates.jsonl.gz"), "wt") as g:
    for r in out:
        g.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- summary
H = pd.DataFrame([r["hit"] for r in out]); n = len(out)
nc = H.sum(1)
def vote(r):
    c = collections.Counter(o for o in r["option"] if o not in ("Z", None, False))
    if not c:
        return r["hit"][0]
    top = c.most_common(); best = [o for o, m in top if m == top[0][1]]
    # tie between options: VisualPRM score of the best candidate with that option decides
    o = max(best, key=lambda o: max(p for p, q in zip(r["prm"], r["option"]) if q == o))
    return int(o == r["answer"])
S = dict(dataset=a.dataset, postprocess=a.postprocess, questions=n, greedy_acc=H[0].mean(), mean_candidate_acc=H.values.mean(),
         prm_selected_acc=sum(r["hit"][r["prm_slot"]] for r in out) / n, majority_vote_acc=sum(vote(r) for r in out) / n,
         pass_at_n=(nc > 0).mean(), all_correct=(nc == a.n).mean(), none_correct=(nc == 0).mean())
pairs = collections.Counter()
for r in out:
    h = r["hit"]
    for p in range(a.n):
        for q in range(p + 1, a.n):
            pairs["decided" if h[p] != h[q] else ("both_correct" if h[p] else "both_wrong")] += 1
    for p in range(a.n):
        for q in range(p + 1, a.n):
            if h[p] == h[q] == 0:
                pairs["both_wrong_same_option" if r["option"][p] == r["option"][q] else "both_wrong_diff_option"] += 1
tot = sum(pairs[k] for k in ("decided", "both_correct", "both_wrong"))
S.update({f"pairs_{k}": v for k, v in pairs.items()}, pairs_total=tot, pair_tie_share=(pairs["both_correct"] + pairs["both_wrong"]) / tot)
S.update({f"n_correct_{c}": int((nc == c).sum()) for c in range(a.n + 1)})
pd.DataFrame([S]).to_csv(os.path.join(a.out, "summary.csv"), index=False)
for k, v in S.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
