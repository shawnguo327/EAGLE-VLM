"""Best-of-N with VLM judges as rerankers and as preference labelers (downstream use; items from make_bon_items.py).

Each question has 8 candidates (VLMEvalKit InternVL2_5-8B-BoN-8 samples, slot 0 = greedy) in 8 cyclic pairs (k, k+1);
correctness of every candidate comes from VLMEvalKit's evaluator (boxed answer, exact matching) and is stored in the
items' meta.
Table 1, selection accuracy (share of questions whose chosen candidate is correct):
  references  greedy sample, a random candidate (expected), VisualPRM's choice (the official Best-of-8 answer),
              majority vote of the extracted options (ties between options broken by VisualPRM), pass@8
  per judge   pointwise: argmax of the expected 1-9 score (each candidate is scored in both of its pairs; averaged)
              pairwise:  argmax of the candidate's mean order-averaged forced-choice win probability over its two pairs
Table 2, preference pairs (what a judge hands to DPO / RLAIF when it labels sampled pairs):
  forced choice (A/B prompt, order-averaged): every pair gets a preference; tie prompt (A/B/T, order-averaged argmax):
  pairs called T are dropped.  Reported: pairs kept, share of kept pairs whose preference is right (preferred one
  correct, other wrong), share of kept pairs that are ties (both correct / both wrong: no preference exists), accuracy
  on the decided pairs, recall of decided pairs.
usage: python bon_select.py --paths configs/paths.bjserver.yaml --dataset bon_scienceqa --judges a,b,c --out x.md"""
import argparse, collections, os
import numpy as np
from eaglevlm.pipeline.run import load_cfg
from eaglevlm.pipeline.io import Store

ap = argparse.ArgumentParser()
ap.add_argument("--paths", required=True); ap.add_argument("--dataset", default="bon_scienceqa")
ap.add_argument("--judges", default="qwen3vl8b,internvl35_8b,qwen35_9b"); ap.add_argument("--out", required=True)
a = ap.parse_args()
cfg = load_cfg(a.paths); P = cfg["paths"]
store = Store(P["items"], P["outputs"], P.get("items_override"), P.get("snapshot"))
I = store.items(a.dataset, rest=False)
D9 = np.arange(1, 10)

# ---- candidate-level references from the pair metas
Q = collections.defaultdict(dict)                     # question -> slot -> dict(hit, option, prm)
for i, x in I.items():
    m = x["meta"]
    for s, h, o, p in zip(m["slots"], m["hit"], m["option"], m["prm"]):
        Q[m["question_index"]][s] = dict(hit=h, option=o, prm=p)
    Q[m["question_index"]]["_prm_slot"] = m["prm_slot"]
qs = sorted(Q); N = 8
H = np.array([[Q[q][s]["hit"] for s in range(N)] for q in qs]); PRM = np.array([[Q[q][s]["prm"] for s in range(N)] for q in qs])
OPT = [[Q[q][s]["option"] for s in range(N)] for q in qs]; ps = np.array([Q[q]["_prm_slot"] for q in qs])


def vote(h, opt, prm):
    c = collections.Counter(o for o in opt if o not in ("Z", None, False))
    if not c:
        return h[0]
    top = c.most_common(); best = [o for o, m in top if m == top[0][1]]
    o = max(best, key=lambda o: max(p for p, q in zip(prm, opt) if q == o))
    return h[opt.index(o)]


ref = [("greedy sample (slot 0)", H[:, 0].mean()), ("random candidate (expected)", H.mean()),
       ("VisualPRM-8B (official Best-of-8 choice)", H[np.arange(len(qs)), ps].mean()),
       ("majority vote of extracted options", np.mean([vote(H[k], OPT[k], PRM[k]) for k in range(len(qs))])),
       ("pass@8 (oracle)", (H.sum(1) > 0).mean())]
L = [f"# Best-of-8 with VLM judges ({a.dataset}: {len(qs)} questions, {len(I)} cyclic candidate pairs)", "",
     "## Selection accuracy", "", "| selector | judge | n questions | accuracy |", "|---|---|---|---|"]
L += [f"| {name} | - | {len(qs)} | {v:.4f} |" for name, v in ref]
T2 = ["", "## Preference pairs labelled by the judge", "",
      "| judge | prompt | pairs | pairs kept | right preference (share of kept) | ties among kept | accuracy on decided pairs kept | decided pairs kept (recall) |",
      "|---|---|---|---|---|---|---|---|"]
qpos = {q: k for k, q in enumerate(qs)}
for j in a.judges.split(","):
    R = store.load(a.dataset, j, conds={"bin", "tri", "score"})
    S = np.full((len(qs), N, 2), np.nan); W = np.full((len(qs), N, 2), np.nan)
    rows = []
    for i, x in I.items():
        r = R.get(i, {}); m = x["meta"]; k = qpos[m["question_index"]]; s0, s1 = m["slots"]
        if all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
            b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
            p = (b1["A"] / (b1["A"] + b1["B"]) + b2["B"] / (b2["A"] + b2["B"])) / 2
            q3 = np.array([(t1["A"] + t2["B"]) / 2, (t1["B"] + t2["A"]) / 2, (t1["T"] + t2["T"]) / 2])
            W[k, s0, 0] = p; W[k, s1, 1] = 1 - p                      # slot s0 is cand 0 of this pair, s1 is cand 1
            y = {"0": 0, "1": 1, "tie": 2}[str(x["label"])]
            rows.append((y, 0 if p > .5 else 1, int(q3.argmax())))
        for o, s, col in (("A", s0, 0), ("B", s1, 1)):
            if ("score", o) in r:
                d = np.array([r[("score", o)].get(str(v), 0) for v in D9]); S[k, s, col] = d @ D9 / d.sum()
    cov_p = np.isfinite(W).all(2).all(1).mean(); cov_s = np.isfinite(S).any(2).all(1).mean()
    if cov_p < .95 and cov_s < .95:
        L.append(f"| judge pointwise / pairwise | {j} | - | outputs incomplete (pairwise {cov_p:.2f}, pointwise {cov_s:.2f} of questions) |")
        continue
    if cov_s >= .95:
        sc = np.nanmean(S, 2); ok = np.isfinite(sc).all(1)
        sel = np.nanargmax(np.where(np.isfinite(sc), sc, -np.inf), 1)
        L.append(f"| judge pointwise (argmax expected 1-9 score) | {j} | {ok.sum()} | {H[np.arange(len(qs)), sel][ok].mean():.4f} |")
    if cov_p >= .95:
        w = np.nanmean(W, 2); ok = np.isfinite(w).all(1)
        sel = np.nanargmax(np.where(np.isfinite(w), w, -np.inf), 1)
        L.append(f"| judge pairwise (argmax mean win probability over its two pairs) | {j} | {ok.sum()} | {H[np.arange(len(qs)), sel][ok].mean():.4f} |")
        y, fc, ta = map(np.array, zip(*rows)); dec = y != 2
        for name, keep, pref in (("forced choice A/B", np.ones(len(y), bool), fc), ("tie prompt A/B/T (T dropped)", ta != 2, ta)):
            right = keep & dec & (pref == y)
            T2.append(f"| {j} | {name} | {len(y)} | {keep.mean():.3f} | {right.sum() / max(keep.sum(), 1):.3f} | "
                      f"{(keep & ~dec).sum() / max(keep.sum(), 1):.3f} | {right.sum() / max((keep & dec).sum(), 1):.3f} | {(keep & dec).sum() / max(dec.sum(), 1):.3f} |")
open(a.out, "w").write("\n".join(L + T2) + "\n"); print("\n".join(L + T2))
