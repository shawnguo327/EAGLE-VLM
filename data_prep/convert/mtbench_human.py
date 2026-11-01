"""MT-Bench human judgments (lmsys/mt_bench_human_judgments, 'human' split) -> items/mtbench_human.jsonl  [kind=text]
Expert votes on (question, model_a, model_b, turn); several judges per pair -> aggregated: per unordered model pair the
votes are counted (winner model / tie). Label = majority (tie if the tie votes are the plurality or the two models
tie in votes). meta.votes keeps the vote counts: the human aleatoric target (agreement) for the text domain.
Turn 1 only (single-turn context, like the multimodal sets)."""
import sys, pathlib, collections; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from common import *

DS = "mtbench_human"
SRC = RAW / "lmsys__mt_bench_human_judgments" / "data"
CRIT = "Prefer the response that is more helpful, relevant, accurate, detailed and appropriate for the user's request."
d = pd.read_parquet(sorted(SRC.glob("human-*.parquet"))[0])
d = d[d.turn == 1]
G = collections.defaultdict(lambda: {"votes": collections.Counter(), "conv": {}})
for r in d.itertuples():
    a, b = sorted([r.model_a, r.model_b]); g = G[(int(r.question_id), a, b)]
    if str(r.winner).startswith("tie"):
        g["votes"]["tie"] += 1
    else:
        g["votes"]["A" if r.winner == ("model_a" if r.model_a == a else "model_b") else "B"] += 1
    conv = {r.model_a: r.conversation_a, r.model_b: r.conversation_b}
    g["conv"] = conv
recs = []
for (qid, a, b), g in G.items():
    ca, cb = g["conv"][a], g["conv"][b]
    q = [m["content"] for m in ca if m["role"] == "user"][0]
    ra = [m["content"] for m in ca if m["role"] == "assistant"][0]; rb = [m["content"] for m in cb if m["role"] == "assistant"][0]
    v = g["votes"]; nA, nB, nT = v["A"], v["B"], v["tie"]
    tie = nT >= max(nA, nB) or nA == nB
    win, lose = ((ra, a), (rb, b)) if nA >= nB else ((rb, b), (ra, a))
    recs.append(dict(id=f"{DS}/{qid}_{a}_{b}", dataset=DS, subset="turn1", kind="text", question=q, context=[],
                     criterion=CRIT, win=win, lose=lose, tie=tie,
                     meta=dict(question_id=qid, votes={"win": max(nA, nB), "lose": min(nA, nB), "tie": nT}, n_votes=nA + nB + nT)))
recs = orient(stratified_cap(recs))


def mat(r):
    r["meta"]["models"] = [c[1] for c in r["cand"]]; r["cand"] = [c[0] for c in r["cand"]]; return r


write(DS, recs, mat, {"pairs_turn1": len(G), "votes_turn1": len(d)}, "Turn 1 only; votes aggregated per unordered model pair.")
