"""LMArena human preference 55k (lmarena-ai/arena-human-preference-55k, public) -> items/arena55k.jsonl  [kind=text]
Text-domain analogue of GenAI-Arena: one crowd vote per battle (winner_model_a / winner_model_b / winner_tie).
Filter: single-turn, mostly-ASCII prompt and responses (English proxy), non-empty, <= 6000 chars each. Water-filled
over vote type (win / tie) to EAGLEVLM_CAP pairs in total (pack: 3000 -> 1500 ties + 1500 strict)."""
import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from common import *

DS = "arena55k"
SRC = RAW / "lmarena-ai__arena-human-preference-55k" / "train.csv"
CRIT = "Prefer the response that is more helpful, relevant, accurate and appropriate for the user's request."
d = pd.read_csv(SRC)
full = {"battles": len(d)}


def one(x):
    try:
        v = json.loads(x)
    except Exception:
        return None
    return v if isinstance(v, list) else None


def ascii_share(t):
    return sum(c.isascii() for c in t) / max(1, len(t))


recs = []
for r in d.itertuples():
    P, A, B = one(r.prompt), one(r.response_a), one(r.response_b)
    if not P or not A or not B or len(P) != 1 or len(A) != 1 or len(B) != 1:
        continue
    q, a, b = P[0], A[0], B[0]
    if not all(isinstance(x, str) and x.strip() for x in (q, a, b)):
        continue
    if max(len(q), len(a), len(b)) > 6000 or min(ascii_share(q), ascii_share(a), ascii_share(b)) < .97:
        continue
    tie = bool(r.winner_tie)
    win, lose = ((a, r.model_a), (b, r.model_b)) if (tie or r.winner_model_a) else ((b, r.model_b), (a, r.model_a))
    recs.append(dict(id=f"{DS}/{r.id}", dataset=DS, subset="tie" if tie else "win", kind="text", question=q, context=[],
                     criterion=CRIT, win=win, lose=lose, tie=tie, meta=dict(orig_id=int(r.id))))
full["single_turn_english_usable"] = len(recs)
recs = orient(stratified_cap(recs))


def mat(r):
    r["meta"]["models"] = [c[1] for c in r["cand"]]; r["cand"] = [c[0] for c in r["cand"]]; return r


write(DS, recs, mat, full, "single-turn, ASCII-dominant, <= 6000 chars; subset = vote type (win/tie), water-filled.")
