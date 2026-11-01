"""Chatbot Arena conversations (lmsys/chatbot_arena_conversations) -> items/chatbot_arena.jsonl  [kind=text]
Text-domain analogue of GenAI-Arena: one crowd vote per battle; winner model_a / model_b / tie / tie (bothbad).
Filter: English, single turn, not flagged toxic/redacted; ties kept as "tie" (meta.vote_type), capped stratified by
vote type so ties are well represented (EAGLEVLM_CAP per vote type, default 1500)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from common import *

DS = "chatbot_arena"
SRC = RAW / "lmsys__chatbot_arena_conversations" / "data"
CRIT = "Prefer the response that is more helpful, relevant, accurate and appropriate for the user's request."
d = pd.read_parquet(sorted(SRC.glob("train-*.parquet"))[0])
full = {"battles": len(d)}
d = d[(d.language == "English") & (d.turn == 1)]
for col in ("toxic_chat_tag", "openai_moderation"):
    if col in d:
        d = d[[not (isinstance(x, dict) and (x.get("flagged") or any((v or {}).get("flagged") for v in x.values() if isinstance(v, dict)))) for x in d[col]]]
full["english_single_turn_clean"] = len(d)
recs = []
for r in d.itertuples():
    ua = [m["content"] for m in r.conversation_a if m["role"] == "user"]; ra = [m["content"] for m in r.conversation_a if m["role"] == "assistant"]
    rb = [m["content"] for m in r.conversation_b if m["role"] == "assistant"]
    if not ua or not ra or not rb or not ra[0].strip() or not rb[0].strip():
        continue
    v = r.winner; tie = v.startswith("tie")
    win, lose = ((ra[0], r.model_a), (rb[0], r.model_b)) if v != "model_b" else ((rb[0], r.model_b), (ra[0], r.model_a))
    vt = {"model_a": "win", "model_b": "win", "tie": "tie", "tie (bothbad)": "bothbad"}.get(v, v)
    recs.append(dict(id=f"{DS}/{r.question_id}", dataset=DS, subset=vt, kind="text", question=ua[0], context=[],
                     criterion=CRIT, win=win, lose=lose, tie=tie, meta=dict(vote_type=v, judge=r.judge)))
full["usable"] = len(recs)
recs = orient(stratified_cap(recs, cap=3 * CAP))


def mat(r):
    r["meta"]["models"] = [c[1] for c in r["cand"]]; r["cand"] = [c[0] for c in r["cand"]]; return r


write(DS, recs, mat, full, "English single-turn, moderation-clean; subset = vote type (win/tie/bothbad), water-filled.")
