"""Three-level decomposition sets built on the controlled tie set (make_synth_ties.py; uses its .gen.jsonl).
Each DV pair differs in exactly one visual fact (phrase o in the correct answer R, n in W1).  We ask the same judge:
  L1 acquisition   synth_perc  "which of these two statements is accurate about the image?"  S(o) vs S(n), where S(.) is
                               the sentence of R containing the phrase.  Both edits of every base (e1 = W1's, e2 = W2's).
  L2 localisation  synth_loc   the original pair, with the difference pointed out in the criterion (order-free wording),
                               so the judge no longer has to find and align the differing claims.  DV, TB and DT pairs.
  L3 verdict       synth_tie   the original pair (already run: S1_synth).
Per item:  L1 wrong -> acquisition failure;  L1 right, L3 wrong, L2 right -> localisation / comparison failure;
L1 right, L2 wrong -> weighting / criterion failure.
usage: python make_synth_levels.py --gen items/synth_tie.jsonl.gen.jsonl --tie items/synth_tie.jsonl --outdir items"""
import argparse, json, re, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True); ap.add_argument("--tie", required=True); ap.add_argument("--outdir", required=True)
a = ap.parse_args()
G = {json.loads(l)["base"]: json.loads(l) for l in open(a.gen)}
T = [json.loads(l) for l in open(a.tie)]
base_item = {t["meta"]["base"]: t for t in T if t["subset"] == "DV"}


def sent(text, span):
    """the sentence of text containing span"""
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        if span in s:
            return s.strip()
    return span


flip = lambda iid: zlib.crc32(iid.encode()) % 2 == 1
perc, loc = [], []
for b, g in G.items():
    if b not in base_item:
        continue
    t = base_item[b]; R = g["R"]
    so = {}
    for e in ("E1", "E2"):
        o, n = g[e]
        s_o = sent(R, o); s_n = s_o.replace(o, n); so[e] = (s_o, s_n)
        iid = f"synth_perc/{e.lower()}/{b}"
        f = flip(iid); cand = [s_n, s_o] if f else [s_o, s_n]
        perc.append({"id": iid, "dataset": "synth_perc", "subset": e.lower(), "kind": "resp",
                     "question": "Which of these two statements is accurate about the image?", "context": t["context"],
                     "cand": cand, "criterion": "Prefer the statement that is accurate about the image.",
                     "label": 1 if f else 0, "meta": {"base": b, "orig": o, "repl": n}})
    (o1, n1), (o2, n2) = so["E1"], so["E2"]
    flaw = g["F"][len(R.rstrip()):].strip()
    hints = {"DV": f"Note: the two responses are identical except that one of them says \"{o1}\" where the other says \"{n1}\".",
             "TB": (f"Note: the two responses are identical except in two places: one of them says \"{n1}\" where the other says "
                    f"\"{o1}\", and one of them says \"{o2}\" where the other says \"{n2}\"."),
             "DT": f"Note: the two responses are identical except that one of them adds the sentence \"{flaw}\"."}
    for typ in ("DV", "TB", "DT"):
        src = next(x for x in T if x["id"] == f"synth_tie/{typ}/{b}")
        it = dict(src); it["id"] = f"synth_loc/{typ}/{b}"; it["dataset"] = "synth_loc"
        it["criterion"] = (src.get("criterion") or "Prefer the response that is more accurate about the image.") + " " + hints[typ]
        loc.append(it)
open(f"{a.outdir}/synth_perc.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in perc))
open(f"{a.outdir}/synth_loc.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in loc))
print(len(perc), "perception items;", len(loc), "localised pairs")
