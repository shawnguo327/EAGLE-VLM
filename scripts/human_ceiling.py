"""Human ceiling for pairwise claims: how well does ONE human agree with the others on the same pair?
If a single human's 'A better' / 'tie' claims disagree with the other annotators' label at a rate above alpha, no judge
can make many strict claims at that alpha on that benchmark - a low coverage there reflects the labels, not the method.
  GenAI-Bench   3 ratings (1-5) per image: rater k's preference (sign of r_k(A) - r_k(B), tie if equal) against the label
                from the mean of the other two raters (tie if equal means).  Rater k is the k-th rating of each image,
                not necessarily the same person across images (an upper bound on single-human noise).
  MT-Bench      pairs with >= 2 expert votes: leave-one-vote-out against the majority of the remaining votes.
Reported: share of strict human claims that are wrong, share of tie claims that are wrong, 3-way agreement, and the
same numbers when the reference label is decided (no tie).
usage (judgebench dir): python human_ceiling.py --out paper_tables/human_ceiling.md"""
import argparse, ast, collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper_tables/human_ceiling.md"); a = ap.parse_args()


def sgn(d):
    return 0 if d > 1e-9 else (1 if d < -1e-9 else 2)          # 0 = A better, 1 = B better, 2 = tie


def summarize(pairs):
    """pairs: list of (human_claim, reference) with values in {0, 1, 2}"""
    h = np.array([x for x, _ in pairs]); r = np.array([y for _, y in pairs])
    st = h != 2; ti = h == 2
    return dict(n=len(h), strict_share=st.mean(), strict_err=(h[st] != r[st]).mean() if st.any() else np.nan,
                tie_share=ti.mean(), tie_err=(r[ti] != 2).mean() if ti.any() else np.nan, agree3=(h == r).mean(),
                ref_tie=(r == 2).mean(), strict_err_on_decided=((h != r) & st & (r != 2)).sum() / max((st & (r != 2)).sum(), 1))


L = ["# Human ceiling: one annotator against the others", "",
     "strict err = share of a single human's 'A better / B better' claims that the other annotators' label contradicts (a tie label counts as an error, as for a judge); "
     "tie err = share of a single human's tie claims where the others decided; agree3 = 3-way agreement.", "",
     "| dataset | reference | n (pair x held-out annotator) | reference tie rate | human strict share | human strict err | human strict err (reference decided) | human tie share | human tie err | 3-way agreement |",
     "|---|---|---|---|---|---|---|---|---|---|"]
# ---- GenAI-Bench
I = items("genaibench", rest=False); P = []
for x in I.values():
    ra, rb = ast.literal_eval(x["meta"]["ratings"]) if isinstance(x["meta"]["ratings"], str) else x["meta"]["ratings"]
    if len(ra) != 3 or len(rb) != 3:
        continue
    for k in range(3):
        oa = np.mean([ra[t] for t in range(3) if t != k]); ob = np.mean([rb[t] for t in range(3) if t != k])
        P.append((sgn(ra[k] - rb[k]), sgn(oa - ob)))
d = summarize(P)
L.append(f"| GenAI-Bench | mean of the other 2 raters | {d['n']} | {d['ref_tie']:.2f} | {d['strict_share']:.2f} | {d['strict_err']:.3f} | {d['strict_err_on_decided']:.3f} | "
         f"{d['tie_share']:.2f} | {d['tie_err']:.3f} | {d['agree3']:.3f} |")
# ---- MT-Bench (votes stored as counts of win / lose / tie for candidate A)
I = items("mtbench_human"); P = []
for x in I.values():
    v = x["meta"].get("votes"); v = ast.literal_eval(v) if isinstance(v, str) else v
    if not v:
        continue
    votes = [0] * v.get("win", 0) + [1] * v.get("lose", 0) + [2] * v.get("tie", 0)
    if len(votes) < 2:
        continue
    for k in range(len(votes)):
        rest = votes[:k] + votes[k + 1:]; c = collections.Counter(rest); top = c.most_common()
        ref = 2 if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]     # split majority -> tie
        P.append((votes[k], ref))
d = summarize(P)
L.append(f"| MT-Bench | majority of the other expert votes | {d['n']} | {d['ref_tie']:.2f} | {d['strict_share']:.2f} | {d['strict_err']:.3f} | {d['strict_err_on_decided']:.3f} | "
         f"{d['tie_share']:.2f} | {d['tie_err']:.3f} | {d['agree3']:.3f} |")
open(a.out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
