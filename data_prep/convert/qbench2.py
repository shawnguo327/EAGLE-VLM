"""Q-Bench2 (q-future/Q-Bench2-HF, dev split = only split with answers) -> items/qbench2.jsonl
Low-level quality MCQ over two images. Only questions whose gold answer identifies one image as better/worse on a
quality attribute with an unambiguous 'good' direction are converted:
  type 0 (yes/no)  "Is the first image sharper than the second?"  -> only gold 'Yes' (a 'No' may mean 'equal')
  type 1 (which)   "Which image is more affected by noise?"        -> only when options are whole images
  type 2 (compare) "Compared to the first image, how is the clarity of the second image?" -> skip 'about the same'
Skipped: 'both images ...' questions, region-level questions, distortion-type questions, attributes without a
quality direction (brightness, lighting strength, size)."""
import sys, pathlib, re, glob; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pandas as pd
from common import *

DS = "qbench2"
SELF = [  # phrase -> polarity (+1 = the described image is better), longest match first
    ("less aesthetically pleasing", -1), ("more insufficient", -1), ("composition is better", 1), ("a lot of fake", -1),
    ("higher authenticity", 1), ("more monotonous", -1), ("more colorful", 1), ("more realistic", 1), ("less realistic", -1),
    ("more authentic", 1), ("less authentic", -1), ("more real", 1), ("less real", -1), ("looks more realistic", 1),
    ("more vivid", 1), ("less vivid", -1), ("more sufficient", 1), ("less sufficient", -1), ("more adequate", 1),
    ("less adequate", -1), ("less clear", -1), ("more blurry", -1), ("more blurred", -1), ("a bit blurry", -1),
    ("blurrier", -1), ("not as good", -1), ("less sharp", -1), ("more truthful", 1), ("clearer", 1), ("sharper", 1), ("better", 1), ("worse", -1), ("poorer", -1), ("not been", 1)]
MAG = [("more", 1), ("most", 1), ("higher", 1), ("richer", 1), ("rich", 1), ("abundant", 1), ("severe", 1), ("serious", 1),
       ("larger", 1), ("bigger", 1), ("less", -1), ("lower", -1), ("slighter", -1), ("smaller", -1)]
NEG_ATTR = r"weak light|nois|blur|overexpos|underexpos|distort|artifact|monoton|fake"
POS_ATTR = r"colou?r|texture|detail|clarity|clear|sharp|realis|authentic|\breal\b|focus|composition|aesthetic|sufficien|adequa|\btrue\b|truth|fidelity"
SUBSETS = [("noise", r"nois"), ("exposure", r"overexpos|underexpos|exposure"), ("clarity", r"clar|clear|sharp|blur|focus"),
           ("realism", r"realis|authentic|\breal\b|\btrue\b|truth|fidelity|fake"), ("color", r"colou?r|vivid|monoton"),
           ("texture", r"texture|detail"), ("composition", r"composition"), ("lighting", r"weak light|light|illumin|sufficien|adequa"),
           ("aesthetics", r"aesthetic")]
CRIT = {"noise": "Prefer the image with less noise.",
        "exposure": "Prefer the image with better exposure (less over- or underexposure).",
        "clarity": "Prefer the image that is sharper and clearer (less blurry).",
        "realism": "Prefer the image that looks more realistic and authentic.",
        "color": "Prefer the image with richer, more vivid color.",
        "texture": "Prefer the image with richer texture detail.",
        "composition": "Prefer the image with better composition.",
        "lighting": "Prefer the image with more sufficient lighting.",
        "aesthetics": "Prefer the image that is more aesthetically pleasing."}
QTEXT = "Compare the low-level visual quality of the two images."


def polarity(phrase, ctx):
    p = re.sub(r"\b(much|slightly|significantly|a little|very|quite|far)\b", " ", phrase.lower())
    p = re.sub(r"\s+", " ", p).strip()
    if re.search(r"same|similar|\bequal", p):
        return None
    for k, v in SELF:
        if k in p:
            return v
    attr = 0
    if re.search(NEG_ATTR, ctx + " " + p): attr = -1
    elif re.search(POS_ATTR, ctx + " " + p): attr = 1
    for k, v in MAG:
        if re.search(rf"\b{k}", p):
            return v * attr if attr else None
    return None


def which(s):
    s = s.lower()
    f, sc = "first" in s, "second" in s
    return 0 if f and not sc else 1 if sc and not f else None


def subset_of(text):
    t = text.lower()
    for name, rx in SUBSETS:
        if re.search(rx, t):
            return name
    return None


def parse(r):
    """-> (better_index, subset) or (None, reason)."""
    q = re.sub(r"\s+", " ", r.question).strip()
    ql = q.lower()
    ans = r["option" + str("ABCD".index(r.correct_choice))]
    if re.search(r"\bboth\b|these two images|the two images", ql):
        return None, "both-images question"
    if r.question_type == 0:
        if ans.strip().lower() != "yes":
            return None, "yes/no with gold No"
        ql2 = ql.replace("photo", "image")
        m = re.match(r"^(?:is|does|has|are) the (?:(.+?) (?:of|in|on) the )?(first|second) images? (.+?)(?:(?: compared)? (?:than|to)(?: that of| that on| those in| in| on| the .+? of)? the (first|second) image.*|\?)$", ql2)
        m2 = re.match(r"^compared to the (first|second) image, (?:is|does|has) the (?:(.+?) of the )?(first|second) image (.+?)\?$", ql2)
        if m2:
            m = type("M", (), {"group": lambda self, i, g=(None, m2.group(2), m2.group(3), m2.group(4)): g[i]})()
        if not m:
            return None, "unparsed yes/no"
        attr, subj, comp = m.group(1) or "", m.group(2), m.group(3)
        pol = polarity(comp, attr + " " + comp)
        if pol is None:
            return None, "no quality direction"
        s = 0 if subj == "first" else 1
        return (s if pol > 0 else 1 - s), subset_of(attr + " " + comp)
    if r.question_type == 1:
        opts = [r[f"option{i}"] for i in range(4) if r[f"option{i}"] and r[f"option{i}"] != "N/A"]
        if not (ql.startswith("which") and "image" in ql.split("?")[0][:40]) or any(
                which(o) is None or len(o.split()) > 4 for o in opts):
            return None, "region/distortion-type question"
        m = re.match(r"^which (?:of the following )?images? (?:below )?(.+?)\?", ql)
        pred = m.group(1) if m else ql
        neg = bool(re.search(r"\bnot\b|n't", pred))
        pred2 = re.sub(r"does not have|has not been|not", "", pred)
        pol = polarity(pred2, pred2)
        if pol is None and re.search(NEG_ATTR, pred2):
            pol = -1  # "has a serious overexposure problem"
        if pol is None:
            return None, "no quality direction"
        if neg:
            pol = -pol
        s = which(ans)
        return (s if pol > 0 else 1 - s), subset_of(pred)
    # type 2
    pol = polarity(ans, ql)
    if pol is None:
        return None, "same / no quality direction"
    mentions = re.findall(r"\b(first|second) (?:image|photo)", ql)
    if len(mentions) < 2 or mentions[0] == mentions[-1]:
        return None, "unparsed compare"
    target = mentions[-1] if re.match(r"^(compared|relative|in comparison)", ql) else mentions[0]
    t = 0 if target == "first" else 1
    return (t if pol > 0 else 1 - t), subset_of(ql + " " + ans)


fs = sorted(glob.glob(str(RAW / "q-future__Q-Bench2-HF/data/dev-*.parquet")))
d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
recs, reasons = [], collections.Counter()
for ix, r in d.iterrows():
    b, s = parse(r)
    if b is None or s is None:
        reasons[s if b is None else "no subset"] += 1
        continue
    ims = [(r.image1, 1), (r.image2, 2)]
    ans = r["option" + str("ABCD".index(r.correct_choice))]
    recs.append(dict(id=f"{DS}/{r.id}", dataset=DS, subset=s, kind="img", question=QTEXT, context=[], criterion=CRIT[s],
                     win=ims[b], lose=ims[1 - b],
                     meta=dict(orig_id=int(r.id), question_type=int(r.question_type), orig_question=r.question,
                               options=[r[f"option{i}"] for i in range(4)], answer=ans, better_orig_image=b + 1)))
full = {"total_dev": len(d), "convertible": len(recs), "skipped": dict(reasons)}
recs = orient(stratified_cap(recs))


def mat(r):
    base = r["id"].split("/")[1]
    r["meta"]["orig_image_order"] = [c[1] for c in r["cand"]]
    r["cand"] = [save_img(c[0], DS, f"{base}_img{c[1]}") for c in r["cand"]]
    return r


if __name__ == "__main__":
    if "--debug" in sys.argv:
        for r in recs:
            m = r["meta"]
            print(f"{r['subset']:11s} better=img{m['better_orig_image']} | {m['orig_question']} -> {m['answer']}")
        sys.exit()
    write(DS, recs, mat, full, "Only answers that identify one image as better on an attribute with a clear quality direction; see script docstring.")
