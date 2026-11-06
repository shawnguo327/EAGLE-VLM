"""Data access for the analysis pipeline: items (human labels) and raw judge outputs (token log-probs per condition/order).
Source directories are an explicit, ordered list taken from the experiment config; for duplicate (item, condition, order)
records the first source in that order wins (same rule as the original scripts/tie_mech.load).  Every file read is
recorded in a manifest (path, lines, md5) that is saved next to the results."""
import collections, glob, hashlib, json, math, os, sys

SUFFIXES = ("", "_edit", "_s32", "_rest", "_rest2")      # item-file complements that share ids/orientation with the base set
NO_REST = {"mllm_judge_sp", "genaibench", "polaris", "agiqa3k"}


def softmax_lp(lp):
    m = max(lp.values()); e = {k: math.exp(v - m) for k, v in lp.items()}; z = sum(e.values())
    return {k: v / z for k, v in e.items()}


class Store:
    def __init__(self, items_dirs, output_dirs, items_override=None, snapshot=None):
        self.items_dirs = list(items_dirs); self.output_dirs = list(output_dirs); self.snapshot = snapshot
        self.items_override = items_override or {}      # dataset -> explicit item file (e.g. the 600-prompt GenAI-Bench snapshot)
        self.manifest = {}

    def _note(self, path, kind):
        if path in self.manifest:
            return
        h = hashlib.md5(); n = 0
        with open(path, "rb") as f:
            for line in f:
                h.update(line); n += 1
        self.manifest[path] = dict(kind=kind, path=path, lines=n, md5=h.hexdigest())

    # ---------------- items
    def items(self, ds, rest=None):
        rest = (ds not in NO_REST) if rest is None else rest
        out = {}
        if self.snapshot:                                     # frozen snapshot (eaglevlm.pipeline.freeze)
            import gzip
            for name in [ds] + ([ds + "_rest", ds + "_rest2"] if rest else []):
                p = f"{self.snapshot}/items/{name}.jsonl.gz"
                if os.path.exists(p):
                    self._note(p, "items"); out.update({r["id"]: r for r in map(json.loads, gzip.open(p, "rt"))})
            return out
        if ds in self.items_override:
            p = self.items_override[ds]; self._note(p, "items")
            return {r["id"]: r for r in map(json.loads, open(p))}
        for name in [ds] + ([ds + "_rest", ds + "_rest2"] if rest else []):
            for d in self.items_dirs:
                p = f"{d}/{name}.jsonl"
                if os.path.exists(p):
                    self._note(p, "items"); out.update({r["id"]: r for r in map(json.loads, open(p))}); break
        return out

    # ---------------- judge outputs
    def load(self, ds, model, conds=None):
        if self.snapshot:
            p = f"{self.snapshot}/outputs/dataset={ds}/judge={model}.parquet"
            if not os.path.exists(p):
                return {}
            import pyarrow.parquet as pq
            self._note(p, "outputs"); t = pq.read_table(p).to_pydict(); R = {}
            toks = [k for k in t if k not in ("item_id", "cond", "order")]
            for n in range(len(t["item_id"])):
                if conds is None or t["cond"][n] in conds:
                    R.setdefault(t["item_id"][n], {})[(t["cond"][n], t["order"][n])] = {k: t[k][n] for k in toks if t[k][n] is not None}
            return R
        R, src = {}, {}
        for d in [d for pat in self.output_dirs for d in sorted(glob.glob(pat))]:
            for suf in SUFFIXES:
                for f in glob.glob(f"{d}/{ds}{suf}__{model}.jsonl") + glob.glob(f"{d}/{ds}{suf}__{model}.s*.jsonl"):
                    used = False
                    for l in open(f):
                        try:
                            x = json.loads(l)
                        except ValueError:          # a line cut by a running writer
                            continue
                        if x.get("cond") == "score" and "/runs/" not in d and "a100_sync" not in d:
                            continue                # 11-01 pointwise runs used the old (pairwise-criterion) prompt
                        if "lp" in x and (conds is None or x["cond"] in conds):
                            k = (x["id"], x["cond"], x["order"])
                            if k in src and src[k] != f:
                                continue            # first source wins
                            src[k] = f; R.setdefault(x["id"], {})[(x["cond"], x["order"])] = softmax_lp(x["lp"]); used = True
                    if used:
                        self._note(f, "outputs")
        return R

    def write_manifest(self, path):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["kind", "path", "lines", "md5"]); w.writeheader()
            for v in sorted(self.manifest.values(), key=lambda v: (v["kind"], v["path"])):
                w.writerow(v)


# ---------------- evaluation item sets (same corrections as scripts/label_noise_lib.eval_items)
def _strat(x):
    return (x["subset"], str(x["label"]) == "tie")


def proportional(store, ds, I, seed=0):
    import random
    full_path = next((f"{d}/{ds}_full.jsonl" for d in store.items_dirs if os.path.exists(f"{d}/{ds}_full.jsonl")), None)
    full = collections.Counter(_strat(json.loads(l)) for l in open(full_path))
    nf = sum(full.values()); have = collections.defaultdict(list)
    for i, x in I.items():
        have[_strat(x)].append(i)
    N = min(len(have[s]) * nf // c for s, c in full.items() if c)
    rng = random.Random(seed); keep = {}
    for s, c in full.items():
        q = round(N * c / nf)
        for i in rng.sample(sorted(have[s]), min(q, len(have[s]))):
            keep[i] = I[i]
    return keep


def dedup_pairs(I):
    g = collections.defaultdict(list)
    for i in sorted(I):
        g[frozenset(I[i]["cand"])].append(i)
    keep = {}
    for ids in g.values():
        labs = {str(I[i]["label"]) if str(I[i]["label"]) == "tie" else I[i]["cand"][int(I[i]["label"])] for i in ids}
        if len(labs) == 1:
            keep[ids[0]] = I[ids[0]]
    return keep


def eval_items(store, ds, I):
    if ds in ("imagerewarddb", "mllm_judge"):
        full_path = next((f"{d}/{ds}_full.jsonl" for d in store.items_dirs if os.path.exists(f"{d}/{ds}_full.jsonl")), None)
        if full_path is None or len(I) == sum(1 for _ in open(full_path)):
            return I
        return proportional(store, ds, I)
    if ds == "pickapic":
        return dedup_pairs(I)
    return I


def dataset_items(store, spec):
    """'<dataset>[/<subset>]' -> evaluation items"""
    ds, sub = (spec.split("/") + [None])[:2]
    I = eval_items(store, ds, store.items(ds))
    return ds, {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
