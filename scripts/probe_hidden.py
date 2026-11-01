"""Phase B analysis: is "close" vs "cannot tell" linearly readable from the judge's last-token hidden state even
though the A/B logits do not show it?  Inputs: <dir>/<ds>__<model>*.hid/*.npz from judge_bench --hidden.
Per pair the feature is the order-averaged hidden state (AB, BA) of one condition at one layer, reduced by PCA (fit on
the training folds only). L2-logistic, 5-fold cross-fitting. Targets: human tie; judge error (strict pairs);
MMRB2 annotator split. Also cross-dataset transfer of the tie probe (train on the other tie datasets).
usage (from $EAGLEVLM_DATA): python probe_hidden.py --model qwen3vl8b --dirs runs_h100/B/hidden [--cond bin] [--pca 256]"""
import argparse, ast, collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from eaglev_stats import auroc, load_records

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--dirs", required=True)
ap.add_argument("--cond", default="bin"); ap.add_argument("--pca", type=int, default=256); ap.add_argument("--l2", type=float, default=10.0)
ap.add_argument("--results", default="results", help="dir with <ds>__<model>.jsonl for the bin verdict (error target)")
ap.add_argument("--ds", default="genai_arena,mllm_judge,pp,editreward,mmrb2,vlrb,mmrb,mjbench,imagerewarddb,qbench2,mmrlhf")
a = ap.parse_args()


def load_hidden(ds):
    H = collections.defaultdict(dict); layers = None
    for d in a.dirs.split(","):
        for f in glob.glob(f"{d}/{ds}__{a.model}*.hid/*.npz"):
            z = np.load(f); layers = z["layers"]
            for i, c, o, h in zip(z["id"], z["cond"], z["order"], z["h"]):
                if c == a.cond:
                    H[str(i)][str(o)] = h.astype(np.float32)
    return {i: (v["AB"] + v["BA"]) / 2 for i, v in H.items() if "AB" in v and "BA" in v}, layers


def fit_predict(Xtr, ytr, Xte, k, l2):
    mu = Xtr.mean(0); U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); P = Vt[:k].T
    Ztr, Zte = (Xtr - mu) @ P, (Xte - mu) @ P; sd = Ztr.std(0) + 1e-6; Ztr, Zte = Ztr / sd, Zte / sd
    Ztr, Zte = np.c_[np.ones(len(Ztr)), Ztr], np.c_[np.ones(len(Zte)), Zte]; w = np.zeros(Ztr.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-Ztr @ w)); g = Ztr.T @ (p - ytr) + l2 * np.r_[0, w[1:]]
        Hs = (Ztr * (p * (1 - p))[:, None]).T @ Ztr + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(Hs, g)
    return Zte @ w


def cv(X, y, k=5):
    s = np.zeros(len(y)); idx = np.random.default_rng(0).permutation(len(y))
    for f in range(k):
        te = idx[f::k]; tr = np.setdiff1d(idx, te)
        s[te] = fit_predict(X[tr], y[tr].astype(float), X[te], min(a.pca, len(tr) - 1), a.l2)
    return s


data = {}; out = {}
for ds in a.ds.split(","):
    Hd, layers = load_hidden(ds)
    if not Hd:
        continue
    items = {json.loads(l)["id"]: json.loads(l) for l in open(f"items/{ds}.jsonl")}
    recs = {r["id"]: r for r in load_records(f"items/{ds}.jsonl", f"{a.results}/{ds}__{a.model}.jsonl")} \
        if os.path.exists(f"{a.results}/{ds}__{a.model}.jsonl") else {}
    ids = [i for i in Hd if i in items]
    X = np.stack([Hd[i] for i in ids])                   # n x L x d
    tie = np.array([items[i]["label"] == "tie" for i in ids])
    y = np.array([items[i]["label"] == 0 for i in ids])
    pa = np.array([recs[i]["pa"] if i in recs else np.nan for i in ids])
    err = (~tie) & ((pa > .5) != y) & ~np.isnan(pa)
    data[ds] = (X, tie, layers)
    res = {"n": len(ids), "layers": layers.tolist()}
    for li, lf in enumerate(layers):
        Xl = X[:, li, :]; r = {}
        if tie.sum() >= 15:
            r["tie_auroc"] = round(auroc(cv(Xl, tie), tie), 3)
        st = ~tie & ~np.isnan(pa)
        if err[st].sum() >= 15:
            r["err_auroc"] = round(auroc(cv(Xl[st], err[st]), err[st]), 3)
            r["err_auroc_logit_baseline"] = round(auroc(np.abs(pa[st] - .5) * -1, err[st]), 3)
        if ds == "mmrb2":
            ha = []
            for i in ids:
                mt = items[i]["meta"]; mt = ast.literal_eval(mt) if isinstance(mt, str) else mt
                h = mt["human_annotations"]; ha.append(ast.literal_eval(h) if isinstance(h, str) else h)
            ha = np.array(ha, float); split = ~(((ha <= 3).all(1)) | ((ha >= 4).all(1)))
            r["split_auroc"] = round(auroc(cv(Xl, split), split), 3)
        res[f"layer{lf}"] = r
    out[ds] = res; print(ds, json.dumps(res), flush=True)

# cross-dataset tie transfer: train on the other tie datasets, test on one
tie_ds = [d for d, (X, t, _) in data.items() if t.sum() >= 15]
if len(tie_ds) >= 2:
    L = len(next(iter(data.values()))[2]); out["tie_transfer"] = {}
    for d in tie_ds:
        for li in range(L):
            Xtr = np.concatenate([data[e][0][:, li] for e in tie_ds if e != d]); ytr = np.concatenate([data[e][1] for e in tie_ds if e != d])
            s = fit_predict(Xtr, ytr.astype(float), data[d][0][:, li], min(a.pca, len(ytr) - 1), a.l2)
            out["tie_transfer"][f"{d}|layer{data[d][2][li]}"] = round(auroc(s, data[d][1]), 3)
    print("tie transfer", out["tie_transfer"])
os.makedirs("paper_tables", exist_ok=True)
json.dump(out, open(f"paper_tables/probe_hidden_{a.model}_{a.cond}.json", "w"), indent=1)
