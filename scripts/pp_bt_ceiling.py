"""How predictable is a single Place Pulse vote from image identity alone? (empirical side of Prop. 'aleatoric cap')
Per category: drop near-duplicate votes (same pair within a day), fit an L2 Bradley-Terry model with a left-bias term on
90% of votes, and report held-out accuracy and explained variance 1 - MSE/Var(y).
usage: python pp_bt_ceiling.py [annotations_dir]"""
import collections, gzip, json, sys
from datetime import datetime
import numpy as np
from scipy import optimize, sparse

ANN = sys.argv[1] if len(sys.argv) > 1 else "/media/shawn/C/papers/VLM/place-pulse-2.0/annotations"
rows = collections.defaultdict(list); seen = {}; dup = 0
for l in gzip.open(f"{ANN}/pairs_binary.jsonl.gz", "rt"):
    d = json.loads(l); a, b = d["left_id"], d["right_id"]
    t = datetime.fromisoformat(d["timestamp"][:19]).timestamp(); k = (d["category"], min(a, b), max(a, b))
    if k in seen and t - seen[k] < 86400:
        dup += 1; continue
    seen[k] = t
    rows[d["category"]].append((a, b, 1 if d["choice"] == "left" else 0))
print("dropped near-duplicates", dup)
rng = np.random.default_rng(0)
for cat, R in sorted(rows.items()):
    ids = {}
    for a, b, _ in R:
        ids.setdefault(a, len(ids)); ids.setdefault(b, len(ids))
    A = np.array([ids[a] for a, _, _ in R]); B = np.array([ids[b] for _, b, _ in R]); y = np.array([r[2] for r in R], float)
    te = rng.random(len(R)) < 0.1; tr = ~te; m = tr.sum()
    X = sparse.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[A[tr], B[tr]])), shape=(m, len(ids)))
    best = None
    for l2 in (0.5, 2.0):
        def f(w):
            th, b0 = w[:-1], w[-1]; z = X @ th + b0; p = 1 / (1 + np.exp(-z))
            return np.sum(np.logaddexp(0, z) - y[tr] * z) + l2 * th @ th, np.r_[X.T @ (p - y[tr]) + 2 * l2 * th, np.sum(p - y[tr])]
        w = optimize.minimize(f, np.zeros(len(ids) + 1), jac=True, method="L-BFGS-B", options={"maxiter": 300}).x
        p = 1 / (1 + np.exp(-(w[:-1][A[te]] - w[:-1][B[te]] + w[-1])))
        mse = np.mean((y[te] - p) ** 2)
        if best is None or mse < best[0]:
            best = (mse, np.var(y[te]), np.mean((p > .5) == (y[te] == 1)), l2, w[-1])
    mse, vy, acc, l2, b0 = best
    print(f"{cat:11s} votes={len(R):7d} images={len(ids):6d} heldout_acc={acc:.3f} explained={1 - mse / vy:.3f} left_bias={b0:+.3f} l2={l2}")
