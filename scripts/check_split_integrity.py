"""Split integrity of a finished certify / router / ablate run (review 11-07, N1): every split index is an integer in
0 .. n_splits - 1, no (dataset, judge, method or variant/rule, alpha, split) key repeats, and no key is missing a split.
Standard library only.

    python scripts/check_split_integrity.py /media/shawn/C/papers/VLM/results/certify_strong/latest [...more runs]
"""
import collections
import csv
import gzip
import os
import sys


def check(run):
    path = os.path.join(run, "splits.csv.gz")
    n_splits = None
    cfg = os.path.join(run, "config.yaml")
    if os.path.exists(cfg):
        for line in open(cfg):
            if line.startswith("n_splits:"):
                n_splits = int(line.split(":", 1)[1])
    seen, per_key, bad = set(), collections.defaultdict(set), []
    rows = 0
    with gzip.open(path, "rt") as fh:
        rd = csv.DictReader(fh)
        cols = [c for c in ("dataset", "judge", "method", "variant", "rule", "alpha") if c in rd.fieldnames]
        for r in rd:
            rows += 1
            s = r["split"]
            if not s.lstrip("-").isdigit():
                bad.append(s[:40]); continue
            key = tuple(r[c] for c in cols)
            if key + (int(s),) in seen:
                bad.append(f"duplicate {key} split {s}")
            seen.add(key + (int(s),)); per_key[key].add(int(s))
    n = n_splits or max(max(v) for v in per_key.values()) + 1
    missing = {k: sorted(set(range(n)) - v)[:5] for k, v in per_key.items() if v != set(range(n))}
    ok = not bad and not missing
    print(f"{'OK  ' if ok else 'FAIL'} {run}: {rows} rows, {len(per_key)} keys x {n} splits; "
          f"{len(bad)} non-integer or duplicate, {len(missing)} keys with missing splits")
    for b in bad[:5]:
        print("   ", b)
    for k, v in list(missing.items())[:5]:
        print("    missing", k, v)
    return ok


if __name__ == "__main__":
    sys.exit(0 if all([check(os.path.realpath(p)) for p in sys.argv[1:]]) else 1)
