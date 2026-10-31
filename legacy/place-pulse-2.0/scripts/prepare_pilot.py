#!/usr/bin/env python3
"""Create a reproducible, image-disjoint Place Pulse pair pilot (stdlib only)."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import statistics
import tempfile

CATEGORIES = ("beautiful", "boring", "depressing", "lively", "safety", "wealthy")


def normalized_vote(row):
    """Return canonical image IDs/paths and a vote in canonical coordinates."""
    left, right = row["left_id"], row["right_id"]
    if left == right:
        raise ValueError("Self-comparison in usable source: " + str(row.get("vote_id")))
    canonical_left = left < right
    if canonical_left:
        a, b, ap, bp = left, right, row["left_image"], row["right_image"]
    else:
        a, b, ap, bp = right, left, row["right_image"], row["left_image"]
    choice = row["choice"]
    if choice == "equal":
        vote = "equal"
    elif choice in ("left", "right"):
        vote = "a" if ((choice == "left") == canonical_left) else "b"
    else:
        raise ValueError("Unexpected choice in usable source")
    return row["category"], a, b, ap, bp, vote


def priority(seed, category, a, b):
    return hashlib.sha256(f"{seed}\0{category}\0{a}\0{b}".encode()).hexdigest()


def prepare(root, out, per_category=50, seed=20261030):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if per_category < 1:
        raise ValueError("per-category must be positive")
    questions = {}
    source_counts = collections.Counter()
    with tempfile.TemporaryDirectory(prefix="pair-aggregation-", dir=out) as temp:
        db = sqlite3.connect(str(Path(temp) / "pairs.sqlite3"))
        db.executescript("""
          PRAGMA journal_mode=OFF;
          PRAGMA synchronous=OFF;
          CREATE TABLE pairs (
            category TEXT, a TEXT, b TEXT, ap TEXT, bp TEXT, priority TEXT,
            n_a INTEGER, n_b INTEGER, n_equal INTEGER DEFAULT 0,
            PRIMARY KEY (category,a,b)
          ) WITHOUT ROWID;
        """)
        batch = []
        insert = """INSERT INTO pairs(category,a,b,ap,bp,priority,n_a,n_b)
          VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(category,a,b) DO UPDATE SET
          n_a=n_a+excluded.n_a, n_b=n_b+excluded.n_b"""
        with gzip.open(root / "annotations/pairs_binary.jsonl.gz", "rt") as f:
            for line in f:
                row = json.loads(line)
                source_counts["binary_votes"] += 1
                if row["left_id"] == row["right_id"]:
                    source_counts["excluded_binary_self_comparisons"] += 1
                    continue
                category, a, b, ap, bp, vote = normalized_vote(row)
                if category not in CATEGORIES or vote not in ("a", "b"):
                    raise ValueError("Unexpected binary source row")
                if category in questions and questions[category] != row["question"]:
                    raise ValueError("Inconsistent question for " + category)
                questions[category] = row["question"]
                batch.append((category, a, b, ap, bp, priority(seed, category, a, b),
                              int(vote == "a"), int(vote == "b")))
                if len(batch) >= 10000:
                    db.executemany(insert, batch)
                    batch.clear()
        if batch:
            db.executemany(insert, batch)
        db.commit()
        print("Aggregated binary votes:", source_counts["binary_votes"], flush=True)
        batch = []
        update = "UPDATE pairs SET n_equal=n_equal+1 WHERE category=? AND a=? AND b=?"
        before = db.total_changes
        with gzip.open(root / "annotations/pairs_ties.jsonl.gz", "rt") as f:
            for line in f:
                row = json.loads(line)
                source_counts["tie_votes"] += 1
                if row["left_id"] == row["right_id"]:
                    source_counts["excluded_tie_self_comparisons"] += 1
                    continue
                category, a, b, ap, bp, vote = normalized_vote(row)
                if vote != "equal":
                    raise ValueError("Non-tie in tie source")
                batch.append((category, a, b))
                if len(batch) >= 10000:
                    db.executemany(update, batch)
                    batch.clear()
        if batch:
            db.executemany(update, batch)
        source_counts["tie_votes_on_binary_candidate_pairs"] = db.total_changes - before
        db.commit()
        populations = dict(db.execute("SELECT category,COUNT(*) FROM pairs GROUP BY category"))
        if any(populations.get(c, 0) < per_category for c in CATEGORIES):
            raise ValueError("Not enough candidate pairs: " + repr(populations))
        db.execute("CREATE INDEX ordering ON pairs(category,priority)")
        category_order = list(CATEGORIES)
        random.Random(seed).shuffle(category_order)
        cursors = {c: db.execute("SELECT a,b,ap,bp,n_a,n_b,n_equal FROM pairs WHERE category=? ORDER BY priority", (c,))
                   for c in category_order}
        used_images, selected = set(), []
        skipped = collections.Counter()
        for _ in range(per_category):
            for c in category_order:
                for a, b, ap, bp, n_a, n_b, n_equal in cursors[c]:
                    if a in used_images or b in used_images:
                        skipped[c] += 1
                        continue
                    if not (root / ap).is_file() or not (root / bp).is_file():
                        raise ValueError("Selected pair references a missing image")
                    used_images.update((a, b))
                    selected.append({
                        "id": f"{c}:{a}:{b}", "category": c, "question": questions[c],
                        "a_id": a, "b_id": b, "a_image": ap, "b_image": bp,
                        "n_a": n_a, "n_b": n_b, "n_equal": n_equal,
                        "majority": "A" if n_a > n_b else "B" if n_b > n_a else None,
                    })
                    break
                else:
                    raise ValueError("Image-disjoint matching exhausted category " + c)
        db.close()
    payload = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in selected)
    pilot_path = out / "pilot.jsonl"
    pilot_path.write_text(payload)
    def group_stats(rows):
        return {
            "pairs": len(rows),
            "unresolved_majority": sum(r["majority"] is None for r in rows),
            "single_vote_pairs": sum(r["n_a"] + r["n_b"] + r["n_equal"] == 1 for r in rows),
            "single_vote_fraction": sum(r["n_a"] + r["n_b"] + r["n_equal"] == 1 for r in rows) / len(rows),
            "median_total_votes_per_pair": statistics.median(r["n_a"] + r["n_b"] + r["n_equal"] for r in rows),
            "median_binary_votes_per_pair": statistics.median(r["n_a"] + r["n_b"] for r in rows),
            "binary_votes": sum(r["n_a"] + r["n_b"] for r in rows),
            "equal_votes": sum(r["n_equal"] for r in rows),
            "majority_A": sum(r["majority"] == "A" for r in rows),
            "majority_B": sum(r["majority"] == "B" for r in rows),
        }
    info = {
        "seed": seed, "per_category": per_category, "root": str(root.resolve()),
        "pilot_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "unique_images": len(used_images), "source_counts": dict(source_counts),
        "candidate_unique_pairs_by_category": populations,
        "sampling": "Seeded SHA256 priority within category; shuffled category round-robin greedy matching; no image ID reused globally; no filtering by label, vote count or consensus.",
        "category_round_robin_order": category_order,
        "image_reuse_candidates_skipped": dict(skipped),
        "orientation": "A is lexicographically smaller image ID; pair_ab displays A then B, pair_ba displays B then A. Predictions A/B always name display position, not canonical image.",
        "population": "Unique unordered category/image pairs with at least one usable binary vote; tie-only pairs are not candidates. All available equal votes on candidates are added.",
        "overall": group_stats(selected),
        "by_category": {c: group_stats([r for r in selected if r["category"] == c]) for c in CATEGORIES},
    }
    (out / "sample_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(info, indent=2), flush=True)
    return info


def self_test():
    r = {"left_id": "z", "right_id": "a", "left_image": "z.jpg", "right_image": "a.jpg", "category": "safety", "choice": "left"}
    assert normalized_vote(r) == ("safety", "a", "z", "a.jpg", "z.jpg", "b")
    r["choice"] = "right"
    assert normalized_vote(r)[-1] == "a"
    r["choice"] = "equal"
    assert normalized_vote(r)[-1] == "equal"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--per-category", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    self_test()
    if args.self_test:
        print("self-test passed")
    else:
        if args.root is None or args.out is None:
            p.error("--root and --out are required")
        prepare(args.root, args.out, args.per_category, args.seed)
