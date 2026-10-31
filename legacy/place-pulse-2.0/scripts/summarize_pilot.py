#!/usr/bin/env python3
"""Summarize paired Place Pulse VLM pilots with stratified pair bootstraps."""
import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

TASKS = ("pair_ab", "pair_ba", "score_a", "score_b")
METRICS = (
    "pair_ab_accuracy", "pair_ba_accuracy", "pair_mean_accuracy",
    "pair_ab_valid_rate", "pair_ba_valid_rate", "pair_valid_rate",
    "swap_consistency", "swap_consistency_among_valid",
    "score_a_valid_rate", "score_b_valid_rate", "score_valid_rate", "score_pair_valid_rate",
    "score_accuracy_tie_halfcredit", "score_tie_rate", "score_non_tie_accuracy",
    "pair_ab_vote_agreement", "pair_ba_vote_agreement", "pair_mean_vote_agreement", "score_vote_agreement",
    "pair_first_image_rate_among_valid",
)
DIFF_METRICS = ("pair_mean_accuracy", "score_accuracy_tie_halfcredit", "swap_consistency",
                "pair_mean_vote_agreement", "score_vote_agreement")


def valid_score(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 1 <= x <= 10 and float(x).is_integer()


def canonical_choice(prediction, task):
    if prediction not in ("A", "B"):
        return None
    if task == "pair_ba":
        return "B" if prediction == "A" else "A"
    return prediction


def pair_contributions(pair, predictions):
    """Each metric contributes a numerator/denominator for one UNIQUE pair."""
    raw_ab = predictions.get("pair_ab", {}).get("prediction")
    raw_ba = predictions.get("pair_ba", {}).get("prediction")
    ab = canonical_choice(raw_ab, "pair_ab")
    ba = canonical_choice(raw_ba, "pair_ba")
    sa = predictions.get("score_a", {}).get("prediction")
    sb = predictions.get("score_b", {}).get("prediction")
    va, vb, vsa, vsb = ab is not None, ba is not None, valid_score(sa), valid_score(sb)
    both_scores = vsa and vsb
    score_tie = both_scores and sa == sb
    score_choice = ("A" if sa > sb else "B") if both_scores and not score_tie else None
    gold = pair["majority"]
    resolved = gold in ("A", "B")
    votes = pair["n_a"] + pair["n_b"] + pair["n_equal"]
    if votes <= 0:
        raise ValueError("Pair has no votes")
    pa = (pair["n_a"] + 0.5 * pair["n_equal"]) / votes
    def agreement(choice):
        return pa if choice == "A" else 1 - pa if choice == "B" else 0.0
    ca = float(resolved and ab == gold)
    cb = float(resolved and ba == gold)
    cs = (0.5 if score_tie else float(score_choice == gold)) if resolved and both_scores else 0.0
    score_agreement = (0.5 if score_tie else agreement(score_choice)) if both_scores else 0.0
    d = {
        "pair_ab_accuracy": (ca, int(resolved)),
        "pair_ba_accuracy": (cb, int(resolved)),
        "pair_mean_accuracy": ((ca + cb) / 2, int(resolved)),
        "pair_ab_valid_rate": (int(va), 1), "pair_ba_valid_rate": (int(vb), 1),
        "pair_valid_rate": ((int(va) + int(vb)) / 2, 1),
        "swap_consistency": (int(va and vb and ab == ba), 1),
        "swap_consistency_among_valid": (int(va and vb and ab == ba), int(va and vb)),
        "score_a_valid_rate": (int(vsa), 1), "score_b_valid_rate": (int(vsb), 1),
        "score_valid_rate": ((int(vsa) + int(vsb)) / 2, 1),
        "score_pair_valid_rate": (int(both_scores), 1),
        "score_accuracy_tie_halfcredit": (cs, int(resolved)),
        "score_tie_rate": (int(score_tie), int(both_scores)),
        "score_non_tie_accuracy": (float(resolved and both_scores and not score_tie and score_choice == gold), int(resolved and both_scores and not score_tie)),
        "pair_ab_vote_agreement": (agreement(ab), 1),
        "pair_ba_vote_agreement": (agreement(ba), 1),
        "pair_mean_vote_agreement": ((agreement(ab) + agreement(ba)) / 2, 1),
        "score_vote_agreement": (score_agreement, 1),
        "pair_first_image_rate_among_valid": (int(raw_ab == "A") + int(raw_ba == "A"), int(va) + int(vb)),
    }
    return tuple(d[m] for m in METRICS)


def totals(vectors, indices, weights=None):
    out = [[0.0, 0.0] for _ in METRICS]
    for i in indices:
        w = 1 if weights is None else weights[i]
        if not w:
            continue
        for j, (num, den) in enumerate(vectors[i]):
            out[j][0] += w * num
            out[j][1] += w * den
    return out


def estimates(sums):
    return [num / den if den else None for num, den in sums]


def percentile(values, p):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    at = (len(v) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return v[lo] + (v[hi] - v[lo]) * (at - lo)


def ci(values):
    return [percentile(values, .025), percentile(values, .975)] if any(x is not None for x in values) else None


def read_inputs(experiment):
    pilot_bytes = (experiment / "pilot.jsonl").read_bytes()
    pairs = [json.loads(line) for line in pilot_bytes.splitlines() if line.strip()]
    if not pairs or len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError("Empty pilot or duplicate IDs")
    ids = {p["id"]: p for p in pairs}
    model_records = {}
    for path in sorted((experiment / "results").glob("*/predictions.jsonl")):
        per_pair, elapsed, invalid_elapsed = {}, [], 0
        rows = 0
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["id"] not in ids or r["task"] not in TASKS:
                    raise ValueError("Unknown prediction ID/task in " + str(path))
                if r["category"] != ids[r["id"]]["category"]:
                    raise ValueError("Prediction category disagrees with pilot")
                bucket = per_pair.setdefault(r["id"], {})
                if r["task"] in bucket:
                    raise ValueError("Duplicate prediction for " + r["id"] + "/" + r["task"])
                bucket[r["task"]] = r
                rows += 1
                t = r.get("elapsed_seconds")
                if isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t >= 0:
                    elapsed.append(t)
                else:
                    invalid_elapsed += 1
        model_records[path.parent.name] = {
            "predictions": per_pair,
            "record_count": rows,
            "expected_record_count": len(pairs) * len(TASKS),
            "missing_record_count": len(pairs) * len(TASKS) - rows,
            "sum_inference_elapsed_seconds": sum(elapsed),
            "median_inference_elapsed_seconds": statistics.median(elapsed) if elapsed else None,
            "records_without_valid_elapsed_seconds": invalid_elapsed,
            "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if not model_records:
        raise ValueError("No results/*/predictions.jsonl files found")
    info_path = experiment / "sample_info.json"
    sample_info = json.loads(info_path.read_text()) if info_path.exists() else {}
    sample_sha = hashlib.sha256(pilot_bytes).hexdigest()
    if sample_info.get("pilot_sha256", sample_sha) != sample_sha:
        raise ValueError("pilot.jsonl hash differs from sample_info")
    return pairs, model_records, sample_info, sample_sha


def summarize(experiment, bootstraps=2000, seed=20261030):
    experiment = Path(experiment)
    pairs, models, sample_info, sample_sha = read_inputs(experiment)
    by_category = collections.defaultdict(list)
    for i, pair in enumerate(pairs):
        by_category[pair["category"]].append(i)
    categories = sorted(by_category)
    groups = {"overall": list(range(len(pairs))), **{c: by_category[c] for c in categories}}
    vectors = {name: [pair_contributions(p, model["predictions"].get(p["id"], {})) for p in pairs]
               for name, model in models.items()}
    point_sums = {name: {g: totals(v, ix) for g, ix in groups.items()} for name, v in vectors.items()}
    samples = {name: {g: [[] for _ in METRICS] for g in groups} for name in models}
    rng = random.Random(seed)
    for _ in range(bootstraps):
        weights = [0] * len(pairs)
        for ix in by_category.values():
            for _ in ix:
                weights[ix[rng.randrange(len(ix))]] += 1
        for name, v in vectors.items():
            category_sums = {c: totals(v, by_category[c], weights) for c in categories}
            overall_sums = [[sum(category_sums[c][j][k] for c in categories) for k in (0, 1)] for j in range(len(METRICS))]
            for g, sums in [("overall", overall_sums), *category_sums.items()]:
                for j, value in enumerate(estimates(sums)):
                    samples[name][g][j].append(value)
    summary = {
        "pilot_sha256": sample_sha, "sample_info": sample_info,
        "bootstrap": {"replicates": bootstraps, "seed": seed, "method": "Percentile 95% CI; resample unique image-disjoint pairs within each category; identical resamples for all models and tasks."},
        "metric_definitions": {
            "pair_orientation": "A/B predictions mean display positions. pair_ba is inverted to canonical image IDs before scoring.",
            "majority": "A if n_a>n_b, B if n_b>n_a, otherwise unresolved. Equal votes do not break a binary-vote tie.",
            "invalid": "Missing/unparseable predictions count as 0 for primary accuracy, vote agreement and swap consistency. Conditional metrics explicitly restrict denominators.",
            "pair_mean_accuracy": "Mean of AB and restored BA correctness on resolved-majority pairs; each unique pair remains one bootstrap unit.",
            "score_accuracy_tie_halfcredit": "Independent integer 1-10 scores; greater score wins; equal valid scores receive 0.5, invalid score pairs 0, unresolved-majority pairs excluded.",
            "score_tie_rate": "Equal scores / pairs with two valid scores.",
            "score_non_tie_accuracy": "Correct / resolved-majority pairs with two valid and unequal scores; diagnostic conditional metric.",
            "swap_consistency": "Same actual image chosen in AB and BA / all pairs; invalid predictions count as inconsistent.",
            "vote_agreement": "For each pair p_A=(n_a+0.5*n_equal)/(n_a+n_b+n_equal); choosing canonical A scores p_A, B scores 1-p_A, score tie 0.5, invalid 0. Average over unique pairs, including unresolved ones; not model confidence.",
            "overall_weighting": "Pooled unique-pair denominators; equal sampled category counts. Resolved-majority denominators can differ by category.",
        },
        "models": {}, "paired_model_differences": [],
    }
    for name, model in models.items():
        result = {k: v for k, v in model.items() if k != "predictions"}
        result["complete"] = result["missing_record_count"] == 0
        for g in groups:
            metrics = {}
            for j, metric in enumerate(METRICS):
                num, den = point_sums[name][g][j]
                metrics[metric] = {"estimate": num / den if den else None, "numerator": num,
                                   "denominator": den, "ci95": ci(samples[name][g][j])}
            if g == "overall":
                result["overall"] = metrics
            else:
                result.setdefault("by_category", {})[g] = metrics
        summary["models"][name] = result
    for first, second in itertools.combinations(models, 2):
        comparison = {"first_model": first, "second_model": second, "direction": "first minus second", "overall": {}, "by_category": {}}
        for g in groups:
            metrics = {}
            for metric in DIFF_METRICS:
                j = METRICS.index(metric)
                x = estimates(point_sums[first][g])[j]
                y = estimates(point_sums[second][g])[j]
                draws = [a - b if a is not None and b is not None else None for a, b in zip(samples[first][g][j], samples[second][g][j])]
                metrics[metric] = {"estimate": x - y if x is not None and y is not None else None, "ci95": ci(draws)}
            if g == "overall":
                comparison["overall"] = metrics
            else:
                comparison["by_category"][g] = metrics
        summary["paired_model_differences"].append(comparison)
    (experiment / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (experiment / "REPORT.md").write_text(report(summary, pairs))
    return summary


def fmt(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def cell(metric, interval=False):
    value = fmt(metric["estimate"])
    if interval and metric.get("ci95"):
        value += " [" + ", ".join(fmt(x) for x in metric["ci95"]) + "]"
    return value


def report(summary, pairs):
    categories = sorted({p["category"] for p in pairs})
    unresolved = sum(p["majority"] is None for p in pairs)
    single = sum(p["n_a"] + p["n_b"] + p["n_equal"] == 1 for p in pairs)
    lines = ["# Place Pulse VLM pilot", "",
        f"{len(pairs)} unique image-disjoint pairs across {len(categories)} categories. {unresolved} unresolved-majority pairs; {single}/{len(pairs)} pairs have only one recorded vote. Seed: {summary['bootstrap']['seed']}.", "",
        "This is a zero-shot pilot, not a claim to reproduce a particular paper's protocol. Labels reflect subjective image perceptions. Small category samples and sparse votes limit model-ranking conclusions.", "",
        "## Overall results", "",
        "Percentile 95% CIs use stratified pair bootstrap; AB/BA stay together. Invalid or missing predictions receive zero in primary metrics. Majority-unresolved pairs are excluded from accuracy but retained for vote agreement.", "",
        "| Model | Direct AB/BA mean accuracy [95% CI] | Score-derived accuracy [95% CI] | Swap consistency | Direct valid | Score-pair valid | Score tie rate |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in summary["models"].items():
        m = result["overall"]
        lines.append(f"| {name} | {cell(m['pair_mean_accuracy'], True)} | {cell(m['score_accuracy_tie_halfcredit'], True)} | {cell(m['swap_consistency'])} | {cell(m['pair_valid_rate'])} | {cell(m['score_pair_valid_rate'])} | {cell(m['score_tie_rate'])} |")
    lines += ["", "## Category results", "", "| Model | Category | AB accuracy | BA restored accuracy | Mean direct accuracy | Score accuracy (ties 0.5) | Score non-tie accuracy |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, result in summary["models"].items():
        for category in categories:
            m = result["by_category"][category]
            lines.append(f"| {name} | {category} | {cell(m['pair_ab_accuracy'])} | {cell(m['pair_ba_accuracy'])} | {cell(m['pair_mean_accuracy'])} | {cell(m['score_accuracy_tie_halfcredit'])} | {cell(m['score_non_tie_accuracy'])} |")
    lines += ["", "## Vote agreement and coverage", "", "Vote agreement assigns half-credit to human equal votes and averages per unique pair. It is agreement with the observed votes, not calibrated model confidence. Score-derived ties also receive 0.5 agreement.", "", "| Model | Direct vote agreement | Score vote agreement | Records / expected | Sum inference time (seconds) |", "|---|---:|---:|---:|---:|"]
    for name, result in summary["models"].items():
        m = result["overall"]
        lines.append(f"| {name} | {cell(m['pair_mean_vote_agreement'])} | {cell(m['score_vote_agreement'])} | {result['record_count']} / {result['expected_record_count']} | {result['sum_inference_elapsed_seconds']:.1f} |")
    if summary["paired_model_differences"]:
        lines += ["", "## Paired model differences", "", "Differences are first model minus second model, with identical bootstrap pairs. Values are percentage points.", "", "| Comparison | Direct accuracy difference [95% CI] | Score accuracy difference [95% CI] |", "|---|---:|---:|"]
        for d in summary["paired_model_differences"]:
            lines.append(f"| {d['first_model']} − {d['second_model']} | {cell(d['overall']['pair_mean_accuracy'], True)} | {cell(d['overall']['score_accuracy_tie_halfcredit'], True)} |")
    lines += ["", "## Reproducibility and interpretation", "",
        f"- Pilot SHA-256: `{summary['pilot_sha256']}`.",
        f"- Bootstrap: {summary['bootstrap']['replicates']} replicates, seed {summary['bootstrap']['seed']}; category sample counts held fixed.",
        "- `pilot.jsonl` stores canonical image IDs, images, questions and all available votes on selected pairs. A is the lexicographically smaller image ID.",
        "- `pair_ab` displays canonical A then B; `pair_ba` displays B then A. A/B model predictions name display positions; BA is inverted during scoring.",
        "- `score_a` and `score_b` independently score the target attribute on an integer 1–10 scale. Higher boring/depressing scores mean more boring/depressing.",
        "- `summary.json` contains all denominators, validity rates, conditional metrics, per-category CIs, model differences, sample provenance and prediction hashes.",
        "- Human ties never break binary-vote ties. Missing/invalid model outputs are not silently omitted from primary metrics.",
        "- No model training or train/test split is performed. Pair-level image disjointness prevents exact image reuse within this pilot; it does not establish geographic independence or absence from pretraining.",
        "- Raw Q-scores are not used as pairwise ground truth. Preserve model revisions, prompts, dtype and runtime configuration alongside each model's predictions.", ""]
    return "\n".join(lines)


def self_test():
    p = {"majority": "A", "n_a": 3, "n_b": 1, "n_equal": 2}
    pred = {"pair_ab": {"prediction": "A"}, "pair_ba": {"prediction": "B"},
            "score_a": {"prediction": 5}, "score_b": {"prediction": 5}}
    def metric(rows, name):
        num, den = rows[METRICS.index(name)]
        return num / den if den else None
    v = pair_contributions(p, pred)
    assert metric(v, "pair_mean_accuracy") == 1
    assert metric(v, "swap_consistency") == 1
    assert metric(v, "score_accuracy_tie_halfcredit") == .5
    assert metric(v, "score_tie_rate") == 1
    assert metric(v, "score_non_tie_accuracy") is None
    assert math.isclose(metric(v, "pair_mean_vote_agreement"), 2 / 3)
    pred["pair_ba"]["prediction"] = "A"  # Same POSITION, opposite actual image.
    v = pair_contributions(p, pred)
    assert metric(v, "pair_mean_accuracy") == .5
    assert metric(v, "swap_consistency") == 0
    pred["pair_ba"]["prediction"] = None
    pred["score_b"]["prediction"] = None
    v = pair_contributions(p, pred)
    assert metric(v, "pair_mean_accuracy") == .5  # Invalid retained in denominator.
    assert metric(v, "score_accuracy_tie_halfcredit") == 0
    assert metric(v, "score_tie_rate") is None
    p.update(majority=None, n_a=1, n_b=1, n_equal=2)
    v = pair_contributions(p, pred)
    assert metric(v, "pair_mean_accuracy") is None
    assert metric(v, "pair_mean_vote_agreement") == .25
    assert valid_score(1) and valid_score(10.0)
    assert not valid_score(True) and not valid_score("5") and not valid_score(5.5)
    assert percentile([0, 1], .5) == .5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("self-test passed")
    else:
        if args.experiment is None:
            parser.error("--experiment is required")
        if args.bootstraps < 1:
            parser.error("--bootstraps must be positive")
        result = summarize(args.experiment, args.bootstraps, args.seed)
        print(json.dumps({name: {"complete": r["complete"], "pair_mean_accuracy": r["overall"]["pair_mean_accuracy"], "score_accuracy_tie_halfcredit": r["overall"]["score_accuracy_tie_halfcredit"]} for name, r in result["models"].items()}, indent=2))
