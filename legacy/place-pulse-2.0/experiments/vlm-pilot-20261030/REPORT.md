# Place Pulse VLM pilot

300 unique image-disjoint pairs across 6 categories. 0 unresolved-majority pairs; 299/300 pairs have only one recorded vote. Seed: 20261030.

This is a zero-shot pilot, not a claim to reproduce a particular paper's protocol. Labels reflect subjective image perceptions. Small category samples and sparse votes limit model-ranking conclusions.

## Overall results

Percentile 95% CIs use stratified pair bootstrap; AB/BA stay together. Invalid or missing predictions receive zero in primary metrics. Majority-unresolved pairs are excluded from accuracy but retained for vote agreement.

| Model | Direct AB/BA mean accuracy [95% CI] | Score-derived accuracy [95% CI] | Swap consistency | Direct valid | Score-pair valid | Score tie rate |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | 63.0% [58.0%, 68.0%] | 64.0% [59.3%, 68.7%] | 78.0% | 100.0% | 100.0% | 23.3% |
| Qwen3.5-9B | 66.7% [61.7%, 71.3%] | 62.2% [58.0%, 66.3%] | 82.7% | 100.0% | 100.0% | 39.0% |

## Category results

| Model | Category | AB accuracy | BA restored accuracy | Mean direct accuracy | Score accuracy (ties 0.5) | Score non-tie accuracy |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | beautiful | 68.0% | 70.0% | 69.0% | 70.0% | 77.8% |
| Qwen3-VL-8B-Instruct | boring | 58.0% | 56.0% | 57.0% | 54.0% | 55.0% |
| Qwen3-VL-8B-Instruct | depressing | 50.0% | 60.0% | 55.0% | 57.0% | 59.0% |
| Qwen3-VL-8B-Instruct | lively | 52.0% | 54.0% | 53.0% | 58.0% | 62.5% |
| Qwen3-VL-8B-Instruct | safety | 78.0% | 74.0% | 76.0% | 78.0% | 83.3% |
| Qwen3-VL-8B-Instruct | wealthy | 66.0% | 70.0% | 68.0% | 67.0% | 70.7% |
| Qwen3.5-9B | beautiful | 74.0% | 74.0% | 74.0% | 68.0% | 82.1% |
| Qwen3.5-9B | boring | 62.0% | 64.0% | 63.0% | 61.0% | 66.7% |
| Qwen3.5-9B | depressing | 56.0% | 52.0% | 54.0% | 51.0% | 52.0% |
| Qwen3.5-9B | lively | 60.0% | 58.0% | 59.0% | 56.0% | 60.0% |
| Qwen3.5-9B | safety | 82.0% | 82.0% | 82.0% | 72.0% | 86.7% |
| Qwen3.5-9B | wealthy | 68.0% | 68.0% | 68.0% | 65.0% | 70.3% |

## Vote agreement and coverage

Vote agreement assigns half-credit to human equal votes and averages per unique pair. It is agreement with the observed votes, not calibrated model confidence. Score-derived ties also receive 0.5 agreement.

| Model | Direct vote agreement | Score vote agreement | Records / expected | Sum inference time (seconds) |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | 63.0% | 64.0% | 1200 / 1200 | 34.5 |
| Qwen3.5-9B | 66.7% | 62.2% | 1200 / 1200 | 57.1 |

## Paired model differences

Differences are first model minus second model, with identical bootstrap pairs. Values are percentage points.

| Comparison | Direct accuracy difference [95% CI] | Score accuracy difference [95% CI] |
|---|---:|---:|
| Qwen3-VL-8B-Instruct − Qwen3.5-9B | -3.7% [-6.7%, -0.7%] | 1.8% [-1.5%, 5.3%] |

## Reproducibility and interpretation

- Pilot SHA-256: `c78a100f23c6acda79bf288e7c8f4d40b9e937bbccdce57d2731efee3e3ad01d`.
- Bootstrap: 2000 replicates, seed 20261030; category sample counts held fixed.
- `pilot.jsonl` stores canonical image IDs, images, questions and all available votes on selected pairs. A is the lexicographically smaller image ID.
- `pair_ab` displays canonical A then B; `pair_ba` displays B then A. A/B model predictions name display positions; BA is inverted during scoring.
- `score_a` and `score_b` independently score the target attribute on an integer 1–10 scale. Higher boring/depressing scores mean more boring/depressing.
- `summary.json` contains all denominators, validity rates, conditional metrics, per-category CIs, model differences, sample provenance and prediction hashes.
- Human ties never break binary-vote ties. Missing/invalid model outputs are not silently omitted from primary metrics.
- No model training or train/test split is performed. Pair-level image disjointness prevents exact image reuse within this pilot; it does not establish geographic independence or absence from pretraining.
- Raw Q-scores are not used as pairwise ground truth. Preserve model revisions, prompts, dtype and runtime configuration alongside each model's predictions.
