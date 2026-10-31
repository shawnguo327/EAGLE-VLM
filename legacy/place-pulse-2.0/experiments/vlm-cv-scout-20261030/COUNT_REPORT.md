# Objective counting and visual comparison pilot

60 image-disjoint COCO pairs / 120 images. Each configuration performs two independent counts and two order-swapped comparisons per pair.

This is a diagnostic pilot. COCO box counts are the annotation reference, not a guarantee that every visually discernible instance is labelled. Images with target-class crowd annotations are excluded.

Always predicting count zero gives 40.0% exact count accuracy and 1.533 MAE. Always choosing the same displayed position gives 50% AB/BA-averaged comparison accuracy.

| Configuration | Count exact | Count MAE (valid only) | Direct AB/BA accuracy | Swap consistency | Count validity | Pair validity |
|---|---:|---:|---:|---:|---:|---:|
| MiniCPM-V-4.6-16x | 75.0% | 0.467 | 70.0% | 43.3% | 100.0% | 100.0% |
| MiniCPM-V-4.6-4x | 74.2% | 0.458 | 70.8% | 45.0% | 100.0% | 100.0% |
| Qwen3-VL-8B-Instruct | 84.2% | 0.250 | 91.7% | 96.7% | 100.0% | 100.0% |
| Qwen3.5-9B | 83.3% | 0.242 | 96.7% | 100.0% | 100.0% | 100.0% |

## When both independent counts are correct

These subsets are selected separately by model output; differences between rows are descriptive, not causal.

| Configuration | Pairs | Direct accuracy | Swap consistency |
|---|---:|---:|---:|
| MiniCPM-V-4.6-16x | 34 | 67.6% | 35.3% |
| MiniCPM-V-4.6-4x | 32 | 71.9% | 43.8% |
| Qwen3-VL-8B-Instruct | 42 | 96.4% | 97.6% |
| Qwen3.5-9B | 41 | 100.0% | 100.0% |

## Counting beyond object presence

Positive-GT count exact uses all positive-GT images, with invalid outputs counted as incorrect. MAE uses valid positive-GT outputs. Presence F1/balanced accuracy use valid outputs only; class-specific coverage and complete confusion matrices are in the JSON.

| Configuration | Positive-GT images | Positive count exact | Positive MAE (valid n) | Zero-GT specificity (all n) | Presence balanced accuracy | Presence F1 | Valid coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM-V-4.6-16x | 72 | 58.3% | 0.778 (72) | 100.0% (48) | 96.5% | 96.4% | 100.0% |
| MiniCPM-V-4.6-4x | 72 | 56.9% | 0.764 (72) | 100.0% (48) | 96.5% | 96.4% | 100.0% |
| Qwen3-VL-8B-Instruct | 72 | 75.0% | 0.389 (72) | 97.9% (48) | 98.3% | 98.6% | 100.0% |
| Qwen3.5-9B | 72 | 72.2% | 0.403 (72) | 100.0% (48) | 97.9% | 97.9% | 100.0% |

## Presence versus genuine count comparison

| Configuration | Pair group | Pairs | Direct accuracy | Swap consistency |
|---|---|---:|---:|---:|
| MiniCPM-V-4.6-16x | both_positive | 12 | 70.8% | 58.3% |
| MiniCPM-V-4.6-16x | zero_vs_positive | 48 | 69.8% | 39.6% |
| MiniCPM-V-4.6-4x | both_positive | 12 | 66.7% | 50.0% |
| MiniCPM-V-4.6-4x | zero_vs_positive | 48 | 71.9% | 43.8% |
| Qwen3-VL-8B-Instruct | both_positive | 12 | 70.8% | 91.7% |
| Qwen3-VL-8B-Instruct | zero_vs_positive | 48 | 96.9% | 97.9% |
| Qwen3.5-9B | both_positive | 12 | 91.7% | 100.0% |
| Qwen3.5-9B | zero_vs_positive | 48 | 97.9% | 100.0% |

## Near versus far count gaps

| Configuration | Gap | Pairs | Count exact | Direct accuracy | Swap consistency |
|---|---|---:|---:|---:|---:|
| MiniCPM-V-4.6-16x | far | 30 | 65.0% | 80.0% | 60.0% |
| MiniCPM-V-4.6-16x | near | 30 | 85.0% | 60.0% | 26.7% |
| MiniCPM-V-4.6-4x | far | 30 | 65.0% | 76.7% | 53.3% |
| MiniCPM-V-4.6-4x | near | 30 | 83.3% | 65.0% | 36.7% |
| Qwen3-VL-8B-Instruct | far | 30 | 80.0% | 95.0% | 96.7% |
| Qwen3-VL-8B-Instruct | near | 30 | 88.3% | 88.3% | 96.7% |
| Qwen3.5-9B | far | 30 | 80.0% | 100.0% | 100.0% |
| Qwen3.5-9B | near | 30 | 86.7% | 93.3% | 100.0% |

## Compression: MiniCPM-V-4.6-16x minus MiniCPM-V-4.6-4x

Same image pairs, frozen weights, and question templates. This changes the supported visual compression mode, not necessarily only token count. Differences below are percentage points, with pair bootstrap 95% intervals.

| Metric | Difference (pp) | 95% CI (pp) |
|---|---:|---:|
| count_exact_accuracy | 0.83 | [-2.50, 4.17] |
| pair_mean_accuracy | -0.83 | [-4.17, 3.33] |
| swap_consistency | -1.67 | [-8.33, 6.67] |
| both_counts_exact_rate | 3.33 | [-3.33, 10.00] |
| count_induced_pair_accuracy_ties_half | 0.83 | [-0.83, 2.50] |

On the fixed 32-pair subset where 4x counted both images correctly, direct accuracy is 71.9% at 4x and 68.8% at 16x. This baseline-defined diagnostic is not proof of mediation or causality.

## Coverage and reproducibility

- Pair manifest SHA-256: `546c8a1a90e7d8c2c2ad93f6333f8ed90c154f09cd6610143e6f75e39d973a2c`.
- `count_pair_info.json` records fixed-source sample provenance, exclusions, pair matching and actual category/gap distributions.
- `count_summary.json` includes all denominators, category results, paired model/mode deltas and conditioning diagnostics.
- Invalid or missing outputs are zero for primary accuracies and consistency; invalid counts are excluded from MAE and coverage is reported.
- Counts are integers 0–99. They must not be parsed as the Place Pulse 1–10 rating scale.
- A/B comparison outputs refer to displayed positions; swapped BA outputs are restored before scoring.
- Near means count gap exactly 1; far means gap at least 2. No tied-GT pair is included.
- Do not infer that stable single-image factual outputs fully explain or mediate pairwise choices.
- MiniCPM-V-4.6-16x: 240/240 records; complete=True.
- MiniCPM-V-4.6-4x: 240/240 records; complete=True.
- Qwen3-VL-8B-Instruct: 240/240 records; complete=True.
- Qwen3.5-9B: 240/240 records; complete=True.
