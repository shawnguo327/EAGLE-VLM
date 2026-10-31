

# Follow-up round (evidence-augmented judge, hybrid gate, Qwen3-14B)

## A/C. Context-free judge: accuracy of picking the real day (unmatched pairs, n=150 per rung, order-averaged)

| rung | 8B plain | 8B +stats | 8B +stats+real-day primer | 8B stats-only+primer | 14B plain | 14B +stats | 14B stats-only+primer | LR per-rung (same pairs) | LR pooled (same pairs) | LR AUC per-rung |
|---|---|---|---|---|---|---|---|---|---|---|
| gauss1 | 0.52 [0.44, 0.60] | 0.51 [0.43, 0.59] | 0.58 [0.49, 0.66] | 0.65 [0.58, 0.73] | 0.43 [0.35, 0.51] | 0.53 [0.45, 0.61] | 0.66 [0.59, 0.73] | 0.89 | 0.83 | 0.907 |
| gauss2 | 0.51 [0.43, 0.59] | 0.59 [0.52, 0.67] | 0.67 [0.59, 0.75] | 0.76 [0.69, 0.83] | 0.54 [0.46, 0.63] | 0.59 [0.51, 0.66] | 0.81 [0.74, 0.87] | 0.94 | 0.91 | 0.962 |
| gauss4 | 0.61 [0.54, 0.69] | 0.61 [0.54, 0.69] | 0.76 [0.69, 0.82] | 0.89 [0.85, 0.94] | 0.71 [0.63, 0.78] | 0.74 [0.67, 0.81] | 0.89 [0.83, 0.93] | 0.98 | 0.94 | 0.976 |
| gauss8 | 0.73 [0.65, 0.79] | 0.69 [0.62, 0.77] | 0.86 [0.81, 0.91] | 0.95 [0.92, 0.99] | 0.83 [0.77, 0.89] | 0.89 [0.83, 0.93] | 0.95 [0.91, 0.98] | 0.99 | 0.91 | 0.993 |
| tshuffle | 0.47 [0.39, 0.55] | 0.46 [0.38, 0.55] | 0.55 [0.47, 0.63] | 0.63 [0.55, 0.71] | 0.68 [0.60, 0.75] | — | — | 0.93 | 0.63 | 0.895 |
| markov | 0.57 [0.49, 0.65] | 0.61 [0.54, 0.69] | 0.63 [0.55, 0.71] | 0.67 [0.60, 0.74] | 0.47 [0.39, 0.56] | — | — | 0.95 | 0.31 | 0.945 |
| llmgen | 0.42 [0.33, 0.50] | 0.45 [0.37, 0.53] | 0.42 [0.34, 0.50] | 0.51 [0.43, 0.59] | 0.51 [0.43, 0.59] | 0.47 [0.39, 0.55] | 0.47 [0.40, 0.55] | 0.71 | 0.31 | 0.693 |
| uniform | 0.93 [0.89, 0.97] | — | — | — | 0.99 [0.98, 1.00] | — | — | 1.00 | 1.00 | 1.000 |

**Calibration / position bias (real vs the 7 non-real rungs above; null = real vs real)**

| judge | n | acc | ECE | AUROC(Δp→correct) | acc@Δp≥0.6 (cov) | null mean Δp | P(A) | same pick both orders |
|---|---|---|---|---|---|---|---|---|
| 8B plain | 1050 | 0.548 | 0.170 | 0.565 | 0.607 (0.34) | 0.48 | 0.48 | 0.50 |
| 8B +stats | 1050 | 0.561 | 0.162 | 0.575 | 0.629 (0.35) | 0.50 | 0.47 | 0.52 |
| 8B +stats+real-day primer | 1050 | 0.639 | 0.125 | 0.636 | 0.724 (0.46) | 0.56 | 0.44 | 0.60 |
| 8B stats-only+primer | 1050 | 0.724 | 0.086 | 0.634 | 0.830 (0.29) | 0.41 | 0.20 | 0.35 |
| 14B plain | 1050 | 0.596 | 0.233 | 0.602 | 0.651 (0.64) | 0.64 | 0.56 | 0.76 |
| 14B +stats | 750 | 0.643 | 0.148 | 0.603 | 0.710 (0.47) | 0.47 | 0.76 | 0.49 |
| 14B stats-only+primer | 750 | 0.755 | 0.098 | 0.759 | 0.854 (0.69) | 0.51 | 0.67 | 0.67 |

**Same-subset comparison (gauss1/2/4/8 + llmgen only, n≤750):**

| judge | acc | ECE | AUROC(Δp) | acc@Δp≥0.6 (cov) |
|---|---|---|---|---|
| 8B plain | 0.559 | 0.159 | 0.588 | 0.645 (0.34) |
| 8B +stats | 0.571 | 0.154 | 0.591 | 0.662 (0.35) |
| 8B +stats+real-day primer | 0.659 | 0.111 | 0.654 | 0.752 (0.48) |
| 8B stats-only+primer | 0.753 | 0.102 | 0.640 | 0.848 (0.31) |
| 14B plain | 0.604 | 0.234 | 0.620 | 0.667 (0.64) |
| 14B +stats | 0.643 | 0.148 | 0.603 | 0.710 (0.47) |
| 14B stats-only+primer | 0.755 | 0.098 | 0.759 | 0.854 (0.69) |

(14B +stats was run only on gauss1/2/4/8, llmgen and the null pairs; its row covers those 5 rungs. Plain-8B row covers all 7.)

Spearman(judge p_real, pooled-LR pair probability) on the same pairs: 8B plain: -0.01, 8B +stats: 0.04, 8B +stats+real-day primer: 0.26, 8B stats-only+primer: 0.30, 14B plain: 0.02, 14B +stats: 0.06, 14B stats-only+primer: 0.44

**Qwen3-14B Bradley–Terry over 9 rungs:** gauss1 0.91 > real 0.85 > markov 0.80 > gauss2 0.77 > llmgen 0.70 > gauss4 0.29 > tshuffle 0.21 > gauss8 -0.48 > uniform -4.06; Kendall τ on noise ladder = 0.87 [0.73, 1.00]
14B adjacent-rung pair accuracy: gauss1>gauss2 0.53, gauss2>gauss4 0.69, gauss4>gauss8 0.76, gauss8>uniform 0.99

## B. Hybrid Δp-gated LR/judge (τ chosen on one random half of pairs, evaluated on the other, both directions averaged; pooled LR = one model for all degradations)

| judge | rung | judge | LR pooled | judge→LR (judge if Δp≥τ) | LR→judge (LR if Δ_LR≥τ) | avg-prob ensemble | LR per-rung | J→LRrung | LRrung→J |
|---|---|---|---|---|---|---|---|---|---|
| 8B plain | ALL (n=1050) | 0.548 | 0.693 | 0.687 | 0.686 | 0.683 | 0.912 | 0.912 | 0.896 |
| 8B plain | gauss1 (n=150) | 0.520 | 0.833 | 0.813 | 0.827 | 0.767 | 0.887 | 0.867 | 0.913 |
| 8B plain | tshuffle (n=150) | 0.467 | 0.633 | 0.613 | 0.613 | 0.527 | 0.933 | 0.933 | 0.933 |
| 8B plain | markov (n=150) | 0.573 | 0.313 | 0.553 | 0.567 | 0.467 | 0.953 | 0.953 | 0.947 |
| 8B plain | llmgen (n=150) | 0.420 | 0.313 | 0.400 | 0.420 | 0.320 | 0.707 | 0.707 | 0.707 |
| 8B +stats | ALL (n=1050) | 0.561 | 0.693 | 0.689 | 0.690 | 0.701 | 0.912 | 0.912 | 0.912 |
| 8B +stats | gauss1 (n=150) | 0.513 | 0.833 | 0.820 | 0.813 | 0.753 | 0.887 | 0.873 | 0.900 |
| 8B +stats | tshuffle (n=150) | 0.460 | 0.633 | 0.600 | 0.633 | 0.567 | 0.933 | 0.933 | 0.933 |
| 8B +stats | markov (n=150) | 0.613 | 0.313 | 0.613 | 0.613 | 0.480 | 0.953 | 0.947 | 0.953 |
| 8B +stats | llmgen (n=150) | 0.447 | 0.313 | 0.400 | 0.433 | 0.353 | 0.707 | 0.700 | 0.707 |
| 8B +stats+real-day primer | ALL (n=1050) | 0.639 | 0.693 | 0.695 | 0.702 | 0.713 | 0.912 | 0.912 | 0.912 |
| 8B +stats+real-day primer | gauss1 (n=150) | 0.580 | 0.833 | 0.833 | 0.780 | 0.747 | 0.887 | 0.887 | 0.887 |
| 8B +stats+real-day primer | tshuffle (n=150) | 0.553 | 0.633 | 0.633 | 0.627 | 0.613 | 0.933 | 0.933 | 0.933 |
| 8B +stats+real-day primer | markov (n=150) | 0.627 | 0.313 | 0.627 | 0.620 | 0.493 | 0.953 | 0.940 | 0.953 |
| 8B +stats+real-day primer | llmgen (n=150) | 0.420 | 0.313 | 0.320 | 0.313 | 0.347 | 0.707 | 0.707 | 0.660 |
| 8B stats-only+primer | ALL (n=1050) | 0.724 | 0.693 | 0.746 | 0.719 | 0.735 | 0.912 | 0.904 | 0.909 |
| 8B stats-only+primer | gauss1 (n=150) | 0.653 | 0.833 | 0.827 | 0.833 | 0.853 | 0.887 | 0.867 | 0.913 |
| 8B stats-only+primer | tshuffle (n=150) | 0.627 | 0.633 | 0.680 | 0.593 | 0.647 | 0.933 | 0.933 | 0.933 |
| 8B stats-only+primer | markov (n=150) | 0.673 | 0.313 | 0.673 | 0.667 | 0.480 | 0.953 | 0.947 | 0.953 |
| 8B stats-only+primer | llmgen (n=150) | 0.507 | 0.313 | 0.507 | 0.507 | 0.333 | 0.707 | 0.660 | 0.647 |
| 14B plain | ALL (n=1050) | 0.596 | 0.693 | 0.682 | 0.693 | 0.657 | 0.912 | 0.912 | 0.912 |
| 14B plain | gauss1 (n=150) | 0.433 | 0.833 | 0.833 | 0.833 | 0.633 | 0.887 | 0.887 | 0.900 |
| 14B plain | tshuffle (n=150) | 0.680 | 0.633 | 0.647 | 0.627 | 0.620 | 0.933 | 0.933 | 0.933 |
| 14B plain | markov (n=150) | 0.473 | 0.313 | 0.467 | 0.387 | 0.453 | 0.953 | 0.953 | 0.953 |
| 14B plain | llmgen (n=150) | 0.513 | 0.313 | 0.513 | 0.513 | 0.413 | 0.707 | 0.707 | 0.653 |
| 14B +stats | ALL (n=750) | 0.643 | 0.781 | 0.781 | 0.781 | 0.749 | 0.900 | 0.900 | 0.897 |
| 14B +stats | gauss1 (n=150) | 0.527 | 0.833 | 0.833 | 0.807 | 0.700 | 0.887 | 0.887 | 0.880 |
| 14B +stats | llmgen (n=150) | 0.473 | 0.313 | 0.473 | 0.473 | 0.380 | 0.707 | 0.660 | 0.673 |
| 14B stats-only+primer | ALL (n=750) | 0.755 | 0.781 | 0.799 | 0.784 | 0.784 | 0.900 | 0.900 | 0.900 |
| 14B stats-only+primer | gauss1 (n=150) | 0.660 | 0.833 | 0.800 | 0.833 | 0.787 | 0.887 | 0.887 | 0.920 |
| 14B stats-only+primer | llmgen (n=150) | 0.473 | 0.313 | 0.473 | 0.453 | 0.340 | 0.707 | 0.607 | 0.673 |

## A/C. Context-conditioned judge: P(actual next day chosen)

| candidate | 8B plain | 8B +stats | 14B plain | 14B +stats |
|---|---|---|---|---|
| llmgen | 0.13 [0.09, 0.17] | 0.17 [0.12, 0.22] | 0.09 [0.05, 0.13] | 0.12 [0.08, 0.16] |
| copy7 | 0.18 [0.13, 0.23] | 0.27 [0.21, 0.32] | 0.10 [0.06, 0.14] | 0.14 [0.10, 0.19] |
| other_day | 0.54 [0.48, 0.60] | 0.51 [0.45, 0.57] | 0.54 [0.48, 0.60] | 0.53 [0.46, 0.59] |
| markov | 0.86 [0.81, 0.90] | — | 0.80 [0.75, 0.85] | — |
| other_user | 0.90 [0.87, 0.94] | — | 0.96 [0.93, 0.98] | — |
| stayhome | 0.98 [0.96, 1.00] | — | 0.88 [0.83, 0.92] | — |

| judge | typicality (picks candidate closer to history; llmgen/copy7/other_day pairs) | ECE | AUROC(Δp) | acc | P(A) | same pick both orders |
|---|---|---|---|---|---|---|
| 8B plain | 0.77 [0.74, 0.80] | 0.498 | 0.383 | 0.283 | 0.54 | 0.72 |
| 8B +stats | 0.73 [0.69, 0.76] | 0.405 | 0.456 | 0.318 | 0.68 | 0.52 |
| 14B plain | 0.84 [0.81, 0.86] | 0.667 | 0.302 | 0.242 | 0.48 | 0.84 |
| 14B +stats | 0.81 [0.78, 0.84] | 0.648 | 0.343 | 0.263 | 0.53 | 0.86 |

## Best-of-6 (DTW km/step vs actual next day; lower is better)

| judge | random | judge pick | medoid | oracle | judge − random (paired) | GEO-BLEU-like judge − random | slot-err judge − random | P(A) |
|---|---|---|---|---|---|---|---|---|
| bon | 5.24 | 5.17 | 4.89 | 3.40 | -0.07 [-0.44, 0.29] | 0.009 [0.000, 0.019] | -0.13 [-0.51, 0.28] | 0.76 |
| bone | 5.24 | 4.86 | 4.89 | 3.40 | -0.37 [-0.72, -0.02] | -0.004 [-0.013, 0.005] | -0.32 [-0.69, 0.03] | 0.84 |
| bone_14b | 5.24 | 4.88 | 4.89 | 3.40 | -0.36 [-0.67, -0.03] | 0.019 [0.011, 0.029] | -0.55 [-0.89, -0.21] | 0.71 |