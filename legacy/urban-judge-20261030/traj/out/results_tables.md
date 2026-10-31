## Generation quality vs actual next day (mean [95% bootstrap CI]; DTW and slot error in km, lower better; GEO-BLEU-like higher better)

| method | DTW km/step | median DTW | GEO-BLEU-like | slot error km | median slot err |
|---|---|---|---|---|---|
| copy1 | 5.66 [4.75, 6.59] | 2.81 | 0.250 [0.224, 0.275] | 7.52 [6.37, 8.84] | 3.89 |
| copy7 | 5.80 [4.77, 6.85] | 2.91 | 0.233 [0.209, 0.257] | 7.14 [6.09, 8.27] | 3.91 |
| llm_gen(mean of 6 samples) | 5.24 [4.42, 6.15] | 3.03 | 0.241 [0.218, 0.263] | 6.73 [5.75, 7.79] | 3.82 |
| llm_gen(sample 0) | 4.97 [4.18, 5.81] | 2.42 | 0.244 [0.221, 0.266] | 6.46 [5.43, 7.49] | 3.37 |
| markov | 11.40 [10.17, 12.61] | 8.82 | 0.052 [0.044, 0.061] | 13.16 [11.92, 14.57] | 9.63 |
| other_day | 5.60 [4.75, 6.44] | 2.89 | 0.224 [0.202, 0.247] | 7.29 [6.36, 8.31] | 4.53 |
| stayhome | 7.87 [6.68, 9.15] | 4.53 | 0.000 [0.000, 0.000] | 7.69 [6.49, 9.04] | 4.16 |

Paired DTW difference (LLM mean-of-6 minus copy Day -7): -0.56 [-1.14, 0.05] km; minus copy Day -1: -0.42 [-1.05, 0.21] km.

Distinct location sequences among the 6 samples per user-day: 3.73 [3.50, 3.95]; mean DTW of LLM samples to the Day -7 record: 4.34 [3.67, 5.08] km; parse-fail (<2 records) rate 0.000.

## Q1: context-free judge, real vs rung (pairs from different source user-days; order-averaged)

| rung | n | judge acc (p_real>0.5) | mean p_real | mean Δp | order-consistent (same pick both orders) | LR-feature AUC (5-fold) |
|---|---|---|---|---|---|---|
| gauss1 | 150 | 0.520 [0.440, 0.600] | 0.513 [0.475, 0.556] | 0.419 | 0.49 | 0.907 |
| gauss2 | 150 | 0.513 [0.433, 0.593] | 0.546 [0.503, 0.591] | 0.442 | 0.49 | 0.962 |
| gauss4 | 150 | 0.613 [0.540, 0.693] | 0.575 [0.534, 0.618] | 0.431 | 0.47 | 0.976 |
| gauss8 | 150 | 0.727 [0.653, 0.800] | 0.641 [0.607, 0.678] | 0.433 | 0.45 | 0.993 |
| llmgen | 150 | 0.420 [0.340, 0.493] | 0.469 [0.425, 0.513] | 0.448 | 0.53 | 0.693 |
| markov | 150 | 0.573 [0.500, 0.653] | 0.543 [0.497, 0.586] | 0.419 | 0.49 | 0.945 |
| real2 | 150 | 0.420 [0.347, 0.500] | 0.489 [0.441, 0.535] | 0.480 | 0.59 | nan |
| tshuffle | 150 | 0.467 [0.387, 0.547] | 0.480 [0.433, 0.526] | 0.461 | 0.51 | 0.895 |
| uniform | 150 | 0.933 [0.887, 0.973] | 0.773 [0.740, 0.803] | 0.564 | 0.57 | 1.000 |

## Q1: Bradley-Terry scores over all rungs (soft outcomes, 200 bootstrap) 

| rung | BT score [95% CI] |
|---|---|
| llmgen | 0.33 [0.27, 0.38] |
| tshuffle | 0.32 [0.27, 0.37] |
| real | 0.26 [0.21, 0.31] |
| gauss1 | 0.19 [0.14, 0.24] |
| markov | 0.14 [0.09, 0.19] |
| gauss2 | 0.09 [0.04, 0.15] |
| gauss4 | -0.03 [-0.08, 0.01] |
| gauss8 | -0.31 [-0.36, -0.27] |
| uniform | -0.98 [-1.03, -0.93] |

Kendall τ of BT scores vs true ladder order ['real', 'gauss1', 'gauss2', 'gauss4', 'gauss8', 'uniform']: 1.00 [0.87, 1.00]

| ladder pair (better vs worse) | n | P(better preferred) | mean p_better |
|---|---|---|---|
| real > gauss1 | 150 | 0.520 [0.440, 0.600] | 0.513 |
| real > gauss2 | 150 | 0.513 [0.427, 0.593] | 0.546 |
| real > gauss4 | 150 | 0.613 [0.540, 0.693] | 0.575 |
| real > gauss8 | 150 | 0.727 [0.660, 0.793] | 0.641 |
| real > uniform | 150 | 0.933 [0.887, 0.967] | 0.773 |
| gauss1 > gauss2 | 150 | 0.507 [0.433, 0.580] | 0.507 |
| gauss1 > gauss4 | 150 | 0.560 [0.480, 0.640] | 0.554 |
| gauss1 > gauss8 | 150 | 0.693 [0.620, 0.767] | 0.621 |
| gauss1 > uniform | 150 | 0.980 [0.953, 1.000] | 0.768 |
| gauss2 > gauss4 | 150 | 0.513 [0.433, 0.593] | 0.524 |
| gauss2 > gauss8 | 150 | 0.660 [0.587, 0.733] | 0.592 |
| gauss2 > uniform | 150 | 0.947 [0.907, 0.980] | 0.740 |
| gauss4 > gauss8 | 150 | 0.607 [0.526, 0.680] | 0.565 |
| gauss4 > uniform | 150 | 0.940 [0.900, 0.973] | 0.725 |
| gauss8 > uniform | 150 | 0.840 [0.773, 0.893] | 0.668 |

## Q3: Δp calibration

### Δp calibration: context-free, all ladder pairs (real, gauss1-8, uniform)

| Δp bin | n | acc | mean conf (=(1+Δp)/2) |
|---|---|---|---|
| [0.0,0.1) | 537 | 0.577 | 0.519 |
| [0.1,0.2) | 247 | 0.668 | 0.573 |
| [0.2,0.4) | 356 | 0.669 | 0.649 |
| [0.4,0.6) | 308 | 0.685 | 0.747 |
| [0.6,0.8) | 313 | 0.783 | 0.850 |
| [0.8,1.0) | 489 | 0.847 | 0.955 |

ECE=0.069; AUROC(Δp → correct)=0.638; overall acc=0.704, n=2250

- Δp≥0.1: coverage 0.76, acc 0.743
- Δp≥0.2: coverage 0.65, acc 0.756
- Δp≥0.4: coverage 0.49, acc 0.784
- Δp≥0.6: coverage 0.36, acc 0.822

### Δp calibration: context-free, real vs every non-real rung

| Δp bin | n | acc | mean conf (=(1+Δp)/2) |
|---|---|---|---|
| [0.0,0.1) | 271 | 0.502 | 0.518 |
| [0.1,0.2) | 130 | 0.569 | 0.575 |
| [0.2,0.4) | 188 | 0.548 | 0.648 |
| [0.4,0.6) | 171 | 0.608 | 0.750 |
| [0.6,0.8) | 148 | 0.642 | 0.851 |
| [0.8,1.0) | 292 | 0.695 | 0.959 |

ECE=0.130; AUROC(Δp → correct)=0.589; overall acc=0.596, n=1200

- Δp≥0.1: coverage 0.77, acc 0.623
- Δp≥0.2: coverage 0.67, acc 0.632
- Δp≥0.4: coverage 0.51, acc 0.658
- Δp≥0.6: coverage 0.37, acc 0.677

## Q2: context-conditioned judge — actual next day vs candidate (given past 7 days)

| candidate | n | judge acc (picks actual) | mean p_actual | mean Δp | order-consistent |
|---|---|---|---|---|---|
| copy7 | 240 | 0.179 [0.129, 0.225] | 0.262 | 0.638 | 0.68 |
| llmgen | 240 | 0.129 [0.092, 0.171] | 0.258 | 0.584 | 0.63 |
| markov | 240 | 0.858 [0.812, 0.900] | 0.783 | 0.687 | 0.75 |
| other_day | 240 | 0.542 [0.483, 0.604] | 0.534 | 0.469 | 0.56 |
| other_user | 240 | 0.904 [0.863, 0.942] | 0.832 | 0.736 | 0.79 |
| stayhome | 240 | 0.979 [0.958, 0.996] | 0.948 | 0.905 | 0.93 |

Typicality check: fraction of context-conditioned decisions where the judge picks the candidate closer (min DTW) to one of the 7 context days: 0.760 [0.735, 0.781]; by candidate: copy7 0.82, llmgen 0.82, markov 0.83, other_day 0.66, other_user 0.91, stayhome 0.51. Fraction where the actual day is closer to context than the candidate: copy7 0.00, llmgen 0.10, markov 0.85, other_day 0.50, other_user 0.97, stayhome 0.49


### Δp calibration: context-conditioned, actual vs all candidates

| Δp bin | n | acc | mean conf (=(1+Δp)/2) |
|---|---|---|---|
| [0.0,0.1) | 152 | 0.401 | 0.522 |
| [0.1,0.2) | 103 | 0.408 | 0.575 |
| [0.2,0.4) | 137 | 0.555 | 0.649 |
| [0.4,0.6) | 122 | 0.574 | 0.753 |
| [0.6,0.8) | 156 | 0.545 | 0.849 |
| [0.8,1.0) | 770 | 0.686 | 0.974 |

ECE=0.236; AUROC(Δp → correct)=0.633; overall acc=0.599, n=1440

- Δp≥0.1: coverage 0.89, acc 0.622
- Δp≥0.2: coverage 0.82, acc 0.641
- Δp≥0.4: coverage 0.73, acc 0.652
- Δp≥0.6: coverage 0.64, acc 0.662

## Q2: best-of-n selection among the 6 LLM samples (context-conditioned judge, round-robin, soft win-rate)

| selector | DTW km/step | GEO-BLEU-like | slot error km |
|---|---|---|---|
| random | 5.24 [4.45, 6.21] | 0.241 [0.219, 0.263] | 6.73 [5.78, 7.72] |
| judge | 5.17 [4.34, 6.15] | 0.250 [0.228, 0.273] | 6.60 [5.60, 7.61] |
| oracle | 3.40 [2.85, 3.98] | 0.266 [0.243, 0.288] | 5.17 [4.43, 6.05] |
| worst | 7.24 [6.18, 8.38] | 0.214 [0.193, 0.237] | 8.28 [7.14, 9.56] |
| medoid | 4.89 [4.10, 5.79] | 0.235 [0.211, 0.257] | 6.59 [5.68, 7.52] |
| gated0.2 | 5.10 [4.25, 6.08] | 0.247 [0.223, 0.271] | 6.60 [5.64, 7.60] |
| gated0.4 | 4.89 [4.12, 5.79] | 0.245 [0.222, 0.268] | 6.36 [5.38, 7.30] |
| gated0.6 | 4.97 [4.18, 5.83] | 0.239 [0.217, 0.262] | 6.53 [5.60, 7.46] |

Paired DTW(judge pick) − DTW(random): -0.069 [-0.409, 0.313] km (n=240); gate margins: mean 0.34; fraction ≥0.2: 0.60, ≥0.4: 0.36, ≥0.6: 0.21
Judge pick is the sample closest to a context day in 0.48 of cases.
(gatedτ: use judge pick if its margin 2·winrate−1 ≥ τ, else medoid of samples)

## Q1b: context-free judge, MATCHED pairs (real day vs its own perturbation)

| rung | n | acc (p_real>0.5; ties count 0.5) | mean p_real | order-consistent | parse ok |
|---|---|---|---|---|---|
| gauss1 | 150 | 0.677 [0.597, 0.750] | 0.550 | 0.22 | 1.00 |
| gauss2 | 150 | 0.730 [0.667, 0.797] | 0.579 | 0.26 | 1.00 |
| gauss4 | 150 | 0.843 [0.783, 0.893] | 0.630 | 0.35 | 1.00 |
| gauss8 | 150 | 0.867 [0.807, 0.920] | 0.716 | 0.51 | 1.00 |
| tshuffle | 150 | 0.553 [0.473, 0.633] | 0.514 | 0.19 | 1.00 |

Spearman(p_real, DTW distortion of the perturbed copy) over gauss rungs: 0.339 (n=600)

### Δp calibration: matched real vs perturbation

| Δp bin | n | acc | mean conf (=(1+Δp)/2) |
|---|---|---|---|
| [0.0,0.1) | 259 | 0.595 | 0.518 |
| [0.1,0.2) | 113 | 0.673 | 0.571 |
| [0.2,0.4) | 159 | 0.755 | 0.647 |
| [0.4,0.6) | 86 | 0.860 | 0.742 |
| [0.6,0.8) | 58 | 0.862 | 0.850 |
| [0.8,1.0) | 75 | 0.893 | 0.958 |

ECE=0.086; AUROC(Δp → correct)=0.679; overall acc=0.721, n=750

- Δp≥0.1: coverage 0.65, acc 0.788
- Δp≥0.2: coverage 0.50, acc 0.823
- Δp≥0.4: coverage 0.29, acc 0.872
- Δp≥0.6: coverage 0.18, acc 0.880

## Q1d: context-free judge with an explicit hint describing jitter artefacts (unmatched pairs)

| rung | n | acc (p_real>0.5; ties count 0.5) | mean p_real | order-consistent | parse ok |
|---|---|---|---|---|---|
| gauss1 | 150 | 0.570 [0.490, 0.640] | 0.565 | 0.57 | 1.00 |
| gauss2 | 150 | 0.703 [0.627, 0.770] | 0.668 | 0.56 | 1.00 |
| gauss4 | 150 | 0.767 [0.700, 0.833] | 0.697 | 0.53 | 1.00 |
| gauss8 | 150 | 0.883 [0.830, 0.930] | 0.759 | 0.57 | 1.00 |
| llmgen | 150 | 0.367 [0.287, 0.440] | 0.430 | 0.65 | 1.00 |
| tshuffle | 150 | 0.453 [0.377, 0.530] | 0.485 | 0.54 | 1.00 |

## Q2b: context-conditioned judge with a de-biasing instruction ('the real day is often not a copy of history')

| rung | n | acc (p_real>0.5; ties count 0.5) | mean p_real | order-consistent | parse ok |
|---|---|---|---|---|---|
| copy7 | 240 | 0.175 [0.129, 0.225] | 0.268 | 0.68 | 1.00 |
| llmgen | 240 | 0.175 [0.129, 0.225] | 0.285 | 0.64 | 1.00 |
| markov | 240 | 0.877 [0.835, 0.917] | 0.773 | 0.74 | 1.00 |
| other_day | 240 | 0.552 [0.490, 0.613] | 0.544 | 0.55 | 1.00 |

## Q1c: thinking-mode judge (Qwen3-8B, enable_thinking, hard vote per order), unmatched pairs

| rung | n | acc (p_real>0.5; ties count 0.5) | mean p_real | order-consistent | parse ok |
|---|---|---|---|---|---|
| gauss2 | 40 | 0.550 [0.438, 0.675] | 0.531 | 0.55 | 0.74 |
| gauss8 | 40 | 0.675 [0.562, 0.775] | 0.625 | 0.55 | 0.60 |
| llmgen | 40 | 0.500 [0.388, 0.600] | 0.500 | 0.47 | 0.80 |
| markov | 40 | 0.537 [0.425, 0.650] | 0.525 | 0.45 | 0.62 |
| tshuffle | 40 | 0.650 [0.537, 0.762] | 0.594 | 0.68 | 0.79 |

## Replication on City B (6000-user subsample; same pipeline, context-free judge)

| pair (first should win) | n | acc | mean p_first | order-consistent | LR AUC |
|---|---|---|---|---|---|
| real > gauss1 | 150 | 0.540 [0.453, 0.620] | 0.517 | 0.51 | 0.870 |
| real > gauss2 | 150 | 0.620 [0.540, 0.693] | 0.588 | 0.41 | 0.946 |
| real > gauss4 | 150 | 0.707 [0.627, 0.773] | 0.617 | 0.51 | 0.964 |
| real > gauss8 | 150 | 0.773 [0.707, 0.840] | 0.656 | 0.47 | 0.994 |
| real > tshuffle | 150 | 0.493 [0.420, 0.573] | 0.493 | 0.48 | 0.893 |
| real > markov | 150 | 0.600 [0.520, 0.673] | 0.568 | 0.53 | 0.891 |
| real > uniform | 150 | 0.907 [0.860, 0.947] | 0.707 | 0.40 | 1.000 |
| gauss1 > gauss2 | 150 | 0.613 [0.540, 0.693] | 0.557 | 0.43 | nan |
| gauss2 > gauss4 | 150 | 0.647 [0.567, 0.727] | 0.571 | 0.48 | nan |
| gauss4 > gauss8 | 150 | 0.593 [0.513, 0.667] | 0.552 | 0.39 | nan |
| gauss8 > uniform | 150 | 0.833 [0.773, 0.893] | 0.569 | 0.11 | nan |
| real > gauss1_matched | 150 | 0.540 [0.460, 0.620] | 0.522 | 0.23 | nan |
| real > gauss2_matched | 150 | 0.727 [0.653, 0.800] | 0.581 | 0.33 | nan |
| real > gauss4_matched | 150 | 0.820 [0.753, 0.880] | 0.631 | 0.30 | nan |
| real > gauss8_matched | 150 | 0.873 [0.820, 0.920] | 0.674 | 0.38 | nan |

## Position bias by job

| job | mean P(A) | same item both orders | A both orders | B both orders |
|---|---|---|---|---|
| cf | 0.478 | 0.50 | 0.22 | 0.25 |
| cfm | 0.447 | 0.31 | 0.25 | 0.40 |
| cc | 0.543 | 0.72 | 0.18 | 0.09 |
| bon | 0.758 | 0.31 | 0.61 | 0.06 |
| cfh | 0.406 | 0.57 | 0.11 | 0.30 |
| ccd | 0.545 | 0.65 | 0.21 | 0.11 |
| cfB | 0.301 | 0.40 | 0.06 | 0.49 |