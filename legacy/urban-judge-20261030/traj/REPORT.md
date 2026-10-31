# LLM-as-judge for human-mobility trajectories (LYMob-4Cities, Qwen3-8B) — preliminary

Workspace: `/media/shawn/C/papers/VLM/urban-judge-20261030/traj/` (bjserver). Run 2026-10-30, 13:15–15:25 server time.
All numbers below come from `out/results_tables.md` / `out/summary.json` (produced by `scripts/analyze.py`). Brackets = 95% bootstrap CI (1000 resamples over pairs or user-days).

## TL;DR
* **Q1 (realness / quality ladder):** Qwen3-8B as a context-free pairwise judge **cannot tell real days from mildly perturbed ones**. On real vs Gaussian-noise σ=1/2 cells it scores 0.52/0.51 (chance), vs 0.91/0.96 AUC for a 10-feature logistic regression. It only gets reliable at σ=8 (0.73) and with uniform-random cells (0.93). Aggregated over many pairs, Bradley–Terry scores do put the noise ladder in the right order (Kendall τ=1.00 [0.87,1.00]). So the judge is usable as a **population-level** quality signal for heavy noise, but **not as a per-trajectory evaluator** at LDP-relevant noise levels. It rates its own generations (BT 0.33) and time-shuffled days (0.32) as "more real" than real days (0.26).
* Making the comparison easier helps but does not close the gap. **Matched pairs** (a real day vs its own perturbed copy) give 0.68/0.73/0.84/0.87 for σ=1/2/4/8. A **hint prompt** that describes jitter gives 0.57/0.70/0.77/0.88. **Thinking mode** did not help (σ=2: 0.55, σ=8: 0.68, n=40, 20–40% unparsed). The pattern replicates on **City B** (σ=1/2/4/8: 0.54/0.62/0.71/0.77; LR AUC 0.87–0.99).
* **Q2 (generation vs judge):** Qwen3-8B next-day generation was not significantly better than copying history. Mean-of-6 DTW was 5.24 km/step vs 5.80 for copy-Day−7, a paired difference of −0.56 [−1.14, 0.05]. A single sample (sample 0) scored 4.97. All of these beat the population-Markov baseline (11.4). Given the user's 7-day history, the judge **prefers the LLM's own generation over the true next day 87% of the time** (acc 0.13 [0.09,0.17]). It does the same with a verbatim copy of Day −7 (acc 0.18), which suggests a **typicality bias** more than self-preference. The judge picks the candidate closer to a context day 76% [74,78] of the time. A de-biasing instruction did not change this (0.175 / 0.175). **Best-of-6 selection with the judge did not help:** paired DTW change vs random pick was −0.07 [−0.41, 0.31] km. The oracle reaches 3.40 against 5.24 for random, so the headroom is there but the judge cannot find it. P(A)=0.76 in this setting: position bias dominates when candidates are similar.
* **Q3 (Δp calibration):** Δp is informative but overconfident. On real-vs-real null pairs, mean Δp = 0.48 where it should be about 0. On the ladder: ECE 0.069 and AUROC(Δp→correct) 0.64; the Δp≥0.6 subset has 82% accuracy at 36% coverage. For real vs non-real: ECE 0.13, AUROC 0.59. For the context-conditioned judge: ECE 0.24, AUROC 0.63. EAGLE-style Δp gating gives modest selective-accuracy gains, but Δp is not a calibrated probability here.

![ladder](ladder.png)

## Setup
* **Data:** City D (6,000 users, 75 days, 200×200 grid of 500 m cells, 30-min slots; no 999-masked rows). Users were split 80/20 (seed 0). A first-order population Markov chain over cells was fitted on the training users' within-day transitions. Everything evaluated comes from held-out test users (1,200).
* **Representation (text):** each user-day = consecutive same-cell records merged into stays: `HH:MM-HH:MM (x,y) [top-2 POI categories of cell] moved D km`. POI names come from `POI_datacategories.csv` with category k → line k (1-based, 85 names). Records are sparse phone pings (median 20/day). The real data itself shows oscillation between neighbouring cells, which makes small noise hard to see.
* **Quality ladder:** 600 held-out user-days with ≥6 records (one per user). Rungs, all derived from real days:
  * real
  * `gauss{1,2,4,8}`: i.i.d. N(0,σ²) noise per record on x and y, rounded and clipped to [1,200]
  * `tshuffle`: the same set of time slots with cells randomly reassigned to them
  * `markov`: population-Markov sample using the real day's time slots
  * `uniform`: uniform random cells at the real time slots
  * `llmgen`: Qwen3-8B next-day generations, sample 0 (see below)
* **Context-free judge (cf):** prompt asks "Which of these two is more likely to be a real person's actual, unaltered day of movement?" Both orders are run. p = first-token logprob over A/B (thinking off, first-token A/B mass = 1.00). p̃ = mean over orders, Δp = |2p̃−1|. For every pair of the 9 rungs there are 150 pairs drawn from *different* source user-days (5,550 pairs incl. 150 real-vs-real null pairs).
  * Variants: `cfm` = matched (real day vs its own perturbation), `cfh` = hint prompt, `cft` = thinking mode (hard vote per order, max 4000 tokens), `cfB` = City B replication (6,000-user subsample).
  * Deviation: the first prompt version ("Answer (A or B):") made the model emit "Answer: **A**", so the first-token A/B mass was ~0.0005. That run was killed and discarded, and the prompt was changed to "Respond with only the single letter A or B".
* **Generation:** 240 test user-days (target day d∈[8,69], ≥6 records, ≥5 of the previous 7 days with data). The prompt contains the past 7 days in the same text format. Qwen3-8B (no thinking), T=0.8, top-p 0.95, n=6.
  * Baselines: copy Day −7 (same weekday), copy Day −1, population Markov starting at home with Day −7's time slots, stay-home, and another later real day of the same user.
  * Metrics vs the actual day: DTW over stay sequences (km per aligned step); a GEO-BLEU-like score (n=1..3 soft n-gram precision, sim=exp(−0.5·km), greedy matching, brevity penalty — my implementation, not the official `geobleu` package); slot error (km at each true record slot, using the latest generated stay).
* **Context-conditioned judge (cc):** same history in the prompt, then "Which candidate is more likely to be this person's actual next day?" The actual next day is compared against each of: LLM sample 0, copy7, markov, another day of the same user, another user's real day, and stay-home. `ccd` is the same with a de-biasing sentence added.
* **Best-of-n:** round-robin over the 6 samples (15 pairs × 2 orders) with the cc judge, scored by soft win-rate. Compared against random (mean over samples), oracle (best DTW), medoid (min mean DTW to the other samples), and margin-gated selection (use the judge pick if 2·winrate−1 ≥ τ, else the medoid).
* **Baseline discriminator:** logistic regression on 10 features: #records, #stays, #unique cells, radius of gyration, mean/max/median jump, fraction of jumps >5 km, top-cell share, stays/records. Real (indices 0–299) vs rung (300–599), 5-fold CV AUC.
* **Calls:** 31,540 API calls, 69.9M prompt tokens and 1.76M completion tokens in total (1.15M of the completion tokens are the thinking run). Breakdown in `summary.json["usage"]`. Generation = 240 calls × n=6.

## Results

### Q1. Real vs rung (context-free, unmatched pairs, City D, n=150 each)
| rung | judge acc | mean p_real | mean Δp | LR AUC |
|---|---|---|---|---|
| gauss1 | 0.520 [0.440,0.600] | 0.513 | 0.42 | 0.907 |
| gauss2 | 0.513 [0.433,0.593] | 0.546 | 0.44 | 0.962 |
| gauss4 | 0.613 [0.540,0.693] | 0.575 | 0.43 | 0.976 |
| gauss8 | 0.727 [0.653,0.800] | 0.641 | 0.43 | 0.993 |
| uniform | 0.933 [0.887,0.973] | 0.773 | 0.56 | 1.000 |
| tshuffle | 0.467 [0.387,0.547] | 0.480 | 0.46 | 0.895 |
| markov | 0.573 [0.500,0.653] | 0.543 | 0.42 | 0.945 |
| llmgen (Qwen3-8B sample) | 0.420 [0.340,0.493] | 0.469 | 0.45 | 0.693 |
| real vs real (null) | — | 0.489 | **0.48** | — |

* **Bradley–Terry over all 36 rung pairs:** llmgen 0.33 > tshuffle 0.32 > **real 0.26** > gauss1 0.19 > markov 0.14 > gauss2 0.09 > gauss4 −0.03 > gauss8 −0.31 > uniform −0.98.
* Kendall τ on the noise ladder (real > g1 > g2 > g4 > g8 > uniform) = 1.00 [0.87, 1.00].
* Adjacent rungs are at chance pair by pair: g1>g2 0.51, g2>g4 0.51, g4>g8 0.61. The ordering only emerges in aggregate.

**Variants (acc of picking real):**
| rung | matched (cfm) | hint (cfh) | thinking (cft, n=40) | City B unmatched | City B matched |
|---|---|---|---|---|---|
| gauss1 | 0.677 [0.60,0.75] | 0.570 [0.49,0.64] | — | 0.540 | 0.540 |
| gauss2 | 0.730 [0.67,0.80] | 0.703 [0.63,0.77] | 0.550 | 0.620 | 0.727 |
| gauss4 | 0.843 [0.78,0.89] | 0.767 [0.70,0.83] | — | 0.707 | 0.820 |
| gauss8 | 0.867 [0.81,0.92] | 0.883 [0.83,0.93] | 0.675 | 0.773 | 0.873 |
| tshuffle | 0.553 | 0.453 | 0.650 | 0.493 | — |
| markov | — | — | 0.537 | 0.600 | — |
| llmgen | — | 0.367 | 0.500 | — | — |

In the matched setting, Spearman(p_real, DTW distortion of the perturbed copy) = 0.34. The judge's preference tracks the actual per-trajectory distortion only weakly.

### Q2. Generation (240 user-days; per-user-day means)
| method | DTW km/step (median) | GEO-BLEU-like | slot err km (median) |
|---|---|---|---|
| LLM sample 0 | 4.97 [4.18,5.81] (2.42) | 0.244 | 6.46 (3.37) |
| LLM mean of 6 | 5.24 [4.42,6.15] (3.03) | 0.241 | 6.73 (3.82) |
| copy Day −1 | 5.66 [4.75,6.59] (2.81) | 0.250 | 7.52 (3.89) |
| copy Day −7 | 5.80 [4.77,6.85] (2.91) | 0.233 | 7.14 (3.91) |
| other real day (same user) | 5.60 (2.89) | 0.224 | 7.29 (4.53) |
| stay home | 7.87 (4.53) | 0.000* | 7.69 (4.16) |
| population Markov | 11.40 (8.82) | 0.052 | 13.16 (9.63) |

\* stay-home produces a single stay, so it has no 2- or 3-grams and the BLEU-like score is ~0.

Further observations:
* The LLM's samples have low diversity: 3.7 distinct stay-sequences out of 6.
* Mean DTW from a sample to Day −7 is 4.3 km/step, so the samples are mostly recombinations of the history.
* The parse-failure rate was 0.

**Context-conditioned judge — P(actual next day chosen):**
| candidate | plain prompt | de-biased prompt |
|---|---|---|
| LLM's own generation | **0.129** [0.09,0.17] | 0.175 |
| copy Day −7 (verbatim in context) | **0.179** [0.13,0.23] | 0.175 |
| another real day of same user | 0.542 [0.48,0.60] | 0.552 |
| population Markov | 0.858 | 0.877 |
| another user's real day | 0.904 | — |
| stay home | 0.979 | — |

* **Self-preference vs typicality:** the judge prefers its own sample over the truth about as often as it prefers a verbatim copy of Day −7. It prefers the candidate closer to the history 76% of the time (0.82 for both llmgen and copy7).
* No second generator model was available on the server, so self-preference cannot be separated from typicality preference. Running a different LLM as the generator is the needed control.
* The judge is at chance between the actual day and another real day of the same user. It is useful for coarse checks such as "is this this user's kind of day?" (other user: 0.90; Markov: 0.86), but not for picking the actual day out of plausible days.
* Spearman(p_candidate, candidate DTW to truth) = −0.20.

**Best-of-6 (n=240):**
| selector | DTW | GEO-BLEU-like | slot err |
|---|---|---|---|
| random | 5.24 [4.45,6.21] | 0.241 | 6.73 |
| judge (round-robin) | 5.17 [4.34,6.15] | 0.250 | 6.60 |
| gated τ=0.4 (else medoid) | 4.89 | 0.245 | 6.36 |
| medoid only | 4.89 [4.10,5.79] | 0.235 | 6.59 |
| oracle (by DTW) | 3.40 [2.85,3.98] | 0.266 | 5.17 |

* Judge minus random, paired DTW: −0.07 [−0.41, 0.31] km. This is not significant; the gated selector equals the plain medoid.
* Strong position bias in this setting (P(A)=0.76; the same sample wins in both orders only 31% of the time).

### Q3. Δp calibration
| setting | n | acc | ECE | AUROC(Δp→correct) | acc at Δp≥0.6 (coverage) |
|---|---|---|---|---|---|
| cf ladder pairs (real,g1–g8,uniform) | 2250 | 0.704 | 0.069 | 0.638 | 0.822 (0.36) |
| cf real vs any non-real | 1200 | 0.596 | 0.130 | 0.589 | 0.677 (0.37) |
| cf matched | 750 | 0.721 | 0.086 | 0.679 | 0.880 (0.18) |
| cc actual vs candidates | 1440 | 0.599 | 0.236 | 0.633 | 0.662 (0.64) |

* Null pairs (real vs real) have mean Δp 0.48 and a 90th percentile of 0.94. The order-averaged first-token probabilities are near-deterministic, so whenever the two orders agree Δp is close to 1 regardless of whether there is any real difference.
* Reliability tables per bin are in `out/results_tables.md`.

### Position bias (P(A) over both orders; same-item-both-orders rate)
| judge | P(A) | same item both orders |
|---|---|---|
| cf | 0.48 | 0.50 |
| cfm | 0.45 | 0.31 |
| cc | 0.54 | 0.72 |
| bon | **0.76** | 0.31 |
| cfh | 0.41 | 0.57 |
| City B cf | **0.30** | 0.40 |

Order averaging is essential. In half or more of the context-free decisions, the pick flips when the order is swapped.

## Caveats
* The data is coarse: a 500 m grid, 30-min slots, and sparse pings. Real days already contain jitter, so σ=1–2 cells of noise is genuinely subtle in text. The LLM reads coordinate text, not maps, and `moved X km` / POI tags are my representation choices; no ablation was run on them.
* Gaussian noise is i.i.d. per record. That is LDP-*style* noise, not the paper's diffusion-forward-process noise or its privacy-budget mapping.
* Sample sizes are small: 150 pairs per cell, 240 user-days, 40 pairs for the thinking run. The thinking run left 20–40% of answers unparsed (often truncated at 4000 tokens); these were counted as ties.
* There is a single judge and generator model (Qwen3-8B), so self-preference is confounded with typicality.
* GEO-BLEU is a simplified re-implementation, not the official package. DTW is computed on stay sequences, not per-slot sequences. Means are outlier-sensitive, so medians are reported as well.
* BT and Kendall-τ CIs come from a 200-replicate bootstrap.
* City B used a 6,000-user subsample, and its `gauss8>uniform` pairs show strong position bias.

## Next steps (tying to paper 62 / EAGLE)
1. **The judge as a complement to aggregate metrics, not a replacement.** A Qwen3-8B judge only reaches ~0.6–0.7 accuracy at the noise levels where paper 62's JSD/SSIM metrics are informative. A judge+feature hybrid is more promising: give the LLM computed per-trajectory statistics (radius of gyration, jump distribution, dwell share), or use a learned discriminator, with the LLM handling only semantic plausibility (POI/time-of-day consistency). Test it on LDP-DiffTraj outputs against DiffTraj and the noisy baseline as a per-trajectory privacy–utility score.
2. **Matched comparisons for privacy evaluation.** Judging an original next to its perturbed or reconstructed version is far more sensitive (0.68 at σ=1 vs 0.52 unmatched). That suits paper 62's reconstruction framing: ask "which reconstruction is closer to plausible human behaviour" between LDP-DiffTraj and baseline outputs for the same noisy input.
3. **Judge-guided sampling only after de-biasing.** Best-of-n is limited by typicality and position bias. Before using an LLM judge for diffusion guidance or reranking, (a) separate generator and judge models to test self-preference properly, (b) debias with order averaging and calibration on real-vs-real null pairs, and (c) prefer a pointwise or feature-grounded score.
4. **EAGLE-style Δp gating needs recalibration in this domain.** First-token Δp is overconfident (null Δp 0.48). Use temperature-scaled or verbalized-plus-logprob confidence, fit τ on null and real-vs-σ8 pairs, and fall back to the LR-feature discriminator rather than the medoid when Δp<τ.
5. **Scale the judge.** Try Qwen3-14B/32B or a VLM given rendered trajectory maps (the "urban-judge" angle) to see whether the σ=1–2 regime becomes detectable. Add semantic perturbations that aggregate JSD cannot see, such as a home/work swap or a night-time hospital visit, where an LLM's world knowledge should beat count-based metrics.

## Files
* `scripts/`:
  * `common.py`: representation, perturbations, metrics, features
  * `prep.py`: splits, Markov fit, ladder, generation targets; `CITY=B` for City B
  * `gen.py`: generation
  * `judge.py`: jobs cf / cfm / cft / cfh / cc / ccd / bon / cfB
  * `analyze.py`: tables, `summary.json`
  * `plot.py`
  * `llm.py`: a copy of the shared client
* `out/`:
  * `prep.pkl`, `prep_B.pkl`
  * `gens.jsonl`: raw text and parsed generations
  * `judge_*.jsonl`: raw per-pair pA in both orders, p̃, Δp, first-token mass
  * `gen_metrics.csv`
  * `summary.json`
  * `results_tables.md`: all tables, including full calibration bins
  * `usage_*.json`
* `logs/`, `ladder.png`

---

# Follow-up round (2026-10-30, 15:26–16:25 server time)

Full tables: `out/results_followup.md`. Numbers: `summary.json["followup"]`; the round-1 summary is backed up at `out/summary_round1_backup.json`. Script: `scripts/analyze2.py`. New judge jobs in `scripts/judge.py`, raw outputs in `out/judge_<job>[_14b].jsonl`:
- `cfe`: context-free judge with a Stats line under each trajectory.
- `cfe2`: Stats line plus a "what real phone-ping days look like" primer.
- `cfs`: Stats line only (no trajectory text) plus the primer. I added this variant myself.
- `cce`: context-conditioned judge with Stats lines.
- `bone`: best-of-6 with the Stats-augmented context-conditioned judge.
- `MODEL=qwen3-14b`: the same jobs run on Qwen3-14B (port 8014).

The follow-up used 42,060 calls and 107M prompt tokens. All rounds together: 73,600 calls, 177M prompt tokens and 1.84M completion tokens.

**Stats line** (plain units, from the same 10 LR features): `Stats: N pings, S stays, U unique cells; most-visited cell share X% of pings; radius of gyration R km; jumps between stays mean/median/max a/b/c km; P% of jumps over 5 km.`

**Primer:** typical values of real days, measured on held-out real items not used in any pair:
- about half as many unique cells as pings (median 9 for 20 pings)
- most-visited cell holds about 27% of pings
- median jump about 1.8 km

**Prompt bug (harmless, applied consistently):** in `cfe`/`cfe2`/`cce`/`bone`, for both 8B and 14B, the sentence "Each record/candidate is followed by a computed Stats line summarising it." appears twice in the prompt because of an editing slip. `cfs` is not affected.

**Same pairs throughout:** every follow-up job uses exactly the same pairs as round-1 `cf`/`cc`:
- context-free: real item k vs rung item 300+k, 150 pairs per rung, plus 150 real-vs-real null pairs
- context-conditioned: the same 240 user-days
- both orders, first-token logprobs, thinking off

## A. Evidence-augmented judge (Qwen3-8B), accuracy of picking the real day
| rung | plain | +stats | +stats+primer | stats-only+primer | LR per-rung (same pairs) | LR pooled |
|---|---|---|---|---|---|---|
| gauss1 | 0.52 | 0.51 | 0.58 | 0.65 [0.58,0.73] | 0.89 | 0.83 |
| gauss2 | 0.51 | 0.59 | 0.67 | 0.76 [0.69,0.83] | 0.94 | 0.91 |
| gauss4 | 0.61 | 0.61 | 0.76 | 0.89 [0.85,0.94] | 0.98 | 0.94 |
| gauss8 | 0.73 | 0.69 | 0.86 | 0.95 [0.92,0.99] | 0.99 | 0.91 |
| tshuffle | 0.47 | 0.46 | 0.55 | 0.63 [0.55,0.71] | 0.93 | 0.63 |
| markov | 0.57 | 0.61 | 0.63 | 0.67 [0.60,0.74] | 0.95 | 0.31 |
| llmgen | 0.42 | 0.45 | 0.42 | 0.51 [0.43,0.59] | 0.71 | 0.31 |
| **all 7 (n=1050)** | 0.548 | 0.561 | 0.639 | **0.724** | 0.912 | 0.693 |

Calibration on the same 1050 real-vs-fake pairs:
| judge | ECE | AUROC(Δp→correct) | acc at Δp≥0.6 (coverage) | null-pair mean Δp | P(A) |
|---|---|---|---|---|---|
| plain | 0.170 | 0.565 | 0.61 (0.34) | 0.48 | 0.48 |
| +stats | 0.162 | 0.575 | 0.63 (0.35) | 0.50 | 0.47 |
| +stats+primer | 0.125 | 0.636 | 0.72 (0.46) | 0.56 | 0.44 |
| stats-only+primer | 0.086 | 0.634 | 0.83 (0.29) | 0.41 | 0.20 |

* **Adding the stats line alone does nothing** (0.548 → 0.561). Spearman of the judge's p_real with the LR pair probability is 0.04: the 8B model does not use the numbers when they sit under the raw trajectory.
* **The numbers only help when the prompt says what "real" looks like** (+primer: 0.64). They help most when the raw text is removed (stats-only: 0.72, with ECE improving from 0.17 to 0.09). The text representation seems to actively distract the model.
* **Stats do not close the gap to LR.** The best LLM variant reaches 0.72 against 0.91 for a per-rung LR on the same pairs.
* **Nothing helps on llmgen** (best 0.51 vs LR 0.71). Qwen's own generations stay indistinguishable to Qwen.
* Position bias is larger in the stats-only format (P(A)=0.20), but order-averaging absorbs it.

**Context-conditioned + stats (`cce`), P(actual chosen):**
* vs llmgen: 0.13 → 0.17
* vs copy7: 0.18 → 0.27
* vs another day of the same user: 0.54 → 0.51
* Typicality bias (judge picks whichever candidate is closer to a history day): 0.77 [0.74,0.80] → 0.73 [0.69,0.76]. This is a small reduction; the judge still mostly picks the history-like candidate.
* ECE 0.50 → 0.41. In the cc setting Δp is anti-informative: AUROC 0.38 → 0.46, because confident answers are mostly confident typicality errors.

## B. Hybrid Δp gate (τ picked on a random half of pairs, scored on the other half, both splits averaged)

Two LR references:
- **per-rung LR**: one LR trained for each degradation type, so it "knows" which degradation it faces; this is optimistic.
- **pooled LR**: a single LR trained on real vs all 8 fake rungs. It is realistic when the degradation type is unknown, but it learns "jitter = fake" and so ranks smooth fakes (markov, llmgen) above real days: 0.31 on both.

| judge (all 7 rungs, n=1050) | judge | pooled LR | judge→LR (judge if Δp≥τ) | LR→judge | avg-prob | per-rung LR | judge→per-rung LR |
|---|---|---|---|---|---|---|---|
| 8B plain | 0.548 | 0.693 | 0.687 | 0.686 | 0.683 | 0.912 | 0.912 |
| 8B stats-only+primer | 0.724 | 0.693 | **0.746** | 0.719 | 0.735 | 0.912 | 0.904 |
| 14B stats-only+primer (5 rungs, n=750) | 0.755 | 0.781 | **0.799** | 0.784 | 0.784 | 0.900 | 0.900 |

* **Against a per-rung LR, the LLM adds nothing.** Every gate is ≤ LR (0.90–0.91).
* **Against a pooled LR, the evidence-based judge adds a little:**
  * 8B: 0.693 → 0.746
  * 14B: 0.781 → 0.799
  * This comes mainly from the markov rung, where pooled LR scores 0.31 and the judge 0.67.
* **On llmgen** (the case where LR AUC is only 0.69), no judge beats chance. Gates either match LR (0.71) or fall below it (0.61–0.66 when the judge is trusted). The LLM adds nothing on the very case where it was hoped to help.
* The plain-text judge never adds anything to either LR.

## C. Scale: Qwen3-14B
| | 8B plain | 14B plain | 8B +stats | 14B +stats | 8B stats-only+primer | 14B stats-only+primer |
|---|---|---|---|---|---|---|
| acc, gauss1/2/4/8+llmgen (n=750) | 0.559 | 0.604 | 0.571 | 0.643 | 0.753 | 0.755 |
| ECE (same subset) | 0.159 | 0.234 | 0.154 | 0.148 | 0.102 | 0.098 |
| AUROC(Δp→correct) | 0.588 | 0.620 | 0.591 | 0.603 | 0.640 | **0.759** |
| acc at Δp≥0.6 (coverage) | 0.65 (0.34) | 0.67 (0.64) | 0.66 (0.35) | 0.71 (0.47) | 0.85 (0.31) | 0.85 (0.69) |
| null-pair mean Δp | 0.48 | **0.64** | 0.50 | 0.47 | 0.41 | 0.51 |
| P(A) / same pick both orders | 0.48 / 0.50 | 0.56 / 0.76 | 0.47 / 0.52 | 0.76 / 0.49 | 0.20 / 0.35 | 0.67 / 0.67 |

**14B plain, per rung:**
* gauss1 0.43, gauss2 0.54, gauss4 0.71, gauss8 0.83, uniform 0.99
* tshuffle 0.68 (8B: 0.47), markov 0.47, llmgen 0.51
* Bradley–Terry ordering: gauss1 0.91 > real 0.85 > markov 0.80 > gauss2 0.77 > llmgen 0.70 > gauss4 0.29 > tshuffle 0.21 > gauss8 −0.48 > uniform −4.06
* Kendall τ on the noise ladder: 0.87 [0.73,1.00] (8B: 1.00)
* Adjacent-rung pairs: g1>g2 0.53, g2>g4 0.69, g4>g8 0.76, g8>uniform 0.99. These separate better than with 8B (0.51 / 0.51 / 0.61 / 0.84).

**14B, context-conditioned:**
* vs llmgen 0.09, vs copy7 0.10, vs another day of the same user 0.54, vs another user 0.96, vs markov 0.80, vs stay-home 0.88
* Typicality bias gets **worse with scale**: 0.84 [0.81,0.86], vs 0.77 for 8B
* ECE 0.67, AUROC(Δp) 0.30
* With stats: typicality 0.81; llmgen 0.12, copy7 0.14

**Verdict on scale:** 14B helps on coarse and structural degradations (σ≥4, tshuffle, uniform) and gives more order-consistent answers (0.76 vs 0.50). It does not help at σ=1–2 or on llmgen. It is *more* overconfident: on real-vs-real null pairs mean Δp is 0.64, and ECE rises from 0.16 to 0.23. In the context-conditioned setting its typicality bias is stronger.

The best-calibrated configuration overall is **14B stats-only+primer**: AUROC(Δp) 0.76, 85% accuracy at 69% coverage. Its accuracy (0.755) is still no better than 8B's and below LR.

## D. Best-of-6 with the evidence-augmented cc judge (240 user-days; paired differences vs a random pick)
| judge | DTW pick (random 5.24, medoid 4.89, oracle 3.40) | ΔDTW judge−random | ΔGEO-BLEU-like | Δslot-err km | P(A) |
|---|---|---|---|---|---|
| 8B plain (round 1) | 5.17 | −0.07 [−0.44,0.29] | +0.009 [0.000,0.019] | −0.13 [−0.51,0.28] | 0.76 |
| 8B +stats | 4.86 | **−0.37 [−0.72,−0.02]** | −0.004 [−0.013,0.005] | −0.32 [−0.69,0.03] | 0.84 |
| 14B +stats | 4.88 | **−0.36 [−0.67,−0.03]** | **+0.019 [0.011,0.029]** | **−0.55 [−0.89,−0.21]** | 0.71 |

* With stats, judge-based selection now gives a small but CI-excluding-zero DTW gain.
* With 14B it also improves GEO-BLEU-like and slot error.
* The gain is about the same as picking the non-LLM **medoid** (4.89), and only about 20% of the random-to-oracle headroom.
* Interpretation: the stats let the judge reject outlier samples, but it still cannot identify the best one.

## Updated take-aways
1. Look-then-match with computed evidence works only when the evidence is paired with a reference description of real data. It works best when the raw coordinate text is dropped: 8B accuracy goes 0.55 → 0.72, and ECE goes 0.17 → 0.09. Even then it stays well below a 10-feature LR (0.91). On LLM-generated days the LLM judge remains at chance under every variant and model size.
2. As a per-trajectory utility evaluator for paper 62, the LLM is at best a calibrated soft gate on top of a feature model:
   - It gives a small gain over a degradation-agnostic LR: +5 points with 8B, +2 with 14B, mostly on markov-type fakes.
   - It gives no gain over an LR trained for the specific degradation.
3. Scale (8B → 14B) improves coarse discrimination and order-consistency. It does not fix fine-noise detection or self-generated days, and it worsens typicality bias and null-pair overconfidence. EAGLE-style Δp gating should be recalibrated per model before use.
4. In this setup, judge-guided best-of-n buys about the same as a medoid heuristic.
