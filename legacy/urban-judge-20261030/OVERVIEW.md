# Urban × LLM/VLM-as-judge: preliminary experiments (2026-10-30)

Workspace on bjserver: `/media/shawn/C/papers/VLM/urban-judge-20261030/`. Each track has its own `REPORT.md`; this file covers what cuts across them.
Models: Qwen3-VL-8B, Qwen3.5-9B and Qwen2.5-VL-7B (vision); Qwen3-8B and Qwen3-14B (text). All zero-shot unless stated, run locally (vLLM / HF), with no API cost.
Prior-art check: `priorart_urban_judge.md`.

## Three tracks

| Track | Paper link | Data | Question |
|---|---|---|---|
| **pp/**: street-view perception judge | EAGLE → vision | Place Pulse 2.0 (1,860 single votes, 713 multi-vote pairs, 3×150-image ranking pools) | Does the Δp gate transfer to VLM judges? Ties? Pairwise vs pointwise? |
| **poi/**: generation vs judge for POI choice | paper 246 (revealed-choice LLM) | Foursquare NYC, K=20 candidates, 1,200 test instances × 2 negative designs | Is prompt-only failure a protocol artifact? Do judge-style protocols close the gap? |
| **traj/**: LLM judge for synthetic trajectories | paper 62 (LDP-DiffTraj) | LYMob-4Cities City D (plus City B replication) | Can an LLM judge rank a privacy-noise ladder and select generations (best-of-n)? |

## Headline results

### Place Pulse (pp/REPORT.md)
- **Δp transfers.** Accuracy rises monotonically with Δp quintile for all three VLMs (≈.51 → ≈.80), with AUROC .60–.64. But Δp saturates: for Qwen3-VL, 60% of pairs sit at Δp≈1.
- **All judges plateau at .645–.649 against a single human vote.** The human vote agrees with the TrueSkill sign 85% of the time. The three judges' errors are correlated at .65–.73, so a majority vote gains only +0.4 pp. Unanimity is no better a gate than a single judge's Δp at the same coverage.
- **Pointwise ≥ pairwise.**
  - Expected 1–9 scores give .650–.666 accuracy. On the 15–17% of pairs where pairwise and pointwise disagree, pairwise is right only 43–50% of the time.
  - For ranking, 150 pointwise calls (Spearman with TrueSkill .55) beat 2,400 pairwise calls (.49) on safety. Active pair sampling adds ≈0.
  - The EAGLE gate adds +0.5 to +1.5 pp.
- **Ties and disagreement.** The models never choose "about the same" (0–0.1%, against 19% human ties in the sample). Δp is uncorrelated with human disagreement (ρ ≈ 0 on multi-vote pairs).
- **Thinking mode** (Qwen3.5-9B): .642 vs .665 non-thinking. The judge becomes more decisive (swap consistency .93), not more accurate.
- **Per-image facts + regression** reach .661, and all signals combined .677. This replicates UrbanAlign; it is not new.

### POI choice (poi/REPORT.md)
- **Judge-style protocols help, but only with population-sampled negatives** (the 246-style "pool"):
  - Top-1 for the fixed-order listwise readout (P2) is .396, against .440 order-averaged, .542 pointwise and **.577** for a both-order pairwise rerank.
  - With the user's own past places among the negatives ("mix"), every protocol lands at .20–.23, no better than a revisit heuristic.
- **LambdaRank on 17 features dominates** (.756 pool / .381 mix). LLM fusion, stacking and Δp-gated reranks all add 0. The Δp gate correctly learns never to override.
- **Novel destinations**, where semantics should help, still show no LLM advantage (LambdaRank .409 vs pairwise .305 on a novel-only set).
- **The LLM under-uses revealed preference.** When the true venue is the user's most-visited candidate, it picks it 57% of the time; LambdaRank does 99%.
- **Scale (14B):** listwise improves (.371 → .477), so the generation-vs-judge gap shrinks from +0.19 to +0.04. Position bias falls, but Δp calibration does not improve.
- **Prompt-only is not below random here** (.16–.28 even with a 246-like minimal prompt). The 246 baseline is likely partly a readout artifact; it is worth checking with P3/P4 on their K=30 sets.
- **Round 3: LoRA with paper 246's settings, on 2.4k–5.4k training examples, evaluated on 600-instance subsets.**
  - **Listwise fine-tune (FT-L) vs LambdaRank, Top-1 (order-averaged):** mix .207 → .312 vs .370; pool .422 → .535 vs .742; novel-only .212 → .388 vs .409 (within CI). So FT-L closes ≈65% of the gap on mix and ≈35% on pool.
  - **Pairwise-judge fine-tune (FT-P):** loses at equal tokens (2M).
    - Training half its negatives on user-visited places taught it "visited means wrong" (pool .352).
    - It is best on mix novel targets (.326 vs LambdaRank .279, n=172). The 4.5M-token FT-P run diverged.
  - **Revealed-preference pick rate** when the true venue is the most-visited candidate: .50/.56 → .70/.68 (LambdaRank .71/1.00).
  - **Calibration:** after FT-P, Δp is no longer saturated (AUROC .62 → .70), but ECE barely moves (.36 → .33). Position bias falls (argmax agreement across orders .51 → .68 mix, .57 → .77 pool).
  - No gated hybrid beats LambdaRank.

### Trajectories (traj/REPORT.md)
- **The text judge is nearly blind to LDP-scale noise.** Real vs σ=1/2/4/8 cells: .52/.51/.61/.73; uniform noise: .93. A 10-feature logistic regression gets AUC .91–.99.
  - Bradley–Terry still orders the ladder correctly (Kendall τ = 1), but adjacent rungs are at chance.
- **Typicality bias.** Given the user's history, the judge prefers its own generation over the true next day 87% of the time, and a copy of last week 82%. It picks the more history-like candidate 76% of the time.
- **Generation:** LLM next-day trajectories do not beat copying the same weekday (DTW 5.24 vs 5.80 km, CI spans 0).
- **Evidence-augmented judge.** Computed stats plus a primer on "what real days look like", with the raw trajectory removed, reach .72 (8B) and .755 (14B, AUROC(Δp) .76). That is still below the per-rung LR (.91). With stats, best-of-6 improves DTW by −0.37 km, about the same as a non-LLM medoid pick.
- **Δp is overconfident:** on real-vs-real null pairs, Δp = .48 (8B) and .64 (14B). Scale increases both confidence and typicality bias.

## Cross-cutting findings

1. **Δp gives a usable ranking signal everywhere but is badly calibrated as a probability.** Calibration is monotone in Place Pulse; on POI and trajectories, Δp saturates near 1 even on null pairs. EAGLE-style gating therefore needs per-domain recalibration (temperature scaling, null-pair anchoring) before τ means anything.
2. **Zero-shot judges lose to small feature models in every urban task tested:** LambdaRank for POI, a 10-feature LR for trajectories, and facts + LR ≈ the VLM in Place Pulse. The judge adds no complementary signal when fused. This is the opposite of text QA judging, where the judge is the strongest available signal.
3. **The text-judge premises don't carry over.**
   - "Pairwise beats pointwise" fails for street perception and for habit-dominated POI choice.
   - "Thinking helps" fails for visual perception judgments.
   - "Ties carry information" fails because models never tie.
   - "More/different judges help" fails because errors are correlated at .65–.73.
4. **Typicality / regularity bias** is the dominant judge failure in mobility. Judges prefer days and places that look like the user's history (and the model's own generations), not what actually happens. This connects to "plausible ≠ realistic" (arXiv 2606.13835).
5. **What does help:**
   - giving the LLM computed evidence instead of raw sequences;
   - pairwise/pointwise readouts over listwise ones when negatives are population-sampled;
   - order averaging.
   All of these are cheap protocol fixes; none closes the gap to supervised rankers.

## Recommended directions (ranked)

1. **Paper 246 follow-up: protocol ablation plus a pairwise fine-tuning head.**
   - Re-run the K=30 prompt-only baseline with P3/P4/P5; this is cheap and likely changes the "below random" row.
   - Test LoRA on pairwise vs listwise revealed choices (Round 3 results).
2. **Paper 62 follow-up: an LLM/VLM judge as a per-trajectory utility/privacy evaluator.** Prior-art check: mostly open. Requirements:
   - matched comparisons (original vs perturbed: .68 at σ=1 vs .52 unmatched);
   - evidence-augmented prompts;
   - a feature-model fallback gated by recalibrated Δp;
   - semantic perturbations that aggregate JSD can't see (home/work swap, sensitive-place visits).
3. **EAGLE paper 2:** frame the gate as domain-dependent. Calibration transfers in rank but not scale. In perception and mobility, the fallback should be a feature model or pointwise score, not a second LLM channel. Report the "model ties ≠ human ties, Δp ⟂ human disagreement" negative result.
4. **Place Pulse:** stop zero-shot VLM judging; UrbanAlign covers concept + regression. The remaining open piece is disagreement-aware evaluation on the 2,284 pairs with ≥4 votes (models' uncertainty is unrelated to human disagreement). That requires training, e.g., distributional targets.

## Caveats
- All results are small-n preliminary runs (CIs in each report), with Qwen-family models only (no other VLM weights were on the server) and one city for most trajectory analyses.
- FSQ check-ins are not 246's GPS revealed choices. The trajectory noise is i.i.d. Gaussian, not LDP-DiffTraj's diffusion mechanism. Place Pulse labels are single votes for most pairs.
