# Place Pulse 2.0 × VLM-as-judge (EAGLE transfer test) — 2026-10-30

Judges: Qwen3-VL-8B-Instruct, Qwen3.5-9B, Qwen2.5-VL-7B-Instruct (HF transformers, bf16, logits of the first answer token, no thinking).
Data: `sample.jsonl` = 1,860 image-disjoint votes (250 strict + 60 equal per attribute × 6), with TrueSkill gap from `qscores.tsv`;
`multi.jsonl` = 720 pairs with ≥4 human votes (mean 7.3); `rank_pairs.jsonl` = 150 images × 3 attributes, 1,200 random pairs each.
Conditions: 2-way pairwise (AB+BA), 3-way with "C = about the same", 48×36 low-res pairwise, pointwise 1–9 expected score, per-image facts JSON (12 fields).
Scripts: prep*.py, judge_pp.py, analyze_pp.py, analyze_extra.py, analyze_ens.py, analyze_rank.py. Summaries: summary_pp.json, summary_rank.json.

## Main numbers (1,500 strict votes)
| | Qwen3-VL-8B | Qwen3.5-9B | Qwen2.5-VL-7B |
|---|---|---|---|
| Pairwise acc vs single vote | .645 [.620,.668] | .646 [.622,.670] | .649 [.625,.675] |
| Swap consistency | .811 | .823 | .657 |
| Acc by Δp quintile (low→high) | .52 .56 .60 .73 .81 | .51 .58 .66 .70 .78 | .57 .57 .63 .69 .78 |
| AUROC(Δp → correct) | .640 | .622 | .602 |
| Pointwise expected-score acc | .666 | .650 | .649 |
| Pairwise acc when pointwise disagrees (15–17%) | .428 | .488 | .500 |
| EAGLE gate best (τ sweep) | .660 (+1.5) | .651 (+0.5) | .655 (+0.6) |
| 3-way: model picks "same" | 0.0% | 0.0% | 0.1% (human ties 19% of sample) |
| AUROC p(C) → human tie | .55 | .55 | .55 |

- Human ceiling proxy: single vote agrees with TrueSkill sign 85.2% (68% in the smallest-gap tercile, 98.6% in the largest).
- Facts → per-category logistic (CV): .661 for Qwen3-VL (judge .645); all signals combined .677. (Qwen3.5/2.5 facts not run.) UrbanAlign (2602.19442) already does concept-scoring + regression, so this is a replication/baseline, not a contribution.
- Low-res (48×36) pairwise keeps .61 accuracy and agrees with full-res on 80%; adding low-res Δp does not improve error prediction (CV AUROC .589→.581).

## Multi-vote pairs (713, ≥4 votes)
- Acc vs majority .62–.64; Spearman(p̃, human vote share) ≈ .29–.31.
- **Δp does not track human disagreement**: Spearman(Δp, |2h−1|) = −.002 / .022 / −.005; p("same") vs human tie share ≈ 0. Model uncertainty ≠ crowd (aleatoric) disagreement.

## Heterogeneous judges
- Error correlation between the three judges .65–.73. Majority vote .653 (+0.4 pp). Unanimous (78% of pairs) .683 — identical to a single judge’s top-78% Δp (.682). Echoes the MAS finding: correlated judges add little.
- On the 372 unanimous-wrong pairs, 26% are label noise (vote contradicts TrueSkill), vs 9% on unanimous-right pairs.

## Ranking (Qwen3-VL-8B, 150 images per attribute)
| | safety | lively | wealthy |
|---|---|---|---|
| Pair acc vs TrueSkill (pool) | .670 | .590 | .588 |
| BT, all 1,200 pairs (soft), Spearman vs TrueSkill | .494 | .255 | .303 |
| Pointwise expected score (150 calls), Spearman | **.550** | .205 | .287 |
| BT k=2/img random → active | .373 → .378 | .198 → .194 | .227 → .235 |
| BT k=8/img random → active | .483 → .486 | .255 → .251 | .294 → .303 |
Active (closest-score) sampling and pointwise priors add ≈0.

## Takeaways for EAGLE → vision
1. Δp transfers: monotone calibration on all three VLM judges (≈.51 → ≈.80). But it saturates (Qwen3-VL: 60% of pairs at Δp≈1), so resolution is poor at the top.
2. The EAGLE premise "pairwise > pointwise" does **not** hold here: logit-expected pointwise scores match or beat pairwise, and for ranking 150 pointwise calls beat 2,400 pairwise ones on safety. Pairwise–pointwise disagreement is the most useful error flag (pairwise right only 43–50% there).
3. Ties: VLM judges never use a tie option and their confidence is unrelated to human disagreement → the "ties as aleatoric" idea fails for zero-shot models; it needs training or an explicit disagreement model.
4. The ceiling is ~.65–.68 for all 7–9B judges with highly correlated errors; gains will come from better perception/training (UrbanAlign-style), not gating.

## Thinking mode (Qwen3.5-9B, 300 strict pairs, 50/attribute; judge_think.py)
- 212 pairs parsed in both orders (85 truncated at 1,200 new tokens). Thinking acc .642 vs non-thinking logit judge .665 on the same pairs; swap consistency rises to .925.
- Unlike text judges (IDK quick tests: thinking .674→.823 on JudgeBench), thinking does not help subjective visual judgments: it makes the judge more decisive, not more correct. Caveat: truncation; single model.
