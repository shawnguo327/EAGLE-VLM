# EAGLE-V: novelty check and applications (2026-10-31)

**Scope.** This file checks whether EAGLE-V is new and lists where it could be used. EAGLE-V queries a pairwise VLM judge in both orders, averages the probability that the first candidate wins (p̃), computes Δp = |2p̃−1|, and trusts the pairwise verdict when Δp ≥ τ. Otherwise it falls back to pointwise scoring or escalates to a stronger model or a human, and it aggregates outcomes with Bradley–Terry or Davidson.

**Method.** About 30 web searches (arXiv, OpenReview, ACL Anthology, CVF, PMLR, Google-Scholar-style queries), seeded from `../vlm-judge-boundary/judgemap_1.md`. Every arXiv ID below was checked against the arXiv API or the abs page on 2026-10-31 (title and first-version date). Venues come from arXiv comments or proceedings pages; anything I could not confirm is marked "venue unverified". As in BOUNDARY.md, "not found" means the searches missed it, not that it cannot exist.

---

## 1. Novelty verdict

### 1.1 The one finding that changes the framing

**SCOPE (2602.13110, ICML 2026) already uses EAGLE's confidence signal.** Its "Bidirectional Preference Entropy" (Sec. 2.2, Eq. 2–3) works like this:

- p_fwd = P(A | original order), p_rev = P(B | swapped order), p̄ = (p_fwd + p_rev)/2;
- score s = H(p̄), the binary entropy of p̄; confidence = max(p̄, 1−p̄).

H(p̄) is a strictly decreasing function of |2p̄−1| = Δp. So **EAGLE's Δp and SCOPE's BPE produce identical rankings, and therefore identical risk–coverage curves and identical selective sets for any threshold.** SCOPE also includes "Swap-and-Aggregate" as a baseline.

SCOPE is text-only (MT-Bench, RewardBench, Chatbot Arena, JudgeBench, PKU-SafeRLHF). It has **no pointwise fallback**; the paper names "extending selective guarantees to point-wise evaluation" as future work. It has no human escalation and no Bradley–Terry aggregation.

**Consequence: EAGLE-V cannot claim the order-averaged confidence as a contribution.** It must present Δp as "BPE (Badshah et al., 2026)" or explicitly as equivalent to it.

### 1.2 Closest 10 works

| # | Work (arXiv, v1 date, venue) | Modality | What it does | How EAGLE-V differs |
|---|---|---|---|---|
| 1 | **SCOPE**, Badshah, Emami, Sajjad. 2602.13110, 2026-02-13, **ICML 2026** (PMLR 306) | Text | Pairwise judge in both orders → BPE (entropy of order-averaged p). Conformal calibration of an accept threshold so that error among accepted judgments ≤ α under exchangeability; abstains otherwise. Up to 2.4× more accepted judgments than baselines at α = 0.1. | **Same signal.** Differences: images, pointwise fallback instead of plain abstention, BT/Davidson aggregation. EAGLE-V's saturation finding (60% of pairs at Δp≈1) means BPE is a weak conformal score for VLMs; SCOPE does not report this. |
| 2 | **Trust or Escalate**, Jung, Brahman, Choi. 2407.18370, 2024-07-25, **ICLR 2025** | Text | Selective evaluation with a provable human-agreement guarantee. "Simulated Annotators" confidence (in-context persona ensemble). **Cascaded Selective Evaluation**: Mistral-7B → GPT-3.5 → GPT-4, with per-stage thresholds set by fixed-sequence testing. | This is the template for "escalate to a stronger model". EAGLE-V is the multimodal version, with a pointwise fallback rung and human as the last rung. Simulated Annotators must be a baseline. |
| 3 | **Judge, Retrieve, or Abstain**, Badshah et al. 2608.17994, 2026-08-18, **COLM 2026** | Text | Risk-controlled selective judging with Clopper–Pearson bounds. When uncertain, the judge retrieves web evidence and re-judges, then abstains to a human. | Its intermediate rung is "retrieve evidence"; EAGLE-V's is "re-ask pointwise". Same authors as SCOPE, so they are likely to extend to vision next. |
| 4 | **When the Judge Acts**, Seo. 2610.01243, 2026-10-01, preprint | **VLM** | Audits VLM judges picking the best of several generated images (300 culturally situated prompts). Repeats each choice with candidates reordered. For a 4B judge, **keeping only order-stable decisions** gives much better than random accuracy; for an 8B judge the same filter mostly throws away good decisions. Reports risk–coverage operating points but says explicitly that they are **not calibrated and carry no guarantee**. | **Closest multimodal work.** It uses a hard order-consistency filter and no guarantee; EAGLE-V uses a soft Δp with a calibrated τ and a fallback. Its finding that "filters must be re-audited per judge" matches EAGLE-V's saturation problem. Must be cited and discussed. |
| 5 | **SHIP**, Yin et al. 2608.21748, 2026-08-22, preprint | T2I | Post-selection risk control for verifier-guided T2I search: replays the deployed best-of-N policy on held-out prompts, judges the released image with an independent judge, and picks the most permissive threshold with risk upper bound ≤ budget. Finite-sample validity on a fixed grid. GenEval2/FLUX: released risk 0.310 → 0.162. | Guarantees cover the *generator's released output*, not *judge–human agreement on pairs*. It shows that the CV community already accepts LTT/CRC-style certificates, which supports EAGLE-V's guarantee framing. It also directly competes with the best-of-N application. |
| 6 | **VLM Judges Can Rank but Cannot Score**, Kumar et al. 2604.25235, 2026-04-28, preprint (venue unverified) | **VLM** | Split-conformal prediction intervals from score-token log-probabilities, 3 judges × 14 task categories. "Ranking–scoring decoupling": judges rank well, but intervals are wide (~40–70% of the scale). | Pointwise intervals only: no pairwise, no selection, no escalation. Its finding argues *against* pointwise fallback as the default. EAGLE-V must explain why pointwise helps on low-Δp pairs (it does sometimes match pairwise in the preliminary data). |
| 7 | **Multimodal Bayesian Prompt Ensembles**, Slyman et al. 2509.08777, 2025-09-10, **ICCV 2025** | VLM (T2I judging) | Calibrates MLLM-judge confidence with image-cluster-conditioned Bayesian prompt ensembles. Improves ECE on T2I judging. | Calibration only: no selection rule, no guarantee, no escalation. A strong *alternative confidence score* to compare against Δp, and it may fix Δp's scale miscalibration. |
| 8 | **K-Sort Eval**, Li et al. 2602.09411, 2026-02-10, **ICLR 2026** | VLM (visual generation) | Uses a VLM judge in place of K-Sort Arena human votes, with **posterior correction** (adaptive, based on VLM–human agreement) inside Bayesian updating plus dynamic matching. Rankings match the arena with <90 model runs. | Heuristic correction with no CI or coverage guarantee and no per-item gate. Main baseline for the arena and leaderboard applications. |
| 9 | **Consensus Entropy**, Zhang et al. 2504.11101, 2025-04-15, **CVPR 2026** | VLM (OCR) | Agreement entropy across several VLMs as a training-free reliability score. Routes low-agreement OCR outputs. +42.1% F1 over VLM-as-judge for verification. | A multi-model signal versus EAGLE-V's single-model order swap. Covers strings only, with no guarantee. The researcher's measured cross-judge error correlation of .65–.73 is the counter-argument. Natural baseline. |
| 10 | **CORA**, Feng et al. 2604.09155, 2026-04-10, preprint | VLM agents (mobile GUI) | Guardian model scores per-step risk. **Conformal risk control** sets an execute/abstain threshold under a harm budget; rejected actions go to a VLM "Diagnostician" (confirm / reflect / abort). Introduces the Phone-Harm benchmark. | Selective control of *actions*, not *judgments*. Shows that CRC + VLM + escalation exists in the agent space, so the agent-verification application is less open than the judge cell. |

**Also must cite (verified IDs):**

*Text-side selective and cascaded judges, 2025–26*
- **JEV-as-a-Judge**: Li et al. 2609.26550, 2026-09-22, preprint. Decision-only judge outputs label probabilities; accept if confident, escalate to a reasoning judge; ~41% of GPT-6's cost.
- **Codifying the Judge**: 2607.22561, preprint. Program judges first, LLM fallback on low-confidence pairs.
- **Localize-Then-Decide**: Li et al. 2608.25824, Findings EMNLP 2026. Conformal shortlist then select-or-abstain with agreement guarantees.
- **Cascaded LMs for Cost-effective Human–AI Decision-Making**: Fanconi & van der Schaar 2506.11887, NeurIPS 2025 per proceedings. Base → large → human, with online-tuned abstention.
- **Decomposing LLM-Judge Uncertainty to Target Expert Labels**: Lail 2609.06444, preprint. Separates aleatoric (human-disagreement) from epistemic uncertainty for escalation. Directly relevant to the finding that Δp is uncorrelated with human disagreement.

*Conformal judges*
- Sheng et al. 2509.18658 (EMNLP 2025), interval evaluations.
- Gupta 2604.15302 (under review), conformal sets plus transitivity violations; relevant to BT aggregation.

*Order bias*
- Judging the Judges, 2406.07791 (AACL-IJCNLP 2025).
- **EditJudgeBias**, 2610.01670 (2026-10-01): order swaps flip up to 60.9% of MLLM pairwise editing verdicts. Strong motivation for the problem.
- MLLM-as-a-Judge, 2402.04788 (ICML 2024).

*Pairwise vs pointwise*
- **Pairwise or Pointwise?**, Tripathi et al. 2504.14716 (venue unverified). Pairwise flips ~35% vs pointwise ~9% under distractors. Needed to justify the fallback.

*VLM selective prediction, not judges*
- Conformal abstention policies for LLMs and VLMs, 2502.06884.
- Look Again Before You Abstain, 2606.16667.
- ReCoVERR, 2402.15610 (ACL Findings 2024).

*Statistics*
- **PPE** (Frick et al.), 2410.14872 (ICLR 2025).
- **PPI++**, 2311.01453.
- **Active Statistical Inference** (Zrnic & Candès), 2403.03208 (ICML 2024).
- **Stratified PPI** (Fisch et al.), 2406.04291 (NeurIPS 2024). Uses autorater confidence for strata.
- **Prediction-Powered Ranking** (Chatzi et al.), 2402.17826 (NeurIPS 2024). PPI rank-sets from human + LLM pairwise comparisons; the direct template for BT + PPI.
- **AutoEval Done Right**, 2403.07008 (includes PPI on Bradley–Terry; venue unverified).
- **Multiple-PPI**, 2603.27414 (ICLR 2026).
- **Active Multiple-PPI**, 2605.08429.

### 1.3 Status of each sub-question (as of 2026-10-31)

| Sub-question | Status | Evidence |
|---|---|---|
| (i) Selective, abstaining, conformal or risk-controlled **VLM judge / multimodal RM with a human-agreement guarantee** | **Still OPEN, but closing.** Nearest: 2610.01243 (selective, no guarantee), 2604.25235 (conformal, pointwise, no selection), 2608.21748 (risk control on generator output). | No VLM SCOPE or Trust-or-Escalate found. |
| (ii) Uncertainty-gated **cascade** of VLM judges (cheap→expensive→human) | **OPEN for judges.** Exists for VLM *answering* (MLLM cascades are mature enough to be attacked: Forced Deferral 2606.15308) and for GUI *actions* (CORA). | — |
| (iii) **Order-swap confidence** for VLM pairwise judges | **PARTIAL.** Hard swap-consistency filters appear in 2610.01243 and as method details in T2I/3D papers (e.g. 2606.20364 de-biased VLM-as-3D-judge). Soft order-averaged probability = SCOPE BPE (text). No paper calibrates it on images. | — |
| (iv) **PPI / active evaluation for VLM benchmarks** | **THIN.** K-Sort Eval (heuristic, no CIs). No PPI-corrected T2I/VQA/editing leaderboard with valid CIs found. DatBench 2601.02316 does subset selection without human labels. | — |
| (v) Text analogues | **Saturated**, see the list above. | — |

### 1.4 One-paragraph verdict

The **cell** EAGLE-V targets is still empty: no paper gives a VLM/MLLM judge a finite-sample human-agreement (or error-rate) guarantee with abstention or escalation, and no paper builds a VLM judge cascade that ends in a human. But the **ingredients** are not new.

- The order-averaged confidence is exactly SCOPE's BPE (ICML 2026).
- Cascaded escalation is Trust-or-Escalate (ICLR 2025).
- Order-consistency filtering of VLM image judges appeared a month ago (2610.01243).
- Risk-controlled VLM verifiers appeared in T2I (SHIP, Aug 2026) and GUI agents (CORA, Apr 2026).

So the window is open but short. The SCOPE authors already extended to "Judge, Retrieve, or Abstain", and two October-2026 VLM papers circle the same idea.

What is genuinely new and defensible:

1. **The first application of SCOPE / Trust-or-Escalate guarantees to VLM judges**, plus an analysis of *why BPE degrades in vision*. The preliminary data supports this: 60% of pairs saturate at Δp≈1, AUROC is about .62, Δp is uncorrelated with human disagreement, and judge errors are correlated across VLMs at .65–.73. This is a substantive negative or diagnostic result that SCOPE does not report.
2. **A pairwise→pointwise→stronger-model→human ladder** with a joint guarantee. SCOPE lists pointwise as future work; Trust-or-Escalate escalates only across models.
3. **Coupling the gate to BT/Davidson aggregation and PPI**, so that the guarantee is about the *ranking*, not the individual pair.

Claiming a "novel confidence measure" or a "novel gating method" would be rejected by any reviewer who knows SCOPE. Also, the current +0.5–1.5 pp gain is too small to carry a methods paper on its own. It needs either the guarantee (coverage at a target α) or the label-efficiency (PPI) story.

**Must cite (minimum set):** 2602.13110, 2407.18370, 2608.17994, 2610.01243, 2608.21748, 2604.25235, 2509.08777, 2602.09411, 2504.11101, 2604.09155, 2609.26550, 2504.14716, 2406.07791, 2610.01670, 2410.14872, 2311.01453, 2403.03208, 2406.04291, 2402.17826, 2509.18658.

---

## 2. Applications

**Scoring.** Impact (I) and feasibility (F) are each scored 1–5. Feasibility assumes one PhD student, 1–2 GPUs, 3 months, and open-weight VLM judges (Qwen2.5/3-VL-7B, InternVL3-8B) with an optional paid API rung. Ranked by I×F.

**Caveat for every row.** With AUROC ≈ .62 and 60% saturation, the *realised* savings will be at the low end until the confidence signal improves (for example by ensembling with 2509.08777-style prompt ensembles, or by using pointwise-score margins as a second signal). The PPI-based applications (#1, #2) are the most robust to a weak signal: PPI stays valid with a bad judge and only loses efficiency.

| Rank | Application | Who uses it | Current practice | What the calibrated gate buys | Demo dataset / pipeline | Key cite | I | F | I×F |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Label-efficient benchmark evaluation (PPI / active inference) for VLM judging and T2I leaderboards** | Benchmark builders, evaluation papers | Either all-human evaluation, or VLM-judge numbers reported with no CI | Δp becomes the **sampling propensity** for which items get human labels (active inference) or the **strata** (StratPPI). The result is valid CIs on win-rates and BT scores with fewer human labels; the judge's bias cannot leak into the estimate. | Place Pulse 2.0 multi-vote pairs (2,284 pairs + ties) as the human anchor; also GenAI-Arena votes, MJ-Bench | Zrnic & Candès 2403.03208; Chatzi 2402.17826 | 4 | 5 | **20** |
| 2 | **Human-annotation triage for perception surveys** (urban perception, aesthetics) | Urban science, geography, HCI survey designers | Random pair sampling or TrueSkill-style active sampling, all labelled by crowds | The VLM labels high-Δp pairs; humans see only low-Δp pairs. Report "x% fewer votes at y% agreement" plus a Davidson-model tie probability for ambiguous pairs. | Place Pulse 2.0 (Dubey et al. 1608.01769); urban VLM benchmark 2509.14574 | 1608.01769; 2606.00871 | 4 | 5 | **20** |
| 3 | **Arena leaderboards with fewer human votes** (T2I, editing, video arenas) | Arena operators (GenAI-Arena, K-Sort Arena) | All human votes; K-Sort Eval replaces them with a heuristically corrected VLM | Route only low-Δp battles to humans and fit BT/Davidson on the mixed votes with a PPI correction. You get a vote-budget curve against K-Sort Eval, plus rank-set coverage. | GenAI-Arena public votes (2406.04485); K-Sort Arena data (2408.14468) | 2602.09411 (baseline) | 4 | 4 | **16** |
| 4 | **Best-of-N / reranking for image generation** | T2I and inference-time-scaling researchers | Pointwise reward (ImageReward, VQAScore) or a VLM tournament with no stopping rule | A pairwise tournament where confident matches are resolved cheaply and uncertain ones go to pointwise scoring or a stronger judge. Fewer expensive calls per prompt, plus a "release / abstain" decision. | GenEval / GenEval2 with SDXL or FLUX-schnell, N = 4–16; inference-time scaling 2501.09732 | 2608.21748 (SHIP, competitor) | 4 | 4 | **16** |
| 5 | **Preference-data curation for multimodal DPO** | MLLM alignment researchers | RLAIF-V / Silkie / MM-RLHF: AI-labelled pairs kept by score margin or length heuristics | Keep high-Δp pairs and send low-Δp pairs to a stronger judge or drop them. Fewer noisy pairs gives the same DPO gain with fewer API calls, and the error rate of the kept labels is certified. | RLAIF-V data (2405.17220), VLFeedback (2312.10665); LoRA-DPO on a 7B model, eval on MMHal / AMBER | 2405.17220 | 5 | 3 | **15** |
| 6 | **Agent trajectory verification** (web/GUI) | Agent researchers using VLM success detectors | Single VLM judge (Pan et al. 2404.06474) or rule checks | The verifier abstains on uncertain trajectories and escalates to a human or a stronger model. Fewer false "success" labels in rejection-sampling or RL data. | AgentRewardBench (2504.08942): human-labelled web trajectories, offline | 2404.06474; CORA 2604.09155 | 4 | 3 | **12** |
| 7 | **Abstaining reward model for multimodal RL (GRPO)** | Multimodal RL researchers (R1-Reward, UnifiedReward users) | Always-on generative RM; reward hacking is handled by KL or ensembles | Zero or neutral reward when Δp < τ, so the policy is not pushed by coin-flip judgments. Gives stabler RL and less reward hacking. | UnifiedReward (2503.05236) as the RM; small GRPO on a 3B VLM | 2505.02835 | 5 | 2 | **10** |
| 8 | **Remote-sensing caption/VQA review** | Earth-observation VLM builders | Reference metrics (BLEU, CIDEr) or unvalidated GPT judges | A gate that flags captions the judge cannot verify from the image (small objects, counts) for expert review | VRSBench (2406.12384) captions + human-checked references | 2406.12384 | 3 | 3 | **9** |
| 9 | **Medical report review (radiology)** | Clinical AI evaluation | Text-only judges (GREEN) that never see the image; radiologist audit of samples | Image-grounded pairwise judge plus abstention, so radiologist time goes to uncertain reports | ReXVal / MIMIC-CXR (credentialed access: slow); calibrated confidence for reports 2603.29492 | 2405.03595 | 4 | 2 | **8** |
| 10 | **Image–text data filtering (DataComp-style)** | Pretraining-data researchers | CLIPScore, DFN, MLM-Filter (2403.02677) | Spend VLM calls only on pairs where CLIPScore is uncertain; abstained pairs get a fallback rule | DataComp small (12.8M); judging even 1% twice is costly, and CLIP training needs multiple GPUs | 2304.14108 | 3 | 2 | **6** |

**Notes by rank**

1. **PPI / active evaluation.** This is the best fit. The gate's weakness (AUROC .62) costs only efficiency here, never validity. Δp drives *where* humans label: inverse-propensity-weighted active inference (2403.03208), or StratPPI strata (2406.04291). PPR rank-sets (2402.17826) carry the result through to BT rankings. No VLM leaderboard reports PPI-corrected CIs yet (BOUNDARY.md "THIN"). The work is mostly CPU statistics on existing judge outputs.

2. **Survey triage.** This is the researcher's home domain, and Place Pulse's 203k ties make Davidson natural. The deliverable: "Δp-routed collection reaches the same TrueSkill/BT ranking of cities with N% of votes." The reliability-aware benchmark paper (2606.00871) asks for exactly this kind of reliability reporting.

3. **Arenas.** Same statistics as #1, applied to model rankings. Compare directly with K-Sort Eval's heuristic posterior correction: EAGLE-V plus PPI gives CIs, K-Sort Eval does not.

4. **Best-of-N.** High visibility in CV. SHIP (2608.21748) already certifies released-image risk, so position EAGLE-V as the *cost* side (fewer strong-judge calls per tournament) and the pairwise side, not as the first guarantee.

5. **DPO curation.** Highest downstream impact. But you must show that a DPO gain survives with LoRA on one or two GPUs, and baseline label-noise filters (margin filtering in RLAIF-V) already exist. Do it after #1–#3.

6. **Agents.** AgentRewardBench supplies human labels, so this is an offline experiment. CORA already covers CRC for GUI actions, so the novelty is limited to judge verdicts on trajectories.

7. **Abstaining RM in RL.** Exciting, but RL compute and noise make three months risky.

8–10. **Remote sensing, medical, DataComp.** Domain value is real, but each is bottlenecked by data access, expert labels, or compute. Best kept as one-paragraph "generalisation" demos.

---

## 3. Suggested paper framing

**A. Diagnostic + guarantee (recommended).** *"Trust or Escalate, Seen: Selective VLM Judges with Human-Agreement Guarantees, and Why Order-Swap Confidence Saturates in Vision."*

Claim: SCOPE- and Trust-or-Escalate-style guarantees carry over to VLM judges (valid risk at α), but bidirectional confidence saturates on images, so coverage collapses at tight α. A pairwise→pointwise→strong-model→human ladder recovers coverage at the same certified risk. Evaluate across T2I preference, image editing, VQA judging, and urban perception.

Baselines: BPE/Δp, Simulated Annotators, Consensus Entropy, prompt ensembles (2509.08777), and the order-consistency filter (2610.01243).

**B. Statistics / label efficiency.** *"Spend Humans Where the VLM Is Unsure: Confidence-Routed Prediction-Powered Ranking for Visual Preference Data."*

Claim: using a VLM judge's order-swap confidence as the human-labeling propensity in active PPI yields valid Bradley–Terry/Davidson rank-sets with X% fewer human comparisons. Shown on Place Pulse 2.0, GenAI-Arena, and K-Sort Arena; the comparison is against uniform sampling, plain PPI, and K-Sort Eval. This claim is the most robust to the weak AUROC.

**C. Protocol paper (weakest alone).** *"Pairwise When Sure, Pointwise When Not: Confidence-Gated Hybrid Protocols for Vision-Language Judges."*

Claim: on low-confidence pairs, pointwise scoring is more reliable than pairwise for VLMs. This connects Tripathi et al. 2504.14716 with Kumar et al. 2604.25235. The gated hybrid dominates both pure protocols on accuracy-per-call.

This only stands if the gain grows well beyond +0.5–1.5 pp, or if it is folded into A as the fallback rung.
