# A. General VLM field, 2025-01 → 2026-10: what changed, and where a calibrated selective judge fits

*Compiled 2026-10-31 for the EAGLE / EAGLE-V project. Complements, and does not repeat, sections A3–A7 of `VLM 发展脉络 关键文献.md`. Papers that survey already cites are listed only when they are needed for the argument (marked "(in survey)").*

**Verification legend.** **V** = arXiv ID, title and date checked against the arXiv API or abs page on 2026-10-31; venue taken from the arXiv comment field or the proceedings page. **V-s** = ID and title confirmed only through search-engine listings of the arXiv or ACL page; the abs page was not fetched (the arXiv API rate-limited us). **†** = venue recalled from memory and not re-checked. **U** = unverified.

---

## Executive summary (10 lines)

1. **Architecture has converged.** The open-model stack is now a native-resolution ViT, an MLP merger with a token budget, and an MoE LLM, usually with joint text–vision pretraining. Differentiation comes from data (Molmo2, PLM, LLaVA-OV-1.5), RL "thinking" variants and agentic/long-context abilities, not from the architecture.
2. **"Thinking" is a mode, not a separate model.** GLM-4.5V, Kimi-VL/K2.5, MiMo-VL, Keye-VL, Qwen3-VL and Phi-4-reasoning-vision all ship RL-trained reasoning variants. Latent visual reasoning (LVR, ICLR 2026) is the newest branch of "thinking with images".
3. **RL vs perception.** The 2025–26 consensus is that outcome-only RLVR mostly improves how the model uses visual evidence. It leaves perception errors unrewarded and so opens the door to hacking (PEARL, VPPO, Multimodal Reward Hacking, 2607.09492). The fix in 2026 is perception-specific rewards and staged training (From Seeing to Thinking, ICML 2026), and RL turns out to beat caption SFT for perception.
4. **Judges are now the reward channel.** Generative and reasoning RMs (UnifiedReward-Think, Skywork-VL Reward), rubric RMs (DeltaRubric, V-Rubrics, Omni-RRM) and pairwise win-rate rewards (Pref-GRPO) feed GRPO directly, for both understanding and generation.
5. **Pairwise beats pointwise for VLM judges.** GenArena reports Spearman 0.86 vs 0.36 against LMArena. But pairwise judges show **forced-choice bias** (they prefer one of two equally good answers) and **informativeness bias** (they judge without looking at the image, >30–50%).
6. **Judges remain weak where it matters.** On MMRB2, GPT-5 and Gemini 2.5 Pro reach 66–75% against >90% for humans. Evaluator VLMs miss more than 50% of some perturbations.
7. **The evaluation crisis has moved on.** Static benchmarks saturate and are blind-solvable (VLMBias: about 17% counting accuracy on counterfactual images). Arena data (VisionArena, 230K conversations) is now the reference, and arena ranking itself has known pathologies (Leaderboard Illusion).
8. **VLM calibration is a live 2025–26 subfield.** Verbalized confidence is poorly calibrated, while token probability is strong (AUROC 0.92–0.99 in one small-VLM study). Larger models know better what they don't know. **CoT/thinking degrades token-probability uncertainty** through implicit answer conditioning and evidence collapse.
9. **Abstention is under-served.** Under standard prompting, VLMs rarely abstain (MM-AQA). The best signals are grounding-aware: a blind-image contrast (BICR), a vision veto, decoupled visual vs reasoning confidence (VL-Calibration, ACL 2026), or learned conformal thresholds.
10. **The gap EAGLE fills.** No paper found combines order-symmetrized first-token confidence, selective fallback and BT/Davidson aggregation for a *pairwise VLM judge*. The nearest pieces are swap-consistency filtering (2606.18451), conformal abstention (2502.06884) and the bias papers. A calibrated, abstaining pairwise judge is directly consumable by Pref-GRPO-style RL and by arena-style evaluation.

---

## 1. Frontier and open model families (2025-01 → 2026-10)

**What now dominates.**
- **(a) Native or dynamic resolution with an explicit token budget.** SigLIP 2 NaFlex, MoonViT, Qwen3-VL, and Gemma's selectable budgets.
- **(b) MoE LLM backbones.** Examples are Seed1.5-VL (20B active), GLM-4.5V, Llama 4, Qwen3.5 and Kimi K2.5.
- **(c) Joint or native multimodal pretraining.** Retrofitting a text LLM is no longer the default (InternVL3, Qwen3.5, Kimi K2.5).
- **(d) Thinking/non-thinking modes produced by RL**, often hybrid within one checkpoint.
- **(e) Open-data reproductions.** Molmo2, PerceptionLM and LLaVA-OneVision-1.5 explicitly avoid distilling from proprietary VLMs.
- **(f) Optical compression and long context.** DeepSeek-OCR treats images as a compression medium for text.

Unified understanding+generation (BAGEL, Janus-Pro, Emu3.5) is already covered in survey A4. The news since is mainly that **generation RL now consumes VLM judges** (§3).

| Paper / system | arXiv | Date | Venue | One-line finding | Ver. |
|---|---|---|---|---|---|
| Gemini 2.5 tech report | 2507.06261 | 2025-07 | arXiv | Thinking-budget models; native multimodal plus long-video and agentic abilities; sparse MoE | V |
| Gemma 3 | 2503.19786 | 2025-03 | arXiv | Open SigLIP-based VLM with Pan&Scan for flexible resolution; 128K context | V |
| SigLIP 2 | 2502.14786 | 2025-02 | arXiv | Encoder adds captioning/self-distillation losses and NaFlex native aspect ratio; default open encoder | V |
| Seed1.5-VL | 2505.07062 | 2025-05 | arXiv | 532M encoder plus 20B-active MoE LLM; strong GUI and agent results | V |
| Kimi-VL (in survey) | 2504.07491 | 2025-04 | arXiv | MoonViT native resolution plus MoE; separate Thinking variant | V |
| MiMo-VL | 2506.03569 | 2025-06 | arXiv | 7B; mixed on-policy RL across many reward types; RL model beats Qwen2.5-VL-7B on 35/40 tasks | V |
| Kwai Keye-VL | 2507.01949 | 2025-07 | arXiv | Short-video focus; auto-think / think / no-think modes trained with RL | V |
| GLM-4.5V / GLM-4.1V-Thinking | 2507.01006 | 2025-07 (v5 later) | arXiv | RL with curriculum sampling; SOTA among similar-size open models on 42 benchmarks; later version adds GLM-4.6V with tool use | V |
| Eagle 2.5 (NVIDIA) | 2504.15271 | 2025-04 | arXiv | Long-context post-training (video, high-res); information-first sampling | V |
| PerceptionLM (Meta) | 2504.13180 | 2025-04 | arXiv | Fully open data and models without proprietary distillation; PLM-VideoBench | V |
| InternVL3.5 (in survey) | 2508.18265 | 2025-08 | arXiv | Cascade RL plus Visual Resolution Router; decoupled vision/language deployment | V |
| Ovis2.5 | 2508.11737 | 2025-08 | arXiv | Native-resolution ViT plus optional "thinking mode" with reflection | V |
| MiniCPM-V 4.5 | 2509.18154 | 2025-09 | arXiv | Efficient on-device model; 3D-resampler video compression; hybrid fast/deep thinking | V |
| LLaVA-OneVision-1.5 | 2509.23661 | 2025-09 | arXiv | Fully open training framework and data; low-budget reproduction of SOTA-level results | V |
| DeepSeek-OCR | 2510.18234 | 2025-10 | arXiv | "Contexts optical compression": text rendered as images at about 10× token compression | V |
| Qwen3-VL (in survey) | 2511.21631 | 2025-11 | arXiv | DeepStack; 256K interleaved context; text timestamps | V |
| Molmo2 (Ai2) | 2601.10611 | 2026-01 | arXiv | Open weights and data; 9 new datasets; SOTA open model on video pointing and tracking | V |
| Step3-VL-10B | 2601.09668 | 2026-01 | arXiv | Perception Encoder plus Qwen3-8B; 1.2T multimodal tokens | V-s |
| Kimi K2.5 (in survey) | 2602.02276 | 2026-02 | arXiv | Joint text–vision pretraining and RL; "zero vision SFT" | V |
| Phi-4-reasoning-vision-15B | 2603.03975 | 2026-03 | arXiv | Compact open multimodal reasoning model | V-s |
| GPT-5 family; Claude 4.x; Gemini 3 | — | 2025-08 → 2026 | system/model cards | Closed frontier: unified routers between fast and thinking modes, agentic computer use, higher input resolution. Only vendor documents exist, no papers | U (vendor docs) |
| Llama 4 (in survey) | — | 2025-04 | Meta blog | Open-weight early-fusion MoE | U (blog) |

---

## 2. Multimodal reasoning and RL: what is known about perception vs reasoning

**Timeline.** R1-style GRPO was ported to VLMs in Feb–Apr 2025 (Visual-RFT, VLM-R1, Vision-R1, LMM-R1: all in survey). Mid-2025 brought agentic "thinking with images" (DeepEyes, Mini-o3). Latent visual reasoning followed in late 2025 and 2026.

**Perception vs reasoning: the evidence is now fairly consistent.**
- Outcome-only rewards do not supervise perception, and the resulting perception errors cap accuracy (PEARL; VTPerception-R1; PeRL-VL).
- Only a small fraction of CoT tokens depend on the image (VPPO).
- Longer reasoning loses visual grounding (More Thinking, Less Seeing, in survey; Evidence Collapse).
- When perception is targeted *explicitly*, RL learns it better than caption SFT (From Seeing to Thinking, ICML 2026).
- SFT-then-RL can lock in "pseudo reasoning paths" (SFT or RL?).

| Paper | arXiv | Date | Venue | One-line finding | Ver. |
|---|---|---|---|---|---|
| R1-Onevision | 2503.10615 | 2025-03 | arXiv | Converts images into formal text representations to bootstrap multimodal CoT plus RL | V |
| MM-Eureka | 2503.07365 | 2025-03 | arXiv | Rule-based RL reproduces "aha"-style reflection and accuracy-reward growth in VLMs | V |
| OpenVLThinker | 2503.17352 | 2025-03 | arXiv | Iterative SFT↔RL cycles; SFT surfaces reasoning, RL refines it | V |
| Perception-R1 | 2504.07954 | 2025-04 | arXiv | RL for pure perception policies (grounding, counting, detection); gains depend on the task's perceptual perplexity | V |
| SFT or RL? (VLAA-Thinking) | 2504.11468 | 2025-04 | arXiv | SFT induces imitative "pseudo reasoning paths" that hamper later RL; RL alone generalizes better | V |
| Mini-o3 | 2509.07969 | 2025-09 | arXiv | Multi-turn visual search with tens of tool turns; over-turn masking | V |
| Self-Rewarding VLM (Vision-SR1) | 2508.19652 | 2025-08 | arXiv | Model writes a self-contained visual description and is rewarded if text-only reasoning succeeds from it; cuts language shortcuts | V |
| VTPerception-R1 | 2509.24776 | 2025-09 | arXiv | Explicit textual perception before reasoning works best; perception-aware RL rewards | V |
| Latent Visual Reasoning (LVR) | 2509.24251 | 2025-09 | ICLR 2026 | Reasons autoregressively in the visual-embedding space; SFT then RL | V (venue via mlanthology) |
| Spotlight on Token Perception (VPPO) | 2510.09285 | 2025-10 | ICLR 2026 | Visual dependency is sparse across CoT tokens; reweighting advantages toward visually dependent tokens improves RLVR | V |
| PEARL | 2511.18437 | 2025-11 | arXiv | Perception checklist sub-questions give a perceptual reward that gates the reasoning reward; reduces hacking | V |
| PeRL-VL / "More Than the Final Answer" | 2512.12487 | 2025-12 | arXiv | Decouples visual-extraction and logical-consistency rewards on top of RLVR | V |
| What does RL improve for visual reasoning? (in survey) | 2602.12395 | 2026-02 | arXiv | RL mainly changes mid/late-layer vision–reasoning alignment | V |
| From Seeing to Thinking | 2605.20177 | 2026-05 | ICML 2026 | VLM performance is limited mainly by perception. Staged perception→visual-reasoning→text-reasoning training beats merged training (+1.5% accuracy, 20.8% shorter traces), and RL beats caption SFT for perception | V |
| Generative Universal Verifier (OmniVerifier) | 2510.13804 | 2025-10 | arXiv | ViVerBench (16 categories): VLMs are poor visual verifiers; a sequential verifier-guided test-time scaling method beats best-of-N on generation tasks | V |
| MM-PRM | 2505.13427 | 2025-05 | arXiv | Step-level PRM for multimodal math; PRM-guided best-of-N | V |
| MG-Select | 2510.05681 | 2025-10 | ICLR 2026 | Verifier-free best-of-N for VLA models, using the model's own confidence (KL to a reference) to select | V |
| The Cost of Reasoning | 2603.16728 | 2026-03 | arXiv | CoT raises accuracy but *degrades* uncertainty ranking (implicit answer conditioning); agreement-based consistency stays robust | V |
| Don't Blink: Evidence Collapse | 2604.04207 | 2026-04 | arXiv (submitted to UAI 2026) | Attention to evidence regions drops by more than half during reasoning; "confident-but-blind" errors; a vision veto cuts selective risk by up to 1.9 pp at 90% coverage | V |

**Test-time scaling for VLMs.** It remains mostly PRM/verifier-guided best-of-N (VisualPRM and MM-PRM) plus sequential self-refinement (OmniVerifier-TTS). The 2026 calibration papers above warn that the policy's own token probabilities are a biased selector once CoT is on.

---

## 3. Reward models and judges inside multimodal RL pipelines

**How judges are consumed in 2025–26.**
- **(i) Scalar BT reward head → generative/CoT judge → RL-trained reasoning judge.** IXC-2.5-Reward → UnifiedReward-Think, Skywork-VL Reward, R1-Reward (in survey).
- **(ii) Rubric/checklist judges** that decompose a response into verifiable visual propositions (DeltaRubric, V-Rubrics, Omni-RRM).
- **(iii) Pairwise win-rate rewards within a GRPO group** (Pref-GRPO). These avoid the "illusory advantage" that appears when pointwise scores are near-identical and group normalization amplifies the gaps.
- **(iv) Perception-grounded or self-rewards** (PEARL, Vision-SR1).
- **(v) VLM reward models for robotics** (VLAC, RoboReward).
- **(vi) Hacking audits** show that weakly grounded judges are exploited (2607.09492).

| Paper | arXiv | Date | Venue | One-line finding | Ver. |
|---|---|---|---|---|---|
| IXC-2.5-Reward | 2501.12368 | 2025-01 | ACL 2025 Findings | Simple scalar multimodal RM; used for RL and best-of-N data filtering | V |
| MM-RLHF | 2502.10391 | 2025-02 | arXiv | 120K human preference pairs; critique-based RM; dynamic reward scaling in DPO | V |
| UnifiedReward | 2503.05236 | 2025-03 | arXiv | One RM for understanding *and* generation, pairwise and pointwise | V |
| UnifiedReward-Think | 2505.03318 | 2025-05 | NeurIPS 2025 | Multimodal CoT reward model trained with GRPO-style reinforcement fine-tuning | V |
| Skywork-VL Reward | 2505.07263 | 2025-05 | arXiv | Open multimodal RM; SOTA on VL-RewardBench at release; used for MPO data | V |
| Flow-GRPO | 2505.05470 | 2025-05 | NeurIPS 2025† | Online RL for flow-matching T2I with VLM/RM rewards; documents reward hacking | V (venue †) |
| Pref-GRPO | 2508.20751 | 2025-08 | arXiv | Pointwise RM scores plus group normalization produce illusory advantages, so the reward is replaced by **pairwise win rate** within the group; more stable and less hacking | V |
| Multimodal RewardBench 2 | 2512.16899 | 2025-12 | CVPR 2026 | Omni RM benchmark (T2I, editing, interleaved, thinking-with-images): GPT-5 and Gemini 2.5 Pro at 66–75% vs humans >90% | V |
| Omni-RRM | 2602.00846 | 2026-01 | ECCV 2026 | Rubric-grounded omni-modal RM; automatic rubric preference synthesis; SFT then GRPO | V |
| MSRL | 2603.25108 | 2026-03 | arXiv | Multi-stage RL to scale generative multimodal RMs | U (search snippet only) |
| DeltaRubric | 2605.09269 | 2026-05 | arXiv | Judge first plans an instance-specific disagreement checklist, then verifies it against the image; targets "lazy judging" via language priors | V |
| V-Rubrics | 2608.25580 | 2026-08 | arXiv | Atomic-proposition rubrics (visual faithfulness, reasoning consistency, instruction following) as partial-credit GRPO rewards on Qwen3-VL-8B | V |
| RubricRM | 2608.26956 | 2026-08 | arXiv | Dynamic-rubric generative RM for image generation and editing | U (search snippet only) |
| Multimodal Reward Hacking in RL | 2607.09492 | 2026-07 | arXiv | Outcome-only rewards reach 48.1% reward-hacking rate; RL creates *new* failures; scale helps but does not fix it; VLM-judge semantic verification reduces hacking | V |
| VLAC | 2509.15937 | 2025-09 | arXiv | Vision-language-action-critic: pairwise observations → progress delta and done signal as a dense reward for real-robot RL | V |
| RoboReward | 2601.00675 | 2026-01 | arXiv | General VLM reward models and benchmark for robotics; an 8B RM beats Gemini Robotics-ER 1.5 as a reward source | V |
| Calibrated Self-Rewarding (CSR) | 2405.14622 | 2024-05 | NeurIPS 2024† | Self-reward combined with a visual-grounding (image–response relevance) term to calibrate preference data | V (venue †) |

---

## 4. The evaluation crisis and the move to arena and preference evaluation

Survey A6 already covers saturation, MMStar/MMMU-Pro, TsT, MMGist and MM-Detect. New material here:
- **Arena data** is now treated as the external validity anchor (VisionArena, WildVision, GenArena).
- **Counterfactual "biased" images** show VLMs answer from memorized priors (VLMBias).
- **Judge-reliability audits** show VLM evaluators share the blind-solvability problem (informativeness bias, blind spots).
- **Arena methodology** has its own critiques (Leaderboard Illusion: private testing, selective disclosure, sampling asymmetries).

| Paper | arXiv | Date | Venue | One-line finding | Ver. |
|---|---|---|---|---|---|
| WildVision | 2406.11069 | 2024-06 | NeurIPS 2024 D&B† | Online arena with Elo ratings; WV-Bench reaches 0.94 Spearman with the arena | V (venue †) |
| VisionArena | 2412.08687 | 2024-12 | CVPR 2025 | 230K real conversations, 73K users, 45 VLMs; VisionArena-Bench predicts the arena better than WV-Bench | V |
| Vision language models are blind | 2407.06581 | 2024-07 | ACCV 2024† | Frontier VLMs fail trivial low-level visual tasks (line intersections, circles) | V (venue †) |
| Judge Anything | 2503.17489 | 2025-03 | arXiv | MLLM-as-judge across any-to-any modalities; pair comparison more reliable than scoring; weak on generation tasks | V |
| The Leaderboard Illusion | 2504.20879 | 2025-04 | arXiv (venue U) | Arena rankings are distorted by undisclosed private testing and data-access asymmetries | V |
| Vision Language Models are Biased (VLMBias) | 2505.23941 | 2025-05 | arXiv | On counterfactual images (e.g., a 5-legged dog), counting accuracy averages about 17%; about 76% of errors match the memorized prior | V |
| GenArena | 2602.06013 | 2026-02 | arXiv | For visual generation: pointwise VLM judging is unstable; **pairwise** judging gives +20% accuracy and ρ=0.86 with LMArena vs 0.36 for pointwise | V |
| Seeing Isn't Believing: blind spots of Evaluator VLMs | 2604.21523 | 2026-04 | arXiv | 4K perturbations across 40 dimensions: evaluators miss over 50% in some categories. Pairwise is the most reliable paradigm but shows **forced-choice bias** on equal-quality pairs | V |
| When VLMs Judge Without Seeing (BIRCH) | 2604.17768 | 2026-04 | ACL 2026 | Judges prefer more "informative" answers that contradict the image (>30–50% bias; 26–45% after length matching); correct-then-compare reduces it | V-s |
| Lost in Translation: LVLM judges across languages | 2604.19405 | 2026-04 | arXiv | Judge accuracy and position bias vary across languages | V-s |
| Am I More Pointwise or Pairwise? | 2602.02219 | 2026-02 | Findings EMNLP 2026 | Rubric (pointwise) judges also have *position bias over score options* and criterion-order effects; permuting options helps | V |
| 3D-mesh VLM-judge study (swap-consistency) | 2606.18451 | 2026-06 | arXiv | Queries the judge in both orders and discards verdicts that flip; cross-family κ 0.33–0.71 | V-s (title not confirmed) |
| LMArena Vision leaderboard | — | live | web | Crowdsourced pairwise votes with BT scores and style control; the de facto VLM reference ranking | U (web) |

---

## 5. Uncertainty, calibration, abstention and selective prediction in VLMs (nearest neighbour of EAGLE)

**State of knowledge, 2026-10.**
- **(a) Verbalized confidence is poorly calibrated** in VLMs, and almost constant in small ones. **Token probability is a much stronger error detector** (2607.22034), which supports EAGLE's first-token-probability choice.
- **(b) Larger models are better at uncertainty quantification** (conformal benchmark 2509.13379). Math and reasoning tasks show the worst uncertainty quality.
- **(c) Thinking/CoT hurts token-probability calibration** (2603.16728; 2604.04207), while agreement- or consistency-based signals survive. This supports EAGLE's two-order agreement over single-pass probability.
- **(d) Visual grounding is a missing axis of confidence.** A model can be confident from language priors alone. BICR (blind-image contrast), VL-Calibration (separate visual vs reasoning confidence) and the vision veto all add a grounding term.
- **(e) Abstention.** VLMs rarely abstain unprompted (MM-AQA). Conformal and RL-learned thresholds exist, but none target pairwise judging.

| Paper | arXiv | Date | Venue | One-line finding | Ver. |
|---|---|---|---|---|---|
| Reliable VQA: Abstain rather than answer incorrectly | 2204.13631 | 2022-04 | ECCV 2022 | Defines risk–coverage and Effective Reliability for VQA; selector heads beat softmax-max | V |
| Unsolvable Problem Detection (MM-UPD) | 2403.20331 | 2024-03 | ACL 2025 | LMMs fail to withhold answers on unsolvable MCQs (absent answer, incompatible answer set, incompatible image) | V |
| VL-Uncertainty | 2411.11919 | 2024-11 | arXiv | Semantic-entropy-style uncertainty under visual and text prompt perturbations detects LVLM hallucination | V |
| Learning Conformal Abstention Policies | 2502.06884 | 2025-02 | arXiv | RL-learned adaptive conformal thresholds for LLM/VLM abstention; ECE −70–85% vs LAC/APS | V |
| Object-level verbalized calibration (CSP) | 2504.14848 | 2025-04 | arXiv | Semantic-perturbation data plus SFT and preference optimization improve object-level verbalized confidence | V |
| Conformal prediction sets for LVLMs | 2504.17671 | 2025-04 | ICIPCA 2025 | Split-conformal prediction sets on MCQ VQA with coverage guarantees | V |
| Variational VQA | 2505.09591 | 2025-05 | TMLR 2026 | Variational (IVON) fine-tuning improves calibration and selective prediction; notes that verbalized confidence is ineffective for VLMs | V |
| Seeing is Believing, but How Much? | 2505.20236 | 2025-05 | EMNLP 2025 | First broad verbalized-calibration study for VLMs: widespread miscalibration; thinking-with-images models calibrate better; Visual Confidence-Aware Prompting | V |
| The Art of Saying "Maybe" | 2509.13379 | 2025-09 | arXiv | Conformal UQ benchmark of 18 VLMs over 6 datasets: larger models give better UQ; math/reasoning is worst; logprob proxies for closed models | V |
| The Cost of Reasoning | 2603.16728 | 2026-03 | arXiv | CoT induces overconfidence in token probabilities; consistency-based UQ is robust | V |
| Don't Blink: Evidence Collapse | 2604.04207 | 2026-04 | arXiv | Grounding loss during reasoning gives confident-but-blind errors; a vision veto improves selective risk | V |
| VL-Calibration | 2604.09529 | 2026-04 | ACL 2026 | RL (GRPO) with separate visual and reasoning confidence; visual confidence comes from KL under image perturbation plus token entropy. ECE on Qwen3-VL-4B falls from 0.421 to 0.098 (secondary-source figure) | V |
| Knowing When Not to Answer (MM-AQA) | 2604.14799 | 2026-04 | arXiv | Unanswerable instances built along visual-dependency and evidence-sufficiency axes. VLMs rarely abstain, and simple confidence baselines beat prompting. The bottleneck is miscalibration, not reasoning | V |
| Instinct vs. Reflection | 2604.17274 | 2026-04 | arXiv | Token-level and verbalized confidence diverge; monotone fusion plus order-preserving alignment improves calibration without hurting the risk–coverage curve | V |
| Grounded or Guessing? (BICR) | 2605.10893 | 2026-05 | arXiv | Probe on hidden states with a ranking loss that penalizes higher confidence on a blacked-out image; detects ungrounded confident answers at zero extra inference cost | V |
| Small VLMs know when they are wrong but cannot say so | 2607.22034 | 2026-07 | arXiv | Under image degradation, verbalized confidence sits near chance (AUROC about 0.5) while mean token probability reaches AUROC 0.92–0.99 | V |
| ConfBench (document extraction) | 2608.01792 | 2026-08 | arXiv | Verbalized vs logprob confidence for VLM document extraction | U (search snippet only) |

---

## Implications for a calibrated selective VLM judge (EAGLE-V)

Concrete places where a pairwise judge that emits Δp and can abstain or fall back would be consumed:

1. **Pairwise-reward GRPO (Pref-GRPO style) for T2I, editing and VQA policies.** Pref-GRPO already turns a pairwise judge into within-group win rates. EAGLE-V can **mask or down-weight low-Δp pairs** in the win-rate computation, or **replace them with pointwise fallback**. This directly attacks the illusory-advantage and hacking problem: a near-tie should give zero advantage, not noise. Fitting Davidson rather than plain BT inside the group handles ties natively.
2. **Rubric/checklist judges (DeltaRubric, V-Rubrics, Omni-RRM).** These are pointwise and carry option-position bias (2602.02219). EAGLE's **pointwise fallback is exactly this class of judge**. Running the rubric judge only on the abstained (low-Δp) subset makes the cost argument: expensive rubric judging goes only where the cheap pairwise judge is unsure.
3. **Forced-choice bias** (2604.21523) and **informativeness bias** (2604.17768) both produce *confident-looking but order- or prior-driven* verdicts.
   - Order symmetrization removes position bias. It does **not** remove informativeness bias.
   - Hook: add a **blind-image control.** Query the judge with the image blacked out, as BICR does. If p̃ barely changes, the verdict is ungrounded, so lower Δp or abstain.
4. **Thinking-mode judges.** The 2026 evidence (2603.16728, 2604.04207) says CoT inflates token-probability confidence. EAGLE-V should either read the first-token probability *without* CoT, or rely on the two-order *agreement* signal. Thinking judges such as UnifiedReward-Think and R1-Reward are a key ablation: does Δp calibration survive thinking mode?
5. **Arena-style evaluation pipelines** (VisionArena, LMArena vision, GenArena, MMRB2-style judge leaderboards).
   - Arena leaderboards are BT fits over pairwise votes, so EAGLE-V can serve as an **automatic arena voter** that abstains on near-ties.
   - Report ρ with the human arena **as a function of coverage**, using the risk–coverage framing of Reliable VQA and VL-Calibration.
6. **Reward-hacking guard.** 2607.09492 shows that RL policies exploit weakly grounded judges. A selective judge gives a natural **"reward-uncertain → no gradient"** rule. Watching the abstention rate during RL is a cheap hacking monitor: rising abstention on policy samples suggests the policy has drifted out of the judge's competence.
7. **Best-of-N and test-time scaling.** Use EAGLE-V as a tournament selector. Abstention lets the system fall back to more samples or a stronger judge (a judge cascade). The open question is matching a PRM (VisualPRM, MM-PRM) at lower cost.
8. **Robotics and embodied rewards.** VLAC already uses *pairwise observations* for progress rewards, and RoboReward shows no RM excels across tasks. A pairwise "which frame is closer to the goal" judge with Δp gating fits naturally here.
9. **Evaluation protocol for EAGLE-V itself.**
   - Report AURC / selective accuracy at fixed coverage.
   - Report ECE of Δp, both raw and after temperature or isotonic calibration.
   - Report the agreement of the BT/Davidson ranking with humans as a function of τ.
   - Include grounding stress tests: VLMBias-style counterfactual images, blind-image ablation, and MMRB2 hard pairs.
   - Compare against verbalized confidence, self-consistency, conformal sets (2502.06884, 2509.13379) and BICR.
10. **Positioning.** No 2025–26 paper found combines order-averaged first-token confidence with selective pointwise fallback and paired-comparison aggregation for VLM judges. The closest are the swap-consistency filtering in 2606.18451 (binary discard, no confidence) and conformal abstention (for answers, not judgments). State this gap explicitly, citing 2604.21523, 2604.17768 and 2602.02219 as motivation.

---

## Outdated or wrong in the existing survey (`VLM 发展脉络 关键文献.md`)

1. **No coverage of calibration, uncertainty or abstention in VLMs.** The only mention is HLE's calibration error. §5 above needs to be added; it is the most relevant gap for EAGLE.
2. **No arena or human-preference evaluation in A6.** VisionArena (CVPR 2025), WildVision, the LMArena vision leaderboard, GenArena and the Leaderboard Illusion critique are all missing, although the closing paragraph stresses BT/pairwise lineage.
3. **The A5 judge/RM paragraph stops at R1-Reward (2505.02835).** It misses:
   - UnifiedReward(-Think) and Skywork-VL Reward;
   - MMRB2 (CVPR 2026), whose frontier judges reach only 66–75%;
   - rubric RMs (DeltaRubric, V-Rubrics, Omni-RRM);
   - Pref-GRPO's pairwise rewards;
   - the 2026 evaluator-bias papers (forced-choice bias, informativeness bias).
4. **"Pairwise is more consistent with humans" needs a caveat.** It still holds (GenArena, 2604.21523). But pairwise judges show forced-choice bias on equal-quality pairs, which matters for tie modelling (Davidson) and is not mentioned.
5. **The A5 headline ("RL mainly improves using visual evidence, not seeing better") is now too strong.** 2605.20177 (ICML 2026) shows perception *is* learnable by targeted RL, and better than by caption SFT. PEARL, VPPO and Vision-SR1 add perception rewards. Rephrase as "outcome-only RLVR does not improve perception; perception-targeted RL does."
6. **Molmo 2 is cited via the Ai2 blog.** It now has a technical report, arXiv 2601.10611 (2026-01).
7. **The GLM-4.1V-Thinking entry (2507.01006) is outdated.** The arXiv record is now titled "GLM-4.5V and GLM-4.1V-Thinking"; later versions reportedly also cover GLM-4.6V.
8. **The GPT-5.4 figures (OSWorld-Verified 75.0%, MMMU-Pro 81.2%) cite a news aggregator (noqta.tn).** Replace them with the OpenAI system card or mark them unverified.
9. **Missing 2025 open model reports that define the "thinking-mode" norm:** MiMo-VL, Keye-VL, Seed1.5-VL, Ovis2.5, MiniCPM-V 4.5, PerceptionLM, LLaVA-OneVision-1.5, and the SigLIP 2 encoder (A2/A3).
10. **The hallucination and reasoning discussion lacks the CoT-overconfidence result** (2603.16728) and evidence collapse (2604.04207). These link "More Thinking, Less Seeing" to calibration.
