# B. Application domains for a calibrated, selective pairwise VLM judge (EAGLE-V)
*Literature review, compiled 2026-10-31. Scope: medicine, science/documents, education/accessibility, niche visual domains.*

Legend: **[V]** = arXiv ID/venue checked against the arXiv abs page, PhysioNet, proceedings, or a matching search hit during this review. **[M]** = cited from memory or a secondary aggregator only; re-check before citing. Venue "preprint" means no venue was listed on arXiv when checked.

---

## Executive summary (10 lines)

1. **Radiology report generation (RRG)** has the most mature "judge" ecosystem: lexical → entity (RadGraph-F1, RaTEScore) → LLM judges (GREEN, FineRadScore, CheXprompt, RadFact, LLM-RadJudge, MRScore). Almost all are **reference-based, pointwise, and text-only**. None look at the image, and none abstain.
2. Expert-agreement data is **small and mostly error-count based, not pairwise**: ReXVal (200 pairs, 6 radiologists, credentialed), RadEvalX (100 reports, CC BY-NC-SA, open), RaTE-Eval (2,215 sentence ratings + paragraph ratings), RadEval expert set (~450 error labels). Pairwise preferences can be derived from ReXVal (4 candidates per study, so 6 pairs per study).
3. **Medical multimodal judging is only starting**: Med-RewardBench (1,026 expert-annotated cases, Aug 2025) shows that 32 MLLMs align poorly with experts. Med VLM DPO/RLHF uses *automatic* preferences (CheXalign, GREEN-based) because radiologist labels are too expensive.
4. **Selective prediction in medicine is active in 2026** but covers the *generator*, not the *judge*: SCUQ-RRG, RadFlag, ReXTrust, Conformal Alignment (radiology), certified-coverage medical VQA (2606.15910), "Uncertainty is not a safety net" (2606.16583). They find little certified coverage at strict error tolerances.
5. **Judges with guarantees exist only in general NLP** (Trust-or-Escalate, ICLR 2025). Nobody yet offers a **pairwise, order-symmetric, image-grounded judge with a calibrated abstain/escalate rule in a medical domain**. That gap is the main opening.
6. **Science/documents**: chart judges (Judging the Judges 2505.08468, ChartAnno, DEEPCHART), SciCap-Eval (3,600 human-rated figure captions), process reward models for visual math (VisualPRM / VisualProcessBench), and document-parsing judges (DOCR-Inspector). **Consensus Entropy (CVPR 2026)** is the closest analogue to EAGLE: an agreement-based gate that routes uncertain OCR outputs and beats VLM-as-judge.
7. **Education**: handwritten-math grading (FERMAT, ACL 2025; GPT-4 on probability exams, LAK25; GPT-5 on calculus, 2510.05162). The calculus study already uses a **confidence filter** and sends ~70% of items back to humans, so a better-calibrated judge would directly shrink that fraction.
8. **Accessibility**: BLV-preference evaluation (ContextRef, VL-GUIDE / 2510.00766, Kreiss et al.) shows that reference-based metrics misalign with blind users. Datasets are small, which suits PPI-style evaluation.
9. **Niche domains** (agriculture, ecology, materials, remote sensing) mostly use unvalidated GPT judges or MCQ. Examples: AgML zero-shot study (LLM judge changes rankings), RealBirdID (abstention AUC ≤ 0.3), ScoreRS (92.6% judge–human agreement). The judges are unvalidated, so calibrated selective judging is easy to add and novel there.
10. **Best first targets for EAGLE-V**: (i) RRG pairwise judging on ReXVal/RadEvalX/RaTE-Eval with risk-controlled escalation to a radiologist; (ii) Med-RewardBench and Multimodal RewardBench as pairwise judge benchmarks; (iii) PPI with a selective judge for cheap expert-validated leaderboards (ReXrank-style).

---

## 1. Medicine

### 1a. What is evaluated and how

- **RRG (chest X-ray dominant).** Metrics fall into four tiers:
  - n-gram metrics: BLEU, ROUGE.
  - embedding metrics: BERTScore.
  - clinical-entity metrics: CheXbert F1, RadGraph-F1, RaTEScore, and the RadCliQ composite.
  - LLM judges: GREEN, FineRadScore, CheXprompt, RadFact, LLM-RadJudge, MRScore.

  Humans provide the ground truth through radiologist **error counts** (clinically significant vs insignificant, 6–8 categories). Pairwise preferences are rare: they appear only in large pharma/Google-style reader studies (Flamingo-CXR, 16 radiologists), and those annotations were not released.
- **Medical VQA / clinical answers.** VQA-RAD, SLAKE, PathVQA, PathMMU, GMAI-MMBench, and DermaBench are mostly scored by exact match or MCQ. Open-ended answers are scored by GPT judges and rarely validated against clinicians: one κ≈0.71 check used a *single medical student*.
- **Reward models / RLHF for medical VLMs.** These rely almost entirely on synthetic or metric-derived preferences, because radiologist feedback is too expensive. CheXalign states this explicitly.
- **Uncertainty / deferral.** Work targets the generator (hallucination flagging, report-level abstention, conformal selection of trustworthy reports, certified coverage for VQA) or diagnosis-level deferral (Codoc). None of it applies to the *judge*.

### 1b. Papers

| Paper | arXiv | Date | Venue | Finding relevant to EAGLE-V |
|---|---|---|---|---|
| RadGraph (Jain et al.) | 2106.14463 [M] | 2021-06 | NeurIPS D&B 2021 [M] | Entity/relation graph used by RadGraph-F1, the de-facto factual metric. |
| Evaluating progress in automatic CXR RRG (Yu, Endo, …, Rajpurkar): RadCliQ, ReXVal | medRxiv 10.1101/2022.08.30.22279318 [V] | 2022-08 / 2023-08 | *Patterns* 4(9) 2023 [V] | Introduces RadGraph-F1 and RadCliQ. Radiologist error-count study released as ReXVal. |
| GREEN (Ostmeier et al.) | 2405.03595 [V] | 2024-05 | Findings EMNLP 2024 [M] | Open <7B LM judge trained on 100k GPT-4-labelled pairs. Better alignment with expert error counts **and expert preferences** than GPT-4. Pointwise, reference-based, text-only. |
| RaTEScore (Zhao et al.) | 2406.16845 [V] | 2024-06 | EMNLP 2024 [M] | Entity-embedding metric with Kendall 0.527 on ReXVal. Releases **RaTE-Eval** (sentence + paragraph radiologist ratings, multi-modality). |
| FineRadScore (Huang et al.) | 2405.20613 [V] | 2024-05 | MLHC 2024 (PMLR 252) [V] | Line-level corrections with severity scores by Claude-3 Opus / GPT-4. Authors call for multi-rater datasets. |
| LLaVA-Rad + **CheXprompt** (Chaves et al.) | 2403.08002 [V] | 2024-03 | *Nature Comms* 2025 [M] | GPT-4 factuality judge; parity with radiologist error assessment (Kendall τ_b on ReXVal). |
| MAIRA-2 + **RadFact** (Bannur et al.) | 2406.04449 [V] | 2024-06 | preprint (MSR) | LLM-based sentence-level entailment, precision/recall, with or without grounding boxes. |
| LLM-RadJudge (Wang et al.) | 2404.00998 [V] | 2024-04 | preprint | GPT-4 judge reaches near-radiologist consistency; distilled to a small model. |
| MRScore (Liu et al.) | 2404.17778 [V] | 2024-04 | MICCAI 2024 [V] | **Reward model trained on radiologist-labelled accepted/rejected report pairs**. This is the closest existing pairwise human preference data for RRG; release status is unclear. |
| Radiology-aware model-based metric / **RadEvalX** (Calamida et al.) | 2311.16764 [V via PhysioNet] | 2023-11 | preprint | Two radiologists, 100 IU-Xray reports, 8 error categories. Open access. |
| ReXrank (Zhang et al.) | 2411.15122 [V] | 2024-11 | preprint | Public RRG leaderboard: 8 metrics, 4 datasets including the 10k private ReXGradient set. A natural consumer of a cheaper, validated judge. |
| RadEval (Xu, Delbrouck et al.) | 2509.18030 [V] | 2025-09 | EMNLP 2025 Demos [V] | Unified metric toolkit plus a radiologist set with ~450 clinically significant error labels; multi-modality GREEN. |
| ReXErr (Rao et al.) | 2409.10829 [V] | 2024-09 | PSB 2025 [V via PSB PDF] | LLM-injected clinically meaningful errors. Gives controlled pairs (clean vs corrupted) for judge stress-tests. |
| CorBenchX (Zou et al.) | 2505.12057 [V] | 2025-05 | preprint | 26,326 erroneous CXR reports for error detection; best model o4-mini only 50.6% detection. |
| Consensus, dissensus & synergy (Flamingo-CXR, Tanno et al.) | 2311.18260 [V] | 2023-11 | *Nature Medicine* 2024 [M] | 16-radiologist **pairwise preference** reader study, AI vs human reports. Shows high inter-radiologist variance, a case for abstention. Data not released. |
| SCUQ-RRG (Wang et al.) | 2412.04606 [V] | 2024-12 | Findings NAACL 2025 [V] | Semantic-consistency uncertainty: abstaining on 20% of reports raises factuality by 10%. Applies to the generator, not the judge. |
| RadFlag (Zhang et al.) | 2411.00299 [V] | 2024-11 | ML4H 2024 (PMLR 259) [V] | Black-box sampling-based hallucination flagging enabling report-level selective prediction. |
| ReXTrust (Hardy et al.) | 2412.15264 [V] | 2024-12 | AAAI Bridge / PMLR 281, 2025 [V] | White-box finding-level hallucination detector, AUROC 0.875. |
| Conformal Alignment (Gui, Jin, Ren) | 2405.10301 [V] | 2024-05 | NeurIPS 2024 [V] | **FDR-controlled selection of trustworthy outputs**, illustrated on CXR RRG. The closest statistical template for EAGLE's τ rule. |
| Conformal Selective Prediction w/ General Risk Control | 2603.24704 [M] | 2026-03 | preprint | Reuses the Gui et al. MIMIC-CXR setup for LLM abstention with general risks. |
| DPO for hallucinated prior exams (Banerjee et al.) | 2406.06496 [V] | 2024-06 | preprint [M: ML4H?] | DPO with GPT-4-edited preference pairs; 3.2–4.8× fewer hallucinated priors. |
| CheXalign (Hein et al.) | 2410.07025 [V] | 2024-10 | preprint [M: later venue] | Automated preference pairs from reference-based metrics (GREEN). Documents **reward over-optimization via length** and proposes length-controlled GREEN. |
| Med-RewardBench (Ding et al.) | 2508.21430 [V] | 2025-08 | preprint | **1,026 expert-annotated** pairwise/judgment cases across 13 organ systems and 6 dimensions. 32 MLLMs struggle to match experts. A ready-made pairwise judge benchmark. |
| LLaVA-Med (Li et al.) | 2306.00890 [M] | 2023-06 | NeurIPS 2023 D&B [M] | Baseline biomedical VLM, often used as the policy in medical DPO. |
| Med-Gemini (Saab et al.; Yang et al.) | 2404.18416 / 2405.03162 [M] | 2024-04/05 | preprint | Includes 2D/3D RRG with radiologist acceptability ratings. |
| MedGemma Technical Report (Sellergren et al.) | 2507.05201 [V] | 2025-07 | preprint | Open 4B multimodal / 27B models plus MedSigLIP. A natural open backbone for a medical EAGLE-V judge. |
| RadFM (Wu et al.) | 2308.02463 [M] | 2023-08 | *Nature Comms* 2025 [M] | 2D/3D radiology generalist; RadBench includes a human rating study. |
| BiomedCLIP (Zhang et al.) | 2303.00915 [M] | 2023-03 | *NEJM AI* 2024 [M] | Contrastive encoder, useful as a cheap image–text grounding signal for a hybrid judge. |
| CheXagent / CheXbench (Chen et al.) | 2401.12208 [M] | 2024-01 | preprint | CXR foundation model; CheXbench includes radiologist evaluation of findings. |
| PathChat (Lu et al.) | — (Nature 2024) [M] | 2024 | *Nature* 2024 [M] | Pathologist-preferred open-ended responses in blinded rating. Rating data not public. |
| PathMMU (Sun et al.) | 2401.16355 [M] | 2024-01 | ECCV 2024 [V] | 33k pathologist-validated MCQs. GPT-4V 51.7% vs pathologists 71.4%. |
| DermaBench | 2601.14084 [V] | 2026-01 | preprint | 656 dermatologist-annotated images, Fitzpatrick I–VI, open-ended answers scored by an atomic-claim LLM judge [M]. Released via Harvard Dataverse. |
| Calibrated Triage, Not Autonomy → retitled "Adapt or Calibrate? How Local Labels Buy Certified Coverage in Medical VQA" (Khanmohammadi et al.) | 2606.15910 [V] | 2026-06 (v3 2026-10) | preprint | Certified selective prediction: at 20% error tolerance, best pooled certificate covers 11.5% of SLAKE and 0% of PathVQA. 1k local labels raise this to 37.6%. **Nothing is certified at 5–10%.** |
| Uncertainty Is Not a Safety Net for Clinical VQA (Fazla et al.) | 2606.16583 [V] | 2026-06 | preprint | 8 UE methods × 12 VLMs. Uncertainty does not react to none-of-the-above (NOTA) perturbations, but clean-input uncertainty anticipates collapse. |
| Medical VLM overconfidence (Byun et al.) | 2604.02543 [V] | 2026-04 | preprint | Overconfidence persists across scale and prompting. Platt scaling beats verbalized confidence. |
| Answer, Clarify, or Abstain (fine-grained selective prediction for medical VLMs) | no arXiv found [M] | 2026 | ICML 2026 [V: ICML virtual page] | Three-way answer / clarify / abstain decision with an MLP reliability probe on hidden states. |
| LLM-as-a-Judge in Healthcare: scoping review (Li et al.) | 2605.25273 [V] | 2026-05 | preprint | 134 studies. Alignment with experts is moderate to strong but task-dependent. The abstract does not mention abstention/uncertainty as a theme. |
| Clinician-Level Agreement Without Clinical Caution (MedQADE) | 2607.01103 [M] | 2026-07 | preprint | Physician κ≈0.61. LLM evaluators approach it but show bias and **low abstention**. |
| Trust or Escalate (Jung, Brahman, Choi) | 2407.18370 [V] | 2024-07 | ICLR 2025 (oral) [V] | Pairwise judge with fixed-sequence-testing guarantee P(agree with human \| judged) ≥ 1−α plus a cascade. **The direct general-domain baseline for EAGLE.** Not yet applied to medical or multimodal settings. |
| Codoc: complementarity-driven deferral to clinicians (Dvijotham et al.) | — | 2023 | *Nature Medicine* 2023 [M] | Learns when to defer AI to the clinician in mammography/TB. A template for the escalation economics. |

### 1c. Selective / calibrated judging with guarantees?
Guarantees exist for **generators**: Conformal Alignment (FDR), certified-coverage VQA (2606.15910), Gui-style risk control (2603.24704). There are heuristic flags too (SCUQ, RadFlag, ReXTrust). **No published medical judge abstains with a guarantee.** GREEN, CheXprompt, RadFact, FineRadScore and MRScore always output a score. Med-RewardBench reports only accuracy and no coverage–risk curves.

### 1d. Pain points EAGLE-V solves
1. **Radiologist triage of AI drafts.** A pairwise judge (draft vs alternative/reference, or draft A vs draft B) with Δp<τ escalates *only uncertain studies* to a radiologist. Report a selective-risk guarantee in the style of Trust-or-Escalate or Conformal Alignment. 2606.15910 shows certified coverage is hard, which makes it a strong motivating result.
2. **Cheaper RLHF/DPO preferences.** CheXalign and GREEN-DPO use unfiltered automatic preferences and suffer length hacking. Keeping only high-Δp pairs, and sending low-Δp pairs to experts, is a cleaner reward signal. Test on Med-RewardBench.
3. **Leaderboards with fewer expert labels.** ReXrank-style rankings via Bradley–Terry/Davidson over a selective judge, with prediction-powered inference (PPI) correction from a small radiologist set such as ReXVal or RadEvalX.
4. **Image-grounded judging.** All RRG judges ignore the image. A VLM judge can catch errors that a reference report omits, such as reference-free findings.

---

## 2. Science and documents

### 2a. What is evaluated and how
- **Charts/figures.**
  - Chart QA uses relaxed exact match.
  - Chart summaries and captions are scored by GPT judges, with small human studies (UniChart: 150 charts, Krippendorff α=0.54).
  - Figure captions use SciCap-Eval (3,600 human ratings).
  - Chart generation/editing uses VLM judges with ICC/Pearson against humans.
- **Math with diagrams.** Answers are checked by exact match. Process reward models (VisualPRM) score each step, and VisualProcessBench supplies human step labels.
- **OCR/documents.** Scoring uses edit distance or tree-edit distance (TEDS). Newer work adds VLM judges (DOCR-Inspector, 28 error types) and LLM table judges (r=0.93 with humans versus TEDS). Consensus Entropy uses *agreement across VLMs* as a training-free reliability score with threshold routing.

### 2b. Papers

| Paper | arXiv | Date | Venue | Finding |
|---|---|---|---|---|
| MLLM-as-a-Judge (Chen et al.) | 2402.04788 [M] | 2024-02 | ICML 2024 [M] | General benchmark: VLM judges are good at pairwise comparison and poor at scoring/batch ranking, with position bias. Supports EAGLE's pairwise-both-orders design. |
| VL-RewardBench (Li et al.) | 2411.17451 [V] | 2024-11 | CVPR 2025 [V] | 1,250 pairs including math/reasoning. GPT-4o reaches 65.4%. |
| Multimodal RewardBench (Yasunaga et al.) | 2502.14191 [V] | 2025-02 | preprint (Meta FAIR) | 5,211 expert-annotated triplets incl. knowledge, reasoning (math/code), VQA. |
| Judging the Judges: LVLMs for chart comprehension (Laskar et al.) | 2505.08468 [V] | 2025-05 | ACL 2025 Industry [V via ACL PDF] | 13 open LVLM judges on chart tasks. Agreement with GPT-4 ranges from <10% to ~80%. Pairwise and pointwise. Releases judgments. |
| GPT-4 as zero-shot evaluator for figure captions / **SciCap-Eval** (Hsu et al.) | 2310.15405 [V] | 2023-10 | Findings EMNLP 2023 [V] | 3,600 captions × 600 figures rated by humans. GPT-4 Kendall 0.401 with PhD rankings; good at flagging bad captions. |
| Five Years of SciCap | 2512.21789 [M] | 2025-12 | preprint | Retrospective: LLM judgments match PhD raters; reference metrics are unreliable. |
| ChartAnno | 2608.03464 [M] | 2026-08 | preprint | GPT judge vs experts on chart annotation: Spearman 0.73–0.87, with stability measured by ICC(3,1). |
| DEEPCHART | 2608.26757 [M] | 2026-08 | preprint | VLM judge vs human labels on 180 charts: 94.1% accuracy, κ=0.85. |
| ChartSync | 2607.10301 [M] | 2026-07 | preprint | Chart editing benchmark; human ICC 0.87, VLM judge Pearson 0.915. |
| VisualPRM / VisualProcessBench (Wang et al.) | 2503.10291 [V] | 2025-03 | preprint | 8B multimodal PRM plus a human step-labelled benchmark. A step-level judge for math/diagram reasoning. |
| MaCBench (Alampara et al.) | 2411.16955 [V] | 2024-11 | NeurIPS 2024 AI4Mat workshop; later version [M] | Chemistry/materials figures, spectra, AFM. VLMs extract data well but fail at spatial and multi-step reasoning. MCQ only, no judge. |
| Consensus Entropy (Zhang et al.) | 2504.11101 [V] | 2025-04 (v3 2026-03) | CVPR 2026 [V] | Agreement entropy across VLMs gates OCR outputs. +15.2% F1 over VLM-as-judge for quality verification; only 7.3% of inputs routed. **The conceptually closest prior to EAGLE's Δp gate**, but not pairwise and with no guarantee. |
| DOCR-Inspector | 2512.10619 [V] | 2025-12 | preprint | 7B VLM judge for document parsing with 28 error types. Outperforms Gemini 2.5 Pro. DOCRcaseBench has 882 human-annotated cases. |
| LLM-as-judge for table extraction | 2603.18652 [M] | 2026-03 | preprint | 1,554 human ratings on 518 table pairs; LLM judge Pearson 0.93 vs humans, far above TEDS. |
| OmniDocBench (Ouyang et al.) | 2412.07626 [M] | 2024-12 | CVPR 2025 [M] | Document parsing benchmark with metric-based scoring. A candidate target for judge-based re-evaluation. |

### 2c. Selective/guaranteed judging?
Consensus Entropy uses a threshold gate but gives **no statistical guarantee** and is not pairwise. VisualPRM is used for Best-of-N without abstention. No chart or figure judge reports risk–coverage curves.

### 2d. Pain points
- **Rankings in science benchmarks with expensive experts.** Chart and figure judges disagree wildly across judge models (2505.08468). A judge that knows when it disagrees with itself across orders, and abstains there, would stabilize rankings.
- **OCR/document QA pipelines.** Replace or augment Consensus Entropy's multi-model ensemble with a single model's order-symmetric pairwise confidence, which is cheaper, and add a coverage guarantee.
- **Verifying visual math** in RL data curation: keep high-Δp preference pairs and escalate the rest.

---

## 3. Education and accessibility

### 3a. What is evaluated and how
- **Handwritten grading.** GPT-4o/GPT-5/Gemini graders are compared with TA rubrics. Agreement is high on objective items (κ≈0.90) and low on drawings and multi-step calculus. Confidence filtering is already used in the calculus study.
- **Alt-text / descriptions for BLV users.** Reference metrics (CIDEr, CLIPScore) misalign with BLV preferences. Small BLV rating studies (8–N participants) have produced learned evaluators (VL-GUIDE-S).

### 3b. Papers

| Paper | arXiv | Date | Venue | Finding |
|---|---|---|---|---|
| FERMAT: Can VLMs evaluate handwritten math? (Nath et al.) | 2501.07244 [V] | 2025-01 | ACL 2025 [V] | 2,244 handwritten solutions with injected errors in 4 dimensions. Best error correction is Gemini-1.5-Pro at 77%. Handwriting hurts. Open release. |
| Evaluating GPT-4 at grading handwritten math exams (Caraeni, Scarlatos, Lan) | 2411.05231 [V] | 2024-11 | LAK 2025 [V] | Probability-course exams. Even with rubric, scores are off by 7.66% on average, too low for deployment. |
| Seeing the Big Picture: MLLMs grading handwritten student work (Henkel et al.) | 2510.05538 [V] | 2025-10 | preprint | Objective items κ=0.90; drawings much lower, rising to κ=0.47 with human descriptions. |
| AI grading assistance for handwritten calculus (Kortemeyer et al.) | 2510.05162 [V] | 2025-10 | preprint | GPT-5 with **confidence filter (partial-credit threshold + IRT risk)** reaches human-level accuracy but leaves ~70% of items for humans. |
| ContextRef (Kreiss et al.) | 2309.11710 [V] | 2023-09 | ICLR 2024 [M] | Referenceless metrics vs BLV-aligned ratings on 204 Wikipedia alt-texts; context matters. |
| Context matters for image descriptions (Kreiss et al.) | 2205.10646 [V] | 2022-05 | EMNLP 2022 [M] | BLV participants' ratings; context-free referenceless metrics fail to capture them. |
| Are LVLMs ready to guide BLV individuals? (VL-GUIDE; Kim, An et al.) | 2510.00766 [V] | 2025-10 | preprint | Supersedes withdrawn 2502.14883 (Eye4B). BLV-user study yields a preference dataset; the VL-GUIDE-S evaluator beats LVLM judges in human alignment. |
| VizWiz-Captions (Gurari et al.) | 2002.08565 [V] | 2020-02 | ECCV 2020 [M] | 39k captions of blind-user photos. Reference data, no preferences. |
| Reference-based metrics are biased against BLV preferences | — | 2024 | NLP4PI @ ACL 2024 [V: ACL anthology] | Length and content preferences differ between BLV and sighted raters. |
| Context-aware descriptions via VizWiz-LF questions | 2510.01576 [M] | 2025-10 | preprint | Pairwise preference human eval (3 labellers, 92 items). |

### 3c. Selective/guaranteed judging?
The Kortemeyer calculus study does heuristic confidence-based deferral, with IRT risk. There is **no formal coverage/risk guarantee and no pairwise judge**. Accessibility work has no abstention at all.

### 3d. Pain points
- **Grading.** Reduce the ~70% human-graded share at a guaranteed error rate. Pairwise comparison of student work against exemplar solutions mirrors how TAs normalise partial credit.
- **Alt-text QA at scale.** Publishers and platforms need to flag alt-text that probably fails BLV needs. A selective judge auto-accepts confident wins and routes ambiguous cases to BLV raters. The human data is tiny, so PPI-corrected estimates of "% of alt-texts adequate" are valuable.

---

## 4. Agriculture, ecology, materials, remote sensing

| Paper | arXiv | Date | Venue | Finding |
|---|---|---|---|---|
| Are VLMs ready to zero-shot replace supervised models in agriculture? (Ranario & Earles) | 2512.15977 [V] | 2025-12 | preprint | 27 AgML datasets. Open-ended accuracy <25%. **LLM semantic judging lifts accuracy (21→30%) and changes rankings**, yet the judge was not validated against humans [M on numbers]. |
| AgroBench (Shinoda et al.) | 2507.20519 [V] | 2025-07 | ICCV 2025 [V] | Agronomist-annotated: 203 crops, 682 diseases; MCQ. |
| RealBirdID | 2603.27033 [M] | 2026-03 | preprint | Fine-grained bird ID with abstention. Uncertainty-threshold AUC ≤ 0.3, so VLMs abstain poorly in ecology. |
| ScoreRS (remote-sensing preference model) | — [M: arXiv ID not found] | 2025 | preprint [M] | GPT-4o judge with 92.6% agreement with humans on 1,000 samples; preference model for RS captions. |
| RSVLM-QA | 2508.07918 [M] | 2025-08 | preprint | GPT-4.1 as correctness and caption judge, with no reported human agreement. |
| MaCBench (see §2) | 2411.16955 [V] | 2024-11 | NeurIPS workshop | Materials/chemistry imagery; MCQ only. |

**(a)** Evaluation is mostly MCQ or accuracy; open-ended answers go to GPT judges with little or no validation. **(b)** There are almost no human preference datasets; AgroBench and AgML give labels, not preferences. **(c)** Abstention appears only as a capability probe (RealBirdID), never in a judge. **(d)** Pain point: open-ended expert answers (disease diagnosis advice, species ID rationale) cannot be graded by exact match. A validated selective judge with few-expert PPI calibration would make open-ended agri/eco leaderboards trustworthy.

---

## 5. Cross-cutting statistical tools (for the EAGLE-V guarantee story)

| Paper | arXiv | Venue | Use |
|---|---|---|---|
| Prediction-Powered Inference (Angelopoulos et al.) | 2301.09633 [M] | *Science* 2023 [M] | Unbiased estimates from many judge labels plus few expert labels. |
| PPI++ (Angelopoulos, Duchi, Zrnic) | 2311.01453 [M] | preprint | Power-tuned PPI. |
| Confidence-driven inference with unconfident LLM annotations (Gligorić et al.) | 2408.15204 [M] | NAACL 2025 [M] | Uses annotator *confidence* to choose which items get human labels. This matches the Δp-based escalation. |
| Efficient inference for noisy LLM-as-judge evaluation | 2601.05420 [M] | ICML 2026 [M: ICML poster page seen] | Compares Rogan–Gladen-style correction with PPI for judge-based evaluation. |
| Trust or Escalate | 2407.18370 [V] | ICLR 2025 | Fixed-sequence testing for selective judge agreement. |
| Conformal Alignment | 2405.10301 [V] | NeurIPS 2024 | FDR-controlled selection, illustrated on RRG. |

---

## 6. Dataset table: usable for testing a VLM judge

| Dataset | Domain | Content / size | Human signal | Pairwise derivable? | Access / license | URL |
|---|---|---|---|---|---|---|
| **ReXVal** | CXR RRG | 50 MIMIC-CXR studies × 4 candidates = 200 report pairs | 6 board-certified radiologists, significant/insignificant error counts, 6 categories | **Yes**: 6 candidate pairs per study, ranked by mean error count; ties map to Davidson | PhysioNet **credentialed** (CITI + DUA), Credentialed Health Data License 1.5.0. Images need MIMIC-CXR access too [V] | https://physionet.org/content/rexval-dataset/1.0.0/ |
| **RadEvalX** | CXR RRG (IU-Xray) | 100 reports (80 abnormal, 20 normal) | 2 radiologists (consensus), 8 error categories; metric scores included | Partial: one candidate per study, so pairs must be cross-study or against injected variants | PhysioNet **open**, CC BY-NC-SA 4.0 [V]. IU-Xray images are public | https://physionet.org/content/rad-eval-x/ |
| **RaTE-Eval** | Multi-modality radiology (MIMIC-IV, Radiopaedia) | 2,215 sentence-level ratings + paragraph-level 5-point ratings + synthetic-report comparison | Radiologists | Yes, within reference groups | Hugging Face; license field blank [M]. **Text only, no images** | https://huggingface.co/datasets/Angelakeke/RaTE-Eval |
| RadEval expert set | CXR | >450 clinically significant error labels | Board-certified radiologists [M] | Partial | GitHub, open [M] | https://github.com/jbdel/RadEval |
| MRScore preference pairs | CXR | Accepted/rejected report pairs | Radiologist-guided | **Natively pairwise** | "Will be released"; unverified [M] | arXiv 2404.17778 |
| ReXErr / ReXErr-v1 | CXR | Synthetic clinically meaningful errors | LLM-generated, clinician-reviewed taxonomy | Yes (clean vs corrupted) | PhysioNet open listing seen (rexerr-v1) [M on license] | https://physionet.org/content/rexerr-v1/ |
| CorBenchX | CXR | 26,326 erroneous reports | Synthetic (DeepSeek-R1) | Yes | Check repo [M] | arXiv 2505.12057 |
| **Med-RewardBench** | Multi-organ medical VQA | 1,026 cases, 6 dimensions, 13 organ systems | Expert-annotated | **Natively pairwise** | Paper does not link data [V]; ask the authors | arXiv 2508.21430 |
| DermaBench | Dermatology VQA | 656 images, 570 patients, Fitzpatrick I–VI, 22 question types | Dermatologist-written answers | No (answers, not preferences) | Harvard Dataverse, metadata-only (images from upstream) [V] | arXiv 2601.14084 |
| PathMMU | Pathology | 33k MCQs, 24k images | 7 pathologists validated | No | Public [M] | arXiv 2401.16355 |
| VQA-RAD / SLAKE / PathVQA | Medical VQA | Closed/open QA | Clinician-written answers | Via model answers + correctness | Public | — |
| **Multimodal RewardBench** | General incl. reasoning/math/knowledge | 5,211 triplets | Expert-annotated | **Native** | GitHub facebookresearch/multimodal_rewardbench [V] | arXiv 2502.14191 |
| **VL-RewardBench** | General incl. math | 1,250 pairs | AI-assisted + human-verified | **Native** | HF dataset [M] | arXiv 2411.17451 |
| MLLM-as-a-Judge | General | ~4.4k pairs/scores [M] | Human | Native | Public [M] | arXiv 2402.04788 |
| **SciCap-Eval** | Scientific figure captions | 3,600 captions × 600 figures | PhD and undergrad ratings | Yes: 6 captions per figure, rankings | GitHub Crowd-AI-Lab/SciCap-Eval [M on license] | arXiv 2310.15405 |
| Judging-the-Judges chart judgments | Charts | Judge outputs across 13 LVLMs | GPT-4 reference (not human) | Yes | Released with paper [M] | arXiv 2505.08468 |
| VisualProcessBench | Multimodal math reasoning | Step-level human labels | Human | Step-correctness (pointwise) | HF / OpenGVLab [M] | arXiv 2503.10291 |
| DOCRcaseBench | Document parsing | 882 cases | Manual annotations | Pointwise errors | GitHub ZZZZZQT/DOCR-Inspector [V] | arXiv 2512.10619 |
| FERMAT | Handwritten math | 2,244 solutions, 609 problems | Curated errors | Yes (correct vs erroneous) | Open release [V] | arXiv 2501.07244 |
| VL-GUIDE data (supersedes Eye4B) | BLV scene descriptions | "Large-scale" image-request-response-score | BLV-informed, user-simulated | Yes (scores) | Check repo [M] | arXiv 2510.00766 |
| ContextRef | Alt-text | 204 Wikipedia images + context | BLV-aligned rating scheme | Yes | Public [M] | arXiv 2309.11710 |

**Recommended judge test-suite order:** ReXVal (pairwise from error counts, image-grounded via MIMIC-CXR), then Med-RewardBench, then RadEvalX and RaTE-Eval for open/text-only robustness. After that, run Multimodal RewardBench and VL-RewardBench for general calibration, SciCap-Eval for science, and FERMAT for education.

---

## 7. Opportunities for EAGLE-V (ranked)

1. **Risk-controlled RRG judge with radiologist escalation (highest impact, clear gap).** Convert ReXVal (and RaTE-Eval/RadEvalX) to pairwise preferences and run EAGLE-V with an image-grounded VLM such as MedGemma-4B or Qwen-VL. Calibrate τ with fixed-sequence or conformal risk control so that P(agree with radiologist majority | judged) ≥ 1−α. Report risk–coverage against GREEN, RaTEScore, CheXprompt, and Trust-or-Escalate. **Selling point**: the first selective judge with a guarantee in medicine. 2606.15910 shows that certified coverage is otherwise tiny.
2. **Expert-efficient leaderboards via PPI and Bradley–Terry/Davidson.** For ReXrank-style rankings, the judge labels all pairs, experts label a small calibration slice, and PPI or confidence-driven inference (Gligorić et al.) gives valid CIs. Escalation should go to the low-Δp pairs. Quantify "radiologist hours saved per CI width".
3. **Filtered preference data for medical DPO/RLHF.** Keep high-Δp pairs and route low-Δp pairs to experts. Test whether this reduces the length hacking documented by CheXalign and improves Med-RewardBench accuracy compared with GREEN-derived preferences.
4. **Medical multimodal reward benchmarking.** Evaluate on Med-RewardBench with selective accuracy curves. Position bias and order-swap inconsistency are untested there, and EAGLE's both-orders averaging addresses them directly.
5. **OCR/document and chart verification.** Compare a single-model, order-symmetric Δp gate with Consensus Entropy (multi-VLM ensemble, CVPR 2026) at matched routing rate, using DOCRcaseBench and Judging-the-Judges chart data. EAGLE-V's edge would be lower cost plus a guarantee.
6. **Handwritten-grading triage.** On FERMAT (pairwise: correct vs erroneous solution) and Kortemeyer-style exams, reduce the ~70% human-graded share at a fixed error rate.
7. **BLV-aligned alt-text judging.** Pairwise preferences from VL-GUIDE/ContextRef give a small-data regime where abstention plus PPI matters most. Reference-free judging is a natural fit.
8. **Niche domains (agriculture/ecology/remote sensing).** Validate the unvalidated GPT judges used in AgML open-ended evaluation and RSVLM-QA. Low competition but also low reviewer pull; good for a workshop or appendix.

**Key risks to note:** medical images under MIMIC need credentialed access, so no API judges on MIMIC data are allowed under the DUA; use local models such as MedGemma. Inter-radiologist disagreement is high (Flamingo-CXR, CXRMate-2 slight κ), so define the "human majority" target carefully and use Davidson ties for low-agreement pairs. Most text-only RRG judges ignore images, so ablate image vs no-image to show what the VLM adds.
