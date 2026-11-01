# C. VLM/LLM judges, rewards and uncertainty in urban, geospatial, driving, embodied, GUI and video domains

*Literature review for EAGLE / EAGLE-V. Compiled 2026-10-31.*

**Verification conventions**
- Every arXiv ID in the tables was resolved through the arXiv API in this session. Title and first-submission date match, and venue is taken from the arXiv comment field where one exists.
- Numbers from the abstract carry no mark.
- † marks numbers or dataset sizes recalled from memory or taken from secondary summaries. Check these before citing.
- "(in notes)" means the paper is already covered in `priorart_urban_judge.md` or in B1–B5 of `VLM 发展脉络 关键文献.md`. Those papers are listed only for context and are not re-reviewed.

---

## Executive summary

1. **Every domain has now built pairwise or binary human-preference sets for judges. Almost none of them pair a judge with a risk guarantee.** The pairwise human-labelled judge benchmarks are:

   | Domain | Pairwise human-labelled judge benchmark |
   |---|---|
   | Driving (2025–26) | DriveCritic (5,730 NAVSIM trajectory pairs†) and DriveJudge (33,577 samples) |
   | Robotics | RoboReward, FailBench (2,197 attempts) and ROBORMBENCH (2,390 trajectories) |
   | Web | AgentRewardBench (1,302 trajectories†) and Online-Mind2Web |
   | Long video | SLVMEval and PlaylistEval |
   | Urban perception | Place Pulse 2.0 (UrbanAlign) |

   The only domain-specific selective or conformal work I found is one 5-page driving extended abstract, SLLCP (2610.02765). The generic precedents are SCOPE (2602.13110, ICML'26), Trust-or-Escalate (2407.18370) and conformal intervals for VLM judges (2604.25235). **No paper in urban, remote sensing, robotics, GUI or video applies order-symmetrized, abstaining pairwise judging with a guarantee.**

2. **Judges are wrong often, and the errors are systematic, not random noise.**
   - *Robotics.* The best of 13 VLM failure detectors reaches only 0.77 balanced accuracy, near chance on contact-rich tasks, and leans towards predicting "success" when evidence is ambiguous (FailBench). Paraphrasing the instruction can flip success and failure (ROBORMBENCH).
   - *Driving.* GPT scores barely change when the input image is black (DriveBench†).
   - *Urban perception.* GPT-4o reaches 56.7% (κ=0.15) zero-shot on PP2 pairs. Model judgments are culturally skewed towards Europe and North America (2604.20048).
   - *Long video.* Frontier judges reach 75.4% pairwise accuracy at day-scale (PlaylistEval).
   - *GUI.* Single judges miss evidence, which is why multi-agent critics such as OS-Themis exist.

   These are exactly the cases where a confidence-gated judge should abstain.

3. **Token-probability confidence is becoming the standard readout, but nobody calibrates it pairwise.** TOPReward (robotics) uses VLM token probabilities as the reward. RoboJev uses a "calibrated two-token readout". 2604.25235 builds conformal intervals from score-token log-probs. EAGLE's Δp is the pairwise, order-debiased version of this readout. The novelty is not in reading the token probability. It is in using both orders, abstaining, falling back to pointwise scores, and aggregating with Bradley–Terry/Davidson.

4. **Pairwise beats pointwise in several domains, which matches the "rank but cannot score" finding.**
   - RL-VLM-F gets preferences from pairwise VLM comparisons of image pairs.
   - DriveCritic and DriveJudge evaluate trajectory pairs.
   - WorldReward is a pairwise reward model.
   - Yo'City evaluates generated cities pairwise, with both VLM and human judges.
   - GPT-4o pairwise judgments with Thurstone scaling reach ρ=0.76 on PP2 attractiveness (Buildings 2025†).

   Together these support EAGLE's design choice of pairwise judging as the primary mode with a pointwise fallback.

5. **The strongest application pitches for EAGLE-V, in order:**
   - (i) Cheap, risk-controlled reward and success labels for robot RL and success detection, escalating ambiguous contact-rich episodes to humans.
   - (ii) A driving-planner evaluator that flags context-dependent trajectory pairs for human review.
   - (iii) An auto-evaluator for urban generative models (3D cities, synthetic street views, mobility) with a tie model and a stated abstention rate.
   - (iv) A GUI/web trajectory judge with a human-escalation budget.

---

## 1. Urban and geospatial

**(a) What gets judged, and by what.** GPT-4o-class MLLMs rate or compare street-view images on PP2 dimensions such as safety, liveliness and beauty, and on walkability, greenspace and wellbeing. Benchmarks test multi-view urban QA. VLMs judge generated 3D cities. LLMs answer planning-exam questions. Mobility-trajectory generators still use statistics or anomaly detectors, not LLM judges (see prior-art notes).

| Paper | arXiv | Date | Venue | Finding relevant to judging |
|---|---|---|---|---|
| UrbanAlign | 2602.19442 | 2026-02 | preprint (in notes) | PP2 pairwise. Zero-shot VLM gets 56.7% (κ 0.15). A post-hoc concept-plus-ridge calibration reaches 72.2% (κ 0.45). It has a tie band of ε but no risk guarantee. |
| Culturally uneven urban perception in LLMs | 2604.20048 | 2026-04 | preprint | Three frontier MLLMs on globally stratified street views. The model's neutral baseline sits closest to Europe and North America framings. Prompting cannot restore human diversity, and the models show a self-favouring sentiment bias. This argues for abstaining on out-of-culture inputs. |
| Greenspace attractiveness (ChatGPT/Claude/Gemini) | 2512.11827 | 2025-12 | preprint | Łódź residents vs. MLLMs on GSV images. Agreement is high on "easy" quadrants (attractive formal, unattractive informal) and low on the counter-intuitive ones. The authors call for human oversight, which is the abstention use case. |
| MLLMs predict urban safety perception but encode demographic priors | 2503.00610 | 2025-03 | preprint (in notes) | MLLM safety scores track humans but carry non-neutral demographic priors. |
| Urban safety perception via MLLMs + SVI | 2407.19719 | 2024-07 | preprint (in notes) | GPT-4o has the closest alignment among MLLMs (R² ≈ 0.40–0.45†). CLIP+kNN does better. |
| Do VLMs see urban scenes as people do? | 2509.14574 | 2025-09 | preprint (in notes) | 100 Montreal images (half synthetic). Reliability-aware and treats abstention as an outcome. Model scores are higher where humans agree. |
| UrbanFeel | 2509.22228 | 2025-09 | preprint (in notes) | Benchmark of temporal and perceptual understanding of city scenes. |
| UrbanWell | 2606.15890 | 2026-06 | preprint (in notes) | Benchmark for spatio-temporal urban-wellbeing analytics. |
| Interpretable human-centred street assessment (Lan) | 2506.05087 | 2025-06 | preprint (in notes) | VLM perception scores combined with interpretable factors. |
| GPT-4o street-view attractiveness (Buildings 15(16):2970) | — (MDPI) | 2025 | Buildings | GPT-4o pairwise judgments with Thurstone scaling reach ρ=0.76 with PP2, vs. 0.60 for a Siamese CNN†. **This is the closest precedent to EAGLE's pairwise-plus-BT protocol on PP2. It has no confidence gating.** |
| UrBench | 2408.17267 | 2024-08 | preprint (AAAI'25†) | Multi-view urban LMM benchmark (about 11.6K questions†). Models fall far short of humans, especially on cross-view tasks. |
| CityBench | 2406.13945 | 2024-06 | KDD'25 D&B (in notes) | Urban LLM tasks. |
| UrbanVideo-Bench | 2503.06157 | 2025-03 | preprint (ACL'25†) (in notes) | Embodied urban aerial video benchmark. |
| CityEQA | 2502.12532 | 2025-02 | preprint (in notes) | Embodied QA in city space. |
| MapEval | 2501.00316 | 2024-12 | ICML'25 Spotlight | Map-based geospatial reasoning. Foundation models trail humans by more than 20 pts†. A source of answer-pair data for a "map QA judge". |
| UrbanPlanBench | 2504.21027 | 2025-04 | KDD'25† | LLMs on planning-exam knowledge show large gaps. This is a knowledge test, not a judge of plans. |
| Yo'City | 2511.18734 | 2025-11 | CVPR'26 | 3D city generation with a self-critic. **Evaluated with pairwise comparisons by both VLM and human judges** on realism (heights, façades, spacing, layout)†. |
| MajutsuCity | 2511.20415 | 2025-11 | preprint (CVPR'26 suppl. exists†) | VLM-based evaluation with absolute scoring (AQS) plus relative dimension ranking (RDR). |
| CityGenAgent | 2602.05362 | 2026-02 | preprint (in notes) | Relies heavily on a GPT-4o judge with only 100 prompts, and the judge is not validated. |
| TrajGenAgent / "When plausible is not realistic" | 2606.12657 / 2606.13835 | 2026-06 | MDM'26† / preprint (in notes) | Mobility realism is judged by anomaly detectors and mobility laws. LLMs reward plausibility over realism. |

**(b) Usable human-label sets.**
- PP2: about 1.17M pairwise votes on about 110K images†.
- SPECS: 1,000 demographically balanced raters (in notes).
- Montreal 100-image set (2509.14574).
- Łódź greenspace survey (2512.11827; check data availability).
- Yo'City human pairwise study (small; check release).

**(c) Calibrated or selective judging.** UrbanAlign calibrates scores post hoc, and 2509.14574 treats abstention as an outcome. Neither offers a guarantee. **No conformal or selective PP2 judge exists.**

**(d) Pain points EAGLE solves.**
- Auto-evaluating urban generative models (3D cities, synthetic street views, trajectory generators) with a validated pairwise judge whose abstention rate is reported. Today's practice is an unvalidated GPT-4o judge on about 100 prompts.
- Large-scale perception mapping with "unsure" pairs routed to local raters. This also addresses cultural bias.

---

## 2. Remote sensing

**(a) What gets judged.** Captions, VQA answers and change captions. Recent RS benchmarks use GPT-class judges for open-ended captions. Learned scorers curate RS image-text data. Process reward models (PRMs) check reasoning faithfulness. No paper treats an RS-specific pairwise judge as an object of study.

| Paper | arXiv | Date | Venue | Finding |
|---|---|---|---|---|
| GeoChat | 2311.15826 | 2023-11 | CVPR'24† | First grounded RS LVLM. Domain baseline. |
| EarthGPT | 2401.16822 | 2024-01 | IEEE TGRS† | Multi-sensor (optical, SAR, IR) RS MLLM. |
| LHRS-Bot | 2402.02544 | 2024-02 | ECCV'24† | Uses VGI/OSM-enhanced alignment. |
| VHM (H2RSVLM) | 2403.20213 | 2024-03 | AAAI'25† (in notes) | **"Honest" RS VLM.** Trained to refuse unanswerable questions, the RS precursor of abstention. |
| SkySenseGPT | 2406.10100 | 2024-06 | preprint | FIT-RS instruction set covering fine-grained relation reasoning. |
| VRSBench | 2406.12384 | 2024-06 | NeurIPS'24 D&B | Captioning, grounding and VQA benchmark. Captions evaluated partly with GPT-4 judges†. |
| EarthDial | 2412.15190 | 2024-12 | CVPR'25† | Multi-sensor, multi-temporal dialogue. |
| XLRS-Bench | 2503.23771 | 2025-03 | CVPR'25 | Ultra-high-resolution RS. Large gaps remain. |
| Ultra-high-res RS MLLM benchmark | 2512.17319 | 2025-12 | preprint | Uses a GPT-5-Thinking judge for captions, with prompts iteratively tuned towards human scores†. |
| SenseBench | 2605.10576 | 2026-05 | preprint | RS low-level perception/IQA with over 10K items. LLM-judge description scores validated against 3 annotators (SRCC/PLCC)†. |
| RSHBench + RADAR | 2603.02754 | 2026-03 | preprint | RS hallucination benchmark. Judge-vs-human κ ranges from about 0.89 (GPT-5.2) to about 0.50 (Qwen3-max) on 100 items†. **Judge choice matters a lot in RS.** |
| ReconScore | 2604.22855 | 2026-04 | preprint | Reference-free caption metric based on reconstructability. Argues that reference metrics mask the ability of foundation models. Kendall τ_b on a UCM-preference set†. |
| RS data scoring model | 2503.00743 | 2025-03 | NeurIPS'25 | Score model trained on **RS vision-language preference data** to curate synthetic RS image-text pairs. The only RS "reward model" I found. |
| GeoSolver / GeoPRM | 2603.09551 | 2026-03 | preprint | Token-level RS process reward model (Geo-PRM-2M) for visual faithfulness in chain-of-thought. |
| GEOBench-VLM | 2411.19325 | 2024-11 | ICCV'25† (in notes) | Geospatial VLM benchmark. |

**(b) Datasets.**
- RS preference data from 2503.00743 (check release).
- VRSBench (open, `github.com/lx709/VRSBench`†).
- RSHBench human hallucination labels.
- SenseBench (3-annotator MOS-style).
- UCM-preference used in ReconScore†.
- **There is no large RS pairwise caption-preference set. This is a gap.**

**(c) Calibrated or selective judging.** VHM handles honesty and refusal at the answering model, not the judge. I found no RS judge with a confidence or abstention guarantee.

**(d) Pain point.** RS caption and VQA leaderboards depend on whichever closed judge each benchmark picked, and κ varies from 0.5 to 0.9 between judges. A selective pairwise judge with a reported coverage-risk curve would give RS benchmarks judge-agnostic, auditable scores. It could also sort synthetic RS instruction data into confident-keep, confident-drop and human-review bins.

---

## 3. Autonomous driving

**(a) What gets judged.** Three things:
- Planner trajectories, in pairs, against human preference (DriveCritic, DriveJudge).
- Free-form driving QA answers (LingoQA's Lingo-Judge, CODA-LM's text judge, DriveBench GPT scores).
- Rewards for RL (CLIP or VLM reward shaping, DriveReward, OpenREAD's LLM critic).

Hazard and corner-case perception is also judged.

| Paper | arXiv | Date | Venue | Finding |
|---|---|---|---|---|
| LingoQA (Lingo-Judge) | 2312.14115 | 2023-12 | ECCV'24 | A learned text classifier judge, Lingo-Judge, reaches Spearman ≈ 0.95† with human ratings and beats METEOR, CIDEr and GPT-4 prompting. This is an early "learned judge for driving QA". |
| DriveLM | 2312.14150 | 2023-12 | ECCV'24 Oral | Graph-VQA driving. Its eval mixes language metrics with a GPT score. |
| CODA-LM | 2404.10595 | 2024-04 | WACV'25 | Corner-case LVLM eval. Shows that a **text-only LLM judge** of LVLM answers aligns with humans better than LVLM judges†. |
| DriveBench ("Are VLMs ready for AD?") | 2501.04003 | 2025-01 | ICCV'25† | 12 VLMs, 19,200 frames and 20,498 QA pairs. VLMs give plausible answers from text priors. **GPT scores barely change under corrupted or black images**†. The current judge protocol cannot detect ungrounded answers. |
| DriveCritic | 2510.13108 | 2025-10 | ICRA'26 | NAVSIM **pairwise human-preference** set (5,730 pairs†) where context matters. Fine-tuned VLM gets 76.0% vs GPT-5 60.9% vs EPDMS 46.6%†. |
| DriveJudge | 2606.17362 | 2026-06 | under review | 33,577 human-annotated samples. A VLM selectively calls deterministic rule functions. It beats EPDMS by 21.23 AUC and DriveCritic by 6.5% on preference selection. |
| DriveReward | 2606.08525 | 2026-06 | preprint | Generative VL reward model with a trajectory-evaluation set augmented by counterfactual failures. |
| OpenREAD | 2512.01830 | 2025-12 | preprint | Uses an LLM-as-critic reward for open-ended driving reasoning in reinforcement fine-tuning (RFT). |
| VLM-RL (driving) | 2412.15544 | 2024-12 | preprint | Contrasting-language-goal CLIP reward for safe-driving RL. |
| CLIP-RLDrive | 2412.16201 | 2024-12 | preprint | CLIP reward shaping at unsignalized intersections. |
| Semantic-Drive | 2512.12012 | 2025-12 | preprint | Long-tail data mining with a multi-model "Judge-Scout" VLM consensus to suppress hallucination. A consensus judge used for data curation. |
| SLLCP: localized conformal safety monitoring with VLMs | 2610.02765 | 2026-10 | extended abstract | **The only domain-specific conformal VLM layer found.** Post-hoc, label-localized conformal prediction sets over frozen VLM safety predictions. It is pointwise, not pairwise. |
| SafeGen | 2607.19701 | 2026-07 | preprint | Generates safety-critical scenario videos for testing VLM-based AD. Generated scenarios themselves need a judge. |

**(b) Datasets.**
- DriveCritic pairwise prefs (`song-jingyu.github.io/DriveCritic`).
- DriveJudge, 33.6K (check release).
- LingoQA human-rated answers (`github.com/wayveai/LingoQA`†).
- CODA-LM (`coda-dataset.github.io/coda-lm`).
- DriveBench (`drive-bench.github.io`).
- NAVSIM / Bench2Drive for generating pairs.

**(c) Calibrated or selective judging.** SLLCP (pointwise conformal sets) is the only one. DriveJudge's "selective invoking of rules" is tool selection, not abstention.

**(d) Pain point.** Rule metrics such as EPDMS disagree with humans in context-dependent scenes, and VLM judges are ambiguous. A Δp-gated pairwise judge would:
- auto-label confident trajectory preferences;
- **flag ambiguous scenario pairs for human safety reviewers**, with a bounded error among accepted labels;
- fall back to rule or pointwise metrics below τ, which mirrors DriveJudge's hybrid design but with a guarantee.

---

## 4. Embodied / robotics: success detectors and reward models

**(a) What gets judged.** Task success (binary), progress (dense value), failure type, and pairwise preference between frames or rollouts. The judges are CLIP-style embeddings (RoboCLIP, VLM-RMs, LIV), prompted VLMs (GVL, RL-VLM-F, TOPReward), and fine-tuned VLM reward models (RoboReward, RoboJev, AHA, Guardian, VLAC).

| Paper | arXiv | Date | Venue | Finding |
|---|---|---|---|---|
| VLMs as Success Detectors (SuccessVQA) | 2303.07280 | 2023-03 | preprint | Fine-tuned Flamingo success detection. Struggles under distribution shift†. |
| LIV | 2306.00958 | 2023-06 | ICML'23 | Language-image value representation used as a reward. |
| REFLECT | 2306.15724 | 2023-06 | CoRL'23 | LLM summarizes robot experience to explain failures. |
| KnowNo ("Robots that ask for help") | 2307.01928 | 2023-07 | CoRL'23 Oral | **Conformal prediction to ask humans for help on LLM plans.** The canonical robotics precedent for "abstain and escalate with a guarantee". It covers planning, not reward judging. |
| RoboCLIP | 2310.07899 | 2023-10 | NeurIPS'23† | Video-language similarity to one demonstration as the reward. |
| VLM-RMs | 2310.12921 | 2023-10 | ICLR'24 | CLIP as a zero-shot reward. Works for simple humanoid poses and scales with model size, but is brittle. |
| RL-VLM-F | 2402.03681 | 2024-02 | ICML'24 | **The VLM gives pairwise preferences over two observation images, and a reward is learned from them.** This is the direct pairwise-VLM-judge-for-RL precedent. It does no order debiasing and no abstention (it uses a "no preference" label†). |
| Video-Language Critic | 2405.19988 | 2024-05 | preprint | Transferable contrastive video-language reward. |
| AHA | 2410.00371 | 2024-10 | ICLR'25† | Failure-detection VLM trained on synthetic failures. Beats GPT-4o by about 10%†. |
| GVL (VLMs are in-context value learners) | 2411.04549 | 2024-11 | ICLR'25† | Gemini predicts per-frame progress on shuffled frames. Value-order correlation is measured over 300+ tasks†. Used for data filtering and success detection. |
| ERL-VLM (offline RL from VLM feedback) | 2411.05273 | 2024-11 | LangRob@CoRL'24; IEEE venue† | Uses VLM ratings instead of preferences for offline RL. Ratings are argued to be more sample-efficient. |
| Code-as-Monitor | 2412.04455 | 2024-12 | CVPR'25 | VLM-generated constraint code for reactive and proactive failure detection. |
| ReWiND | 2505.10911 | 2025-05 | CoRL'25 Oral | Language-guided reward from a small number of demos, with video rewinding for failures. |
| VLAC | 2509.15937 | 2025-09 | preprint | VLA-plus-critic that gives pairwise progress deltas and done signals for real-world RL. |
| FailSafe | 2510.01642 | 2025-10 | preprint | Failure reasoning and recovery for VLAs. |
| Guardian | 2512.01946 | 2025-12 | preprint | GuardianFail-36k synthetic failures. Multi-view VLM verifies planning and execution. |
| **RoboReward** | 2601.00675 | 2026-01 | preprint | Reward dataset and benchmark from OXE and RoboArena with counterfactual negatives. **No VLM excels across tasks.** The 4B/8B RMs beat larger VLMs. Real-robot RL improves over Gemini Robotics-ER 1.5 rewards. |
| **TOPReward** | 2602.19313 | 2026-02 | preprint | Training-free reward that uses the **VLM's token probability that the task is complete**. ManiRewardBench covers 130 tasks on 4 robots. **Same readout family as EAGLE's p̃.** |
| Robot Critics that Sweat the Small Stuff | 2606.21572 | 2026-06 | preprint | Fine-tunes critics on **pairwise progress supervision** from success and failure rollouts. Picks candidate actions through a video model and raises success by 11%. |
| **FailBench** | 2609.03611 | 2026-09 | preprint | 2,197 attempts from 14 sources. The best of 13 detectors reaches **0.77 balanced accuracy**. Near chance (<0.60) on contact-rich assembly. Detectors fine-tuned for failure detection *underperform* general VLMs. **Systematic bias towards "success" under ambiguous evidence.** |
| **ROBORMBENCH** | 2609.05401 | 2026-09 | preprint | 2,390 real trajectories and 21,673 paraphrases. **Paraphrasing the instruction flips success and failure.** Neither scale nor reasoning fixes this, and trained RMs are more stable. |
| ARS: Agentic Reward System | 2609.34484 | 2026-09 | preprint | Agentic reward pipeline for robot learning (abstract not read). |
| Robo-Dopamine 2.0 | 2608.15680 | 2026-08 | preprint | History-conditioned, **OOD-aware** process reward model for manipulation (title-level; abstract not read). |
| **RoboJev** | 2610.33718 | 2026-10 | preprint | Open 27B judge with a **calibrated two-token success/failure readout**. 95.2% balanced accuracy and 0.993 AUROC in-domain, AUROC 0.890 on unseen labs, about $0.19 per 1K predictions. 20,628 predictions released. |
| Perturbation-based epistemic uncertainty for VLA failure detection | 2606.20754 | 2026-06 | preprint | Uncertainty of the *policy*, not the judge (title-level). |

**How often are they wrong?**
- Best general detector: about 23% balanced error on FailBench, and about 40% on assembly.
- Even in-domain-trained RoboJev drops from 0.993 to 0.890 AUROC on unseen labs.
- ROBORMBENCH shows instruction-phrasing instability.
- RoboReward finds no model is uniformly good.

These are precisely the high-variance, asymmetric-bias regimes where an abstaining judge pays off. FailBench's "biased towards success" result also means one-sided errors that a symmetric pairwise query, which compares a rollout against a reference success or failure, could partly cancel.

**(b) Datasets.**
- FailBench (2,197 attempts).
- ROBORMBENCH (2,390 trajectories with progress labels and 21,673 paraphrases).
- RoboReward (OXE+RoboArena; release stated in abstract).
- RoboJev's 20,628 released predictions plus 268/459/135-episode test splits.
- ManiRewardBench (130 tasks).
- GuardianFail-36k.
- AHA dataset†.
- RoboArena pairwise human policy comparisons (arena-style A/B evaluations†): a natural pairwise human benchmark.

**(c) Calibrated or selective judging.** KnowNo (conformal, planning only). RoboJev (calibrated readout, no abstention guarantee). Robo-Dopamine 2.0 (OOD-aware). **No conformal or selective *reward* or *success* judge with a risk bound.**

**(d) Pain point.** Human reward labels are the bottleneck in real-robot RL, and wrong VLM rewards cause reward hacking. EAGLE-V can:
- give **cheap labels on confident pairs**, either rollout vs. rollout or rollout vs. reference;
- **escalate ambiguous contact-rich episodes**;
- provide a BT/Davidson progress ranking across rollouts.

Order-swapping and paraphrase-swapping can be folded into the same consistency signal.

---

## 5. GUI / web agents

**(a) What gets judged.** Whole trajectories (outcome success), steps (process reward), and pairs of rollouts (best-of-N). The judges are GPT-4o or Claude-class MLLMs over screenshots and action histories, trained critic and reward models, and multi-agent critics.

| Paper | arXiv | Date | Venue | Finding |
|---|---|---|---|---|
| Autonomous Evaluation and Refinement of Digital Agents | 2404.06474 | 2024-04 | COLM'24 | VLM/LLM evaluators agree with oracle success 74–92%† and enable Reflexion and filtered behaviour cloning (BC). |
| Agent-as-a-Judge | 2410.10934 | 2024-10 | ICML'25† | Agentic judge on DevAI (55 tasks, 365 requirements†). Close to human-panel agreement, at much lower cost. |
| OS-Genesis | 2412.19723 | 2024-12 | ACL'25 | Reverse task synthesis plus a **trajectory reward model** to filter synthetic GUI data. |
| A3: Android Agent Arena | 2501.01149 | 2025-01 | preprint | Essential-state procedural evaluation with LLM-based evaluation†. |
| Online-Mind2Web / WebJudge ("An Illusion of Progress?") | 2504.01382 | 2025-04 | COLM'25† | 300 tasks on 136 live sites†. The WebJudge LLM judge reaches about 85% agreement with humans†. Earlier benchmarks overstate agent success. |
| AgentRewardBench | 2504.08942 | 2025-04 | preprint (COLM'25†) | 1,302 expert-annotated trajectories across 5 benchmarks†. **No single LLM judge is best. Rule-based checks under-report success, and LLM judges over-report it (low precision)**†. |
| UI-Genie | 2505.21496 | 2025-05 | preprint | Reward model UI-Genie-RM for self-improving mobile agents. |
| GUI-Critic-R1 | 2506.04614 | 2025-06 | preprint | Pre-operative critic (judges the step before acting). |
| Mind2Web 2 | 2506.21506 | 2025-06 | preprint (NeurIPS'25 D&B†) | Rubric-tree **Agent-as-a-Judge** for agentic search. |
| Agent-RewardBench | 2506.21252 | 2025-06 | ACL'25 | Step-level reward eval across perception, planning and safety in 7 scenarios. |
| GUI-Shepherd | 2509.23738 | 2025-09 | preprint | PRM trained on 52k human-scored interactions. +7.7 points on AndroidWorld with PPO. |
| BJudge ("Scaling Agents for Computer Use") | 2510.02250 | 2025-10 | preprint | **Pairwise comparison of rollouts as behaviour narratives.** OSWorld state of the art at 72.6%. Manual audit shows judge accuracy about 92.8% on disputed cases†. |
| MagicGUI-RMS | 2601.13060 | 2026-01 | preprint | Multi-agent reward-model system. |
| OS-Themis + OGRBench | 2603.19191 | 2026-03 | preprint | Milestone decomposition plus a reviewer audit before the verdict. Cross-platform GUI outcome-reward benchmark. +10.3% on AndroidWorld RL. |
| HalluClear | 2604.17284 | 2026-04 | preprint | GUI-agent hallucination diagnosis (uses a VLM judge; title/snippet-level). |
| VisCritic | 2606.24525 | 2026-06 | preprint | Visual pre/post screenshot comparison as process reward. |

**(b) Datasets.**
- AgentRewardBench (`agent-reward-bench.github.io`†).
- Online-Mind2Web human labels (OSU-NLP HF†).
- OGRBench.
- Mind2Web 2 rubrics.
- Agent-RewardBench.
- GUI-Shepherd's 52k human scores (check release).
- OSWorld-Verified.

**(c) Calibrated or selective judging.** None with a guarantee. OS-Themis "verdict calibration" is a review process. Searches for selective prediction or abstention combined with GUI judges returned nothing.

**(d) Pain point.** Best-of-N and RL for computer-use agents need trustworthy pairwise rollout judgments, and BJudge is already pairwise. LLM judges over-report success (AgentRewardBench). An EAGLE-style judge would:
- abstain on low-Δp pairs and fall back to pointwise or rule checkers;
- report precision at a given coverage;
- bound the false-success rate that poisons RL rewards.

---

## 6. Video, image and 3D generation evaluators

**(a) What gets judged.**
- T2I, T2V and 3D outputs, judged by learned preference scorers (ImageReward, PickScore, HPSv2/v3, VisionReward, VideoScore 1/2, VideoAlign), VQA-likelihood scorers (VQAScore, which reads P("Yes")), and generative MLLM reward models (UnifiedReward(-Think), LLaVA-Critic, VR-Thinker).
- Long videos, judged by MLLM judges evaluated by meta-benchmarks.

| Paper | arXiv | Date | Venue | Finding |
|---|---|---|---|---|
| ImageReward | 2304.05977 | 2023-04 | NeurIPS'23† | 137K expert pairwise comparisons†. First general T2I human-preference reward model. |
| Pick-a-Pic / PickScore | 2305.01569 | 2023-05 | NeurIPS'23† | Open user-preference pairs (about 500K–1M†). |
| GPTEval3D | 2401.04092 | 2024-01 | CVPR'24† | GPT-4V pairwise comparisons of text-to-3D outputs, aggregated with Elo, align with humans. **Pairwise VLM judge plus Elo for 3D, with no confidence gating.** |
| MLLM-as-a-Judge | 2402.04788 | 2024-02 | ICML'24 Oral | MLLMs are decent at pairwise comparison and poor at scoring and batch ranking. They show position and egocentric biases. |
| VQAScore | 2404.01291 | 2024-04 | ECCV'24† | P("Yes") from a VQA model as an alignment score. **A first-token-probability judge.** |
| T2IScoreScore (TS2) | 2404.04251 | 2024-04 | preprint | Meta-evaluates prompt-coherence metrics. |
| VideoScore / VideoFeedback | 2406.15252 | 2024-06 | EMNLP'24† | 37.6K† videos with multi-aspect human scores. A learned video metric. |
| GenAI-Bench | 2406.13743 | 2024-06 | preprint | 1,600 compositional prompts with large-scale human ratings†. |
| MJ-Bench | 2407.04842 | 2024-07 | preprint | Multimodal judges for T2I are inconsistent across alignment, safety and quality. Closed VLMs are best overall†. |
| LLaVA-Critic | 2410.02712 | 2024-10 | CVPR'25 | Open generalist multimodal judge. |
| VL-RewardBench | 2411.17451 | 2024-11 | CVPR'25 | 1,250† hard examples. GPT-4o reaches about 65%†. |
| VisionReward | 2412.21059 | 2024-12 | preprint (AAAI'26†) | Multi-dimensional fine-grained preference for images and video. |
| VideoAlign / VideoReward | 2501.13918 | 2025-01 | preprint (NeurIPS'25†) | 182K† annotated video pairs. Uses Bradley–Terry with ties†. **Directly relevant to EAGLE's Davidson tie modelling.** |
| Multimodal RewardBench | 2502.14191 | 2025-02 | preprint | 5,211† triplets across 6 domains. |
| UnifiedReward | 2503.05236 | 2025-03 | preprint | One reward model for understanding and generation, both pairwise and pointwise. |
| 3DGen-Bench | 2503.21745 | 2025-03 | ICCV'25† | Human-preference arena for 3D generation and a learned 3D scorer. |
| Eval3D | 2504.18509 | 2025-04 | CVPR'25 | Interpretable 3D eval with VLM and foundation-model probes. |
| UnifiedReward-Think | 2505.03318 | 2025-05 | NeurIPS'25 | Chain-of-thought multimodal reward model via RL fine-tuning. |
| HPSv3 / HPDv3 | 2508.03789 | 2025-08 | ICCV'25 | 1.08M text-image pairs and 1.17M annotated comparisons†. Uses an uncertainty-aware ranking loss†. |
| VideoScore2 | 2509.22799 | 2025-09 | preprint | "Think before you score". Multi-dimension video eval (VideoFeedback2, about 27K videos†). |
| VR-Thinker | 2510.10518 | 2025-10 | preprint | Video RM that selects frames and keeps a visual memory, for long contexts. |
| SLVMEval | 2603.29186 | 2026-03 | CVPR'26 | Meta-eval for long T2V (up to about 3 h). Pairwise degraded-vs-original pairs filtered by crowdsourcing. Humans reach 84.7–96.8%, while automatic evaluators (VLM judges, CLIPScore) trail humans on 9 of 10 aspects†. |
| VLM Judges Can Rank but Cannot Score | 2604.25235 | 2026-04 | preprint (in notes) | Conformal intervals from score-token log-probs. Interval width covers about 40% of the range on aesthetics and about 70% on charts and math. **Ranking-scoring decoupling.** Strong evidence for EAGLE's pairwise-first design. |
| Informativeness bias (BIRCH) | 2604.17768 | 2026-04 | ACL'26 | VLM judges prefer the more informative answer even when it contradicts the image. |
| Sci-VBench | 2608.09873 | 2026-08 | COLM'26 | 1,253 expert-annotated science T2V items. Rubric MLLM judges agree with experts. Agreement is weaker on low-level fidelity and temporal consistency†. |
| WorldReward | 2609.03952 | 2026-09 | preprint | **VLM pairwise preference reward** for camera-conditioned world models. Local action evidence gets diluted in long contexts. |
| PlaylistEval | 2609.34314 | 2026-09 | preprint | Day-scale (100 h) video judging, 630 pairs. Human agreement is 93.0% (IAA 0.781). **Frontier judges reach 75.4%, and accuracy falls as the collection grows.** |

**(b) Datasets.** ImageReward, Pick-a-Pic, HPDv3, VideoFeedback(2), VideoGen-RewardBench (from VideoAlign), GenAI-Bench, MJ-Bench, VL-RewardBench, Multimodal RewardBench, 3DGen-Bench, SLVMEval, PlaylistEval and Sci-VBench. All are pairwise or rating-based and mostly on HF/GitHub (see table below).

**(c) Calibrated or selective judging.**
- 2604.25235: conformal intervals for pointwise scores.
- HPSv3: uncertainty-aware loss in training, no abstention.
- 2505.21478: an ensemble-variance gate that zeroes the reward when uncertain, in T2I pipeline design.
- **No selective pairwise generation judge with a guarantee.**

**(d) Pain point.** Reward hacking in diffusion RL and unreliable long-video judging. EAGLE-V can:
- supply risk-bounded pairwise preferences for DPO/GRPO data, keeping confident pairs only;
- report leaderboard rankings via BT/Davidson with explicit ties;
- abstain on long-video pairs where evidence retrieval is uncertain.

---

## Dataset table: human-label sets usable as EAGLE-V benchmarks

| Dataset | Domain | Label type | Size | Access |
|---|---|---|---|---|
| Place Pulse 2.0 | Urban perception | Pairwise votes, 6 dims | ~1.17M votes / ~110K images† | via MIT Media Lab / Dubey et al. 1608.01769 |
| SPECS | Urban perception | Demographic-balanced ratings | 1,000 raters (in notes) | 2505.12758 |
| Montreal urban perception (2509.14574) | Urban perception | Ratings, real + synthetic | 100 images | arXiv HTML / authors |
| Yo'City human study | 3D city generation | Pairwise | small† | 2511.18734 |
| MapEval | Map QA | MCQ + open answers | ~700 questions† | github.com/MapEval |
| RS preference data (2503.00743) | Remote sensing | Preference pairs | large† | NeurIPS'25 paper repo |
| RSHBench | Remote sensing | Hallucination labels (human-validated judge) | 100 human-checked† | 2603.02754 |
| DriveCritic | Driving | **Pairwise human prefs** | 5,730 pairs† | song-jingyu.github.io/DriveCritic |
| DriveJudge | Driving | Binary "reasonable" + preference | 33,577 | 2606.17362 (check release) |
| LingoQA | Driving | Human-rated QA answers | ~28K videos / 419K QA† | github.com/wayveai/LingoQA |
| CODA-LM | Driving | Corner-case QA + judge | ~10K images† | coda-dataset.github.io/coda-lm |
| FailBench | Robotics | Success/failure | 2,197 attempts | 2609.03611 |
| ROBORMBENCH | Robotics | Progress + paraphrases | 2,390 traj / 21,673 paraphrases | 2609.05401 |
| RoboReward | Robotics | Reward / progress | large (OXE + RoboArena) | 2601.00675 (released) |
| RoboJev eval release | Robotics | Success predictions + labels | 20,628 predictions | 2610.33718 |
| ManiRewardBench | Robotics | Progress | 130 tasks, 4 robots | 2602.19313 |
| AgentRewardBench | Web agents | Expert success labels | 1,302 trajectories† | 2504.08942 |
| Online-Mind2Web | Web agents | Human success labels | 300 tasks / 136 sites† | 2504.01382 |
| OGRBench | GUI | Outcome reward labels | — | 2603.19191 |
| ImageReward | T2I | Pairwise | 137K† | github.com/THUDM/ImageReward |
| Pick-a-Pic v2 | T2I | Pairwise | ~1M† | HF yuvalkirstain/pickapic_v2 |
| HPDv3 | T2I | Pairwise | 1.17M comparisons† | github.com/MizzenAI/HPSv3 |
| GenAI-Bench | T2I/T2V | Ratings | 1,600 prompts† | linzhiqiu.github.io/papers/genai_bench |
| VideoFeedback / VideoFeedback2 | T2V | Multi-aspect ratings | 37.6K / ~27K† | tiger-ai-lab.github.io/VideoScore |
| VideoGen-RewardBench (VideoAlign) | T2V | Pairwise with ties | 182K training pairs† | github.com/KwaiVGI/VideoAlign |
| SLVMEval | Long T2V | Pairwise (crowd-filtered) | — | 2603.29186 |
| PlaylistEval | Long-video QA judging | Pairwise | 630 pairs (152 human-checked) | playlisteval.github.io |
| MLLM-as-a-Judge / VL-RewardBench / Multimodal RewardBench | General | Pairwise + scores | ~4.4K† / 1,250† / 5,211† | respective GitHub |

---

## Ranked opportunities for EAGLE-V

1. **Selective success and reward judge for robot learning, with a risk bound.** This is the highest impact and the clearest gap.
   - *Evidence:* FailBench (0.77 best, biased towards success), ROBORMBENCH (paraphrase flips), RoboReward ("no model excels") and TOPReward (token-probability reward). KnowNo shows the robotics community values conformal ask-for-help.
   - *What to build:* query rollout A vs. B, or rollout vs. reference, in both orders. Optionally add paraphrase swaps to Δp. Abstain below τ and send those episodes to humans. Rank rollouts with BT.
   - *Metrics:* reward-label cost saved at a fixed false-success rate, and downstream RL success.
   - *Benchmarks:* FailBench, ROBORMBENCH, RoboJev release.

2. **Driving planner evaluator that flags uncertain scenario pairs.**
   - DriveCritic and DriveJudge already provide pairwise human preferences. Rule metrics (EPDMS) and VLMs disagree in context-dependent scenes.
   - The only conformal work, SLLCP, is a pointwise extended abstract.
   - EAGLE-V on DriveCritic pairs would give accuracy vs. coverage curves and fall back to EPDMS below τ, a principled version of DriveJudge's hybrid. The safety framing ("escalate to human review") is compelling for reviewers.

3. **PP2 and urban generative-model auto-evaluation (the lab's home turf).**
   - Combine order-symmetrized Δp gating, a Davidson tie model and active BT on PP2. Compare against UrbanAlign (72.2%) and GPT-4o + Thurstone (ρ=0.76†).
   - Then reuse the judge to evaluate generated cities (Yo'City and MajutsuCity-style) and synthetic street views. Report abstention and coverage alongside the cultural-bias analysis (2604.20048): abstention should rise on out-of-distribution cities.
   - This is still open per the prior-art notes. The novelty is the *combination plus the guarantee*.

4. **GUI/web trajectory judge with an escalation budget.**
   - BJudge shows that pairwise rollout comparison is SOTA for best-of-N. AgentRewardBench shows LLM judges have low precision.
   - EAGLE gives precision at coverage and abstains on low-Δp pairs, falling back to rule checkers.
   - Benchmarks: AgentRewardBench, Online-Mind2Web, OGRBench.

5. **Generation-reward data filtering (T2I/T2V/3D/world models).**
   - Use Δp to keep only confident preference pairs for DPO/GRPO, and Davidson ties for near-equal pairs (VideoAlign already uses ties).
   - Test on HPDv3, VideoGen-RewardBench and WorldReward-style pairs, and on long-video pairs from PlaylistEval and SLVMEval where judges degrade.
   - Crowded field; position as "risk-controlled preference data".

6. **Remote-sensing caption and VQA judge standardization.**
   - Judge κ varies from 0.5 to 0.9 across closed models (RSHBench), and no RS pairwise preference set exists.
   - EAGLE-V could release a small RS pairwise caption-preference set and a selective judge. It is lower impact but nearly empty territory, and a good cheap side contribution.

7. **Mobility trajectory realism judge (text or rendered-map modality).** Still open (see prior-art notes), but the evidence that LLMs reward plausibility over realism (2606.13835) means abstention and calibration must be shown first on real vs. synthetic pairs with ground truth.

**Positioning caveats.**
- SCOPE (2602.13110) already does both-order querying with a bidirectional entropy uncertainty and conformal abstention for text pairwise judges. EAGLE-V's differentiation must be:
  - vision and domain transfer;
  - the pointwise fallback rather than pure abstention;
  - BT/Davidson aggregation with ties;
  - an evaluation of downstream utility (RL reward, planner ranking, city-model ranking).
- TOPReward and RoboJev occupy the "token-probability readout" niche in robotics, so cite them as the pointwise analogues.

---

## Sources (key, beyond arXiv IDs above)
- DriveCritic project: https://song-jingyu.github.io/DriveCritic
- DriveJudge: https://arxiv.org/abs/2606.17362
- UrbanAlign: https://arxiv.org/html/2602.19442v1
- GPT-4o PP2 attractiveness (Buildings 2025): https://mdpi.com/2075-5309/15/16/2970
- PlaylistEval: https://playlisteval.github.io
- SenseBench: https://arxiv.org/pdf/2605.10576 ; RSHBench: https://arxiv.org/pdf/2603.02754
- BJudge: https://arxiv.org/abs/2510.02250 ; OS-Themis: https://arxiv.org/abs/2603.19191
- Urban planning LLM-judge leads (unverified, non-arXiv): Preprints.org 202508.0418; "Planning, Living and Judging" (Ni, Wang, Liu); JPER 2025 plan-evaluation article.
