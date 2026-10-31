# Prior-art check: urban/mobility judges (2026-10-30)

All arXiv IDs below were opened on their abs page (or via WebFetch) in this session. I did not do a full-text read of every paper. Verdicts are based on abstracts, plus the full HTML for UrbanAlign.

## 1. LLM-as-judge for synthetic mobility trajectories: mostly OPEN
- TrajGenAgent, arXiv 2606.12657 (Jun 2026). Hierarchical LLM-agent generator. It names the same gap: aggregate metrics miss how realistic individual trajectories are. Its fix is two trained anomaly detectors (ICAD, BeSTAD), not an LLM judge. **Closest motivation, and it uses a different evaluator.**
- When Plausible Is Not Realistic, arXiv 2606.13835 (Jun 2026). Finds that LLM simulators (AgentSociety, CitySim) produce plausible-sounding mobility that is not realistic. Uses statistical and mobility-law validation, no LLM judge. Strong motivation for "LLM judges may reward plausibility".
- LLM-as-a-Discriminator: When Synthetic Tables Still Look Real, arXiv 2606.09865 (Jun 2026). An LLM labels each tabular row REAL or SYNTHETIC. **Same protocol, different modality (tables, not trajectories).**
- Generators with no judge or reranking: LLMob 2402.14744 (NeurIPS'24), ELLMob 2603.07946 (ICLR'26), COMPASS 2602.16726, EncounterMob 2610.17672 (Oct 2026; Japan/NYC data).
- Surveys with no judge direction: 2603.12420 and 2609.21413.
- I found no LLM real-vs-synthetic judge for individual trajectories, no judge-guided best-of-n, and no work on HuMob/YJMob100K with a judge.

## 2. Position bias and judging protocols for LLM next-POI: PARTIAL
- Hou et al., Large Language Models are Zero-Shot Rankers for Recommender Systems, arXiv 2305.08845 (ECIR 2024). Confirmed: it reports position and popularity bias and mitigates them with prompting and bootstrapping over candidate orders.
- LLMMove ("Where to Move Next"), arXiv 2404.01855. Reports that POI results are sensitive to candidate order and uses shuffle-ensembling. **This directly covers the POI order-sensitivity finding.**
- Bito, Ren & He have a series of papers: 2508.02020 (RISE), 2604.27599 (InvariRank, SIGIR'26), 2608.03091 (pairwise preference instability under permutation). These are general recommendation papers, not POI.
- LLM4POI, arXiv 2404.17591 (SIGIR'24), and 2609.09609 (100-candidate ranking with order randomized) do not study the order effect itself.
- **Open:** a systematic comparison of pointwise vs pairwise vs listwise judging on POI/mobility data, and the generation-vs-discrimination gap. Neither appeared in what I found.

## 3. VLM judge on Place Pulse with Δp gating, active Bradley–Terry, and ties: PARTIAL
- UrbanAlign, arXiv 2602.19442 (PP2.0 pairwise). Uses TrueSkill, filters pairs by human consensus, and predicts "equal" when the score gap is below ε. Zero-shot GPT-4o gets 56.7% with forced choice and no tie option; their method gets 72.2%. **This covers tie handling.**
- SCOPE, arXiv 2602.13110 (ICML'26). Queries the pairwise LLM judge in both orders, measures uncertainty with BPE, and abstains with conformal guarantees. **The Δp gating idea in general form, not urban.**
- Active Bradley–Terry with LLM judges: 2607.02104 (top-k, debiasing BT), 2405.05894 (Product of Experts, about 2% of comparisons), 2610.29416 (active Bayesian BT).
- Urban benchmarks: 2509.14574 and 2606.00871 (reliability-aware, treat abstention as an outcome; Montreal, rating-based, not PP2). Others: Buildings 15(16):2970 (GPT-4o attractiveness, Chongqing), 2407.19719 (MLLM safety + CLIP kNN).
- **Open:** combining PP2 with VLM Δp gating, an active BT budget and a tie model in one study.

## 4. Look-then-match: LARGELY DONE
- UrbanAlign (2602.19442) already mines concepts, has a VLM score each image on them (e.g. Façade Quality, Vegetation Maintenance) and fits a calibrated ridge regression. It beats direct VLM pairwise judgment by 15.5 pp.
- Also related: Lan, arXiv 2506.05087; Tang & Law, arXiv 2604.22103 (levers such as maintenance and mobility infrastructure); Dubey et al., arXiv 1608.01769 (ECCV'16).
- Generic background: VQAScore 2404.01291, Compare2Score 2405.19298, Co-Instruct 2402.16641, and "VLM Judges Can Rank but Cannot Score" 2604.25235.
- **Remaining angle:** a cheap fixed attribute schema with a tiny model as an ablation or baseline against UrbanAlign. It would not be novel as a main claim.
