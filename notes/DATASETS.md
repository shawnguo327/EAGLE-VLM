# Judge-bench items manifest

Unified pairwise items (`items/<dataset>.jsonl`, schema at the top of `judge_bench.py`). Image paths are relative to this folder's parent (the workspace root).
A/B placement is a seeded (0), subset-balanced shuffle; `label` 0 = cand[0] preferred, 1 = cand[1], "tie" = explicit human tie.
Caps: at most 1,500 items per dataset, water-filled across subsets (small subsets kept whole), random sample seed 0.
Regenerate: `.venv/bin/python -I convert/<dataset>.py` for each, then `convert/manifest.py` and `convert/sanity.py`.

| dataset | kind | subsets (written) | items | full size | label 0 / 1 | ties | source | license |
|---|---|---|---|---|---|---|---|---|
| vlrb | resp | general (181), hallucination (749), reasoning (317) | 1247 | 1247 | 625 / 622 (50% A) | 0 | MMInstruction/VL-RewardBench | mit — Research use only. Usage is restricted by the license agreements of GPT-4o and Claude. |
| mmrb | resp | correctness (214), knowledge (214), preference (215), reasoning-coding (214), reasoning-math (214), safety (214), vqa (215) | 1500 | 4711 | 751 / 749 (50% A) | 0 | syhuggingface/multimodal_rewardbench (mirror of facebookresearch/multimodal_rewardbench) | cc-by-nc-4.0 — The dataset is released under the CC-by-NC license. |
| mmrlhf | resp | long (37), mcq (42), safety (32), short (34), video (25) | 170 | 170 | 86 / 84 (51% A) | 0 | yifanzhang114/MM-RLHF-RewardBench | mit |
| genai_arena | img | edition (750), generation (750) | 1500 | 4175 | 482 / 481 (50% A) | 537 | TIGER-Lab/GenAI-Arena-Bench | cc-by-4.0 |
| mjbench | img | alignment (729), safety (672) | 1401 | 1401 | 701 / 700 (50% A) | 0 | MJ-Bench/MJ-Bench | apache-2.0 |
| editreward | img | 2pair (133) | 133 | 133 | 58 / 57 (50% A) | 18 | TIGER-Lab/EditReward-Bench | apache-2.0 |
| mmrb2 | img | edit (750), t2i (750) | 1500 | 2000 | 750 / 750 (50% A) | 0 | rl-research/multimodal-rewardbench-2 | (none in README) — LICENSE.md: This work is licensed under CC BY-NC 4.0 |
| qbench2 | img | clarity (182), color (61), composition (16), exposure (20), lighting (40), noise (13), realism (84), texture (26) | 442 | 1000 | 222 / 220 (50% A) | 0 | q-future/Q-Bench2-HF (dev split; only split with answers) | (none in README) |
| mllm_judge | resp | AesBench (107), ChartQA (107), Concept Caption (107), ScienceQA (107), VisitBench (107), WIT (107), coco (108), diffusiondb (107), infographicsVQA (107), llava_bench (107), mathvista (108), mind2web (107), mm-vet (107), textVQA (107) | 1500 | 7756 | 598 / 589 (50% A) | 313 | ONE-Lab/MLLM-as-a-Judge (Benchmark/pair.jsonl) | mit |
| imagerewarddb | img | Abstract (71), Animals (183), Artifacts (183), Arts (184), Food (86), Illustrations (85), Indoor Scenes (149), Outdoor Scenes (183), People (184), Plants (17), Vehicles (137), World Knowledge (38) | 1500 | 6399 pairs | 754 / 746 (50% A) | 0 | THUDM/ImageRewardDB (test split) | apache-2.0 |
| vlrmbench | – | – | 0 | – | – | – | Winston-Yuan/VLRMBench | apache-2.0 |
| mjbench/bias | – | – | 0 | – | – | – | MJ-Bench/MJ-Bench | apache-2.0 |

**Total items: 10893. Images: 16939 files, 3.3G on disk (`images/`).**

## Per-dataset notes

### vlrb
- Source: `MMInstruction/VL-RewardBench`; license: mit — Research use only. Usage is restricted by the license agreements of GPT-4o and Claude.
- Notes: Subsets from id prefix (official eval): reasoning=mathverse/mmmu, hallucination=hallucination_pair/RLAIF-V/RLHF-V, general=rest. human_ranking lower=better. 9 ids repeat with different pairs; ids get row prefix. Under cap: all kept.
- Full sizes: `{"hallucination": 749, "reasoning": 317, "general": 181, "total": 1247}`

### mmrb
- Source: `syhuggingface/multimodal_rewardbench (mirror of facebookresearch/multimodal_rewardbench)`; license: cc-by-nc-4.0 — The dataset is released under the CC-by-NC license.
- Notes: Subsets: correctness/preference (open_generation), knowledge, reasoning-math (mathvista), reasoning-coding (EMMA-Coding + image2struct), safety (=safety/bias only; Hateful-Memes toxicity part is not in the mirror), vqa. Stratified to ~214/subset.
- Full sizes: `{"vqa": 1200, "preference": 654, "knowledge": 630, "correctness": 623, "reasoning-coding": 582, "reasoning-math": 514, "safety": 508, "total": 4711}`

### mmrlhf
- Source: `yifanzhang114/MM-RLHF-RewardBench`; license: mit
- Notes: Subsets from image dir (mcq/long/short/safety) + video (25 items; frames sorted, 8 uniformly sampled as context). Ids repeat per prompt; row prefix added.
- Full sizes: `{"mcq": 42, "long": 37, "short": 34, "safety": 32, "video": 25, "total": 170}`

### genai_arena
- Source: `TIGER-Lab/GenAI-Arena-Bench`; license: cc-by-4.0
- Notes: image_generation + image_edition (video split not downloaded). tievote (A=B=good) and bothbad_vote (A=B=bad) both -> 'tie' (meta.vote_type). Single vote per item. HF parquet has 3192/983 rows vs 1735/919 in the README text. Stratified 750/750.
- Full sizes: `{"generation": 3192, "generation_votes": {"rightvote": 1219, "leftvote": 1156, "bothbad_vote": 501, "tievote": 316}, "edition": 983, "edition_votes": {"bothbad_vote": 404, "rightvote": 287, "leftvote": 268, "tievote": 24}, "total": 4175}`

### mjbench
- Source: `MJ-Bench/MJ-Bench`; license: apache-2.0
- Notes: alignment + safety. label = index of preferred image (checked visually). bias SKIPPED: image0==image1 byte-identical, rows are demographic variants of an occupation prompt, no gold preference. safety contains NSFW imagery.
- Full sizes: `{"alignment": 729, "safety": 672, "total": 1401, "bias(skipped)": 1548}`

### editreward
- Source: `TIGER-Lab/EditReward-Bench`; license: apache-2.0
- Notes: 2pair config only (3pair/4pair not downloaded). Explicit A=B rankings kept as 'tie'. Per-candidate human scores in meta.
- Full sizes: `{"2pair": 133, "total": 133}`

### mmrb2
- Source: `rl-research/multimodal-rewardbench-2`; license: (none in README) — LICENSE.md: This work is licensed under CC BY-NC 4.0
- Notes: t2i + edit (already pairs; chosen a/b). Edit prompts may have 1-3 input images (all in context; '<image_k>' tags stripped). 2 edit items with multi-image responses skipped. Stratified 750/750. LICENSE.md: CC BY-NC 4.0.
- Full sizes: `{"t2i": 1000, "edit": 1000, "total": 2000, "skipped_multi_image_response": 2}`

### qbench2
- Source: `q-future/Q-Bench2-HF (dev split; only split with answers)`; license: (none in README)
- Notes: Converted only MCQs whose gold answer makes one image better/worse on an attribute with a clear quality direction (yes/no with gold 'Yes'; which-image; compare-second-to-first non-'same'). Skipped: both-images (187), region/distortion-type (210), gold 'No' (114), 'about the same' (~37), brightness/strength/size or unparsed (~10). Subset = attribute. question is a generic prompt; original Q/options/answer in meta.
- Full sizes: `{"total_dev": 1000, "convertible": 442, "skipped": {"both-images question": 187, "same / no quality direction": 37, "region/distortion-type question": 210, "yes/no with gold No": 114, "no quality direction": 6, "unparsed yes/no": 4}}`

### mllm_judge
- Source: `ONE-Lab/MLLM-as-a-Judge (Benchmark/pair.jsonl)`; license: mit
- Notes: Pair split. Human 'C' = explicit tie -> 'tie'. Subset = original_dataset (14). Hard & HQ/ not converted: their (id,pair_id) keys don't align with Benchmark/ (0/116, 19/133 match). Stratified ~107/subset.
- Full sizes: `{"coco": 617, "Concept Caption": 597, "diffusiondb": 300, "infographicsVQA": 571, "mathvista": 1183, "textVQA": 491, "WIT": 445, "ChartQA": 500, "VisitBench": 495, "llava_bench": 497, "mind2web": 592, "ScienceQA": 575, "AesBench": 555, "mm-vet": 338, "total": 7756, "skipped": {}}`

### imagerewarddb
- Source: `THUDM/ImageRewardDB (test split)`; license: apache-2.0
- Notes: All within-prompt pairs with distinct expert rank (rank 1 = best); equal-rank pairs skipped (no explicit tie). Subset = prompt classification, stratified. webp re-encoded to JPEG q95.
- Full sizes: `{"images": 2720, "prompts": 466, "pairs_distinct_rank": 6399, "pairs_equal_rank_skipped": 831, "by_class": {"People": 2820, "Arts": 1763, "Outdoor Scenes": 640, "Artifacts": 334, "Animals": 259, "Indoor Scenes": 149, "Vehicles": 137, "Food": 86, "Illustrations": 85, "Abstract": 71, "World Knowledge": 38, "Plants": 17}}`

### vlrmbench
- SKIPPED. Images live in Image.zip (4.25 GB, not downloaded). No labelled pairwise form: tasks are step-level; multi_solution.jsonl re-lists original vs perturbed solutions with no label, perturbations for most_confidence/redundant_det are stylistic, and the original solution isn't guaranteed correct. See convert/vlrmbench.py.

### mjbench/bias
- SKIPPED subset (1548 rows): no gold preference (see mjbench).

