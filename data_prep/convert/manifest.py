"""Build items/MANIFEST.md from items/*.stats.json + the static info below. Run after all converters."""
import sys, pathlib, json, collections, re, subprocess; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, RAW

INFO = {  # dataset -> (source repo, raw dir, kind, notes)
    "vlrb": ("MMInstruction/VL-RewardBench", "MMInstruction__VL-RewardBench", "resp",
             "Subsets from id prefix (official eval): reasoning=mathverse/mmmu, hallucination=hallucination_pair/RLAIF-V/RLHF-V, general=rest. human_ranking lower=better. 9 ids repeat with different pairs; ids get row prefix. Under cap: all kept."),
    "mmrb": ("syhuggingface/multimodal_rewardbench (mirror of facebookresearch/multimodal_rewardbench)", "syhuggingface__multimodal_rewardbench", "resp",
             "Subsets: correctness/preference (open_generation), knowledge, reasoning-math (mathvista), reasoning-coding (EMMA-Coding + image2struct), safety (=safety/bias only; Hateful-Memes toxicity part is not in the mirror), vqa. Stratified to ~214/subset."),
    "mmrlhf": ("yifanzhang114/MM-RLHF-RewardBench", "yifanzhang114__MM-RLHF-RewardBench", "resp",
               "Subsets from image dir (mcq/long/short/safety) + video (25 items; frames sorted, 8 uniformly sampled as context). Ids repeat per prompt; row prefix added."),
    "genai_arena": ("TIGER-Lab/GenAI-Arena-Bench", "TIGER-Lab__GenAI-Arena-Bench", "img",
                    "image_generation + image_edition (video split not downloaded). tievote (A=B=good) and bothbad_vote (A=B=bad) both -> 'tie' (meta.vote_type). Single vote per item. HF parquet has 3192/983 rows vs 1735/919 in the README text. Stratified 750/750."),
    "mjbench": ("MJ-Bench/MJ-Bench", "MJ-Bench__MJ-Bench", "img",
                "alignment + safety. label = index of preferred image (checked visually). bias SKIPPED: image0==image1 byte-identical, rows are demographic variants of an occupation prompt, no gold preference. safety contains NSFW imagery."),
    "editreward": ("TIGER-Lab/EditReward-Bench", "TIGER-Lab__EditReward-Bench", "img",
                   "2pair config only (3pair/4pair not downloaded). Explicit A=B rankings kept as 'tie'. Per-candidate human scores in meta."),
    "mmrb2": ("rl-research/multimodal-rewardbench-2", "rl-research__multimodal-rewardbench-2", "img",
              "t2i + edit (already pairs; chosen a/b). Edit prompts may have 1-3 input images (all in context; '<image_k>' tags stripped). 2 edit items with multi-image responses skipped. Stratified 750/750. LICENSE.md: CC BY-NC 4.0."),
    "qbench2": ("q-future/Q-Bench2-HF (dev split; only split with answers)", "q-future__Q-Bench2-HF", "img",
                "Converted only MCQs whose gold answer makes one image better/worse on an attribute with a clear quality direction (yes/no with gold 'Yes'; which-image; compare-second-to-first non-'same'). Skipped: both-images (187), region/distortion-type (210), gold 'No' (114), 'about the same' (~37), brightness/strength/size or unparsed (~10). Subset = attribute. question is a generic prompt; original Q/options/answer in meta."),
    "mllm_judge": ("ONE-Lab/MLLM-as-a-Judge (Benchmark/pair.jsonl)", "ONE-Lab__MLLM-as-a-Judge", "resp",
                   "Pair split. Human 'C' = explicit tie -> 'tie'. Subset = original_dataset (14). Hard & HQ/ not converted: their (id,pair_id) keys don't align with Benchmark/ (0/116, 19/133 match). Stratified ~107/subset."),
    "imagerewarddb": ("THUDM/ImageRewardDB (test split)", "THUDM__ImageRewardDB", "img",
                      "All within-prompt pairs with distinct expert rank (rank 1 = best); equal-rank pairs skipped (no explicit tie). Subset = prompt classification, stratified. webp re-encoded to JPEG q95."),
}
SKIPPED = [("vlrmbench", "Winston-Yuan/VLRMBench", "Winston-Yuan__VLRMBench",
            "SKIPPED. Images live in Image.zip (4.25 GB, not downloaded). No labelled pairwise form: tasks are step-level; multi_solution.jsonl re-lists original vs perturbed solutions with no label, perturbations for most_confidence/redundant_det are stylistic, and the original solution isn't guaranteed correct. See convert/vlrmbench.py."),
           ("mjbench/bias", "MJ-Bench/MJ-Bench", "MJ-Bench__MJ-Bench", "SKIPPED subset (1548 rows): no gold preference (see mjbench).")]


def license_of(rawdir):
    p = RAW / rawdir / "README.md"
    t = p.read_text(errors="ignore") if p.exists() else ""
    m = re.search(r"^license:\s*(.+)$", t, re.M)
    out = m.group(1).strip() if m else "(none in README)"
    m2 = re.search(r"## (?:📜 )?License\s*\n+(.+)", t)
    if m2:
        out += " — " + m2.group(1).strip()
    lic = RAW / rawdir / "LICENSE.md"
    if lic.exists():
        out += " — LICENSE.md: " + lic.read_text().strip().splitlines()[0].split(". To view")[0]
    return out


rows, total_items = [], 0
lines = ["# Judge-bench items manifest", "",
         "Unified pairwise items (`items/<dataset>.jsonl`, schema at the top of `judge_bench.py`). Image paths are relative to this folder's parent (the workspace root).",
         "A/B placement is a seeded (0), subset-balanced shuffle; `label` 0 = cand[0] preferred, 1 = cand[1], \"tie\" = explicit human tie.",
         "Caps: at most 1,500 items per dataset, water-filled across subsets (small subsets kept whole), random sample seed 0.",
         "Regenerate: `.venv/bin/python -I convert/<dataset>.py` for each, then `convert/manifest.py` and `convert/sanity.py`.", "",
         "| dataset | kind | subsets (written) | items | full size | label 0 / 1 | ties | source | license |",
         "|---|---|---|---|---|---|---|---|---|"]
details = []
for ds, (src, rawdir, kind, notes) in INFO.items():
    sp = ROOT / "items" / f"{ds}.stats.json"
    if not sp.exists():
        continue
    st = json.loads(sp.read_text())
    sub, lab = collections.Counter(), collections.Counter()
    for k, n in st["by_subset_label"].items():
        s, l = k.rsplit("|", 1)
        sub[s] += n; lab[l] += n
    total_items += st["written"]
    fs = st["full_sizes"]
    full = fs.get("total") or fs.get("total_dev") or fs.get("pairs_distinct_rank")
    fulls = f"{full}" + (" pairs" if ds == "imagerewarddb" else "")
    nb = lab["0"] + lab["1"]
    lines.append(f"| {ds} | {kind} | {', '.join(f'{s} ({n})' for s, n in sorted(sub.items()))} | {st['written']} | {fulls} | "
                 f"{lab['0']} / {lab['1']} ({100 * lab['0'] / max(nb, 1):.0f}% A) | {lab['tie']} | {src} | {license_of(rawdir)} |")
    details += [f"### {ds}", f"- Source: `{src}`; license: {license_of(rawdir)}", f"- Notes: {notes}",
                f"- Full sizes: `{json.dumps(fs)}`", ""]
for name, src, rawdir, why in SKIPPED:
    lines.append(f"| {name} | – | – | 0 | – | – | – | {src} | {license_of(rawdir)} |")
    details += [f"### {name}", f"- {why}", ""]
du = subprocess.run(["du", "-sh", str(ROOT / "images")], capture_output=True, text=True).stdout.split()[0]
nimg = sum(1 for p in (ROOT / "images").rglob("*") if p.is_file())
lines += ["", f"**Total items: {total_items}. Images: {nimg} files, {du} on disk (`images/`).**", "", "## Per-dataset notes", ""] + details
(ROOT / "items" / "MANIFEST.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:22]))
