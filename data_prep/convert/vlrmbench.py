"""VLRMBench (Winston-Yuan/VLRMBench) -- SKIPPED, writes nothing.
Why:
 1. Images are not available: benchmark_data/*.jsonl reference paths like 'dynamath/...jpg' that live in Image.zip
    (4.25 GB), which was intentionally not downloaded.
 2. No labelled pairwise form. The tasks are step-level (step_correctness, redundant_det, most_confidence, ...).
    multi_solution.jsonl (2,986 rows) just re-lists those rows (original step_list + perturbed 'reasoning_error')
    with no preference label. The perturbations for most_confidence / redundant_det are stylistic (over-confident
    wording, redundant steps), not wrong answers, and the original step_list is not guaranteed correct.
Possible later: original vs error-injected solution from step_correctness (995 rows) as a 'resp' pair, once Image.zip
is fetched -- but the 'gold' rests on the unverified assumption that the original solution is the better one."""
import sys, pathlib, json, collections; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import RAW

src = RAW / "Winston-Yuan__VLRMBench/benchmark_data"
for f in sorted(src.glob("*.jsonl")):
    print(f.stem, sum(1 for _ in open(f)))
print("image dir present:", any((RAW / "Winston-Yuan__VLRMBench").glob("Image*")))
print("SKIPPED: see docstring")
