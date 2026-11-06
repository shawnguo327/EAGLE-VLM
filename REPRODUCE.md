# Reproducing the results

The work has two stages. **Stage 1 (GPU)** runs the judges and saves their raw outputs (token log-probabilities per item,
prompt condition and A/B order). **Stage 2 (CPU)** turns the saved outputs into every number in the paper. Stage 2 is fully
config-driven and deterministic; it never runs a model.

## Stage 2: analyses and tables (CPU)

```
bash reproduce.sh            # = the four commands below + the tables
python -m eaglevlm.pipeline.run configs/exp/conventional.yaml     # Table 1
python -m eaglevlm.pipeline.run configs/exp/certify_main.yaml     # Tables 2 (all methods, alpha .1/.2/.3) and 3
python -m eaglevlm.pipeline.run configs/exp/certify_calsize.yaml  # calibration-size ablation
python -m eaglevlm.pipeline.run configs/exp/router_main.yaml      # Table 4
python -m eaglevlm.pipeline.tables configs/tables.yaml            # markdown tables from the summary CSVs only
```

Each run writes `<results>/<experiment>/<timestamp>/` (and a `latest` link):

| file | content |
|---|---|
| `config.yaml` | the fully resolved configuration |
| `splits.csv.gz` | one row per dataset x judge x method x alpha x random split |
| `summary.csv` | aggregated over splits; the tables read only this |
| `manifest.csv` | every input file read, with line count and md5 |
| `env.json` | git commit (and whether the code was modified), package versions, host, command, start / end |

Shared data locations are in `configs/paths.bjserver.yaml` (ordered list of output directories; for duplicate records the
first source wins). Settings live in the experiment YAML: datasets, judges, methods, alphas, delta, split fractions,
number of splits, seed.

| paper table | config | summary columns |
|---|---|---|
| 0 datasets, 1 accuracy / ECE / AUROC / AUPRC / AURC | `conventional.yaml` | `acc_*`, `ece_*`, `err_auroc_*`, `err_auprc_*`, `aurc_*` |
| 2 risk control (SCOPE Table 3, ToE Table 2 columns) | `certify_main.yaml` | `acceptance` (coverage), `risk` (empirical FDR), `human_agreement`, `guarantee_success`, `significant_violation`, `decisive`, `set_size`, `strict_error`, `empty_splits` |
| calibration-size ablation | `certify_calsize.yaml` | same |
| 4 router | `router_main.yaml` | `acceptance`, `informative`, `decisive`, `level`, `level_precision`, `weak`, `risk`, `guarantee_success` |

Equivalence: the pipeline code is a direct port of `scripts/chain.py`, `scripts/c3_diag.py`, `scripts/router_v2.py` and
`scripts/conv_metrics.py`; on a check config (`configs/exp/certify_check.yaml`) all 70 compared cells are identical to the
original scripts' outputs.

Not yet migrated (still run as individual scripts in `scripts/`, results as markdown in `judgebench/paper_tables/`):
mechanism analyses for section 2 (`phenomena.py`, `decompose.py`, `tie_anatomy.py`, `edit_bothbad.py`, `mondrian.py`),
the resolution ablation (`res_ablation.py`), the LoRA evaluation (`lora_eval.py`) and the Best-of-N analysis.

## Stage 1: judge inference (GPU; outputs are saved, do not re-run)

* Items (human labels): converters in `data_prep/convert/` and `scripts/make_*.py` write `items/<dataset>.jsonl`.
* Judge runs: `eaglevlm/judge_bench.py` (prompt conditions: `bin` two-way, `tri` with a tie option, `score` pointwise 1-9,
  `tri_sa1..4` Trust-or-Escalate personas, `tri_p1..3` paraphrases, `*_hires` full resolution, `bin_blind` no image).
  * bjserver: `python pack/ctl.py --config pack/config.bjserver_local.yaml start <phase>`; every phase (datasets x judges x
    conditions) is listed in that YAML and writes `judgebench_local/runs/<phase>/raw/<dataset>__<judge>.jsonl`.
  * rented 4xA100: the runners in `scripts/a100/` (`run_*.sh`, `run_after.sh`) call the same `judge_bench.py`; outputs are
    synced (md5-verified) into `judgebench_local/a100_sync/`.
* Resumable: re-running a judge on an extended item file only judges the new items.
