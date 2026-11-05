# EAGLE-VLM

Risk-controlled, confidence-gated pairwise **vision-language judges**: the EAGLE gate (order-averaged first-token confidence Δp with a pointwise fallback), extended to VLM judges with a joint finite-sample guarantee and prediction-powered rankings.

This repository holds **code only**. Datasets, images, model outputs and logs live in a data directory given by `$EAGLEVLM_DATA` (on bjserver: `/media/shawn/C/papers/VLM/judgebench`).

## Layout

| Path | What |
|---|---|
| `eaglevlm/judge_bench.py` | Unified harness. Both-order pairwise (`bin`) and 1–9 pointwise (`score`) prompts; first-token log-prob readout; HF Transformers; OOM back-off; resumable |
| `eaglevlm/analyze_bench.py` | Per dataset × judge: accuracy, swap consistency, Δp quintiles, AUROC, ECE, pointwise, EAGLE gate (cross-fitted τ), ties |
| `eaglevlm/eaglev_stats.py` | Confidence scores (Δp, log-odds λ, pointwise agreement), Clopper–Pearson fixed-sequence risk control, escalation ladder, PPI Bradley–Terry with routed labelling |
| `eaglevlm/make_paper_tables.py` | Writes `paper_tables/*.tex` and `paper_results.json` from results |
| `data_prep/download.py`, `data_prep/convert/*.py` | Download from Hugging Face and convert 10 judge benchmarks into the common item schema (see `notes/DATASETS.md`) |
| `scripts/run_all.sh` | Full run: 10 datasets × 3 judges |
| `tests/` | Synthetic checks of the risk control and PPI code |
| `paper/` | LaTeX draft (`main.tex`, `sections/`, `refs.bib`) |
| `notes/` | Literature reviews (general VLM; medicine/science; urban/geo/embodied; novelty and applications) and the Chinese synthesis |

## Item schema (`$EAGLEVLM_DATA/items/<dataset>.jsonl`)

```
id, dataset, subset, kind ("resp" | "img"), question, context [image paths], cand [A, B], criterion, label (0 | 1 | "tie")
```

Image paths are relative to `$EAGLEVLM_DATA`.

## Run

```bash
export EAGLEVLM_DATA=/media/shawn/C/papers/VLM/judgebench
PY=/home/shawn/anaconda3/envs/lis-omni/bin/python
cd $EAGLEVLM_DATA
$PY ~/path/to/EAGLE-VLM/eaglevlm/judge_bench.py --model qwen3vl8b --items items/mmrlhf.jsonl \
    --out results/mmrlhf__qwen3vl8b.jsonl --conds bin,score --limit 8      # smoke test
bash ~/path/to/EAGLE-VLM/scripts/run_all.sh                                # everything
```

Model paths are set in `MODELS` in `judge_bench.py`.

## Versions (git tags)
| tag | stage |
|---|---|
| `v0.0-legacy-pilots-20261030` | pre-EAGLE Place Pulse / urban-judge pilots (`legacy/`) |
| `v0.1-harness-pp-pilot-20261031` | judge harness, risk-control stats, Place Pulse pilot, BT / PPI leaderboard |
| `v0.2-ladder-32b-20261101` | multi-rung escalation ladder with Qwen3-VL-32B |
| `v0.3-phaseA-pack-20261101` | vision probes, InternVL, shared errors, phase A (tri / str7), multi-GPU pack + scheduler |
| `v0.4-review-baselines-20261102` | zoom / blind / action-value, gate 0, baseline table with dedicated judges (shared + native protocols) |
| `v0.5-tie-mechanism-20261103` | tie mechanism, failure levels, hidden-state probes, native synthetic tie test |
| `v0.6-diagnostics-certification-20261103` | VLM-vs-LLM selective judging, position bias, certification with ties kept |
| `v0.7-c1c2-chain-20261104` | tie-dropping reproduction chain, ablations, phenomena, decomposition, absolute pointwise prompts |
| `v0.8-a100-router-lora-20261105` | A100 queues, pairwise->pointwise router v2 + control, tie-channel LoRA diagnostic, figures/tables |
| `v0.9-resolution-ablation-20261105` | full-resolution (1536 px) ablation conditions |

Other directories: `scripts/a100/` (rented 4xA100 setup, queues, sync; `deprecated/` = stopped home-grown Best-of-N),
`runs/bjserver/` (historical run chains), `paper_figures/` (main-table generator), `scripts/models/` (dedicated-judge downloads).
The full experiment history (settings, results, decisions) is kept with the data on bjserver, not in this repo.
