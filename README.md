# EGDM-HGPPO

EGDM-HGPPO is a PPO-based heterogeneous-graph deep reinforcement learning solver for a reconfigurable human-robot collaborative assembly scheduling problem. The release contains the implementation, immutable instances, trained checkpoints, training/evaluation logs, and formal CSV results used by the paper experiments.

The package follows the latest advisor protocol: Scheme-2 random-joint training, one formal seed (`0`), no curriculum transitions, fresh online S/M/L instances at every PPO iteration, deterministic fixed validation, and a separate frozen test suite. The advisor's `M/A/R/J` wording is treated as an illustrative parameter example; the implementation uses the canonical structural metadata stored in each instance index.

## 1. Problem And Experimental Contract

The environment models event-gated human/robot-cell reconfiguration and parallel operation-cell dispatch. The objective is minimum total weighted tardiness (TWT), with hard feasibility masks, duration-aware PPO/GAE, graph-based policy/value networks, and exact reward-identity checks. The executable contract is summarized in this README so the release has one authoritative runtime guide.

Training instances are generated online from the S/M/L ranges in `configs/instance.yaml`. Validation and test instances are generated once and stored as one CSV per fixed design unit with an `index.csv`; test files are never read during training. The XL suite is generated separately for zero-shot size generalization. The current formal budget is the `accelerated_3epoch` profile in `configs/scheme2.yaml`: 200 iterations, 32 environments, 4096 events per iteration, logical minibatch 512, physical replay microbatch 128, and 3 PPO epochs.

Repository layout:

```text
agent/        policy, PPO, training, baselines, evaluation, experiments
configs/      reproducible runtime and experiment settings
data/         instance generation plus fixed validation/test instances
environment/  scheduling state, graph, actions, rewards, and simulator
result/       released checkpoints, logs, reports, and CSV results
scripts/      numbered reproducible experiment entrypoints
train.py      short main-training entrypoint
eval.py       fixed-test evaluation entrypoint
README.md     the single setup and reproduction guide
```

The numbered scripts are retained because they represent distinct reproducible paper experiments; the former top-level `training`, `baselines`, `evaluation`, and `experiments` packages are consolidated under `agent/`. No separate `docs/` directory is required.

## 2. Environment

Use a CUDA-enabled PyTorch environment. From the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/run_00_hardware_probe.py
```

The hardware probe must report `torch_cuda_available: true` before long training. The code has been profiled on an RTX 4060-class 8 GB GPU; the active execution profile is also suitable for the target RTX 5060 Ti environment, subject to checking the first iteration's memory and wall time. No absolute machine path is embedded in the code or README.

## 3. Smoke Check

Run the disposable full-stack smoke before formal computation:

```powershell
python scripts/run_00_preflight.py
python scripts/run_00_smoke.py
```

The smoke performs a small random-joint rollout, one PPO update, checkpoint writing, reward-identity checks, and a fixed-instance rule-baseline evaluation. It writes only ignored temporary outputs under `result/` and never replaces a formal checkpoint. Smoke output is an engineering check, not paper evidence.

## 4. Fixed Data

Materialize validation, S/M/L test, and index files once:

```powershell
python scripts/run_01_prepare_data.py
python scripts/run_01b_prepare_structural_effects.py
```

The first command creates `data/instances/val_scheme2/` and `data/instances/test/`. The second creates the nine-cell controlled structural-effects suite under `data/instances/test_structural_effects_v2/`. Sensitivity instances are already stored under `data/instances/sensitivity/` and are reused by the sensitivity runner.

## 5. Main Training

Start the advisor-approved EGDM-HGPPO run:

```powershell
python scripts/run_02_train_formal.py
```

The formal run writes `result/scheme2_runs/egdm_hgppo_scheme2_seed0_n200/`. It saves `latest.pt` every iteration and selects `best_model.pt` using deterministic fixed-validation mean TWT. If the process is interrupted, resume the same budget with:

```powershell
python scripts/run_05_resume_formal.py
```

Judge convergence from `train_log.csv`, `validation_log.csv`, and `validation_instance_log.csv`: fixed-validation mean TWT should improve or stabilize, all six validation cases should remain feasible, reward-identity error should be near zero, and there must be no NaN/Inf or uncaught CUDA OOM. PPO losses do not need to decrease monotonically.

## 6. Baselines And Ablations

Train the learned comparison architectures:

```powershell
python scripts/run_07_train_learned_baselines.py
```

Compute the offline reference and evaluate all available main methods:

```powershell
python scripts/run_08_exact_reference.py
python scripts/run_03_eval_formal.py
```

Run the strict component ablations and evaluate their fixed-test rows:

```powershell
python scripts/run_12_train_ablations.py
python scripts/run_13_eval_ablations.py
```

The released evidence currently has valid validation-selected checkpoints for seven ablation variants. `sequential_dispatch` and `single_critic` are retained as explicitly incomplete/failed variants because they do not have valid `best_model.pt` files; they must not be reported as completed ablations.

## 7. Generalization And Sensitivity

Run the remaining paper experiments:

```powershell
python scripts/run_04_prepare_xl_generalization.py
python scripts/run_06_eval_xl_generalization.py
python scripts/run_10_sensitivity.py
python scripts/run_11_eval_structural_effects.py
```

These commands write `result/generalization.csv`, `result/sensitivity.csv`, and `result/structural_effects.csv`. The sensitivity runner is restartable at the completed-cell level.

## 8. Statistics

Aggregate only completed evaluation-schema CSV files:

```powershell
python scripts/run_09_aggregate_stats.py
```

The output is `result/stats_summary.csv`. This release follows the advisor's one-seed experimental design, so the aggregation is descriptive and does not fabricate multi-seed confidence intervals, p-values, or significance claims. Plotting is intentionally left for the manuscript stage after the advisor confirms the raw results.

## 9. One-Command And PyCharm Launch

For a fresh output directory, the complete dependency-ordered workflow is:

```powershell
python scripts/run_all.py
```

`run_all.py` starts a new main training run; it is not a resume command. On a package that already contains the released checkpoints/results, run the numbered commands selectively, or use `run_05_resume_formal.py` for an interrupted main run. In PyCharm, set the script path to the desired `scripts/run_XX_*.py`, leave parameters empty, set the working directory to the repository root, and select the CUDA-enabled interpreter.

## 10. Output Map

| Purpose | Path |
|---|---|
| Implementation | `agent/` (policy, training, baselines, evaluation, and experiments), `environment/` |
| Runtime and method configuration | `configs/` |
| Fixed validation/test/sensitivity instances | `data/instances/` |
| Formal and baseline checkpoints/logs | `result/*_runs/` |
| Main fixed-test comparison | `result/eval_results.csv` |
| Exact reference | `result/ref_exact.csv` |
| XL generalization | `result/generalization.csv` |
| Sensitivity | `result/sensitivity.csv` |
| Structural effects | `result/structural_effects.csv` |
| Ablation comparison | `result/ablation_results.csv` |
| Descriptive aggregate | `result/stats_summary.csv` |

The package does not include obsolete profiling scripts, K2 diagnostic smoke code, unit-test scaffolding, documentation duplicates, or machine-specific IDE files. All commands above use repository-relative paths.
