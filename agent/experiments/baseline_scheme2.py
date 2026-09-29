"""Single-seed Scheme-2 orchestration for learned comparison algorithms."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent.baselines.learned_variants import LEARNED_BASELINE_METHODS
from agent.baselines.rl_training_scheme2 import LearnedBaselineRandomJointTrainer
from agent.experiments.l1 import (
    L1_CONFIG,
    PROJECT_CONFIGS,
    THROUGHPUT_CONFIG,
    build_formal_project,
    load_phase_l1_config,
    validate_phase_l1_config,
)
from agent.experiments.scheme2 import (
    SCHEME2_CONFIG,
    TRAIN_CONFIG,
    _settings,
    load_scheme2_config,
    scheme2_preflight,
)
from agent.training.config import load_phase_j_config


BASELINE_RUN_ROOT = "result/baseline_scheme2_runs"


def baseline_scheme2_preflight(*, method: str, replay_microbatch: int = 512, cache_mib: int = 2048) -> dict:
    if method not in LEARNED_BASELINE_METHODS:
        raise ValueError(f"unsupported learned baseline: {method}")
    report = dict(scheme2_preflight())
    report.update({
        "method": method,
        "run_root": BASELINE_RUN_ROOT,
        "replay_microbatch": int(replay_microbatch),
        "materialize_cache_max_mib": int(cache_mib),
        "validation_checkpoint_selection": True,
        "resume_supported": True,
        "baseline_action_constraints": (
            "flat_single_edge" if method in {"MLP-PPO", "HGT-PPO-Flat"}
            else "unconstrained"
        ),
    })
    if int(replay_microbatch) <= 0 or int(replay_microbatch) > int(report["logical_minibatch"]):
        raise ValueError("replay microbatch must be within logical minibatch")
    if int(cache_mib) <= 0:
        raise ValueError("cache_mib must be positive")
    return report


def run_baseline_scheme2(
    *,
    method: str,
    mode: str,
    iterations: int | None = None,
    device: str | None = None,
    resume: bool = False,
    replay_microbatch: int = 512,
    cache_mib: int = 2048,
):
    if method not in LEARNED_BASELINE_METHODS:
        raise ValueError(f"unsupported learned baseline: {method}")
    s2 = load_scheme2_config()
    l1 = load_phase_l1_config(L1_CONFIG, project_cfg=None)
    if not l1.formal_training_ready:
        raise RuntimeError("baseline Scheme 2 requires the frozen formal reward protocol")
    project = build_formal_project(l1)
    validate_phase_l1_config(l1, project_cfg=project)
    phase_j = load_phase_j_config(TRAIN_CONFIG, project_cfg=project)

    mode = str(mode)
    safe = method.lower().replace("-", "_")
    if mode == "smoke":
        total = 1
        run_name = f"{safe}_scheme2_smoke_seed{s2.training_seed}"
        validation_every = 10**9
        validate_at_end = False
    elif mode == "confirm":
        total = 1
        run_name = f"{safe}_scheme2_confirm_seed{s2.training_seed}"
        validation_every = 10**9
        validate_at_end = False
    elif mode == "formal":
        if iterations is None or int(iterations) <= 0:
            raise ValueError("formal learned-baseline training requires --iterations N")
        total = int(iterations)
        run_name = f"{safe}_scheme2_seed{s2.training_seed}_n{total}"
        validation_every = s2.validation_every_iterations
        validate_at_end = True
    else:
        raise ValueError("mode must be smoke/confirm/formal")

    run_dir = Path(BASELINE_RUN_ROOT) / run_name
    resume_path = None
    if resume:
        candidate = run_dir / "checkpoints" / "latest.pt"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        resume_path = str(candidate)
    elif (run_dir / "checkpoints" / "latest.pt").is_file():
        raise FileExistsError(
            f"learned-baseline Scheme-2 run already exists: {run_name}. "
            "Use --resume with the same --iterations or remove the diagnostic run."
        )

    settings = _settings(
        s2,
        phase_j,
        total_iterations=total,
        device=device,
        run_name=run_name,
        resume_checkpoint=resume_path,
        validation_every_iterations=validation_every,
    )
    settings = replace(
        settings,
        run_root=BASELINE_RUN_ROOT,
        replay_microbatch_size_override=int(replay_microbatch),
        replay_microbatch_min_size_override=min(int(s2.replay_microbatch_min), int(replay_microbatch)),
        materialize_cache_max_mib=int(cache_mib),
    )
    if mode == "smoke":
        # Execution-only diagnostic; formal budget remains 32/8192/4.
        settings = replace(
            settings,
            parallel_envs=8,
            rollout_events=512,
            ppo_epochs_override=1,
            normalization_episodes=1,
            normalization_max_graphs=64,
            checkpoint_every_iterations=1,
            multiprocess_worker_processes=4,
            rollout_action_batch_size=8,
        )

    trainer = LearnedBaselineRandomJointTrainer(
        project,
        phase_j,
        settings,
        method=method,
        total_iterations=total,
        joint_scale_pool=s2.scales,
        joint_scenario_pool=s2.scenarios,
        joint_load_ratio_pool=s2.load_ratios,
        joint_due_tightness_pool=s2.due_tightness,
        config_paths=PROJECT_CONFIGS + (L1_CONFIG, THROUGHPUT_CONFIG, TRAIN_CONFIG, SCHEME2_CONFIG),
        validate_at_end=validate_at_end,
        fresh_instances_each_iteration=s2.fresh_instances_each_iteration,
        validation_monitor_subset=s2.validation_monitor_subset,
        validation_progress_every_instances=s2.validation_progress_every_instances,
        instance_parameter_table_path=s2.instance_parameter_table_path,
        training_instance_parameter_table_path=(
            s2.instance_parameter_table_path
            if s2.training_structural_source == "parameter_table" else None
        ),
    )
    result = trainer.run()
    return result, trainer.last_iteration_execution_stats


__all__ = [
    "BASELINE_RUN_ROOT",
    "baseline_scheme2_preflight",
    "run_baseline_scheme2",
]
