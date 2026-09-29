"""Teacher Scheme-2 orchestration built on the validated L1.7.6 executor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from data.generator import load_parameter_table

from configs.config import load_config
from agent.experiments.l1 import (
    L1_CONFIG,
    PROJECT_CONFIGS,
    THROUGHPUT_CONFIG,
    build_formal_project,
    load_phase_l1_config,
    validate_phase_l1_config,
)
from agent.training.config import load_phase_j_config
from agent.training.throughput import load_throughput_profile
from agent.training.trainer_j import PhaseJRunSettings
from agent.training.trainer_random_joint import RandomJointTrainer


SCHEME2_CONFIG = "configs/scheme2.yaml"
TRAIN_CONFIG = "configs/train.yaml"


@dataclass(frozen=True, slots=True)
class Scheme2Config:
    budget_profile_name: str
    available_budget_profiles: tuple[str, ...]
    training_seed: int
    run_root: str
    run_name_prefix: str
    formal_iterations: int
    parallel_envs: int
    rollout_events: int
    ppo_epochs: int
    fresh_instances_each_iteration: bool
    replay_microbatch_target: int
    replay_microbatch_min: int
    persist_oom_fallback_across_updates: bool
    materialize_cache_max_mib: int
    tensorized_decoder_scoring: bool
    replay_candidate_score_memoization: bool
    hardware_profile_label: str
    scales: tuple[str, ...]
    scenarios: tuple[str, ...]
    load_ratios: tuple[float, ...]
    due_tightness: tuple[str, ...]
    validation_root: str
    validation_base_seed: int
    validation_instances_per_combination: int
    validation_scales: tuple[str, ...]
    validation_scenarios: tuple[str, ...]
    validation_load_ratios: tuple[float, ...]
    validation_due_tightness: tuple[str, ...]
    validation_monitor_subset: str
    validation_every_iterations: int
    validation_progress_every_instances: int
    validation_min_delta: float
    checkpoint_policy: str
    confirm_iterations: int
    confirm_rollout_events: int
    profile_rollout_events: int
    pilot_iterations: int
    pilot_validation_every_iterations: int
    instance_parameter_table_path: str | None
    training_structural_source: str
    rollout_executor: str
    visdom_enabled: bool
    visdom_server: str
    visdom_port: int
    visdom_env_prefix: str
    visdom_update_every_iterations: int


def load_scheme2_config(
    path: str | Path = SCHEME2_CONFIG,
    *,
    budget_profile: str | None = None,
) -> Scheme2Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["scheme2"]
    profiles = dict(raw.get("budget_profiles") or {})
    selected_profile = str(
        budget_profile or raw.get("active_budget_profile") or "legacy"
    )
    if profiles:
        if selected_profile not in profiles:
            raise ValueError(
                f"unknown Scheme-2 budget profile {selected_profile!r}; "
                f"available={tuple(profiles)}"
            )
        budget = dict(profiles[selected_profile] or {})
    else:
        # Backward-compatible reader for archived config snapshots.
        budget = raw
        selected_profile = "legacy"
    execution = raw["execution"]
    dist = raw["training_distribution"]
    val = raw["validation"]
    evaluation = raw.get("evaluation", {})
    pilot = raw["pilot"]
    visdom = raw.get("monitoring", {}).get("visdom", {})
    cfg = Scheme2Config(
        budget_profile_name=selected_profile,
        available_budget_profiles=tuple(profiles) if profiles else ("legacy",),
        training_seed=int(raw["training_seed"]),
        run_root=str(raw["run_root"]),
        run_name_prefix=str(raw["run_name_prefix"]),
        formal_iterations=int(budget["formal_iterations"]),
        parallel_envs=int(budget["parallel_envs"]),
        rollout_events=int(budget["rollout_events"]),
        ppo_epochs=int(budget["ppo_epochs"]),
        fresh_instances_each_iteration=bool(raw.get("fresh_instances_each_iteration", True)),
        replay_microbatch_target=int(execution["replay_microbatch_target"]),
        replay_microbatch_min=int(execution.get("replay_microbatch_min", 16)),
        persist_oom_fallback_across_updates=bool(
            execution.get("persist_oom_fallback_across_updates", False)
        ),
        materialize_cache_max_mib=int(execution["materialize_cache_max_mib"]),
        tensorized_decoder_scoring=bool(
            execution.get("tensorized_decoder_scoring", False)
        ),
        replay_candidate_score_memoization=bool(
            execution.get("replay_candidate_score_memoization", False)
        ),
        hardware_profile_label=str(execution.get("hardware_profile_label", "unspecified")),
        scales=tuple(map(str, dist["scales"])),
        scenarios=tuple(map(str, dist["scenarios"])),
        load_ratios=tuple(map(float, dist["load_ratios"])),
        due_tightness=tuple(map(str, dist["due_tightness"])),
        validation_root=str(val["root"]),
        validation_base_seed=int(val["base_seed"]),
        validation_instances_per_combination=int(val["instances_per_combination"]),
        validation_scales=tuple(map(str, val["scales"])),
        validation_scenarios=tuple(map(str, val["scenarios"])),
        validation_load_ratios=tuple(map(float, val["load_ratios"])),
        validation_due_tightness=tuple(map(str, val["due_tightness"])),
        validation_monitor_subset=str(val.get("monitor_subset", "full")),
        validation_every_iterations=int(
            budget.get("validation_every_iterations", val["every_iterations"])
        ),
        validation_progress_every_instances=int(val.get("progress_every_instances", 0)),
        validation_min_delta=float(val["min_delta"]),
        checkpoint_policy=str(evaluation.get("checkpoint_policy", "best_validation")),
        confirm_iterations=int(pilot["confirm_iterations"]),
        confirm_rollout_events=int(pilot.get("confirm_rollout_events", 128)),
        profile_rollout_events=int(pilot.get("profile_rollout_events", 32)),
        pilot_iterations=int(pilot["short_iterations"]),
        pilot_validation_every_iterations=int(pilot["validation_every_iterations"]),
        instance_parameter_table_path=(
            str(raw.get("instance_parameter_table_path"))
            if raw.get("instance_parameter_table_path") is not None
            else None
        ),
        training_structural_source=str(raw.get("training_structural_source", "parameter_table")),
        rollout_executor=str(execution.get("rollout_executor", "gpu_batched")),
        visdom_enabled=bool(visdom.get("enabled", True)),
        visdom_server=str(visdom.get("server", "http://localhost")),
        visdom_port=int(visdom.get("port", 8097)),
        visdom_env_prefix=str(visdom.get("env_prefix", "egdm_hgppo_scheme2")),
        visdom_update_every_iterations=int(visdom.get("update_every_iterations", 1)),
    )
    if cfg.training_seed != 0:
        raise ValueError("teacher Scheme 2 currently freezes exactly one training seed: seed 0")
    if set(cfg.scales) != {"S", "M", "L"}:
        raise ValueError("Scheme 2 training distribution must include S/M/L")
    if set(cfg.scenarios) != {"D1", "D2", "D3", "D4", "D5"}:
        raise ValueError("Scheme 2 training distribution must include D1-D5")
    if cfg.instance_parameter_table_path is None:
        raise ValueError("Scheme 2 requires an explicit structural parameter table (S,M,J,H,R,V)")
    if cfg.training_structural_source not in {"scale_ranges", "parameter_table"}:
        raise ValueError("Scheme 2 training_structural_source must be scale_ranges or parameter_table")
    table = load_parameter_table(cfg.instance_parameter_table_path)
    if not table:
        raise ValueError("Scheme 2 parameter table cannot be empty")
    if (
        cfg.formal_iterations <= 0
        or cfg.confirm_rollout_events <= 0
        or cfg.profile_rollout_events <= 0
    ):
        raise ValueError("Scheme 2 formal_iterations must be positive")
    if cfg.parallel_envs <= 0 or cfg.rollout_events <= 0 or cfg.ppo_epochs <= 0:
        raise ValueError("Scheme 2 runtime budgets must be positive")
    if cfg.rollout_events % cfg.parallel_envs != 0:
        raise ValueError("Scheme 2 rollout_events must be divisible by parallel_envs")
    if not 0 < cfg.replay_microbatch_min <= cfg.replay_microbatch_target:
        raise ValueError("Scheme 2 replay microbatch bounds are invalid")
    if cfg.materialize_cache_max_mib <= 0:
        raise ValueError("Scheme 2 materialize cache budget must be positive")
    if cfg.rollout_executor not in {"gpu_batched", "multiprocess_cpu", "serial"}:
        raise ValueError("Scheme 2 rollout_executor must be gpu_batched, multiprocess_cpu, or serial")
    if cfg.validation_monitor_subset not in {"full", "balanced_scale_scenario_15", "stratified_fast_9", "parameter_cases"}:
        raise ValueError("unsupported Scheme 2 validation monitor subset")
    if cfg.validation_every_iterations <= 0 or cfg.validation_progress_every_instances < 0:
        raise ValueError("Scheme 2 validation cadence/progress settings are invalid")
    if cfg.checkpoint_policy not in {"best_validation", "final_iteration"}:
        raise ValueError("Scheme 2 checkpoint_policy must be best_validation or final_iteration")
    if cfg.visdom_port <= 0 or cfg.visdom_update_every_iterations <= 0:
        raise ValueError("Scheme 2 visdom settings are invalid")
    return cfg


def scheme2_preflight(*, budget_profile: str | None = None) -> dict:
    s2 = load_scheme2_config(budget_profile=budget_profile)
    l1 = load_phase_l1_config(L1_CONFIG, project_cfg=None)
    if not l1.formal_training_ready:
        raise RuntimeError("Scheme 2 reuses the frozen formal reward protocol; Phase L1 reward must remain frozen")
    project = build_formal_project(l1)
    validate_phase_l1_config(l1, project_cfg=project)
    throughput = load_throughput_profile(THROUGHPUT_CONFIG)
    table_cases = None
    if s2.instance_parameter_table_path:
        from data.generator import load_parameter_table
        table_cases = load_parameter_table(s2.instance_parameter_table_path)
    fixed_cases = (
        {key: value for key, value in table_cases.items() if str(value["scale"]) in set(s2.scales)}
        if table_cases is not None else None
    )
    # Range-sampled training is not a finite Cartesian product of the
    # immutable validation/test design cells. Report that distinction rather
    # than mislabeling the six fixed cases as the training distribution.
    if s2.training_structural_source == "parameter_table":
        training_structure_mode = "fixed_parameter_cases"
        training_parameter_cases = len(fixed_cases) if fixed_cases is not None else 0
        training_parameter_case_ids = tuple(fixed_cases or {})
        training_distribution_cells = (
            training_parameter_cases * len(s2.scenarios)
            * len(s2.load_ratios) * len(s2.due_tightness)
        )
    else:
        training_structure_mode = "scale_ranges"
        training_parameter_cases = None
        training_parameter_case_ids = ()
        training_distribution_cells = None
    val_cases = (
        sum(1 for item in table_cases.values() if str(item["scale"]) in set(s2.validation_scales))
        if table_cases is not None else len(s2.validation_scales)
    )
    val_combos = (
        val_cases
        if s2.validation_monitor_subset == "parameter_cases"
        else val_cases * len(s2.validation_scenarios) * len(s2.validation_load_ratios) * len(s2.validation_due_tightness)
    )
    if s2.validation_monitor_subset == "balanced_scale_scenario_15":
        monitor_instances = len(s2.validation_scales) * len(s2.validation_scenarios)
    elif s2.validation_monitor_subset == "stratified_fast_9":
        monitor_instances = 9
    elif s2.validation_monitor_subset == "parameter_cases":
        monitor_instances = val_cases
    else:
        monitor_instances = val_combos * s2.validation_instances_per_combination
    return {
        "budget_profile": s2.budget_profile_name,
        "available_budget_profiles": s2.available_budget_profiles,
        "training_seed": s2.training_seed,
        "curriculum_enabled": False,
        "training_distribution_cells": training_distribution_cells,
        "training_structure_mode": training_structure_mode,
        "training_parameter_cases": training_parameter_cases,
        "training_parameter_case_ids": training_parameter_case_ids,
        "training_scales": s2.scales,
        "training_scenarios": s2.scenarios,
        "training_load_ratios": s2.load_ratios,
        "training_due_tightness": s2.due_tightness,
        "formal_iterations": s2.formal_iterations,
        "parallel_envs": s2.parallel_envs,
        "rollout_events": s2.rollout_events,
        "ppo_epochs": s2.ppo_epochs,
        "logical_minibatch": l1.minibatch_size,
        "sampled_events_budget": int(s2.formal_iterations * s2.rollout_events),
        "optimizer_steps_budget": int(
            s2.formal_iterations
            * ((s2.rollout_events + l1.minibatch_size - 1) // l1.minibatch_size)
            * s2.ppo_epochs
        ),
        "replay_microbatch": s2.replay_microbatch_target,
        "replay_microbatch_min": s2.replay_microbatch_min,
        "microbatch_fallback_persists": s2.persist_oom_fallback_across_updates,
        "run_local_microbatch_safe_cap": True,
        "rollout_executor": s2.rollout_executor,
        "hardware_profile_label": s2.hardware_profile_label,
        "multiprocess_rollout": bool(
            throughput.formal_strategy1.enabled and s2.rollout_executor == "multiprocess_cpu"
        ),
        "cross_epoch_materialize_cache": bool(throughput.formal_strategy1.cross_epoch_materialize_cache),
        "materialize_cache_max_mib": s2.materialize_cache_max_mib,
        "tensorized_decoder_scoring": s2.tensorized_decoder_scoring,
        "replay_candidate_score_memoization": s2.replay_candidate_score_memoization,
        "fresh_instances_each_iteration": s2.fresh_instances_each_iteration,
        "instance_parameter_table_path": s2.instance_parameter_table_path,
        "training_structural_source": s2.training_structural_source,
        "validation_pool_instances": val_combos * s2.validation_instances_per_combination,
        "validation_instances": monitor_instances,
        "validation_monitor_subset": s2.validation_monitor_subset,
        "validation_every_iterations": s2.validation_every_iterations,
        "checkpoint_policy": s2.checkpoint_policy,
        "test_visible_to_training": False,
    }


def _settings(
    s2: Scheme2Config,
    phase_j,
    *,
    total_iterations: int,
    device: str | None,
    run_name: str,
    resume_checkpoint: str | None,
    validation_every_iterations: int,
    rollout_events: int | None = None,
    profile_ppo_update_timing: bool = False,
) -> PhaseJRunSettings:
    throughput = load_throughput_profile(THROUGHPUT_CONFIG)
    # stage_iterations is deliberately inert in RandomJointTrainer; the old
    # PhaseJTrainer constructor still validates this legacy field. Keeping a
    # tiny valid tuple avoids touching the accepted curriculum implementation.
    return PhaseJRunSettings(
        stage_iterations=(1, 1, 1, 1),
        parallel_envs=s2.parallel_envs,
        rollout_events=s2.rollout_events if rollout_events is None else int(rollout_events),
        training_seed=s2.training_seed,
        normalization_episodes=8,
        normalization_max_graphs=1024,
        max_episode_decisions=phase_j.runtime.max_episode_decisions,
        device=phase_j.runtime.device if device is None else str(device),
        run_root=s2.run_root,
        run_name=run_name,
        validation_root=s2.validation_root,
        validation_base_seed=s2.validation_base_seed,
        validation_instances_per_combination=s2.validation_instances_per_combination,
        validation_scales=s2.validation_scales,
        validation_scenarios=s2.validation_scenarios,
        validation_load_ratios=s2.validation_load_ratios,
        validation_due_tightness=s2.validation_due_tightness,
        validation_every_iterations=int(validation_every_iterations),
        patience_validations=10**9,
        validation_min_delta=s2.validation_min_delta,
        checkpoint_every_iterations=1,
        restore_best_before_next_stage=False,
        validate_at_stage_end=False,
        ppo_epochs_override=s2.ppo_epochs,
        resume_checkpoint=resume_checkpoint,
        progress_every_events=throughput.progress_every_events,
        normalization_progress_every_graphs=0,
        rollout_storage_mode_override="compact_replay_state",
        replay_microbatch_size_override=s2.replay_microbatch_target,
        replay_microbatch_min_size_override=s2.replay_microbatch_min,
        replay_microbatch_oom_fallback=throughput.cuda_oom_fallback,
        replay_microbatch_persist_oom_fallback=s2.persist_oom_fallback_across_updates,
        rollout_action_batch_size=min(throughput.rollout_action_batch_size, s2.parallel_envs),
        throughput_diagnostics=throughput.diagnostics,
        # The Python discrete-event simulator passed the full-scale 1.5x
        # promotion gate, so formal rollout uses four process-local CPU policy
        # workers. The authoritative PPO learner remains on one GPU and all
        # workers synchronize at the rollout boundary.
        use_multiprocess_rollout=bool(
            throughput.formal_strategy1.enabled
            and s2.rollout_executor == "multiprocess_cpu"
        ),
        multiprocess_worker_processes=int(throughput.formal_strategy1.worker_processes),
        multiprocess_worker_torch_threads=int(throughput.formal_strategy1.worker_torch_threads),
        multiprocess_start_method=str(throughput.formal_strategy1.start_method),
        replay_static_processing_cache=bool(throughput.formal_strategy1.static_processing_cache),
        trusted_replay_normalization=bool(throughput.formal_strategy1.trusted_replay_normalization),
        cross_epoch_materialize_cache=bool(throughput.formal_strategy1.cross_epoch_materialize_cache),
        materialize_cache_max_mib=int(s2.materialize_cache_max_mib),
        tensorized_decoder_scoring=bool(s2.tensorized_decoder_scoring),
        replay_candidate_score_memoization=bool(
            s2.replay_candidate_score_memoization
        ),
        instance_parameter_table_path=s2.instance_parameter_table_path,
        training_instance_parameter_table_path=(
            s2.instance_parameter_table_path
            if s2.training_structural_source == "parameter_table" else None
        ),
        rollout_executor=s2.rollout_executor,
        visdom_enabled=s2.visdom_enabled,
        visdom_server=s2.visdom_server,
        visdom_port=s2.visdom_port,
        visdom_env_prefix=s2.visdom_env_prefix,
        visdom_update_every_iterations=s2.visdom_update_every_iterations,
        hardware_profile_label=s2.hardware_profile_label,
        budget_profile=s2.budget_profile_name,
        profile_ppo_update_timing=bool(profile_ppo_update_timing),
    )


def run_scheme2(
    *,
    mode: str,
    iterations: int | None = None,
    device: str | None = None,
    resume: bool = False,
    budget_profile: str | None = None,
):
    s2 = load_scheme2_config(budget_profile=budget_profile)
    l1 = load_phase_l1_config(L1_CONFIG, project_cfg=None)
    if not l1.formal_training_ready:
        raise RuntimeError("Scheme 2 requires the already frozen reward protocol")
    project = build_formal_project(l1)
    validate_phase_l1_config(l1, project_cfg=project)
    phase_j = load_phase_j_config(TRAIN_CONFIG, project_cfg=project)

    mode = str(mode)
    if mode == "confirm":
        total = s2.confirm_iterations
        run_name = f"scheme2_confirm_seed{s2.training_seed}_n{total}"
        validation_every = 10**9
        validate_at_end = False
        profile_ppo_update_timing = False
    elif mode == "profile":
        total = 1
        run_name = f"scheme2_ppo_profile_seed{s2.training_seed}"
        validation_every = 10**9
        validate_at_end = False
        profile_ppo_update_timing = True
    elif mode == "pilot":
        total = s2.pilot_iterations if iterations is None else int(iterations)
        run_name = (
            f"scheme2_pilot_{s2.budget_profile_name}_seed{s2.training_seed}_n{total}"
        )
        validation_every = s2.pilot_validation_every_iterations
        validate_at_end = True
        profile_ppo_update_timing = False
    elif mode == "formal":
        if iterations is None or int(iterations) <= 0:
            raise ValueError("formal Scheme-2 training requires --iterations N")
        total = int(iterations)
        run_name = f"{s2.run_name_prefix}{s2.training_seed}_n{total}"
        validation_every = s2.validation_every_iterations
        validate_at_end = True
        profile_ppo_update_timing = False
    else:
        raise ValueError("mode must be confirm/profile/pilot/formal")

    resume_path = None
    if resume:
        candidate = Path(s2.run_root) / run_name / "checkpoints" / "latest.pt"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        resume_path = str(candidate)
    elif (Path(s2.run_root) / run_name / "checkpoints" / "latest.pt").is_file():
        raise FileExistsError(
            f"Scheme-2 run already exists: {run_name}. Use --resume with the same --iterations, "
            "or choose a different formal budget."
        )

    settings = _settings(
        s2,
        phase_j,
        total_iterations=total,
        device=device,
        run_name=run_name,
        resume_checkpoint=resume_path,
        validation_every_iterations=validation_every,
        rollout_events=(
            s2.profile_rollout_events
            if mode == "profile"
            else s2.confirm_rollout_events if mode == "confirm" else None
        ),
        profile_ppo_update_timing=profile_ppo_update_timing,
    )
    trainer = RandomJointTrainer(
        project,
        phase_j,
        settings,
        total_iterations=total,
        joint_scale_pool=s2.scales,
        joint_scenario_pool=s2.scenarios,
        joint_load_ratio_pool=s2.load_ratios,
        joint_due_tightness_pool=s2.due_tightness,
        config_paths=PROJECT_CONFIGS + (L1_CONFIG, THROUGHPUT_CONFIG, TRAIN_CONFIG, SCHEME2_CONFIG, "configs/fixed_design_scheme2.yaml"),
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


__all__ = ["SCHEME2_CONFIG", "Scheme2Config", "load_scheme2_config", "run_scheme2", "scheme2_preflight"]
