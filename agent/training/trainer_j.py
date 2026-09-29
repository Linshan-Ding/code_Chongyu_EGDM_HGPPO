"""Phase J four-stage curriculum trainer with fixed validation and checkpoint/resume."""

from __future__ import annotations

import csv
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from agent.policy import EGDMCompositePolicy
from agent.ppo import PPOAgent
from agent.training.checkpoint import (
    create_run_directory,
    load_training_checkpoint,
    restore_model_optimizer_snapshot,
    restore_rng_state,
    save_model_optimizer_snapshot,
    save_training_checkpoint,
    write_run_snapshot,
)
from agent.training.curriculum import (
    CurriculumController,
    build_stage_policy_spec,
    configure_curriculum_stage,
)
from agent.training.normalization import fit_training_normalizer
from agent.training.multiprocess_rollout import (
    MultiprocessRolloutConfig,
    collect_rollout_multiprocess,
)
from agent.training.rollout import RolloutCollector
from agent.training.sampler import OnlineInstanceSampler
from agent.training.trainer import resolve_device
from agent.training.validation import (
    append_validation_csv,
    append_validation_instance_csv,
    ensure_fixed_validation_suite,
    evaluate_fixed_validation,
    filter_validation_records,
)
from agent.training.vector_env import EventVectorEnv


@dataclass(frozen=True, slots=True)
class PhaseJRunSettings:
    stage_iterations: tuple[int, int, int, int]
    parallel_envs: int
    rollout_events: int
    training_seed: int
    normalization_episodes: int
    normalization_max_graphs: int
    max_episode_decisions: int
    device: str
    run_root: str
    run_name: str
    validation_root: str
    validation_base_seed: int
    validation_instances_per_combination: int
    validation_scales: tuple[str, ...]
    validation_scenarios: tuple[str, ...]
    validation_load_ratios: tuple[float, ...]
    validation_due_tightness: tuple[str, ...]
    validation_every_iterations: int
    patience_validations: int
    validation_min_delta: float
    checkpoint_every_iterations: int
    restore_best_before_next_stage: bool
    validate_at_stage_end: bool = True
    validation_stop_after_first_failure: bool = False
    ppo_epochs_override: int | None = None
    stop_after_global_iterations: int | None = None
    resume_checkpoint: str | None = None
    # Phase L1.1 alignment/stress-pilot controls. Formal L1.2 adds memory-safe overrides below.
    start_stage_index: int = 0
    forced_scale_pool: tuple[str, ...] | None = None
    forced_scenario_pool: tuple[str, ...] | None = None
    forced_load_ratio_pool: tuple[float, ...] | None = None
    forced_due_tightness_pool: tuple[str, ...] | None = None
    # Phase L1.2 diagnostics only; zero keeps ordinary runs quiet.
    progress_every_events: int = 0
    normalization_progress_every_graphs: int = 0
    rollout_storage_mode_override: str | None = None
    replay_microbatch_size_override: int | None = None
    replay_microbatch_min_size_override: int | None = None
    replay_microbatch_oom_fallback: bool = False
    # Historical L1.7.6 behavior mutates the configured replay size after OOM.
    # Scheme 2 keeps the configuration immutable and instead uses an in-process
    # run-local safe cap, preserving the logical minibatch and optimizer steps.
    replay_microbatch_persist_oom_fallback: bool = True
    # L1.5 execution-only throughput knobs. They do not change the logical PPO
    # minibatch, rollout event budget, curriculum, reward, or action semantics.
    rollout_action_batch_size: int = 1
    throughput_diagnostics: bool = False
    # L1.6.9 production Strategy-1 executor. Defaults preserve every earlier
    # Phase-J/L1 diagnostic and calibration run unless formal settings opt in.
    use_multiprocess_rollout: bool = False
    multiprocess_worker_processes: int = 4
    multiprocess_worker_torch_threads: int = 2
    multiprocess_start_method: str = "spawn"
    replay_static_processing_cache: bool = False
    trusted_replay_normalization: bool = False
    # L1.7.6 execution-only PPO replay optimization.  It reuses exact CPU
    # materializations across PPO epochs and leaves policy/GAE/PPO semantics intact.
    cross_epoch_materialize_cache: bool = False
    materialize_cache_max_mib: int = 0
    # Execution-only decoder optimization. Candidate generation, hard masks,
    # candidate order and autoregressive trace semantics remain unchanged.
    tensorized_decoder_scoring: bool = False
    replay_candidate_score_memoization: bool = False
    # Scheme-2: training instance parameter table. Validation/test remain fixed.
    instance_parameter_table_path: str | None = None
    # Optional online-training table. None means sample structural dimensions
    # from the S/M/L ranges in configs/instance.yaml. The fixed design table is
    # intentionally not reused for online training.
    training_instance_parameter_table_path: str | None = None
    # Execution backend: serial, gpu_batched, or legacy multiprocess CPU policy.
    rollout_executor: str = "serial"
    # Live monitoring is optional; CSV remains the experiment source of truth.
    visdom_enabled: bool = False
    visdom_server: str = "http://localhost"
    visdom_port: int = 8097
    visdom_env_prefix: str = "egdm_hgppo_scheme2"
    visdom_update_every_iterations: int = 1
    hardware_profile_label: str = "unspecified"
    budget_profile: str = "unspecified"
    # Diagnostic-only timing hooks; never enabled by formal Scheme-2 runners.
    profile_ppo_update_timing: bool = False


@dataclass(frozen=True, slots=True)
class PhaseJIterationMetric:
    global_iteration: int
    stage_index: int
    stage_name: str
    stage_iteration: int
    global_progress: float
    events: int
    completed_episodes: int
    mean_completed_twt: float | None
    reward_mean: float
    reconfiguration_fraction: float
    max_worker_moves: int
    max_robot_moves: int
    policy_loss: float
    value_loss: float
    approx_kl: float
    clip_fraction: float
    learning_rate: float
    encoder_learning_rate: float
    gate_entropy_coef: float
    matching_entropy_coef: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class PhaseJTrainingResult:
    run_dir: str
    finished: bool
    global_iterations: int
    final_stage_index: int
    final_stage_name: str
    metrics_rows: int
    validation_rows: int
    latest_checkpoint: str
    best_model: str | None


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _child_seed(seed: int, *parts: int) -> int:
    seq = np.random.SeedSequence([int(seed), *map(int, parts)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Resumed checkpoints may contain rows from an older schema.  Use the
    # stable first-seen union so adding observational telemetry never breaks a
    # resumed run or silently drops the new fields.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class PhaseJTrainer:
    def __init__(self, cfg, phase_j, settings: PhaseJRunSettings, *, config_paths) -> None:
        self.cfg = cfg
        self.phase_j = phase_j
        self.settings = settings
        self.config_paths = tuple(str(x) for x in config_paths)
        if len(settings.stage_iterations) != 4 or any(int(x) <= 0 for x in settings.stage_iterations):
            raise ValueError("Phase J requires four positive stage iteration budgets")
        if settings.rollout_executor not in {"serial", "gpu_batched", "multiprocess_cpu"}:
            raise ValueError("unsupported rollout_executor")
        self.device = resolve_device(settings.device)
        self.run_dir = create_run_directory(settings.run_root, settings.run_name)
        self.metrics_path = self.run_dir / "train_log.csv"
        self.validation_path = self.run_dir / "validation_log.csv"
        self.validation_instance_path = self.run_dir / "validation_instance_log.csv"
        self.latest_path = self.run_dir / "checkpoints" / "latest.pt"
        self.last_iteration_execution_stats: dict | None = None
        if self.settings.use_multiprocess_rollout:
            MultiprocessRolloutConfig(
                worker_processes=int(self.settings.multiprocess_worker_processes),
                worker_torch_threads=int(self.settings.multiprocess_worker_torch_threads),
                start_method=str(self.settings.multiprocess_start_method),
            ).validate(
                parallel_envs=int(self.settings.parallel_envs),
                rollout_events=int(self.settings.rollout_events),
            )
            if (
                self.settings.rollout_storage_mode_override
                or getattr(self.cfg.algo.ppo_implementation, "rollout_storage_mode", "full_graph_context")
            ) != "compact_replay_state":
                raise ValueError("formal Strategy-1 requires compact_replay_state storage")
            if not self.settings.replay_static_processing_cache:
                raise ValueError("formal Strategy-1 requires the validated static processing cache")
            if not self.settings.trusted_replay_normalization:
                raise ValueError("formal Strategy-1 requires the validated trusted replay-normalization fast path")
            if self.settings.cross_epoch_materialize_cache and int(self.settings.materialize_cache_max_mib) <= 0:
                raise ValueError("formal cross-epoch materialize cache requires a positive MiB budget")

    @property
    def diagnostics_enabled(self) -> bool:
        return bool(
            self.settings.progress_every_events
            or self.settings.normalization_progress_every_graphs
            or self.settings.throughput_diagnostics
        )

    def _diag(self, message: str) -> None:
        if self.diagnostics_enabled:
            print(f"[L1.2] {message}", flush=True)

    def _samplers_for_stage(self, stage_spec):
        scale_pool = self.settings.forced_scale_pool or stage_spec.scale_pool
        scenario_pool = self.settings.forced_scenario_pool or stage_spec.scenario_pool
        load_ratio_pool = self.settings.forced_load_ratio_pool or stage_spec.load_ratio_pool
        due_tightness_pool = self.settings.forced_due_tightness_pool or stage_spec.due_tightness_pool
        return [
            OnlineInstanceSampler(
                self.cfg,
                seed=_child_seed(self.settings.training_seed, 100 + stage_spec.stage_index, env_id),
                scale_pool=scale_pool,
                scenario_pool=scenario_pool,
                load_ratio_pool=load_ratio_pool,
                due_tightness_pool=due_tightness_pool,
                instance_parameter_table_path=self.settings.training_instance_parameter_table_path,
            )
            for env_id in range(self.settings.parallel_envs)
        ]

    def _new_vector_env(self, stage_spec) -> EventVectorEnv:
        return EventVectorEnv(
            self.cfg,
            self._samplers_for_stage(stage_spec),
            max_episode_decisions=self.settings.max_episode_decisions,
        )

    def _collector(self, vector_env, agent, normalizer, stage_spec) -> RolloutCollector:
        return RolloutCollector(
            vector_env=vector_env,
            agent=agent,
            normalizer=normalizer,
            storage_device=self.cfg.algo.ppo_implementation.rollout_storage_device,
            storage_mode=(
                self.settings.rollout_storage_mode_override
                or getattr(
                    self.cfg.algo.ppo_implementation,
                    "rollout_storage_mode",
                    "full_graph_context",
                )
            ),
            progress_every_events=self.settings.progress_every_events,
            action_batch_size=self.settings.rollout_action_batch_size,
            trusted_normalization=bool(self.settings.trusted_replay_normalization),
            trusted_policy_inputs=bool(self.settings.trusted_replay_normalization),
            static_processing_cache=bool(self.settings.replay_static_processing_cache),
            constraints=stage_spec.constraints,
        )

    def _collect_training_rollout(
        self, *, vector_env, collector, policy, normalizer, stage_spec, global_iteration: int
    ):
        """Collect one formal on-policy batch with the selected execution backend.

        L1.6.9 promotes only the already-confirmed execution path.  The returned
        object intentionally has the same buffer/bootstrap/summary/stats surface
        used by the legacy serial collector, so PPO/curriculum/reward code below
        stays untouched.
        """
        if not self.settings.use_multiprocess_rollout:
            return collector.collect(self.settings.rollout_events)

        mp_cfg = MultiprocessRolloutConfig(
            worker_processes=int(self.settings.multiprocess_worker_processes),
            worker_torch_threads=int(self.settings.multiprocess_worker_torch_threads),
            start_method=str(self.settings.multiprocess_start_method),
        )
        rollout = collect_rollout_multiprocess(
            cfg=self.cfg,
            vector_env=vector_env,
            policy=policy,
            normalizer=normalizer,
            stage_spec=stage_spec,
            target_events=int(self.settings.rollout_events),
            training_seed=int(self.settings.training_seed),
            global_iteration=int(global_iteration),
            collector_next_env=int(collector.next_env),
            storage_device=self.cfg.algo.ppo_implementation.rollout_storage_device,
            storage_mode=(
                self.settings.rollout_storage_mode_override
                or getattr(
                    self.cfg.algo.ppo_implementation,
                    "rollout_storage_mode",
                    "full_graph_context",
                )
            ),
            mp_config=mp_cfg,
            commit_vector_env_state=True,
            static_processing_cache=bool(self.settings.replay_static_processing_cache),
            trusted_replay_normalization=bool(self.settings.trusted_replay_normalization),
        )
        if not rollout.vector_env_state_committed:
            raise RuntimeError("formal Strategy-1 rollout did not commit parent vector-env state")
        collector.next_env = int(rollout.next_env)
        return rollout

    def _write_execution_profile_audit(self) -> None:
        payload = {
            "phase": "L1.6.9",
            "executor": (
                "multiprocess_strategy1"
                if self.settings.use_multiprocess_rollout
                else (
                    "gpu_batched"
                    if self.settings.rollout_executor == "gpu_batched"
                    else "serial"
                )
            ),
            "parallel_envs": int(self.settings.parallel_envs),
            "rollout_events": int(self.settings.rollout_events),
            "worker_processes": int(self.settings.multiprocess_worker_processes),
            "worker_torch_threads": int(self.settings.multiprocess_worker_torch_threads),
            "start_method": str(self.settings.multiprocess_start_method),
            "logical_minibatch": int(self.cfg.algo.minibatch_size),
            "replay_microbatch": int(self.settings.replay_microbatch_size_override or 0),
            "static_processing_cache": bool(self.settings.replay_static_processing_cache),
            "trusted_replay_normalization": bool(self.settings.trusted_replay_normalization),
            "cross_epoch_materialize_cache": bool(self.settings.cross_epoch_materialize_cache),
            "materialize_cache_max_mib": int(self.settings.materialize_cache_max_mib),
            "algorithmic_semantics_changed": False,
            "rollout_executor": str(self.settings.rollout_executor),
        }
        (self.run_dir / "execution_profile_l1_6_9.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        payload_l176 = dict(payload)
        payload_l176["phase"] = "L1.7.6"
        (self.run_dir / "execution_profile_l1_7_6.json").write_text(
            json.dumps(payload_l176, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _fit_normalizer(self):
        sampler = OnlineInstanceSampler(
            self.cfg,
            seed=_child_seed(self.settings.training_seed, 7),
            scale_pool=(self.settings.forced_scale_pool or self.phase_j.normalization.scale_pool),
            scenario_pool=(self.settings.forced_scenario_pool or self.phase_j.normalization.scenario_pool),
            load_ratio_pool=(self.settings.forced_load_ratio_pool or self.cfg.instance.load_ratio_choices),
            due_tightness_pool=(self.settings.forced_due_tightness_pool or ("tight", "medium", "loose")),
            instance_parameter_table_path=self.settings.training_instance_parameter_table_path,
        )
        return fit_training_normalizer(
            self.cfg,
            sampler,
            episodes=self.settings.normalization_episodes,
            max_graphs=self.settings.normalization_max_graphs,
            max_episode_decisions=self.settings.max_episode_decisions,
            progress_every_graphs=self.settings.normalization_progress_every_graphs,
        )[0]

    def run(self) -> PhaseJTrainingResult:
        _seed_everything(self.settings.training_seed)
        self._write_execution_profile_audit()
        controller = CurriculumController(
            self.settings.stage_iterations,
            patience_validations=self.settings.patience_validations,
            min_delta=self.settings.validation_min_delta,
        )
        if self.settings.resume_checkpoint is None and int(self.settings.start_stage_index) != 0:
            start_stage = int(self.settings.start_stage_index)
            if not 0 <= start_stage < 4:
                raise ValueError("start_stage_index must be 0..3")
            controller.state.stage_index = start_stage
            controller.state.stage_iteration = 0
            # Preserve the global curriculum schedule location so the stress pilot
            # uses the LR/entropy coefficients appropriate to the selected stage.
            controller.state.global_iteration = int(sum(self.settings.stage_iterations[:start_stage]))
            controller.state.transition_reason = "diagnostic_start_stage_override"
        metrics_rows: list[dict] = []
        validation_rows: list[dict] = []

        validation_records = ensure_fixed_validation_suite(
            self.cfg,
            root=self.settings.validation_root,
            base_seed=self.settings.validation_base_seed,
            scales=self.settings.validation_scales,
            scenarios=self.settings.validation_scenarios,
            load_ratios=self.settings.validation_load_ratios,
            due_tightness=self.settings.validation_due_tightness,
            instances_per_combination=self.settings.validation_instances_per_combination,
        )

        if self.settings.resume_checkpoint is None:
            write_run_snapshot(
                self.run_dir,
                project_config=self.cfg,
                runtime_settings=self.settings,
                config_paths=self.config_paths + ("configs/train.yaml",),
            )
            self._diag("1/6 run snapshot written; starting streaming training-only normalizer")
            normalizer = self._fit_normalizer()
            self._diag("2/6 normalizer fitted; creating curriculum vector environments")
            stage_spec = build_stage_policy_spec(self.cfg, self.phase_j, controller.state.stage_index)
            vector_env = self._new_vector_env(stage_spec)
            self._diag(
                f"3/6 vector env ready: stage={stage_spec.stage_index + 1} "
                f"{stage_spec.stage_name}, envs={len(vector_env)}"
            )
            reference_graph = normalizer.transform(vector_env.graph(0))
            policy = EGDMCompositePolicy(self.cfg, reference_graph).to(self.device)
            configure_curriculum_stage(policy, controller.state.stage_index)
            agent = PPOAgent(self.cfg, policy)
            if self.settings.replay_microbatch_size_override is not None:
                agent.replay_microbatch_size = int(self.settings.replay_microbatch_size_override)
            if self.settings.replay_microbatch_min_size_override is not None:
                agent.replay_microbatch_min_size = int(self.settings.replay_microbatch_min_size_override)
            agent.replay_microbatch_oom_fallback = bool(self.settings.replay_microbatch_oom_fallback)
            agent.replay_microbatch_persist_oom_fallback = bool(
                self.settings.replay_microbatch_persist_oom_fallback
            )
            agent.cross_epoch_materialize_cache = bool(self.settings.cross_epoch_materialize_cache)
            agent.materialize_cache_max_bytes = int(self.settings.materialize_cache_max_mib) * 1024 * 1024
            if self.settings.ppo_epochs_override is not None:
                agent.ppo_epochs = int(self.settings.ppo_epochs_override)
            collector = self._collector(vector_env, agent, normalizer, stage_spec)
        else:
            payload = load_training_checkpoint(self.settings.resume_checkpoint, map_location="cpu")
            controller.load_state_dict(payload["curriculum_state"])
            metrics_rows = list(payload.get("metrics_rows", []))
            validation_rows = list(payload.get("validation_rows", []))
            normalizer = payload["normalizer"]
            vector_env = payload["vector_env"]
            stage_spec = build_stage_policy_spec(self.cfg, self.phase_j, controller.state.stage_index)
            reference_graph = normalizer.transform(vector_env.graph(0))
            policy = EGDMCompositePolicy(self.cfg, reference_graph).to(self.device)
            configure_curriculum_stage(policy, controller.state.stage_index)
            agent = PPOAgent(self.cfg, policy)
            if self.settings.replay_microbatch_size_override is not None:
                agent.replay_microbatch_size = int(self.settings.replay_microbatch_size_override)
            if self.settings.replay_microbatch_min_size_override is not None:
                agent.replay_microbatch_min_size = int(self.settings.replay_microbatch_min_size_override)
            agent.replay_microbatch_oom_fallback = bool(self.settings.replay_microbatch_oom_fallback)
            agent.replay_microbatch_persist_oom_fallback = bool(
                self.settings.replay_microbatch_persist_oom_fallback
            )
            agent.cross_epoch_materialize_cache = bool(self.settings.cross_epoch_materialize_cache)
            agent.materialize_cache_max_bytes = int(self.settings.materialize_cache_max_mib) * 1024 * 1024
            if self.settings.ppo_epochs_override is not None:
                agent.ppo_epochs = int(self.settings.ppo_epochs_override)
            policy.load_state_dict(payload["policy_state"])
            agent.optimizer.load_state_dict(payload["optimizer_state"])
            collector = self._collector(vector_env, agent, normalizer, stage_spec)
            collector.next_env = int(payload["collector_next_env"])
            restore_rng_state(payload["rng_state"])

        best_model_path: Path | None = None

        while not controller.state.finished:
            stage_spec = build_stage_policy_spec(
                self.cfg, self.phase_j, controller.state.stage_index
            )
            configure_curriculum_stage(policy, controller.state.stage_index)
            progress = controller.global_progress()
            # Apply the same global schedule to action sampling and the ensuing
            # PPO update. The first update starts at progress=0.
            agent.set_progress(progress)
            started = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self._diag(
                f"4/6 collecting {self.settings.rollout_events} events "
                f"with storage_mode={self.settings.rollout_storage_mode_override or getattr(self.cfg.algo.ppo_implementation, 'rollout_storage_mode', 'full_graph_context')}, "
                f"action_batch={collector.action_batch_size}; "
                f"executor={'multiprocess_strategy1' if self.settings.use_multiprocess_rollout else 'serial'}"
            )
            rollout_started = time.perf_counter()
            try:
                rollout = self._collect_training_rollout(
                    vector_env=vector_env,
                    collector=collector,
                    policy=policy,
                    normalizer=normalizer,
                    stage_spec=stage_spec,
                    global_iteration=controller.state.global_iteration,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"rollout failed in curriculum stage {stage_spec.stage_index + 1} "
                    f"{stage_spec.stage_name}"
                ) from exc
            rollout_seconds = time.perf_counter() - rollout_started
            self._diag(
                f"5/6 rollout complete in {rollout_seconds:.1f}s "
                f"({self.settings.rollout_events / max(rollout_seconds, 1e-9):.2f} events/s); "
                f"compact replay payload≈{rollout.stats.estimated_replay_megabytes:.1f} MiB; "
                f"starting PPO update (logical minibatch={agent.minibatch_size}, "
                f"replay microbatch={agent.replay_microbatch_size})"
            )
            update_started = time.perf_counter()
            update = agent.update(
                rollout.buffer,
                bootstrap_values=rollout.bootstrap_values,
                progress=progress,
                components=stage_spec.components,
            )
            update_seconds = time.perf_counter() - update_started
            peak_mb = (
                torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
                if self.device.type == "cuda" else 0.0
            )
            self._diag(
                f"PPO update complete in {update_seconds:.1f}s; "
                f"optimizer_steps={update.optimizer_steps}, approx_kl={update.approx_kl:.6g}, "
                f"clip_fraction={update.clip_fraction:.6g}, cuda_peak_alloc≈{peak_mb:.1f} MiB"
            )
            controller.record_iteration()
            if stage_spec.stage_index == 0 and rollout.stats.reconfiguration_fraction != 0.0:
                raise RuntimeError("Stage I executed a forbidden reconfiguration")
            if stage_spec.stage_index == 1:
                if rollout.stats.max_worker_moves > 1 or rollout.stats.max_robot_moves > 1:
                    raise RuntimeError("Stage II exceeded one worker/robot move per event")
            twts = [s.twt for s in rollout.episode_summaries]
            metric = PhaseJIterationMetric(
                global_iteration=controller.state.global_iteration,
                stage_index=stage_spec.stage_index,
                stage_name=stage_spec.stage_name,
                stage_iteration=controller.state.stage_iteration,
                global_progress=progress,
                events=rollout.stats.events,
                completed_episodes=rollout.stats.completed_episodes,
                mean_completed_twt=None if not twts else float(mean(twts)),
                reward_mean=float(rollout.stats.reward_mean),
                reconfiguration_fraction=float(rollout.stats.reconfiguration_fraction),
                max_worker_moves=int(rollout.stats.max_worker_moves),
                max_robot_moves=int(rollout.stats.max_robot_moves),
                policy_loss=float(update.policy_loss),
                value_loss=float(update.value_loss),
                approx_kl=float(update.approx_kl),
                clip_fraction=float(update.clip_fraction),
                learning_rate=float(update.learning_rate),
                encoder_learning_rate=float(update.encoder_learning_rate),
                gate_entropy_coef=float(update.gate_entropy_coef),
                matching_entropy_coef=float(update.matching_entropy_coef),
                wall_seconds=time.perf_counter() - started,
            )
            self.last_iteration_execution_stats = {
                "global_iteration": int(controller.state.global_iteration),
                "rollout_seconds": float(rollout_seconds),
                "ppo_seconds": float(update_seconds),
                "wall_seconds": float(metric.wall_seconds),
                "optimizer_steps": int(update.optimizer_steps),
                "cuda_peak_alloc_mib": float(peak_mb),
                "materialize_cache": agent.last_materialize_cache_stats,
            }
            metrics_rows.append(asdict(metric))
            _write_rows(self.metrics_path, metrics_rows)
            self._diag(
                f"iteration {controller.state.global_iteration} complete in {metric.wall_seconds:.1f}s; "
                f"stage={stage_spec.stage_index + 1}:{stage_spec.stage_name} "
                f"stage_iter={controller.state.stage_iteration}; "
                f"rollout={rollout_seconds:.1f}s, ppo={update_seconds:.1f}s"
            )

            budget_due, budget_reason = controller.transition_due()
            validation_due = (
                controller.state.stage_iteration % self.settings.validation_every_iterations == 0
                or (budget_due and (self.settings.validate_at_stage_end or stage_spec.stage_index == 3))
            )
            if validation_due:
                stage_filter = self.phase_j.validation.stage_filters[stage_spec.stage_name]
                # A diagnostic smoke suite may intentionally be a strict subset of
                # the formal master suite. Intersect instead of inventing data.
                try:
                    stage_records = filter_validation_records(validation_records, stage_filter)
                except ValueError:
                    stage_records = tuple(validation_records)
                summary, instance_results = evaluate_fixed_validation(
                    self.cfg,
                    policy=policy,
                    normalizer=normalizer,
                    records=stage_records,
                    stage_index=stage_spec.stage_index,
                    deterministic=True,
                    device=self.device,
                    max_episode_decisions=self.settings.max_episode_decisions,
                    return_instance_results=True,
                    stop_after_first_failure=self.settings.validation_stop_after_first_failure,
                )
                append_validation_instance_csv(
                    self.validation_instance_path,
                    global_iteration=controller.state.global_iteration,
                    stage_index=stage_spec.stage_index,
                    stage_name=stage_spec.stage_name,
                    results=instance_results,
                )
                if summary.failed_instances:
                    failures = [r for r in instance_results if r.status != "completed"]
                    print(
                        f"[validation] stage={stage_spec.stage_index + 1} {stage_spec.stage_name}: "
                        f"{summary.failed_instances}/{summary.instances} instance(s) failed; "
                        f"selection_metric=inf",
                        flush=True,
                    )
                    for failure in failures[:5]:
                        print(
                            f"[validation] {failure.instance_id}: status={failure.status}, "
                            f"decisions={failure.decisions}, sim_time={failure.simulation_time:.3f}, "
                            f"reconfigurations={failure.reconfigurations}, "
                            f"orders={failure.completed_orders}/{failure.total_orders}, "
                            f"operations={failure.completed_operations}/{failure.total_operations}",
                            flush=True,
                        )
                    if len(failures) > 5:
                        print(
                            f"[validation] ... {len(failures) - 5} more failures; see "
                            f"{self.validation_instance_path}",
                            flush=True,
                        )
                validation_row = {
                    "global_iteration": controller.state.global_iteration,
                    **asdict(summary),
                }
                validation_rows.append(validation_row)
                _write_rows(self.validation_path, validation_rows)

                stage_best = self.run_dir / "checkpoints" / f"best_stage_{stage_spec.stage_index + 1}.pt"
                improved = controller.record_validation(summary.mean_twt, str(stage_best))
                if improved:
                    save_model_optimizer_snapshot(
                        stage_best,
                        policy=policy,
                        optimizer=agent.optimizer,
                        metric=summary.mean_twt,
                        stage_index=stage_spec.stage_index,
                        normalizer=normalizer,
                        global_iteration=controller.state.global_iteration,
                    )
                budget_due, budget_reason = controller.transition_due()

            if budget_due:
                reason = str(budget_reason)
                if (
                    self.settings.restore_best_before_next_stage
                    and controller.state.best_stage_checkpoint
                    and Path(controller.state.best_stage_checkpoint).is_file()
                ):
                    restore_model_optimizer_snapshot(
                        controller.state.best_stage_checkpoint,
                        policy=policy,
                        optimizer=agent.optimizer,
                        map_location=self.device,
                    )
                old_stage = controller.state.stage_index
                controller.advance(reason)
                if controller.state.finished:
                    if controller.state.best_stage_checkpoint:
                        best_model_path = self.run_dir / "checkpoints" / "best_model.pt"
                        shutil.copy2(controller.state.best_stage_checkpoint, best_model_path)
                else:
                    next_spec = build_stage_policy_spec(
                        self.cfg, self.phase_j, controller.state.stage_index
                    )
                    configure_curriculum_stage(policy, controller.state.stage_index)
                    vector_env = self._new_vector_env(next_spec)
                    collector = self._collector(vector_env, agent, normalizer, next_spec)

            if (
                controller.state.global_iteration % self.settings.checkpoint_every_iterations == 0
                or controller.state.finished
            ):
                save_training_checkpoint(
                    self.latest_path,
                    policy=policy,
                    optimizer=agent.optimizer,
                    normalizer=normalizer,
                    vector_env=vector_env,
                    collector_next_env=collector.next_env,
                    curriculum_state=controller.state_dict(),
                    metrics_rows=metrics_rows,
                    validation_rows=validation_rows,
                    run_settings=self.settings,
                )
                self._diag(f"6/6 checkpoint saved: {self.latest_path}")

            if (
                self.settings.stop_after_global_iterations is not None
                and controller.state.global_iteration >= int(self.settings.stop_after_global_iterations)
                and not controller.state.finished
            ):
                if not self.latest_path.is_file():
                    save_training_checkpoint(
                        self.latest_path,
                        policy=policy,
                        optimizer=agent.optimizer,
                        normalizer=normalizer,
                        vector_env=vector_env,
                        collector_next_env=collector.next_env,
                        curriculum_state=controller.state_dict(),
                        metrics_rows=metrics_rows,
                        validation_rows=validation_rows,
                        run_settings=self.settings,
                    )
                    self._diag(f"6/6 checkpoint saved: {self.latest_path}")
                break

        return PhaseJTrainingResult(
            run_dir=str(self.run_dir),
            finished=controller.state.finished,
            global_iterations=controller.state.global_iteration,
            final_stage_index=controller.state.stage_index,
            final_stage_name=controller.state.stage_name,
            metrics_rows=len(metrics_rows),
            validation_rows=len(validation_rows),
            latest_checkpoint=str(self.latest_path),
            best_model=None if best_model_path is None else str(best_model_path),
        )


__all__ = [
    "PhaseJIterationMetric", "PhaseJRunSettings", "PhaseJTrainer",
    "PhaseJTrainingResult",
]
