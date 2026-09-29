"""Teacher Scheme-2 trainer: one seed, online random instances, no curriculum."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import torch

from agent.policy import EGDMCompositePolicy
from agent.ppo import PPOAgent
from agent.training.checkpoint import (
    load_training_checkpoint,
    restore_rng_state,
    save_model_optimizer_snapshot,
    save_training_checkpoint,
    write_run_snapshot,
)
from agent.training.random_joint import (
    JOINT_STAGE_INDEX,
    JOINT_STAGE_NAME,
    RandomJointController,
    build_random_joint_policy_spec,
    configure_random_joint_training,
    select_balanced_validation_monitor,
    select_stratified_validation_monitor_9,
    select_parameter_case_validation_monitor,
)
from agent.training.trainer_j import (
    PhaseJIterationMetric,
    PhaseJRunSettings,
    PhaseJTrainer,
    PhaseJTrainingResult,
    _seed_everything,
    _write_rows,
)
from agent.training.validation import (
    append_validation_instance_csv,
    ensure_fixed_validation_suite,
    evaluate_fixed_validation,
)
from agent.training.runtime_metrics import gpu_runtime_snapshot
from agent.training.ppo_operator_profile import run_profiled_ppo_update
from agent.training.visdom_logger import VisdomLogger, settings_summary


class RandomJointTrainer(PhaseJTrainer):
    """Reuse the validated L1.7.6 execution stack under the new training protocol."""

    def __init__(
        self,
        cfg,
        phase_j,
        settings: PhaseJRunSettings,
        *,
        total_iterations: int,
        joint_scale_pool,
        joint_scenario_pool,
        joint_load_ratio_pool,
        joint_due_tightness_pool,
        config_paths,
        validate_at_end: bool = True,
        fresh_instances_each_iteration: bool = True,
        validation_monitor_subset: str = "full",
        validation_progress_every_instances: int = 0,
        instance_parameter_table_path: str | None = None,
        training_instance_parameter_table_path: str | None = None,
    ) -> None:
        super().__init__(cfg, phase_j, settings, config_paths=config_paths)
        self.total_iterations = int(total_iterations)
        if self.total_iterations <= 0:
            raise ValueError("Scheme 2 total_iterations must be positive")
        self.joint_spec = build_random_joint_policy_spec(
            scale_pool=joint_scale_pool,
            scenario_pool=joint_scenario_pool,
            load_ratio_pool=joint_load_ratio_pool,
            due_tightness_pool=joint_due_tightness_pool,
        )
        self.validate_at_end = bool(validate_at_end)
        self.fresh_instances_each_iteration = bool(fresh_instances_each_iteration)
        self.validation_monitor_subset = str(validation_monitor_subset)
        self.validation_progress_every_instances = max(0, int(validation_progress_every_instances))
        self.instance_parameter_table_path = instance_parameter_table_path
        if training_instance_parameter_table_path is not None:
            self.settings = type(self.settings)(
                **{
                    **asdict(self.settings),
                    "training_instance_parameter_table_path": str(training_instance_parameter_table_path),
                }
            )

    def _diag(self, message: str) -> None:
        if self.diagnostics_enabled:
            print(f"[S2] {message}", flush=True)

    def _build_policy(self, reference_graph):
        """Factory hook used by Scheme-2 learned baselines.

        The default keeps the accepted EGDM-HGPPO policy unchanged. Subclasses
        may replace only the policy architecture while reusing the identical
        random-joint data, rollout, PPO, validation, checkpoint and resume stack.
        """
        return EGDMCompositePolicy(self.cfg, reference_graph)

    def _configure_policy(self, policy):
        """Enable all trainable components from iteration 1."""
        state = configure_random_joint_training(policy)
        if state.frozen_parameters != 0:
            raise RuntimeError("Scheme 2 unexpectedly froze policy parameters")
        if hasattr(policy, "set_tensorized_decoder_scoring"):
            policy.set_tensorized_decoder_scoring(
                self.settings.tensorized_decoder_scoring
            )
        if hasattr(policy, "trusted_replay_gate_feasibility"):
            policy.trusted_replay_gate_feasibility = bool(
                self.settings.trusted_replay_normalization
            )
        for matcher_name in ("worker_matcher", "robot_matcher", "schedule_matcher"):
            matcher = getattr(policy, matcher_name, None)
            if matcher is not None and hasattr(matcher, "memoize_replay_candidate_scores"):
                matcher.memoize_replay_candidate_scores = bool(
                    self.settings.replay_candidate_score_memoization
                )
        return state

    def _configure_agent(self, agent) -> None:
        """Hook for controlled ablations; default formal Scheme-2 is unchanged."""
        return None

    def _write_scheme2_protocol_audit(self) -> None:
        payload = {
            "training_protocol": "teacher_scheme2_random_joint",
            "training_seed": int(self.settings.training_seed),
            "total_iterations": int(self.total_iterations),
            "scale_pool": list(self.joint_spec.scale_pool),
            "scenario_pool": list(self.joint_spec.scenario_pool),
            "load_ratio_pool": list(self.joint_spec.load_ratio_pool),
            "due_tightness_pool": list(self.joint_spec.due_tightness_pool),
            "curriculum_enabled": False,
            "fresh_instances_each_training_iteration": bool(self.fresh_instances_each_iteration),
            "instance_parameter_table_path": self.instance_parameter_table_path,
            "training_instance_parameter_table_path": self.settings.training_instance_parameter_table_path,
            "training_structural_source": (
                "scale_ranges"
                if self.settings.training_instance_parameter_table_path is None
                else "parameter_table"
            ),
            "all_policy_parameters_trainable_from_iteration_1": True,
            "action_constraints": "unconstrained_physical_feasibility_only",
            "ppo_components": "full",
            "fixed_validation_used_for_monitoring": True,
            "validation_monitor_subset": self.validation_monitor_subset,
            "validation_changes_training_distribution": False,
            "validation_early_stops_training": False,
            "test_visible_to_training": False,
            "execution_stack": "L1.7.6",
            "budget_profile": str(self.settings.budget_profile),
            "rollout_events_per_iteration": int(self.settings.rollout_events),
            "ppo_epochs": int(self.settings.ppo_epochs_override or self.cfg.algo.ppo_epochs),
            "parallel_envs": int(self.settings.parallel_envs),
            "rollout_executor": str(self.settings.rollout_executor),
            "hardware_profile_label": str(self.settings.hardware_profile_label),
            "tensorized_decoder_scoring": bool(
                self.settings.tensorized_decoder_scoring
            ),
            "replay_candidate_score_memoization": bool(
                self.settings.replay_candidate_score_memoization
            ),
            "live_static_processing_cache": True,
            "batched_fixed_validation": True,
            "trusted_validation_inputs": True,
            "validation_precision": "fp32",
            "visdom": {
                "enabled": bool(self.settings.visdom_enabled),
                "server": self.settings.visdom_server,
                "port": int(self.settings.visdom_port),
                "env_prefix": self.settings.visdom_env_prefix,
                "update_every_iterations": int(self.settings.visdom_update_every_iterations),
            },
        }
        (self.run_dir / "scheme2_protocol.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def run(self) -> PhaseJTrainingResult:
        _seed_everything(self.settings.training_seed)
        self._write_execution_profile_audit()
        self._write_scheme2_protocol_audit()
        controller = RandomJointController(
            self.total_iterations,
            min_delta=self.settings.validation_min_delta,
        )
        metrics_rows: list[dict] = []
        validation_rows: list[dict] = []
        visdom = VisdomLogger(
            enabled=self.settings.visdom_enabled,
            server=self.settings.visdom_server,
            port=self.settings.visdom_port,
            env_prefix=self.settings.visdom_env_prefix,
            run_name=self.settings.run_name,
            update_every_iterations=self.settings.visdom_update_every_iterations,
            config_summary=settings_summary(self.settings),
        )

        validation_pool = ensure_fixed_validation_suite(
            self.cfg,
            root=self.settings.validation_root,
            base_seed=self.settings.validation_base_seed,
            scales=self.settings.validation_scales,
            scenarios=self.settings.validation_scenarios,
            load_ratios=self.settings.validation_load_ratios,
            due_tightness=self.settings.validation_due_tightness,
            instances_per_combination=self.settings.validation_instances_per_combination,
            instance_parameter_table_path=self.instance_parameter_table_path,
            one_per_parameter_case=True,
        )
        if self.validation_monitor_subset == "balanced_scale_scenario_15":
            validation_records = select_balanced_validation_monitor(validation_pool)
        elif self.validation_monitor_subset == "stratified_fast_9":
            validation_records = select_stratified_validation_monitor_9(validation_pool)
        elif self.validation_monitor_subset == "parameter_cases":
            validation_records = select_parameter_case_validation_monitor(validation_pool)
        elif self.validation_monitor_subset == "full":
            validation_records = validation_pool
        else:
            raise ValueError(f"unsupported validation monitor subset: {self.validation_monitor_subset}")
        self._diag(
            f"fixed validation monitor ready: {len(validation_records)} instances "
            f"(pool={len(validation_pool)}, subset={self.validation_monitor_subset})"
        )

        if self.settings.resume_checkpoint is None:
            write_run_snapshot(
                self.run_dir,
                project_config=self.cfg,
                runtime_settings=self.settings,
                config_paths=self.config_paths,
            )
            self._diag("1/6 snapshot written; fitting training-only S/M/L normalizer")
            normalizer = self._fit_normalizer()
            self._diag("2/6 normalizer fitted; creating full random-joint vector environment")
            vector_env = self._new_vector_env(self.joint_spec)
            self._diag(
                f"3/6 vector env ready: mode={JOINT_STAGE_NAME}, envs={len(vector_env)}, "
                f"scales={self.joint_spec.scale_pool}, scenarios={self.joint_spec.scenario_pool}"
            )
            reference_graph = normalizer.transform(vector_env.graph(0))
            policy = self._build_policy(reference_graph).to(self.device)
            self._configure_policy(policy)
            agent = PPOAgent(self.cfg, policy)
            self._configure_agent(agent)
            if self.settings.replay_microbatch_size_override is not None:
                agent.replay_microbatch_size = int(self.settings.replay_microbatch_size_override)
                agent.safe_replay_microbatch_cap = int(agent.replay_microbatch_size)
            if self.settings.replay_microbatch_min_size_override is not None:
                agent.replay_microbatch_min_size = int(self.settings.replay_microbatch_min_size_override)
            agent.replay_microbatch_oom_fallback = bool(self.settings.replay_microbatch_oom_fallback)
            agent.trusted_replay_inputs = bool(self.settings.trusted_replay_normalization)
            agent.replay_microbatch_persist_oom_fallback = bool(
                self.settings.replay_microbatch_persist_oom_fallback
            )
            agent.cross_epoch_materialize_cache = bool(self.settings.cross_epoch_materialize_cache)
            agent.materialize_cache_max_bytes = int(self.settings.materialize_cache_max_mib) * 1024 * 1024
            agent.profile_update_timing = bool(self.settings.profile_ppo_update_timing)
            if self.settings.ppo_epochs_override is not None:
                agent.ppo_epochs = int(self.settings.ppo_epochs_override)
            collector = self._collector(vector_env, agent, normalizer, self.joint_spec)
        else:
            payload = load_training_checkpoint(self.settings.resume_checkpoint, map_location="cpu")
            controller.load_state_dict(payload["curriculum_state"])
            if controller.state.finished:
                raise ValueError("Scheme-2 checkpoint already completed its fixed training budget")
            metrics_rows = list(payload.get("metrics_rows", []))
            validation_rows = list(payload.get("validation_rows", []))
            normalizer = payload["normalizer"]
            vector_env = payload["vector_env"]
            reference_graph = normalizer.transform(vector_env.graph(0))
            policy = self._build_policy(reference_graph).to(self.device)
            self._configure_policy(policy)
            agent = PPOAgent(self.cfg, policy)
            self._configure_agent(agent)
            if self.settings.replay_microbatch_size_override is not None:
                agent.replay_microbatch_size = int(self.settings.replay_microbatch_size_override)
                agent.safe_replay_microbatch_cap = int(agent.replay_microbatch_size)
            if self.settings.replay_microbatch_min_size_override is not None:
                agent.replay_microbatch_min_size = int(self.settings.replay_microbatch_min_size_override)
            agent.replay_microbatch_oom_fallback = bool(self.settings.replay_microbatch_oom_fallback)
            agent.trusted_replay_inputs = bool(self.settings.trusted_replay_normalization)
            agent.replay_microbatch_persist_oom_fallback = bool(
                self.settings.replay_microbatch_persist_oom_fallback
            )
            agent.cross_epoch_materialize_cache = bool(self.settings.cross_epoch_materialize_cache)
            agent.materialize_cache_max_bytes = int(self.settings.materialize_cache_max_mib) * 1024 * 1024
            agent.profile_update_timing = bool(self.settings.profile_ppo_update_timing)
            if self.settings.ppo_epochs_override is not None:
                agent.ppo_epochs = int(self.settings.ppo_epochs_override)
            policy.load_state_dict(payload["policy_state"])
            agent.optimizer.load_state_dict(payload["optimizer_state"])
            collector = self._collector(vector_env, agent, normalizer, self.joint_spec)
            collector.next_env = int(payload["collector_next_env"])
            restore_rng_state(payload["rng_state"])

        best_model_path = self.run_dir / "checkpoints" / "best_model.pt"
        final_model_path = self.run_dir / "checkpoints" / "final_model.pt"

        while not controller.state.finished:
            # Default EGDM semantics remain configure_random_joint_training(policy);
            # the hook lets comparison policies reuse this execution stack unchanged.
            self._configure_policy(policy)
            progress = controller.global_progress()
            agent.set_progress(progress)
            # Teacher Scheme 2 defines each PPO cycle from newly sampled S/M/L
            # range instances. Do not carry partially progressed instances across
            # iterations; that was an L1.6.8 stateful optimization for the old
            # curriculum protocol and caused graph/replay size to grow each cycle.
            if self.fresh_instances_each_iteration and controller.state.global_iteration > 0:
                vector_env.reset_all()
                collector.next_env = 0
                self._diag(
                    f"fresh random instances sampled for iteration "
                    f"{controller.state.global_iteration + 1}"
                )
            started = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self._diag(
                f"4/6 collecting {self.settings.rollout_events} events; random-joint full policy; "
                f"executor={('multiprocess_strategy1' if self.settings.use_multiprocess_rollout else self.settings.rollout_executor)}"
            )
            rollout_started = time.perf_counter()
            rollout = self._collect_training_rollout(
                vector_env=vector_env,
                collector=collector,
                policy=policy,
                normalizer=normalizer,
                stage_spec=self.joint_spec,
                global_iteration=controller.state.global_iteration,
            )
            rollout_seconds = time.perf_counter() - rollout_started
            self._diag(
                f"5/6 rollout complete in {rollout_seconds:.1f}s "
                f"({self.settings.rollout_events / max(rollout_seconds, 1e-9):.2f} events/s); "
                f"replay~{rollout.stats.estimated_replay_megabytes:.1f} MiB; starting PPO"
            )
            update_started = time.perf_counter()
            update_call = lambda: agent.update(
                rollout.buffer,
                bootstrap_values=rollout.bootstrap_values,
                progress=progress,
                components=self.joint_spec.components,
            )
            operator_profile = None
            if self.settings.profile_ppo_update_timing:
                update, operator_profile = run_profiled_ppo_update(
                    update_call,
                    output_dir=self.run_dir / "operator_profile",
                    use_cuda=self.device.type == "cuda",
                    set_step_callback=lambda callback: setattr(
                        agent, "operator_profiler_step", callback
                    ),
                )
            else:
                update = update_call()
            update_seconds = time.perf_counter() - update_started
            peak_mb = (
                torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
                if self.device.type == "cuda" else 0.0
            )
            peak_reserved_mb = (
                torch.cuda.max_memory_reserved(self.device) / (1024.0 * 1024.0)
                if self.device.type == "cuda" else 0.0
            )
            telemetry = gpu_runtime_snapshot(
                self.device,
                peak_alloc_bytes=int(peak_mb * 1024.0 * 1024.0),
                peak_reserved_bytes=int(peak_reserved_mb * 1024.0 * 1024.0),
            )
            controller.record_iteration()
            twts = [s.twt for s in rollout.episode_summaries]
            metric = PhaseJIterationMetric(
                global_iteration=controller.state.global_iteration,
                stage_index=JOINT_STAGE_INDEX,
                stage_name=JOINT_STAGE_NAME,
                stage_iteration=controller.state.global_iteration,
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
                "sps": float(rollout.stats.events) / max(float(metric.wall_seconds), 1e-9),
                "gpu_util": float(telemetry["gpu_util"]),
                "gpu_mem_gb": float(telemetry["gpu_mem_gb"]),
                "gpu_mem_reserved_gb": float(telemetry["gpu_mem_reserved_gb"]),
                "optimizer_steps": int(update.optimizer_steps),
                "cuda_peak_alloc_mib": float(peak_mb),
                # Runtime action-space audit for teacher Scheme 2.  These are
                # observational diagnostics only; they do not alter sampling,
                # masks, replay, PPO loss, or optimizer behavior.
                "reconfiguration_fraction": float(rollout.stats.reconfiguration_fraction),
                "max_worker_moves": int(rollout.stats.max_worker_moves),
                "max_robot_moves": int(rollout.stats.max_robot_moves),
                "configured_replay_microbatch": int(self.settings.replay_microbatch_size_override or agent.replay_microbatch_size),
                "effective_replay_microbatch": int(agent.last_effective_replay_microbatch_size),
                "safe_replay_microbatch_cap": int(agent.safe_replay_microbatch_cap),
                "oom_fallback_count": int(agent.last_oom_fallback_count),
                "materialize_cache": agent.last_materialize_cache_stats,
                "ppo_update_profile": agent.last_update_profile,
                "torch_operator_profile": operator_profile,
            }
            metrics_rows.append({
                **asdict(metric),
                "rollout_seconds": float(rollout_seconds),
                "ppo_seconds": float(update_seconds),
                "sps": float(rollout.stats.events) / max(float(metric.wall_seconds), 1e-9),
                "gpu_util": float(telemetry["gpu_util"]),
                "gpu_mem_gb": float(telemetry["gpu_mem_gb"]),
                "gpu_mem_reserved_gb": float(telemetry["gpu_mem_reserved_gb"]),
                "optimizer_steps": int(update.optimizer_steps),
                "configured_replay_microbatch": int(
                    self.settings.replay_microbatch_size_override or agent.replay_microbatch_size
                ),
                "effective_replay_microbatch": int(agent.last_effective_replay_microbatch_size),
                "safe_replay_microbatch_cap": int(agent.safe_replay_microbatch_cap),
                "oom_fallback_count": int(agent.last_oom_fallback_count),
            })
            _write_rows(self.metrics_path, metrics_rows)
            visdom.training(
                iteration=controller.state.global_iteration,
                metrics=metrics_rows[-1],
            )
            self._diag(
                f"iteration {controller.state.global_iteration}/{self.total_iterations} complete "
                f"in {metric.wall_seconds:.1f}s; rollout={rollout_seconds:.1f}s, ppo={update_seconds:.1f}s; "
                f"microbatch={agent.last_effective_replay_microbatch_size}; "
                f"oom_fallbacks={agent.last_oom_fallback_count}"
            )

            validation_due = (
                controller.state.global_iteration % self.settings.validation_every_iterations == 0
                or (controller.state.finished and self.validate_at_end)
            )
            if validation_due:
                summary, instance_results = evaluate_fixed_validation(
                    self.cfg,
                    policy=policy,
                    normalizer=normalizer,
                    records=validation_records,
                    stage_index=JOINT_STAGE_INDEX,
                    deterministic=True,
                    device=self.device,
                    max_episode_decisions=self.settings.max_episode_decisions,
                    return_instance_results=True,
                    stop_after_first_failure=self.settings.validation_stop_after_first_failure,
                    constraints_override=self.joint_spec.constraints,
                    stage_name_override=JOINT_STAGE_NAME,
                    progress_every_instances=self.validation_progress_every_instances,
                    progress_prefix="S2-val",
                )
                append_validation_instance_csv(
                    self.validation_instance_path,
                    global_iteration=controller.state.global_iteration,
                    stage_index=JOINT_STAGE_INDEX,
                    stage_name=JOINT_STAGE_NAME,
                    results=instance_results,
                )
                validation_row = {"global_iteration": controller.state.global_iteration, **asdict(summary)}
                validation_rows.append(validation_row)
                _write_rows(self.validation_path, validation_rows)
                visdom.validation(
                    iteration=controller.state.global_iteration,
                    mean_twt=summary.mean_twt,
                )
                improved = controller.record_validation(summary.mean_twt, str(best_model_path))
                if improved:
                    save_model_optimizer_snapshot(
                        best_model_path,
                        policy=policy,
                        optimizer=agent.optimizer,
                        metric=summary.mean_twt,
                        stage_index=JOINT_STAGE_INDEX,
                        normalizer=normalizer,
                        global_iteration=controller.state.global_iteration,
                    )
                self._diag(
                    f"validation: n={summary.instances}, failed={summary.failed_instances}, "
                    f"mean_twt={summary.mean_twt}, improved={improved}"
                )

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

        # Teacher protocol: freeze the network after the fixed training budget.
        final_metric = (
            float(controller.state.best_validation_metric)
            if controller.state.best_validation_metric is not None else float("nan")
        )
        save_model_optimizer_snapshot(
            final_model_path,
            policy=policy,
            optimizer=agent.optimizer,
            metric=final_metric,
            stage_index=JOINT_STAGE_INDEX,
            normalizer=normalizer,
            global_iteration=controller.state.global_iteration,
        )
        return PhaseJTrainingResult(
            run_dir=str(self.run_dir),
            finished=True,
            global_iterations=controller.state.global_iteration,
            final_stage_index=JOINT_STAGE_INDEX,
            final_stage_name=JOINT_STAGE_NAME,
            metrics_rows=len(metrics_rows),
            validation_rows=len(validation_rows),
            latest_checkpoint=str(self.latest_path),
            best_model=(str(best_model_path) if best_model_path.is_file() else None),
        )


__all__ = ["RandomJointTrainer"]
