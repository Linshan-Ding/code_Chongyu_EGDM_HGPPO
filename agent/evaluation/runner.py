"""Shared fixed-test runner for learned policies and non-learning baselines.

Learned policies may expose ``act_batch``. In that case independent fixed-test
environments share the graph-encoder forward while retaining per-instance
autoregressive decoding and simulator transitions. The resulting decision
latency fields are amortized per environment within each batch wave.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
import time

from data.io import load_instance_csv
from environment.env import AssemblyEnv
from agent.evaluation.metrics import (
    EVAL_FIELDS,
    EpisodeMetrics,
    failed_episode_metrics,
    final_episode_metrics,
)


@dataclass(slots=True)
class _BatchedEpisode:
    """Mutable accounting for one independent fixed-test environment."""

    record: object
    env: AssemblyEnv
    reward_sum: float = 0.0
    decision_times_ms: list[float] = field(default_factory=list)
    stage_cv_values: list[float] = field(default_factory=list)
    worker_relocation: float = 0.0
    robot_relocation: float = 0.0
    moved_resources: int = 0
    decisions: int = 0
    # Batch-shared model time is charged amortized to this environment; local
    # graph/diagnostic/step time is charged directly.
    wall_time_seconds: float = 0.0


def _stage_cv(env) -> float:
    graph = env.graph()
    try:
        idx = graph.gate_continuous_names.index("stage_workload_cv")
    except ValueError as exc:
        raise RuntimeError("graph missing stage_workload_cv") from exc
    return float(graph.gate_continuous[0, idx].item())


def _relocation_diagnostics(env, action) -> tuple[float, float, int]:
    context = env.action_context()
    worker_time = 0.0
    robot_time = 0.0
    moved = 0
    for assignment in action.worker_assignments:
        rid = int(assignment.resource_id)
        target = int(assignment.target_cell)
        if target < 0:
            continue
        source = int(context.worker_physical_cell[rid].item())
        if source != target:
            worker_time += float(context.worker_relocation_time[rid, source, target].item())
            moved += 1
    for assignment in action.robot_assignments:
        rid = int(assignment.resource_id)
        target = int(assignment.target_cell)
        if target < 0:
            continue
        source = int(context.robot_physical_cell[rid].item())
        if source != target:
            robot_time += float(context.robot_relocation_time[rid, source, target].item())
            moved += 1
    return worker_time, robot_time, moved


def _progress_signature(env) -> tuple[float, int, int, int]:
    """Return simulator progress counters independent of decision index."""
    completed_orders = sum(1 for order in env.sim.orders if order.completed)
    completed_operations = sum(
        1 for operation in env.sim.operations
        if getattr(operation.status, "value", operation.status) == "DONE"
    )
    return (
        float(env.sim.time),
        int(completed_orders),
        int(completed_operations),
        int(env.sim.reconfiguration_count),
    )


def evaluate_episode(
    cfg,
    *,
    instance_path: str | Path,
    policy,
    run_id: str,
    tier: str,
    max_episode_decisions: int,
    stagnation_patience: int = 0,
    obj_ref: float | None = None,
) -> EpisodeMetrics:
    instance = load_instance_csv(instance_path)
    env = AssemblyEnv(cfg)
    env.reset(instance)
    policy.reset(env)
    reward_sum = 0.0
    decision_times_ms: list[float] = []
    stage_cv_values: list[float] = []
    worker_relocation = 0.0
    robot_relocation = 0.0
    moved_resources = 0
    wall_start = time.perf_counter()

    decisions = 0
    stagnant_decisions = 0
    previous_progress = _progress_signature(env)
    while not env.sim.is_done:
        stage_cv_values.append(_stage_cv(env))
        start = time.perf_counter()
        action = policy.act(env)
        decision_times_ms.append((time.perf_counter() - start) * 1000.0)
        wh, rr, moved = _relocation_diagnostics(env, action)
        worker_relocation += wh
        robot_relocation += rr
        moved_resources += moved
        _, reward, done, info = env.step(action)
        reward_sum += float(info["reward_twt"])
        decisions += 1
        current_progress = _progress_signature(env)
        if float(info.get("delta_t", 0.0)) > 1.0e-12 or current_progress != previous_progress:
            stagnant_decisions = 0
        else:
            stagnant_decisions += 1
        previous_progress = current_progress
        if int(stagnation_patience) > 0 and stagnant_decisions >= int(stagnation_patience):
            return failed_episode_metrics(
                env=env,
                method=policy.method_name,
                run_id=run_id,
                tier=tier,
                decisions=decisions,
                wall_time_seconds=time.perf_counter() - wall_start,
                obj_ref=obj_ref,
                failure_reason=(
                    f"stagnation_patience={int(stagnation_patience)} reached"
                ),
                decision_times_ms=decision_times_ms,
            )
        if decisions > int(max_episode_decisions):
            return failed_episode_metrics(
                env=env,
                method=policy.method_name,
                run_id=run_id,
                tier=tier,
                decisions=decisions,
                wall_time_seconds=time.perf_counter() - wall_start,
                obj_ref=obj_ref,
                failure_reason=(
                    f"exceeded max_episode_decisions={int(max_episode_decisions)}"
                ),
                decision_times_ms=decision_times_ms,
            )
        if done:
            break

    metrics = final_episode_metrics(
        env=env,
        method=policy.method_name,
        run_id=run_id,
        tier=tier,
        reward_sum=reward_sum,
        decision_times_ms=decision_times_ms,
        stage_cv_values=stage_cv_values,
        worker_relocation_time=worker_relocation,
        robot_relocation_time=robot_relocation,
        moved_resources=moved_resources,
        wall_time_seconds=time.perf_counter() - wall_start,
        obj_ref=obj_ref,
    )
    if metrics.reward_identity_error > 1e-6:
        raise RuntimeError(
            f"reward identity failed for {instance.instance_id}/{policy.method_name}: "
            f"{metrics.reward_identity_error}"
        )
    return metrics


def _finalize_batched_episode(
    state: _BatchedEpisode,
    *,
    policy,
    run_id: str,
    tier: str,
    obj_ref: float | None,
) -> EpisodeMetrics:
    metrics = final_episode_metrics(
        env=state.env,
        method=policy.method_name,
        run_id=run_id,
        tier=tier,
        reward_sum=state.reward_sum,
        # A shared encoder forward is charged equally to its independent
        # environments, making this an amortized per-environment latency.
        decision_times_ms=state.decision_times_ms,
        stage_cv_values=state.stage_cv_values,
        worker_relocation_time=state.worker_relocation,
        robot_relocation_time=state.robot_relocation,
        moved_resources=state.moved_resources,
        wall_time_seconds=state.wall_time_seconds,
        obj_ref=obj_ref,
    )
    if metrics.reward_identity_error > 1e-6:
        raise RuntimeError(
            f"reward identity failed for {state.record.instance_id}/{policy.method_name}: "
            f"{metrics.reward_identity_error}"
        )
    return metrics


def _evaluate_batched_policy(
    cfg,
    *,
    records,
    policy,
    run_id: str,
    tier: str,
    max_episode_decisions: int,
    stagnation_patience: int = 0,
    references: dict[str, float],
    batch_size: int,
) -> list[EpisodeMetrics]:
    """Evaluate a learned policy with one shared encoder forward per wave.

    Environment transitions remain individual and use the same ``env.step`` as
    serial evaluation. Batching changes only how independent current graphs are
    passed to the already-batched learned-policy encoder.
    """
    states: list[_BatchedEpisode] = []
    for record in records:
        env = AssemblyEnv(cfg)
        env.reset(load_instance_csv(record.path))
        policy.reset(env)
        states.append(_BatchedEpisode(record=record, env=env))

    completed: dict[int, EpisodeMetrics] = {}
    active = list(range(len(states)))
    stagnant_decisions = {index: 0 for index in active}
    previous_progress = {index: _progress_signature(states[index].env) for index in active}
    while active:
        next_active: list[int] = []
        for offset in range(0, len(active), batch_size):
            indices = active[offset:offset + batch_size]
            batch = [states[index] for index in indices]
            for state in batch:
                cv_start = time.perf_counter()
                state.stage_cv_values.append(_stage_cv(state.env))
                state.wall_time_seconds += time.perf_counter() - cv_start

            start = time.perf_counter()
            actions = tuple(policy.act_batch(tuple(state.env for state in batch)))
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if len(actions) != len(batch):
                raise RuntimeError(
                    f"{policy.method_name}.act_batch returned {len(actions)} actions for "
                    f"{len(batch)} environments"
                )

            amortized_ms = elapsed_ms / len(batch)
            for index, state, action in zip(indices, batch, actions, strict=True):
                state.decision_times_ms.append(amortized_ms)
                step_start = time.perf_counter()
                wh, rr, moved = _relocation_diagnostics(state.env, action)
                state.worker_relocation += wh
                state.robot_relocation += rr
                state.moved_resources += moved
                _, _reward, done, info = state.env.step(action)
                state.wall_time_seconds += (elapsed_ms / 1000.0) / len(batch)
                state.wall_time_seconds += time.perf_counter() - step_start
                state.reward_sum += float(info["reward_twt"])
                state.decisions += 1
                current_progress = _progress_signature(state.env)
                if float(info.get("delta_t", 0.0)) > 1.0e-12 or current_progress != previous_progress[index]:
                    stagnant_decisions[index] = 0
                else:
                    stagnant_decisions[index] += 1
                previous_progress[index] = current_progress
                if int(stagnation_patience) > 0 and stagnant_decisions[index] >= int(stagnation_patience):
                    completed[index] = failed_episode_metrics(
                        env=state.env,
                        method=policy.method_name,
                        run_id=run_id,
                        tier=tier,
                        decisions=state.decisions,
                        wall_time_seconds=state.wall_time_seconds,
                        obj_ref=references.get(state.record.instance_id),
                        failure_reason=(
                            f"stagnation_patience={int(stagnation_patience)} reached"
                        ),
                        decision_times_ms=state.decision_times_ms,
                    )
                    continue
                if state.decisions > int(max_episode_decisions):
                    completed[index] = failed_episode_metrics(
                        env=state.env,
                        method=policy.method_name,
                        run_id=run_id,
                        tier=tier,
                        decisions=state.decisions,
                        wall_time_seconds=state.wall_time_seconds,
                        obj_ref=references.get(state.record.instance_id),
                        failure_reason=(
                            f"exceeded max_episode_decisions={int(max_episode_decisions)}"
                        ),
                        decision_times_ms=state.decision_times_ms,
                    )
                elif done:
                    completed[index] = _finalize_batched_episode(
                        state,
                        policy=policy,
                        run_id=run_id,
                        tier=tier,
                        obj_ref=references.get(state.record.instance_id),
                    )
                else:
                    next_active.append(index)
        active = next_active

    return [completed[index] for index in range(len(states))]


def evaluate_records(
    cfg,
    *,
    records,
    policies,
    run_id: str,
    tier: str,
    max_episode_decisions: int,
    stagnation_patience: int = 0,
    references: dict[str, float] | None = None,
    learned_policy_batch_size: int = 32,
) -> list[EpisodeMetrics]:
    references = references or {}
    records, policies = tuple(records), tuple(policies)
    if int(learned_policy_batch_size) <= 0:
        raise ValueError("learned_policy_batch_size must be positive")

    # Store by positional index so callers retain the historical record-major
    # CSV ordering even when a learned method is evaluated first.
    rows_by_pair: dict[tuple[int, int], EpisodeMetrics] = {}
    for policy_index, policy in enumerate(policies):
        act_batch = getattr(policy, "act_batch", None)
        if callable(act_batch):
            metrics = _evaluate_batched_policy(
                cfg,
                records=records,
                policy=policy,
                run_id=run_id,
                tier=tier,
                max_episode_decisions=max_episode_decisions,
                stagnation_patience=stagnation_patience,
                references=references,
                batch_size=int(learned_policy_batch_size),
            )
            for record_index, metric in enumerate(metrics):
                rows_by_pair[(record_index, policy_index)] = metric
            continue

        for record_index, record in enumerate(records):
            rows_by_pair[(record_index, policy_index)] = evaluate_episode(
                cfg,
                instance_path=record.path,
                policy=policy,
                run_id=run_id,
                tier=tier,
                max_episode_decisions=max_episode_decisions,
                stagnation_patience=stagnation_patience,
                obj_ref=references.get(record.instance_id),
            )
    return [
        rows_by_pair[(record_index, policy_index)]
        for record_index in range(len(records))
        for policy_index in range(len(policies))
    ]


def write_eval_csv(path: str | Path, rows: list[EpisodeMetrics]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    return path


__all__ = ["evaluate_episode", "evaluate_records", "write_eval_csv"]
