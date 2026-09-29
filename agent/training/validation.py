"""Fixed validation-suite materialization and deterministic batched evaluation."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

import numpy as np
import torch

from agent.constraints import PolicyActionConstraints
from data.generator import InstanceGenerator, load_parameter_table
from data.io import load_instance_csv, save_instance_csv
from environment.entities import OperationStatus
from environment.env import AssemblyEnv, DeadlockError
from agent.training.curriculum import stage_action_constraints
from agent.training.numerics import assert_reward_identity, stable_sum


VALIDATION_INDEX_FIELDS = [
    "instance_id", "relative_path", "scale", "scenario", "generation_seed",
    "target_load_ratio", "due_tightness", "num_orders", "num_cells",
    "num_workers", "num_robots", "num_stages", "num_products",
    "relocation_multiplier", "worker_skill_density", "robot_capability_density",
    "S", "M", "A", "J", "H", "R", "V", "parameter_case_id",
]


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    instance_id: str
    path: str
    scale: str
    scenario: str
    generation_seed: int
    target_load_ratio: float
    due_tightness: str
    parameter_case_id: str | None = None
    num_cells: int | None = None
    num_stages: int | None = None
    num_robots: int | None = None
    num_orders: int | None = None
    num_workers: int | None = None
    num_products: int | None = None


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    stage_index: int
    stage_name: str
    instances: int
    completed_instances: int
    failed_instances: int
    nonterminating_instances: int
    deadlocked_instances: int
    failure_rate: float
    # Model-selection metric. Any policy-level failure makes this non-finite so
    # an incomplete policy can never become a best checkpoint.
    mean_twt: float
    median_twt: float
    # Finite diagnostic over completed instances only; not used for selection.
    mean_completed_twt: float | None
    mean_decisions: float
    mean_reconfigurations: float
    max_reward_identity_error: float


@dataclass(frozen=True, slots=True)
class ValidationInstanceResult:
    instance_id: str
    scale: str
    scenario: str
    target_load_ratio: float
    due_tightness: str
    status: Literal["completed", "non_terminating", "deadlock"]
    twt: float | None
    decisions: int
    reconfigurations: int
    simulation_time: float
    completed_orders: int
    total_orders: int
    completed_operations: int
    total_operations: int
    reward_identity_error: float | None
    failure_reason: str
    parameter_case_id: str | None = None
    M: int | None = None
    S: int | None = None
    J: int | None = None
    H: int | None = None
    R: int | None = None
    V: int | None = None


def _seed_for(base_seed: int, combo_index: int, replicate: int) -> int:
    seq = np.random.SeedSequence([int(base_seed), int(combo_index), int(replicate)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def ensure_fixed_validation_suite(
    cfg,
    *,
    root: str | Path,
    base_seed: int,
    scales,
    scenarios,
    load_ratios,
    due_tightness,
    instances_per_combination: int,
    instance_parameter_table_path: str | Path | None = None,
    one_per_parameter_case: bool = False,
) -> tuple[ValidationRecord, ...]:
    """Create the fixed suite once and reuse existing CSVs thereafter."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    count = int(instances_per_combination)
    if count <= 0:
        raise ValueError("instances_per_combination must be positive")
    generator = InstanceGenerator(cfg)
    records: list[ValidationRecord] = []
    combo = 0
    table = load_parameter_table(instance_parameter_table_path) if instance_parameter_table_path else None
    cases = (
        [(case_id, params) for case_id, params in table.items() if str(params["scale"]) in set(map(str, scales))]
        if table is not None else
        [(None, {"scale": scale}) for scale in scales]
    )
    if not cases:
        raise ValueError("validation parameter table has no cases matching requested scales")
    for case_index, (case_id, case_params) in enumerate(cases):
        scale = str(case_params["scale"])
        factor_cells = (
            [
                (
                    tuple(scenarios)[case_index % len(tuple(scenarios))],
                    tuple(load_ratios)[case_index % len(tuple(load_ratios))],
                    tuple(due_tightness)[case_index % len(tuple(due_tightness))],
                    0,
                )
            ]
            if one_per_parameter_case else
            [
                (scenario, rho, due, rep)
                for scenario in scenarios
                for rho in load_ratios
                for due in due_tightness
                for rep in range(count)
            ]
        )
        for scenario, rho, due, rep in factor_cells:
            seed = _seed_for(base_seed, combo, rep)
            case_label = "base" if case_id is None else str(case_id)
            instance_id = (
                f"val_{case_label}_{scenario}_rho{int(round(float(rho)*100)):03d}_"
                f"{due}_{rep:02d}"
            )
            path = root / f"{instance_id}.csv"
            if path.exists():
                instance = load_instance_csv(path)
                # A fixed file is authoritative; fail rather than silently replacing it.
                expected = (str(scale), str(scenario), float(rho), str(due))
                actual = (
                    instance.scale, instance.scenario,
                    float(instance.target_load_ratio), instance.due_tightness,
                )
                if actual != expected:
                    raise ValueError(
                        f"existing validation file {path} does not match requested cell: "
                        f"expected={expected}, actual={actual}"
                    )
            else:
                params = None if case_id is None else dict(case_params)
                instance = generator.sample(
                    scale=str(scale), scenario=str(scenario), seed=seed,
                    load_ratio=float(rho), due_tightness=str(due),
                    instance_id=instance_id,
                    instance_parameters=params,
                )
                save_instance_csv(instance, path)
            records.append(ValidationRecord(
                instance_id=instance.instance_id,
                path=str(path),
                scale=instance.scale,
                scenario=instance.scenario,
                generation_seed=seed,
                target_load_ratio=float(instance.target_load_ratio),
                due_tightness=instance.due_tightness,
                parameter_case_id=(None if case_id is None else str(case_id)),
                num_cells=instance.num_cells,
                num_stages=instance.num_stages,
                num_robots=instance.num_robots,
                num_orders=instance.num_orders,
                num_workers=instance.num_workers,
                num_products=instance.num_products,
            ))
            combo += 1

    index_path = root / "validation_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VALIDATION_INDEX_FIELDS)
        writer.writeheader()
        for record in records:
            inst = load_instance_csv(record.path)
            writer.writerow({
                "instance_id": record.instance_id,
                "relative_path": Path(record.path).relative_to(root).as_posix(),
                "scale": record.scale,
                "scenario": record.scenario,
                "generation_seed": record.generation_seed,
                "target_load_ratio": record.target_load_ratio,
                "due_tightness": record.due_tightness,
                "num_orders": inst.num_orders,
                "num_cells": inst.num_cells,
                "num_workers": inst.num_workers,
                "num_robots": inst.num_robots,
                "num_stages": inst.num_stages,
                "num_products": inst.num_products,
                "relocation_multiplier": inst.relocation_multiplier,
                "worker_skill_density": inst.worker_skill_density,
                "robot_capability_density": inst.robot_capability_density,
                "S": inst.num_stages,
                "M": inst.num_cells,
                # Deprecated alias for pre-Scheme-2 validation consumers.
                "A": inst.num_stages,
                "J": inst.num_orders,
                "H": inst.num_workers,
                "R": inst.num_robots,
                "V": inst.num_products,
                "parameter_case_id": record.parameter_case_id or "",
            })
    return tuple(records)


def filter_validation_records(records, stage_filter: dict) -> tuple[ValidationRecord, ...]:
    scales = set(stage_filter["scales"])
    scenarios = set(stage_filter["scenarios"])
    rhos = {round(float(x), 8) for x in stage_filter["load_ratios"]}
    out = tuple(
        r for r in records
        if r.scale in scales
        and r.scenario in scenarios
        and round(float(r.target_load_ratio), 8) in rhos
    )
    if not out:
        raise ValueError("stage validation filter selected no fixed instances")
    return out


def _validation_summary_from_results(
    results: list[ValidationInstanceResult],
    *,
    stage_index: int,
    stage_name_override: str | None,
) -> ValidationSummary:
    twts = [float(result.twt) for result in results if result.twt is not None]
    completed = sum(1 for result in results if result.status == "completed")
    nonterm = sum(1 for result in results if result.status == "non_terminating")
    deadlocked = sum(1 for result in results if result.status == "deadlock")
    failed = nonterm + deadlocked
    total = len(results)
    finite_mean = None if not twts else float(mean(twts))
    reward_errors = [
        float(result.reward_identity_error)
        for result in results
        if result.reward_identity_error is not None
    ]
    return ValidationSummary(
        stage_index=int(stage_index),
        stage_name=(str(stage_name_override) if stage_name_override is not None else (
            "fixed_configuration_warmup",
            "single_resource_reconfiguration",
            "full_set_reconfiguration",
            "multi_scale_joint_finetuning",
        )[int(stage_index)]),
        instances=total,
        completed_instances=completed,
        failed_instances=failed,
        nonterminating_instances=nonterm,
        deadlocked_instances=deadlocked,
        failure_rate=(0.0 if total == 0 else float(failed / total)),
        mean_twt=(float("inf") if failed else float(finite_mean)),
        median_twt=(
            float("inf") if failed
            else float(np.median(np.asarray(twts, dtype=float)))
        ),
        mean_completed_twt=finite_mean,
        mean_decisions=float(mean([result.decisions for result in results])),
        mean_reconfigurations=float(mean([result.reconfigurations for result in results])),
        max_reward_identity_error=float(max(reward_errors, default=0.0)),
    )


def _evaluate_fixed_validation_batched(
    cfg,
    *,
    policy,
    normalizer,
    records,
    stage_index: int,
    deterministic: bool,
    max_episode_decisions: int,
    return_instance_results: bool,
    stop_after_first_failure: bool,
    constraints: PolicyActionConstraints,
    stage_name_override: str | None,
    progress_every_instances: int,
    progress_prefix: str,
):
    """Advance active fixed instances together and batch every policy forward."""
    states = []
    for record_index, record in enumerate(records):
        instance = load_instance_csv(record.path)
        env = AssemblyEnv(cfg)
        env.reset(instance)
        states.append({
            "record_index": int(record_index),
            "record": record,
            "instance": instance,
            "env": env,
            "reward_terms": [],
            "decisions": 0,
        })
    active = list(range(len(states)))
    completed_results: dict[int, ValidationInstanceResult] = {}
    suite_started = time.perf_counter()
    progress_count = 0

    while active:
        transform_graph = getattr(
            normalizer, "transform_replay_trusted", normalizer.transform
        )
        graphs = [transform_graph(states[index]["env"].graph()) for index in active]
        contexts = [states[index]["env"].action_context() for index in active]
        act_kwargs = {
            "deterministic": bool(deterministic),
            "constraints": constraints,
        }
        if isinstance(policy, torch.nn.Module):
            act_kwargs["validate_inputs"] = False
        outputs = policy.act_batch(graphs, contexts, **act_kwargs)
        if len(outputs) != len(active):
            raise RuntimeError("batched validation policy returned wrong action count")
        next_active: list[int] = []
        failure_seen = False
        for position, index in enumerate(active):
            state = states[index]
            env = state["env"]
            instance = state["instance"]
            record = state["record"]
            state["decisions"] += 1
            status: Literal["completed", "non_terminating", "deadlock"] | None = None
            failure_reason = ""
            twt: float | None = None
            identity_error: float | None = None
            try:
                _, _, done, info = env.step(outputs[position].action)
            except DeadlockError as exc:
                done = False
                info = None
                status = "deadlock"
                failure_reason = str(exc)
            if status is None:
                state["reward_terms"].append(float(info["reward_twt"]))
                if done:
                    twt = float(info["twt"])
                    identity_error = assert_reward_identity(
                        stable_sum(state["reward_terms"]),
                        twt,
                        context=(
                            "validation reward identity failed for "
                            f"{instance.instance_id}"
                        ),
                    )
                    status = "completed"
                elif state["decisions"] >= int(max_episode_decisions):
                    status = "non_terminating"
                    failure_reason = (
                        "exceeded validation safety limit of "
                        f"{int(max_episode_decisions)} decisions"
                    )
            if status is None:
                next_active.append(index)
                continue

            reconfiguration_count = int(env.sim.reconfiguration_count)
            completed_orders = sum(1 for order in env.sim.orders if order.completed)
            completed_operations = sum(
                1 for operation in env.sim.operations
                if operation.status == OperationStatus.DONE
            )
            result = ValidationInstanceResult(
                instance_id=instance.instance_id,
                scale=instance.scale,
                scenario=instance.scenario,
                target_load_ratio=float(instance.target_load_ratio),
                due_tightness=instance.due_tightness,
                status=status,
                twt=twt,
                decisions=int(state["decisions"]),
                reconfigurations=reconfiguration_count,
                simulation_time=float(env.sim.time),
                completed_orders=int(completed_orders),
                total_orders=int(len(env.sim.orders)),
                completed_operations=int(completed_operations),
                total_operations=int(len(env.sim.operations)),
                reward_identity_error=identity_error,
                failure_reason=failure_reason,
                parameter_case_id=record.parameter_case_id,
                M=getattr(record, "num_cells", None) or getattr(instance, "num_cells", None),
                S=getattr(record, "num_stages", None) or getattr(instance, "num_stages", None),
                J=getattr(record, "num_orders", None) or getattr(instance, "num_orders", None),
                H=getattr(record, "num_workers", None) or getattr(instance, "num_workers", None),
                R=getattr(record, "num_robots", None) or getattr(instance, "num_robots", None),
                V=getattr(record, "num_products", None) or getattr(instance, "num_products", None),
            )
            completed_results[index] = result
            progress_count += 1
            if progress_every_instances and (
                progress_count % progress_every_instances == 0
                or progress_count == len(states)
            ):
                suite_elapsed = time.perf_counter() - suite_started
                print(
                    f"[{progress_prefix}] {progress_count}/{len(states)} "
                    f"{instance.instance_id}; status={status}; decisions={state['decisions']}; "
                    f"batched_suite={suite_elapsed:.1f}s",
                    flush=True,
                )
            if status != "completed":
                failure_seen = True
        if bool(stop_after_first_failure) and failure_seen:
            break
        active = next_active

    results = [completed_results[index] for index in sorted(completed_results)]
    summary = _validation_summary_from_results(
        results,
        stage_index=stage_index,
        stage_name_override=stage_name_override,
    )
    if return_instance_results:
        return summary, tuple(results)
    return summary


@torch.inference_mode()
def evaluate_fixed_validation(
    cfg,
    *,
    policy,
    normalizer,
    records,
    stage_index: int,
    deterministic: bool,
    device: torch.device,
    max_episode_decisions: int,
    return_instance_results: bool = False,
    stop_after_first_failure: bool = False,
    constraints_override: PolicyActionConstraints | None = None,
    stage_name_override: str | None = None,
    progress_every_instances: int = 0,
    progress_prefix: str = "validation",
):
    """Evaluate a fixed suite without turning poor policy behavior into a crash.

    ``max_episode_decisions`` is a safety guard, not a surrogate objective. A
    policy that exceeds it is recorded as ``non_terminating`` and receives a
    non-finite *selection* metric; no artificial finite TWT penalty is invented.
    Deadlocks caused by a policy are treated the same way for model selection.
    Unexpected programming errors are intentionally not swallowed.

    ``stop_after_first_failure`` is an evaluation-efficiency control used only by
    validation-only calibration diagnostics. Once one fixed instance is already
    non-terminating/deadlocked, that checkpoint is ineligible; stopping the rest
    of that suite does not invent a finite objective or change training dynamics.
    """
    policy.eval()
    constraints: PolicyActionConstraints = (constraints_override or stage_action_constraints(stage_index))
    progress_every_instances = max(0, int(progress_every_instances))
    if hasattr(policy, "act_batch") and len(records) > 1:
        return _evaluate_fixed_validation_batched(
            cfg,
            policy=policy,
            normalizer=normalizer,
            records=records,
            stage_index=stage_index,
            deterministic=deterministic,
            max_episode_decisions=max_episode_decisions,
            return_instance_results=return_instance_results,
            stop_after_first_failure=stop_after_first_failure,
            constraints=constraints,
            stage_name_override=stage_name_override,
            progress_every_instances=progress_every_instances,
            progress_prefix=progress_prefix,
        )
    twts: list[float] = []
    decisions_list: list[int] = []
    reconfigs: list[int] = []
    reward_errors: list[float] = []
    instance_results: list[ValidationInstanceResult] = []
    total_records = len(records)
    suite_started = time.perf_counter()

    for record_index, record in enumerate(records, start=1):
        instance_started = time.perf_counter()
        instance = load_instance_csv(record.path)
        env = AssemblyEnv(cfg)
        env.reset(instance)
        base_reward_terms: list[float] = []
        decisions = 0
        reconfig_count = 0
        status: Literal["completed", "non_terminating", "deadlock"] | None = None
        failure_reason = ""
        twt: float | None = None
        identity_error: float | None = None

        while True:
            transform_graph = getattr(
                normalizer, "transform_replay_trusted", normalizer.transform
            )
            graph = transform_graph(env.graph())
            context = env.action_context()
            if hasattr(policy, "act_batch"):
                act_kwargs = {
                    "deterministic": bool(deterministic),
                    "constraints": constraints,
                }
                if isinstance(policy, torch.nn.Module):
                    act_kwargs["validate_inputs"] = False
                output = policy.act_batch([graph], [context], **act_kwargs)[0]
            else:
                # Compatibility for simple rule/test doubles that implement only
                # the historical single-action policy interface.
                output = policy.act(
                    graph.to(device), context.to(device),
                    deterministic=bool(deterministic),
                    constraints=constraints,
                )
            decisions += 1
            try:
                _, _, done, info = env.step(output.action)
            except DeadlockError as exc:
                reconfig_count = int(env.sim.reconfiguration_count)
                status = "deadlock"
                failure_reason = str(exc)
                break

            base_reward_terms.append(float(info["reward_twt"]))
            reconfig_count = int(info["reconfiguration_count"])
            if done:
                twt = float(info["twt"])
                base_reward_sum = stable_sum(base_reward_terms)
                identity_error = assert_reward_identity(
                    base_reward_sum,
                    twt,
                    context=f"validation reward identity failed for {instance.instance_id}",
                )
                status = "completed"
                twts.append(twt)
                reward_errors.append(identity_error)
                break
            if decisions >= int(max_episode_decisions):
                status = "non_terminating"
                failure_reason = (
                    f"exceeded validation safety limit of {int(max_episode_decisions)} decisions"
                )
                break

        assert status is not None
        decisions_list.append(decisions)
        reconfigs.append(reconfig_count)
        completed_orders = sum(1 for order in env.sim.orders if order.completed)
        completed_operations = sum(
            1 for op in env.sim.operations if op.status == OperationStatus.DONE
        )
        instance_results.append(ValidationInstanceResult(
            instance_id=instance.instance_id,
            scale=instance.scale,
            scenario=instance.scenario,
            target_load_ratio=float(instance.target_load_ratio),
            due_tightness=instance.due_tightness,
            status=status,
            twt=twt,
            decisions=decisions,
            reconfigurations=reconfig_count,
            simulation_time=float(env.sim.time),
            completed_orders=int(completed_orders),
            total_orders=int(len(env.sim.orders)),
            completed_operations=int(completed_operations),
            total_operations=int(len(env.sim.operations)),
            reward_identity_error=identity_error,
            failure_reason=failure_reason,
            parameter_case_id=record.parameter_case_id,
            M=getattr(record, "num_cells", None) or getattr(instance, "num_cells", None),
            S=getattr(record, "num_stages", None) or getattr(instance, "num_stages", None),
            J=getattr(record, "num_orders", None) or getattr(instance, "num_orders", None),
                H=getattr(record, "num_workers", None) or getattr(instance, "num_workers", None),
            R=getattr(record, "num_robots", None) or getattr(instance, "num_robots", None),
                V=getattr(record, "num_products", None) or getattr(instance, "num_products", None),
        ))
        if progress_every_instances and (
            record_index % progress_every_instances == 0 or record_index == total_records
        ):
            elapsed = time.perf_counter() - instance_started
            suite_elapsed = time.perf_counter() - suite_started
            print(
                f"[{progress_prefix}] {record_index}/{total_records} "
                f"{instance.instance_id}; status={status}; decisions={decisions}; "
                f"instance={elapsed:.1f}s; suite={suite_elapsed:.1f}s",
                flush=True,
            )
        if bool(stop_after_first_failure) and status != "completed":
            break

    completed = sum(1 for r in instance_results if r.status == "completed")
    nonterm = sum(1 for r in instance_results if r.status == "non_terminating")
    deadlocked = sum(1 for r in instance_results if r.status == "deadlock")
    failed = nonterm + deadlocked
    total = len(instance_results)
    finite_mean = None if not twts else float(mean(twts))
    # A failed validation instance means the policy has no valid finite score on
    # the complete suite. Preserve finite completed-instance diagnostics separately.
    selection_mean = float("inf") if failed else float(finite_mean)
    selection_median = float("inf") if failed else float(np.median(np.asarray(twts, dtype=float)))

    summary = ValidationSummary(
        stage_index=int(stage_index),
        stage_name=(str(stage_name_override) if stage_name_override is not None else (
            "fixed_configuration_warmup",
            "single_resource_reconfiguration",
            "full_set_reconfiguration",
            "multi_scale_joint_finetuning",
        )[int(stage_index)]),
        instances=total,
        completed_instances=completed,
        failed_instances=failed,
        nonterminating_instances=nonterm,
        deadlocked_instances=deadlocked,
        failure_rate=(0.0 if total == 0 else float(failed / total)),
        mean_twt=selection_mean,
        median_twt=selection_median,
        mean_completed_twt=finite_mean,
        mean_decisions=float(mean(decisions_list)),
        mean_reconfigurations=float(mean(reconfigs)),
        max_reward_identity_error=float(max(reward_errors, default=0.0)),
    )
    if return_instance_results:
        return summary, tuple(instance_results)
    return summary


def append_validation_csv(path: str | Path, *, global_iteration: int, summary: ValidationSummary) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"global_iteration": int(global_iteration), **asdict(summary)}
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_validation_instance_csv(
    path: str | Path,
    *,
    global_iteration: int,
    stage_index: int,
    stage_name: str,
    results,
) -> None:
    """Append per-instance validation diagnostics for audit/debugging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "global_iteration": int(global_iteration),
            "stage_index": int(stage_index),
            "stage_name": str(stage_name),
            **asdict(result),
        }
        for result in results
    ]
    if not rows:
        return
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "ValidationInstanceResult", "ValidationRecord", "ValidationSummary",
    "append_validation_csv", "append_validation_instance_csv",
    "ensure_fixed_validation_suite", "evaluate_fixed_validation",
    "filter_validation_records",
]
