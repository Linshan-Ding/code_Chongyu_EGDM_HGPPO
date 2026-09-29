"""Per-instance metrics for the paper's fixed-test evaluation contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import mean

import numpy as np

from environment.entities import ExecutionMode


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    instance_id: str
    method: str
    run_id: str
    tier: str
    scale: str
    parameter_case_id: str | None
    # Canonical paper scale symbols: S=stages, M=cells, J=orders,
    # H=workers, R=robots and V=product types.  The result contract keeps
    # descriptive fields in addition to these audit dimensions.
    S: int
    M: int
    J: int
    H: int
    R: int
    V: int
    scenario: str
    rho: float
    target_load_ratio: float
    due_tightness: str
    relocation_multiplier: float
    worker_skill_density: float
    robot_capability_density: float
    twt: float
    obj_ref: float | None
    optimality_gap: float | None
    feasible: bool
    tardy_ratio: float
    mean_flow_time: float
    reconfiguration_count: int
    mean_resources_moved_per_reconfiguration: float
    worker_relocation_time: float
    robot_relocation_time: float
    mean_stage_load_capacity_cv: float
    mode_h_fraction: float
    mode_r_fraction: float
    mode_hr_fraction: float
    mean_decision_time_ms: float
    p95_decision_time_ms: float
    decisions: int
    wall_time_seconds: float
    reward_identity_error: float
    # Non-empty only when a bounded evaluation run could not terminate.  The
    # row is retained so aggregate statistics report the failure in
    # ``feasible_rate`` instead of aborting the whole experiment matrix.
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def optimality_gap(obj: float, ref: float | None) -> float | None:
    if ref is None:
        return None
    ref = float(ref)
    if abs(ref) <= 1e-12:
        return 0.0 if abs(float(obj)) <= 1e-12 else math.inf
    return (float(obj) - ref) / ref


def final_episode_metrics(
    *,
    env,
    method: str,
    run_id: str,
    tier: str,
    reward_sum: float,
    decision_times_ms: list[float],
    stage_cv_values: list[float],
    worker_relocation_time: float,
    robot_relocation_time: float,
    moved_resources: int,
    wall_time_seconds: float,
    obj_ref: float | None,
) -> EpisodeMetrics:
    if not env.sim.is_done:
        raise ValueError("metrics require a completed episode")
    inst = env.sim.instance
    twt = float(env.sim.twt())
    tardy = 0
    flows = []
    for runtime, order in zip(env.sim.orders, inst.orders, strict=True):
        completion = float(runtime.completion_time)
        tardy += int(completion > float(order.due_date) + 1e-10)
        flows.append(completion - float(order.release_time))
    modes = [op.mode for op in env.sim.operations]
    total_modes = max(1, len(modes))
    h = sum(mode == ExecutionMode.H for mode in modes)
    r = sum(mode == ExecutionMode.R for mode in modes)
    hr = sum(mode == ExecutionMode.HR for mode in modes)
    reconfigs = int(env.sim.reconfiguration_count)
    times = np.asarray(decision_times_ms, dtype=float)
    return EpisodeMetrics(
        instance_id=inst.instance_id,
        method=str(method),
        run_id=str(run_id),
        tier=str(tier),
        scale=inst.scale,
        parameter_case_id=(inst.generation_meta.get("parameter_case_id") or None),
        S=int(inst.num_stages),
        M=int(inst.num_cells),
        J=int(inst.num_orders),
        H=int(inst.num_workers),
        R=int(inst.num_robots),
        V=int(inst.num_products),
        scenario=inst.scenario,
        rho=float(inst.target_load_ratio),
        target_load_ratio=float(inst.target_load_ratio),
        due_tightness=inst.due_tightness,
        relocation_multiplier=float(inst.relocation_multiplier),
        worker_skill_density=float(inst.worker_skill_density),
        robot_capability_density=float(inst.robot_capability_density),
        twt=twt,
        obj_ref=None if obj_ref is None else float(obj_ref),
        optimality_gap=optimality_gap(twt, obj_ref),
        feasible=True,
        tardy_ratio=float(tardy) / float(inst.num_orders),
        mean_flow_time=float(mean(flows)),
        reconfiguration_count=reconfigs,
        mean_resources_moved_per_reconfiguration=(
            float(moved_resources) / float(reconfigs) if reconfigs else 0.0
        ),
        worker_relocation_time=float(worker_relocation_time),
        robot_relocation_time=float(robot_relocation_time),
        mean_stage_load_capacity_cv=float(mean(stage_cv_values)) if stage_cv_values else 0.0,
        mode_h_fraction=float(h) / total_modes,
        mode_r_fraction=float(r) / total_modes,
        mode_hr_fraction=float(hr) / total_modes,
        mean_decision_time_ms=float(times.mean()) if times.size else 0.0,
        p95_decision_time_ms=float(np.percentile(times, 95)) if times.size else 0.0,
        decisions=len(decision_times_ms),
        wall_time_seconds=float(wall_time_seconds),
        reward_identity_error=abs(float(reward_sum) + twt),
    )


def failed_episode_metrics(
    *,
    env,
    method: str,
    run_id: str,
    tier: str,
    decisions: int,
    wall_time_seconds: float,
    obj_ref: float | None,
    failure_reason: str,
    decision_times_ms: list[float] | None = None,
) -> EpisodeMetrics:
    """Create a contract-compliant row for a bounded non-terminating run.

    A learned policy that keeps producing actions without completing an
    episode must not be silently dropped and must not abort unrelated methods.
    Non-finite objective fields are ignored by ``agent.evaluation.stats`` while the
    explicit ``feasible=False`` value contributes to the reported failure
    rate.
    """
    inst = env.sim.instance
    times = np.asarray(decision_times_ms or (), dtype=float)
    return EpisodeMetrics(
        instance_id=inst.instance_id,
        method=str(method),
        run_id=str(run_id),
        tier=str(tier),
        scale=inst.scale,
        parameter_case_id=(inst.generation_meta.get("parameter_case_id") or None),
        S=int(inst.num_stages), M=int(inst.num_cells), J=int(inst.num_orders),
        H=int(inst.num_workers), R=int(inst.num_robots), V=int(inst.num_products),
        scenario=inst.scenario,
        rho=float(inst.target_load_ratio),
        target_load_ratio=float(inst.target_load_ratio),
        due_tightness=inst.due_tightness,
        relocation_multiplier=float(inst.relocation_multiplier),
        worker_skill_density=float(inst.worker_skill_density),
        robot_capability_density=float(inst.robot_capability_density),
        twt=float("nan"),
        obj_ref=None if obj_ref is None else float(obj_ref),
        optimality_gap=None,
        feasible=False,
        tardy_ratio=float("nan"),
        mean_flow_time=float("nan"),
        reconfiguration_count=int(env.sim.reconfiguration_count),
        mean_resources_moved_per_reconfiguration=0.0,
        worker_relocation_time=0.0,
        robot_relocation_time=0.0,
        mean_stage_load_capacity_cv=float("nan"),
        mode_h_fraction=float("nan"),
        mode_r_fraction=float("nan"),
        mode_hr_fraction=float("nan"),
        mean_decision_time_ms=float(times.mean()) if times.size else 0.0,
        p95_decision_time_ms=float(np.percentile(times, 95)) if times.size else 0.0,
        decisions=int(decisions),
        wall_time_seconds=float(wall_time_seconds),
        reward_identity_error=float("nan"),
        failure_reason=str(failure_reason),
    )


EVAL_FIELDS = tuple(EpisodeMetrics.__dataclass_fields__.keys())

__all__ = [
    "EVAL_FIELDS", "EpisodeMetrics", "failed_episode_metrics",
    "final_episode_metrics", "optimality_gap",
]
