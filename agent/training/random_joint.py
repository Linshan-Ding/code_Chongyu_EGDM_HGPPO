"""Single-seed random-instance joint-training control for teacher Scheme 2.

This module intentionally leaves the historical four-stage curriculum untouched.
Scheme 2 trains the complete EGDM-HGPPO policy from iteration 1 on fresh
S/M/L range-sampled structures, with D1-D5, load-ratio and due-tightness pools.
The immutable parameter cases are reserved for validation/test design.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from agent.constraints import PolicyActionConstraints
from agent.ppo import PPOComponentMask
from agent.training.curriculum import StagePolicySpec, configure_curriculum_stage


JOINT_STAGE_INDEX = 3  # reuse the already-validated full-action/full-PPO semantics
JOINT_STAGE_NAME = "random_joint_training"


def configure_random_joint_training(policy):
    """Enable every EGDM-HGPPO policy/value parameter from the first update."""
    state = configure_curriculum_stage(policy, JOINT_STAGE_INDEX)
    if state.frozen_parameters != 0:
        raise RuntimeError("random-joint training must not freeze any policy parameters")
    return state


def build_random_joint_policy_spec(
    *,
    scale_pool=("S", "M", "L"),
    scenario_pool=("D1", "D2", "D3", "D4", "D5"),
    load_ratio_pool=(0.65, 0.80, 0.95),
    due_tightness_pool=("tight", "medium", "loose"),
) -> StagePolicySpec:
    scales = tuple(str(x) for x in scale_pool)
    scenarios = tuple(str(x) for x in scenario_pool)
    rhos = tuple(float(x) for x in load_ratio_pool)
    due = tuple(str(x) for x in due_tightness_pool)
    if not scales or not set(scales).issubset({"S", "M", "L"}):
        raise ValueError("random-joint scale_pool must be a non-empty subset of S/M/L")
    if not scenarios or not set(scenarios).issubset({"D1", "D2", "D3", "D4", "D5"}):
        raise ValueError("random-joint scenario_pool is invalid")
    if not rhos or not set(rhos).issubset({0.65, 0.80, 0.95}):
        raise ValueError("random-joint load_ratio_pool is invalid")
    if not due or not set(due).issubset({"tight", "medium", "loose"}):
        raise ValueError("random-joint due_tightness_pool is invalid")
    return StagePolicySpec(
        stage_index=JOINT_STAGE_INDEX,
        stage_name=JOINT_STAGE_NAME,
        scale_pool=scales,
        scenario_pool=scenarios,
        load_ratio_pool=rhos,
        due_tightness_pool=due,
        constraints=PolicyActionConstraints.unconstrained(),
        components=PPOComponentMask.full(),
    )


@dataclass(slots=True)
class RandomJointState:
    global_iteration: int = 0
    validations_without_improvement: int = 0
    best_validation_metric: float | None = None
    best_checkpoint: str | None = None
    finished: bool = False




def select_balanced_validation_monitor(records):
    """Select 15 deterministic fixed monitors: one per S/M/L x D1-D5.

    The chosen load/due cells rotate across scenarios so all three load levels
    and all three due-tightness levels remain represented without evaluating
    the full 135-cell pool during training.  This affects only checkpoint
    monitoring cost; final paper testing still uses the separate full test suite.
    """
    scenario_cells = {
        "D1": (0.65, "tight"),
        "D2": (0.80, "medium"),
        "D3": (0.95, "loose"),
        "D4": (0.95, "tight"),
        "D5": (0.65, "loose"),
    }
    index = {
        (str(r.scale), str(r.scenario), round(float(r.target_load_ratio), 8), str(r.due_tightness)): r
        for r in records
    }
    selected = []
    for scale in ("S", "M", "L"):
        for scenario in ("D1", "D2", "D3", "D4", "D5"):
            rho, due = scenario_cells[scenario]
            key = (scale, scenario, round(float(rho), 8), due)
            try:
                selected.append(index[key])
            except KeyError as exc:
                raise ValueError(f"fixed validation pool is missing monitor cell {key}") from exc
    if len(selected) != 15:
        raise RuntimeError("balanced Scheme-2 validation monitor must contain exactly 15 instances")
    return tuple(selected)


def select_stratified_validation_monitor_9(records):
    """Select 9 deterministic fixed monitors for fast in-training checkpoint selection.

    The subset preserves all three scales and covers D1-D5 globally, while also
    retaining all three load levels and due-tightness levels.  Deliberately, the
    very expensive L-D4 cell is left to the final fixed test suite rather than
    repeated during training monitoring.  This changes monitoring cost only; it
    does not change the random training distribution or final paper test suite.
    """
    cells = (
        ("S", "D1", 0.65, "tight"),
        ("S", "D3", 0.95, "loose"),
        ("S", "D5", 0.65, "loose"),
        ("M", "D2", 0.80, "medium"),
        ("M", "D4", 0.95, "tight"),
        ("M", "D5", 0.65, "loose"),
        ("L", "D1", 0.65, "tight"),
        ("L", "D2", 0.80, "medium"),
        ("L", "D3", 0.95, "loose"),
    )
    index = {
        (str(r.scale), str(r.scenario), round(float(r.target_load_ratio), 8), str(r.due_tightness)): r
        for r in records
    }
    selected = []
    for scale, scenario, rho, due in cells:
        key = (scale, scenario, round(float(rho), 8), due)
        try:
            selected.append(index[key])
        except KeyError as exc:
            raise ValueError(f"fixed validation pool is missing fast monitor cell {key}") from exc
    if len(selected) != 9:
        raise RuntimeError("fast Scheme-2 validation monitor must contain exactly 9 instances")
    return tuple(selected)


def select_parameter_case_validation_monitor(records):
    """Select one deterministic monitor per parameter case and scenario.

    The monitor remains small enough for every-20-iteration checks while
    preventing duplicate ``scale`` labels from hiding distinct structural
    ``(S,M,J,H,R,V)`` combinations
    cases.  Load ratio and due tightness rotate over the available fixed pool.
    """
    groups = {}
    for record in records:
        case_id = getattr(record, "parameter_case_id", None) or str(record.scale)
        groups.setdefault((case_id, str(record.scenario)), []).append(record)
    selected = []
    for key in sorted(groups):
        candidates = sorted(
            groups[key],
            key=lambda r: (float(r.target_load_ratio), str(r.due_tightness), str(r.instance_id)),
        )
        selected.append(candidates[len(selected) % len(candidates)])
    if not selected:
        raise ValueError("fixed validation pool is empty")
    return tuple(selected)


class RandomJointController:
    """Fixed-budget controller with validation monitoring but no curriculum stages.

    Validation is allowed to select/store a best checkpoint, but it never changes
    the training distribution or action space and does not terminate training
    early.  The model is frozen only after the predeclared iteration budget.
    """

    def __init__(self, total_iterations: int, *, min_delta: float = 0.0) -> None:
        total = int(total_iterations)
        if total <= 0:
            raise ValueError("total_iterations must be positive")
        self.total_iterations = total
        self.min_delta = float(min_delta)
        self.state = RandomJointState()

    def global_progress(self) -> float:
        denominator = max(1, self.total_iterations - 1)
        return min(1.0, float(self.state.global_iteration) / denominator)

    def record_iteration(self) -> None:
        if self.state.finished:
            raise RuntimeError("cannot record an iteration after random-joint training finished")
        self.state.global_iteration += 1
        if self.state.global_iteration >= self.total_iterations:
            self.state.finished = True

    def record_validation(self, metric: float, checkpoint_path: str | None = None) -> bool:
        metric = float(metric)
        if not math.isfinite(metric):
            self.state.validations_without_improvement += 1
            return False
        best = self.state.best_validation_metric
        improved = best is None or metric < best - self.min_delta
        if improved:
            self.state.best_validation_metric = metric
            self.state.best_checkpoint = checkpoint_path
            self.state.validations_without_improvement = 0
        else:
            self.state.validations_without_improvement += 1
        return improved

    def state_dict(self) -> dict:
        return {
            "training_mode": "random_joint",
            "total_iterations": int(self.total_iterations),
            "min_delta": float(self.min_delta),
            "state": {
                "global_iteration": int(self.state.global_iteration),
                "validations_without_improvement": int(self.state.validations_without_improvement),
                "best_validation_metric": self.state.best_validation_metric,
                "best_checkpoint": self.state.best_checkpoint,
                "finished": bool(self.state.finished),
            },
        }

    def load_state_dict(self, payload: dict) -> None:
        if payload.get("training_mode") != "random_joint":
            raise ValueError("checkpoint is not a Scheme-2 random-joint checkpoint")
        if int(payload.get("total_iterations", -1)) != self.total_iterations:
            raise ValueError("resume checkpoint total_iterations differs from current Scheme-2 run")
        if float(payload.get("min_delta", self.min_delta)) != self.min_delta:
            raise ValueError("resume checkpoint validation min_delta differs from current Scheme-2 run")
        self.state = RandomJointState(**payload["state"])


__all__ = [
    "JOINT_STAGE_INDEX", "JOINT_STAGE_NAME", "RandomJointController", "RandomJointState",
    "build_random_joint_policy_spec", "configure_random_joint_training",
    "select_balanced_validation_monitor", "select_stratified_validation_monitor_9",
    "select_parameter_case_validation_monitor",
]
