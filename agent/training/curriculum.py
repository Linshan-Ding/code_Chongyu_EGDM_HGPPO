"""Four-stage paper curriculum control for Phase J."""

from __future__ import annotations

from dataclasses import dataclass
import math

from agent.constraints import PolicyActionConstraints
from agent.ppo import PPOComponentMask


STAGE_NAMES = (
    "fixed_configuration_warmup",
    "single_resource_reconfiguration",
    "full_set_reconfiguration",
    "multi_scale_joint_finetuning",
)


@dataclass(frozen=True, slots=True)
class StageTrainability:
    stage_index: int
    stage_name: str
    trainable_parameters: int
    frozen_parameters: int


@dataclass(frozen=True, slots=True)
class StagePolicySpec:
    stage_index: int
    stage_name: str
    scale_pool: tuple[str, ...]
    scenario_pool: tuple[str, ...]
    load_ratio_pool: tuple[float, ...]
    due_tightness_pool: tuple[str, ...]
    constraints: PolicyActionConstraints
    components: PPOComponentMask


def _set_module(module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def configure_curriculum_stage(policy, stage_index: int) -> StageTrainability:
    """Apply the paper's stage-wise parameter freezing.

    Stage I trains encoder/scheduler/scheduling critic only. Stages II-IV jointly
    train all policy/value components; Stage II restricts action cardinality via
    ``PolicyActionConstraints`` rather than freezing resource heads.
    """

    stage_index = int(stage_index)
    if not 0 <= stage_index < 4:
        raise ValueError("stage_index must be 0..3")
    for parameter in policy.parameters():
        parameter.requires_grad_(True)

    if stage_index == 0:
        _set_module(policy.representation.gate, False)
        _set_module(policy.worker_matcher, False)
        _set_module(policy.robot_matcher, False)
        _set_module(policy.representation.critics.v_gate, False)
        _set_module(policy.representation.critics.v_rec, False)

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    if trainable <= 0:
        raise RuntimeError("curriculum stage left no trainable parameters")
    if stage_index == 0 and frozen <= 0:
        raise RuntimeError("Stage I must freeze Gate/resource components")
    if stage_index > 0 and frozen != 0:
        raise RuntimeError("Stages II-IV must jointly train all policy/value parameters")
    return StageTrainability(stage_index, STAGE_NAMES[stage_index], trainable, frozen)


def stage_action_constraints(stage_index: int) -> PolicyActionConstraints:
    stage_index = int(stage_index)
    if stage_index == 0:
        return PolicyActionConstraints.fixed_configuration()
    if stage_index == 1:
        return PolicyActionConstraints.single_resource_reconfiguration()
    if stage_index in (2, 3):
        return PolicyActionConstraints.unconstrained()
    raise ValueError("stage_index must be 0..3")


def stage_component_mask(stage_index: int) -> PPOComponentMask:
    if int(stage_index) == 0:
        return PPOComponentMask.scheduling_warmup()
    if int(stage_index) in (1, 2, 3):
        return PPOComponentMask.full()
    raise ValueError("stage_index must be 0..3")


def build_stage_policy_spec(cfg, phase_j, stage_index: int) -> StagePolicySpec:
    stage_index = int(stage_index)
    if not 0 <= stage_index < 4:
        raise ValueError("stage_index must be 0..3")
    stage = cfg.curriculum.stages[stage_index]
    name = str(stage.name)
    if name != STAGE_NAMES[stage_index]:
        raise ValueError("curriculum stage ordering changed from the paper contract")
    runtime = phase_j.stage_runtime[name]
    return StagePolicySpec(
        stage_index=stage_index,
        stage_name=name,
        scale_pool=tuple(str(x) for x in stage.allowed_scales),
        scenario_pool=tuple(runtime["scenario_pool"]),
        load_ratio_pool=tuple(float(x) for x in runtime["load_ratio_pool"]),
        due_tightness_pool=tuple(str(x) for x in runtime["due_tightness_pool"]),
        constraints=stage_action_constraints(stage_index),
        components=stage_component_mask(stage_index),
    )


# Backward-compatible Phase-I helpers used by earlier tests/imports.
def configure_fixed_configuration_warmup(policy) -> StageTrainability:
    return configure_curriculum_stage(policy, 0)


def fixed_configuration_component_mask() -> PPOComponentMask:
    return stage_component_mask(0)


@dataclass(slots=True)
class CurriculumState:
    stage_index: int = 0
    stage_iteration: int = 0
    global_iteration: int = 0
    validations_without_improvement: int = 0
    best_stage_metric: float | None = None
    best_stage_checkpoint: str | None = None
    transition_reason: str = "start"
    finished: bool = False

    @property
    def stage_name(self) -> str:
        return STAGE_NAMES[self.stage_index]


class CurriculumController:
    """Budget + validation-patience curriculum progression.

    The paper leaves numerical transition thresholds unspecified. Phase J uses
    explicit stage iteration budgets supplied by the run plus a documented
    validation-patience early transition. This is an implementation choice, not
    a paper-prescribed threshold.
    """

    def __init__(self, stage_iterations, *, patience_validations: int, min_delta: float) -> None:
        budgets = tuple(int(x) for x in stage_iterations)
        if len(budgets) != 4 or any(x <= 0 for x in budgets):
            raise ValueError("stage_iterations must contain four positive integers")
        if int(patience_validations) <= 0:
            raise ValueError("patience_validations must be positive")
        self.stage_iterations = budgets
        self.patience_validations = int(patience_validations)
        self.min_delta = float(min_delta)
        self.state = CurriculumState()

    @property
    def total_planned_iterations(self) -> int:
        return sum(self.stage_iterations)

    def global_progress(self) -> float:
        # Progress is evaluated *before* the next PPO update: the first update
        # therefore uses the paper's initial LR/entropy coefficients and the
        # last planned update reaches the final coefficients. Early stage
        # transitions jump to the next planned curriculum boundary.
        completed_budget = sum(self.stage_iterations[: self.state.stage_index])
        local = min(self.state.stage_iteration, self.stage_iterations[self.state.stage_index])
        completed_equivalent = completed_budget + local
        denominator = max(1, self.total_planned_iterations - 1)
        return min(1.0, completed_equivalent / denominator)

    def record_iteration(self) -> None:
        self.state.stage_iteration += 1
        self.state.global_iteration += 1

    def record_validation(self, metric: float, checkpoint_path: str | None = None) -> bool:
        metric = float(metric)
        # Non-finite metrics represent a policy that failed to complete the full
        # fixed validation suite (e.g. non-termination/deadlock). Such a policy
        # must never become a best checkpoint, including on the very first
        # validation when best_stage_metric is still None.
        if not math.isfinite(metric):
            self.state.validations_without_improvement += 1
            return False
        best = self.state.best_stage_metric
        improved = best is None or metric < best - self.min_delta
        if improved:
            self.state.best_stage_metric = metric
            self.state.best_stage_checkpoint = checkpoint_path
            self.state.validations_without_improvement = 0
        else:
            self.state.validations_without_improvement += 1
        return improved

    def transition_due(self) -> tuple[bool, str | None]:
        if self.state.stage_iteration >= self.stage_iterations[self.state.stage_index]:
            return True, "stage_iteration_budget"
        if self.state.validations_without_improvement >= self.patience_validations:
            return True, "validation_patience"
        return False, None

    def advance(self, reason: str) -> None:
        if self.state.stage_index == 3:
            self.state.finished = True
            self.state.transition_reason = reason
            return
        self.state.stage_index += 1
        self.state.stage_iteration = 0
        self.state.validations_without_improvement = 0
        self.state.best_stage_metric = None
        self.state.best_stage_checkpoint = None
        self.state.transition_reason = str(reason)

    def state_dict(self) -> dict:
        return {
            "stage_iterations": self.stage_iterations,
            "patience_validations": self.patience_validations,
            "min_delta": self.min_delta,
            "state": {
                "stage_index": self.state.stage_index,
                "stage_iteration": self.state.stage_iteration,
                "global_iteration": self.state.global_iteration,
                "validations_without_improvement": self.state.validations_without_improvement,
                "best_stage_metric": self.state.best_stage_metric,
                "best_stage_checkpoint": self.state.best_stage_checkpoint,
                "transition_reason": self.state.transition_reason,
                "finished": self.state.finished,
            },
        }

    def load_state_dict(self, payload: dict) -> None:
        if tuple(payload["stage_iterations"]) != self.stage_iterations:
            raise ValueError("resume checkpoint stage budgets differ from current run")
        raw = payload["state"]
        self.state = CurriculumState(**raw)


__all__ = [
    "CurriculumController",
    "CurriculumState",
    "STAGE_NAMES",
    "StagePolicySpec",
    "StageTrainability",
    "build_stage_policy_spec",
    "configure_curriculum_stage",
    "configure_fixed_configuration_warmup",
    "fixed_configuration_component_mask",
    "stage_action_constraints",
    "stage_component_mask",
]
