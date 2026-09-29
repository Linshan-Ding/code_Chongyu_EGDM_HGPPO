"""Phase I execution-only configuration.

Paper/model parameters remain in the four validated project YAML files.  This
module loads only runtime/smoke choices that the paper does not numerically fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class NormalizationRuntime:
    collection_policy: str
    episodes: int
    max_graphs: int


@dataclass(frozen=True, slots=True)
class PhaseISmokeRuntime:
    iterations: int
    parallel_envs: int
    rollout_events: int
    training_seed: int
    scale_pool: tuple[str, ...]
    scenario_pool: tuple[str, ...]
    normalization_episodes: int
    normalization_max_graphs: int
    max_episode_decisions: int
    log_csv: str


@dataclass(frozen=True, slots=True)
class PhaseIRuntime:
    device: str
    optimizer_schedule: str
    scenario_pool: tuple[str, ...]
    max_episode_decisions: int
    log_csv: str


@dataclass(frozen=True, slots=True)
class PhaseIConfig:
    stage_name: str
    runtime: PhaseIRuntime
    normalization: NormalizationRuntime
    smoke: PhaseISmokeRuntime


def _positive_int(value, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _pool(values, name: str, allowed: set[str]) -> tuple[str, ...]:
    out = tuple(str(x) for x in values)
    if not out:
        raise ValueError(f"{name} cannot be empty")
    unknown = set(out).difference(allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown values: {sorted(unknown)}")
    return out


def load_phase_i_config(path: str | Path = "configs/train.yaml") -> PhaseIConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "phase_i" not in raw:
        raise ValueError("configs/train.yaml must contain phase_i")
    root = raw["phase_i"]
    if root.get("stage_name") != "fixed_configuration_warmup":
        raise ValueError("Phase I is intentionally restricted to fixed_configuration_warmup")

    runtime_raw = root["runtime"]
    norm_raw = root["normalization"]
    smoke_raw = root["smoke"]
    normalization = NormalizationRuntime(
        collection_policy=str(norm_raw["collection_policy"]),
        episodes=_positive_int(norm_raw["episodes"], "normalization.episodes"),
        max_graphs=_positive_int(norm_raw["max_graphs"], "normalization.max_graphs"),
    )
    if normalization.collection_policy != "greedy_fixed_configuration":
        raise ValueError("Phase I currently supports greedy_fixed_configuration normalization only")

    runtime = PhaseIRuntime(
        device=str(runtime_raw["device"]),
        optimizer_schedule=str(runtime_raw["optimizer_schedule"]),
        scenario_pool=_pool(runtime_raw["scenario_pool"], "runtime.scenario_pool", {"D1","D2","D3","D4","D5"}),
        max_episode_decisions=_positive_int(runtime_raw["max_episode_decisions"], "runtime.max_episode_decisions"),
        log_csv=str(runtime_raw["log_csv"]),
    )
    if runtime.optimizer_schedule != "hold_start_until_phase_j":
        raise ValueError("Phase I keeps LR/entropy schedules at their start values until Phase J")

    smoke = PhaseISmokeRuntime(
        iterations=_positive_int(smoke_raw["iterations"], "smoke.iterations"),
        parallel_envs=_positive_int(smoke_raw["parallel_envs"], "smoke.parallel_envs"),
        rollout_events=_positive_int(smoke_raw["rollout_events"], "smoke.rollout_events"),
        training_seed=int(smoke_raw["training_seed"]),
        scale_pool=_pool(smoke_raw["scale_pool"], "smoke.scale_pool", {"S","M","L"}),
        scenario_pool=_pool(smoke_raw["scenario_pool"], "smoke.scenario_pool", {"D1","D2","D3","D4","D5"}),
        normalization_episodes=_positive_int(smoke_raw["normalization_episodes"], "smoke.normalization_episodes"),
        normalization_max_graphs=_positive_int(smoke_raw["normalization_max_graphs"], "smoke.normalization_max_graphs"),
        max_episode_decisions=_positive_int(smoke_raw["max_episode_decisions"], "smoke.max_episode_decisions"),
        log_csv=str(smoke_raw["log_csv"]),
    )
    return PhaseIConfig(
        stage_name=str(root["stage_name"]),
        runtime=runtime,
        normalization=normalization,
        smoke=smoke,
    )


__all__ = [
    "NormalizationRuntime",
    "PhaseIConfig",
    "PhaseIRuntime",
    "PhaseISmokeRuntime",
    "load_phase_i_config",
]

# ---------------------------------------------------------------------------
# Phase J complete-curriculum runtime configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PhaseJValidationConfig:
    root: str
    base_seed: int
    instances_per_combination: int
    scales: tuple[str, ...]
    scenarios: tuple[str, ...]
    load_ratios: tuple[float, ...]
    due_tightness: tuple[str, ...]
    every_iterations: int
    patience_validations: int
    min_delta: float
    deterministic: bool
    stage_filters: dict[str, dict[str, tuple]]


@dataclass(frozen=True, slots=True)
class PhaseJNormalizationConfig:
    scale_pool: tuple[str, ...]
    scenario_pool: tuple[str, ...]
    episodes: int
    max_graphs: int


@dataclass(frozen=True, slots=True)
class PhaseJRuntimeConfig:
    device: str
    run_root: str
    max_episode_decisions: int
    checkpoint_every_iterations: int
    restore_best_before_next_stage: bool


@dataclass(frozen=True, slots=True)
class PhaseJSmokeConfig:
    stage_iterations: tuple[int, int, int, int]
    parallel_envs: int
    rollout_events: int
    training_seed: int
    normalization_episodes: int
    normalization_max_graphs: int
    max_episode_decisions: int
    validation_root: str
    validation_instances_per_combination: int
    validation_scales: tuple[str, ...]
    validation_scenarios: tuple[str, ...]
    validation_load_ratios: tuple[float, ...]
    validation_due_tightness: tuple[str, ...]
    validation_every_iterations: int
    patience_validations: int
    validate_at_stage_end: bool
    ppo_epochs_override: int | None


@dataclass(frozen=True, slots=True)
class PhaseJConfig:
    runtime: PhaseJRuntimeConfig
    stage_iterations: tuple[int, int, int, int] | None
    stage_runtime: dict[str, dict[str, tuple]]
    normalization: PhaseJNormalizationConfig
    validation: PhaseJValidationConfig
    smoke: PhaseJSmokeConfig


def _stage_iterations(value) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    values = tuple(int(x) for x in value)
    if len(values) != 4 or any(x <= 0 for x in values):
        raise ValueError("phase_j curriculum.stage_iterations must contain four positive integers or null")
    return values  # type: ignore[return-value]


def _due_pool(values, name: str) -> tuple[str, ...]:
    return _pool(values, name, {"tight", "medium", "loose"})


def _load_ratios(values, allowed_values) -> tuple[float, ...]:
    out = tuple(float(x) for x in values)
    allowed = {float(x) for x in allowed_values}
    if not out or not set(out).issubset(allowed):
        raise ValueError(f"load-ratio pool must be a non-empty subset of {sorted(allowed)}")
    return out


def load_phase_j_config(path: str | Path = "configs/train.yaml", *, project_cfg=None) -> PhaseJConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "phase_j" not in raw:
        raise ValueError("configs/train.yaml must contain phase_j")
    root = raw["phase_j"]
    allowed_rho = (
        [0.65, 0.80, 0.95]
        if project_cfg is None
        else list(project_cfg.instance.load_ratio_choices)
    )

    runtime_raw = root["runtime"]
    runtime = PhaseJRuntimeConfig(
        device=str(runtime_raw["device"]),
        run_root=str(runtime_raw["run_root"]),
        max_episode_decisions=_positive_int(runtime_raw["max_episode_decisions"], "phase_j.runtime.max_episode_decisions"),
        checkpoint_every_iterations=_positive_int(runtime_raw["checkpoint_every_iterations"], "phase_j.runtime.checkpoint_every_iterations"),
        restore_best_before_next_stage=bool(runtime_raw["restore_best_before_next_stage"]),
    )

    cur_raw = root["curriculum"]
    stage_names = (
        "fixed_configuration_warmup",
        "single_resource_reconfiguration",
        "full_set_reconfiguration",
        "multi_scale_joint_finetuning",
    )
    stage_runtime: dict[str, dict[str, tuple]] = {}
    for name in stage_names:
        item = cur_raw[name]
        stage_runtime[name] = {
            "scenario_pool": _pool(item["scenario_pool"], f"phase_j.curriculum.{name}.scenario_pool", {"D1","D2","D3","D4","D5"}),
            "load_ratio_pool": _load_ratios(item["load_ratio_pool"], allowed_rho),
            "due_tightness_pool": _due_pool(item["due_tightness_pool"], f"phase_j.curriculum.{name}.due_tightness_pool"),
        }

    norm_raw = root["normalization"]
    normalization = PhaseJNormalizationConfig(
        scale_pool=_pool(norm_raw["scale_pool"], "phase_j.normalization.scale_pool", {"S","M","L"}),
        scenario_pool=_pool(norm_raw["scenario_pool"], "phase_j.normalization.scenario_pool", {"D1","D2","D3","D4","D5"}),
        episodes=_positive_int(norm_raw["episodes"], "phase_j.normalization.episodes"),
        max_graphs=_positive_int(norm_raw["max_graphs"], "phase_j.normalization.max_graphs"),
    )

    val_raw = root["validation"]
    stage_filters: dict[str, dict[str, tuple]] = {}
    for name in stage_names:
        item = val_raw["stage_filters"][name]
        stage_filters[name] = {
            "scales": _pool(item["scales"], f"phase_j.validation.{name}.scales", {"S","M","L"}),
            "scenarios": _pool(item["scenarios"], f"phase_j.validation.{name}.scenarios", {"D1","D2","D3","D4","D5"}),
            "load_ratios": _load_ratios(item["load_ratios"], allowed_rho),
        }
    validation = PhaseJValidationConfig(
        root=str(val_raw["root"]),
        base_seed=int(val_raw["base_seed"]),
        instances_per_combination=_positive_int(val_raw["instances_per_combination"], "phase_j.validation.instances_per_combination"),
        scales=_pool(val_raw["scales"], "phase_j.validation.scales", {"S","M","L"}),
        scenarios=_pool(val_raw["scenarios"], "phase_j.validation.scenarios", {"D1","D2","D3","D4","D5"}),
        load_ratios=_load_ratios(val_raw["load_ratios"], allowed_rho),
        due_tightness=_due_pool(val_raw["due_tightness"], "phase_j.validation.due_tightness"),
        every_iterations=_positive_int(val_raw["every_iterations"], "phase_j.validation.every_iterations"),
        patience_validations=_positive_int(val_raw["patience_validations"], "phase_j.validation.patience_validations"),
        min_delta=float(val_raw["min_delta"]),
        deterministic=bool(val_raw["deterministic"]),
        stage_filters=stage_filters,
    )

    smoke_raw = root["smoke"]
    smoke = PhaseJSmokeConfig(
        stage_iterations=_stage_iterations(smoke_raw["stage_iterations"]),  # type: ignore[arg-type]
        parallel_envs=_positive_int(smoke_raw["parallel_envs"], "phase_j.smoke.parallel_envs"),
        rollout_events=_positive_int(smoke_raw["rollout_events"], "phase_j.smoke.rollout_events"),
        training_seed=int(smoke_raw["training_seed"]),
        normalization_episodes=_positive_int(smoke_raw["normalization_episodes"], "phase_j.smoke.normalization_episodes"),
        normalization_max_graphs=_positive_int(smoke_raw["normalization_max_graphs"], "phase_j.smoke.normalization_max_graphs"),
        max_episode_decisions=_positive_int(smoke_raw["max_episode_decisions"], "phase_j.smoke.max_episode_decisions"),
        validation_root=str(smoke_raw["validation_root"]),
        validation_instances_per_combination=_positive_int(smoke_raw["validation_instances_per_combination"], "phase_j.smoke.validation_instances_per_combination"),
        validation_scales=_pool(smoke_raw["validation_scales"], "phase_j.smoke.validation_scales", {"S","M","L"}),
        validation_scenarios=_pool(smoke_raw["validation_scenarios"], "phase_j.smoke.validation_scenarios", {"D1","D2","D3","D4","D5"}),
        validation_load_ratios=_load_ratios(smoke_raw["validation_load_ratios"], allowed_rho),
        validation_due_tightness=_due_pool(smoke_raw["validation_due_tightness"], "phase_j.smoke.validation_due_tightness"),
        validation_every_iterations=_positive_int(smoke_raw["validation_every_iterations"], "phase_j.smoke.validation_every_iterations"),
        patience_validations=_positive_int(smoke_raw["patience_validations"], "phase_j.smoke.patience_validations"),
        validate_at_stage_end=bool(smoke_raw.get("validate_at_stage_end", True)),
        ppo_epochs_override=(None if smoke_raw.get("ppo_epochs_override") is None else _positive_int(smoke_raw["ppo_epochs_override"], "phase_j.smoke.ppo_epochs_override")),
    )
    if smoke.stage_iterations is None:
        raise ValueError("phase_j.smoke.stage_iterations cannot be null")

    return PhaseJConfig(
        runtime=runtime,
        stage_iterations=_stage_iterations(cur_raw.get("stage_iterations")),
        stage_runtime=stage_runtime,
        normalization=normalization,
        validation=validation,
        smoke=smoke,
    )


__all__ += [
    "PhaseJConfig", "PhaseJNormalizationConfig", "PhaseJRuntimeConfig",
    "PhaseJSmokeConfig", "PhaseJValidationConfig", "load_phase_j_config",
]
