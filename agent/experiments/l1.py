"""Phase L1: freeze and launch the formal multi-seed EGDM-HGPPO training protocol.

The paper fixes model/PPO hyperparameters and the four curriculum *types*, but it
leaves numerical stage budgets, validation-set size, and the optional potential
shaping coefficient unspecified. L1.3d/L1.3d.1 completed the validation-only
selection pipeline; L1.4 froze the reward protocol, and L1.5 adds an execution-only
throughput profile for the newer GPU without changing the formal research semantics.
"""

from __future__ import annotations

import hashlib
import math
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable
import csv

import yaml
import torch

from agent.policy import EGDMCompositePolicy
from configs.config import load_config
from agent.training.config import load_phase_j_config
from agent.training.checkpoint import load_training_checkpoint
from agent.training.trainer_j import PhaseJRunSettings, PhaseJTrainer
from agent.training.throughput import load_throughput_profile
from agent.training.validation import ensure_fixed_validation_suite, evaluate_fixed_validation
from environment.env import AssemblyEnv


PROJECT_CONFIGS = (
    "configs/instance.yaml",
    "configs/env.yaml",
    "configs/algo.yaml",
    "configs/curriculum.yaml",
)
L1_CONFIG = "configs/phase_l1.yaml"
TRAIN_CONFIG = "configs/train.yaml"
EVAL_CONFIG = "configs/eval.yaml"
THROUGHPUT_CONFIG = "configs/throughput.yaml"


@dataclass(frozen=True, slots=True)
class L1ValidationConfig:
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




@dataclass(frozen=True, slots=True)
class L1BudgetCandidate:
    label: str
    stage_iterations: tuple[int, int, int, int]
    run_root: str


@dataclass(frozen=True, slots=True)
class L1ShapingBudgetSearchConfig:
    output_root: str
    candidates: tuple[L1BudgetCandidate, ...]
    stop_after_first_stable: bool
    stop_budget_after_first_invalid_seed: bool
    validation_stop_after_first_failure: bool


@dataclass(frozen=True, slots=True)
class L1ShapingCalibrationConfig:
    seeds: tuple[int, ...]
    stage_iterations: tuple[int, int, int, int]
    parallel_envs: int
    rollout_events: int
    ppo_epochs: int
    normalization_episodes: int
    normalization_max_graphs: int
    validation_root: str
    validation_base_seed: int
    validation_instances_per_combination: int
    validation_scales: tuple[str, ...]
    validation_scenarios: tuple[str, ...]
    validation_load_ratios: tuple[float, ...]
    validation_due_tightness: tuple[str, ...]
    validation_every_iterations: int
    patience_validations: int
    output_root: str
    resume_existing_runs: bool
    strict_twt_control_gate: bool
    validation_stop_after_first_failure: bool
    stop_candidate_after_first_invalid_seed: bool


@dataclass(frozen=True, slots=True)
class L1ShapingConfirmationConfig:
    validation_root: str
    validation_base_seed: int
    validation_instances_per_combination: int
    validation_scales: tuple[str, ...]
    validation_scenarios: tuple[str, ...]
    validation_load_ratios: tuple[float, ...]
    validation_due_tightness: tuple[str, ...]
    output_root: str


@dataclass(frozen=True, slots=True)
class PhaseL1Config:
    status: str
    formal_training_ready: bool
    stage_iterations: tuple[int, int, int, int]
    training_seeds: tuple[int, ...]
    run_root: str
    run_name_prefix: str
    parallel_envs: int
    rollout_events: int
    minibatch_size: int
    ppo_epochs: int
    normalization_episodes: int
    normalization_max_graphs: int
    memory_rollout_storage_mode: str
    memory_replay_microbatch_size: int
    validation: L1ValidationConfig
    test_root: str
    minimum_test_instances_per_combination: int
    forbid_test_access_during_training: bool
    main_training_reward: str
    potential_shaping_implemented: bool
    selected_reward_variant: str | None
    potential_shaping_enabled: bool
    potential_eta: float | None
    shaping_calibration_candidates: tuple[float, ...]
    shaping_budget_search: L1ShapingBudgetSearchConfig
    shaping_calibration: L1ShapingCalibrationConfig
    shaping_confirmation: L1ShapingConfirmationConfig
    resource_decoder_implementation: str
    fixed_source_order_exists: bool
    hardware_pilot_stop_after: int
    stress_start_stage_index: int
    stress_scale_pool: tuple[str, ...]
    stress_scenario_pool: tuple[str, ...]
    stress_load_ratio_pool: tuple[float, ...]
    stress_due_tightness_pool: tuple[str, ...]
    stress_additional_iterations: int

    @property
    def planned_iterations_per_seed(self) -> int:
        return int(sum(self.stage_iterations))

    @property
    def planned_event_interactions_per_seed(self) -> int:
        return int(self.planned_iterations_per_seed * self.rollout_events)

    @property
    def planned_event_interactions_all_seeds(self) -> int:
        return int(self.planned_event_interactions_per_seed * len(self.training_seeds))

    @property
    def validation_combinations(self) -> int:
        v = self.validation
        return len(v.scales) * len(v.scenarios) * len(v.load_ratios) * len(v.due_tightness)

    @property
    def validation_instances(self) -> int:
        return self.validation_combinations * self.validation.instances_per_combination


def _positive_int(value, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _tuple4_positive(values, name: str) -> tuple[int, int, int, int]:
    out = tuple(int(x) for x in values)
    if len(out) != 4 or any(x <= 0 for x in out):
        raise ValueError(f"{name} must contain four positive integers")
    return out  # type: ignore[return-value]


def load_phase_l1_config(path: str | Path = L1_CONFIG, *, project_cfg=None) -> PhaseL1Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = raw.get("phase_l1")
    if not isinstance(root, dict):
        raise ValueError("configs/phase_l1.yaml must contain phase_l1")

    val = root["validation"]
    test = root["test_protection"]
    reward = root["reward_policy"]
    budget_search = root["shaping_budget_search"]
    cal = root["shaping_calibration"]
    confirm = root["shaping_confirmation"]
    decoder = root["resource_decoder"]
    norm = root["normalization"]
    memory = root["memory_safe_runtime"]
    pilot = root["hardware_pilot"]
    stress = pilot["full_stress"]

    cfg = PhaseL1Config(
        status=str(root["status"]),
        formal_training_ready=bool(root["formal_training_ready"]),
        stage_iterations=_tuple4_positive(root["stage_iterations"], "stage_iterations"),
        training_seeds=tuple(int(x) for x in root["training_seeds"]),
        run_root=str(root["run_root"]),
        run_name_prefix=str(root["run_name_prefix"]),
        parallel_envs=_positive_int(root["parallel_envs"], "parallel_envs"),
        rollout_events=_positive_int(root["rollout_events"], "rollout_events"),
        minibatch_size=_positive_int(root["minibatch_size"], "minibatch_size"),
        ppo_epochs=_positive_int(root["ppo_epochs"], "ppo_epochs"),
        normalization_episodes=_positive_int(norm["episodes"], "normalization.episodes"),
        normalization_max_graphs=_positive_int(norm["max_graphs"], "normalization.max_graphs"),
        memory_rollout_storage_mode=str(memory["rollout_storage_mode"]),
        memory_replay_microbatch_size=_positive_int(
            memory["replay_microbatch_size"], "memory_safe_runtime.replay_microbatch_size"
        ),
        validation=L1ValidationConfig(
            root=str(val["root"]),
            base_seed=int(val["base_seed"]),
            instances_per_combination=_positive_int(val["instances_per_combination"], "validation.instances_per_combination"),
            scales=tuple(str(x) for x in val["scales"]),
            scenarios=tuple(str(x) for x in val["scenarios"]),
            load_ratios=tuple(float(x) for x in val["load_ratios"]),
            due_tightness=tuple(str(x) for x in val["due_tightness"]),
            every_iterations=_positive_int(val["every_iterations"], "validation.every_iterations"),
            patience_validations=_positive_int(val["patience_validations"], "validation.patience_validations"),
            min_delta=float(val["min_delta"]),
            deterministic=bool(val["deterministic"]),
        ),
        test_root=str(test["test_root"]),
        minimum_test_instances_per_combination=_positive_int(
            test["minimum_instances_per_combination"], "test.minimum_instances_per_combination"
        ),
        forbid_test_access_during_training=bool(test["forbid_test_access_during_training"]),
        main_training_reward=str(reward["main_training_reward"]),
        potential_shaping_implemented=bool(reward["potential_shaping_implemented"]),
        selected_reward_variant=(None if reward.get("selected_variant") is None else str(reward["selected_variant"])),
        potential_shaping_enabled=bool(reward["potential_shaping_enabled"]),
        potential_eta=(None if reward.get("potential_eta") is None else float(reward["potential_eta"])),
        shaping_calibration_candidates=tuple(float(x) for x in reward["calibration_candidates"]),
        shaping_budget_search=L1ShapingBudgetSearchConfig(
            output_root=str(budget_search["output_root"]),
            candidates=tuple(
                L1BudgetCandidate(
                    label=str(item["label"]),
                    stage_iterations=_tuple4_positive(
                        item["stage_iterations"],
                        f"shaping_budget_search.candidates[{idx}].stage_iterations",
                    ),
                    run_root=str(item["run_root"]),
                )
                for idx, item in enumerate(budget_search["candidates"])
            ),
            stop_after_first_stable=bool(budget_search.get("stop_after_first_stable", True)),
            stop_budget_after_first_invalid_seed=bool(
                budget_search.get("stop_budget_after_first_invalid_seed", True)
            ),
            validation_stop_after_first_failure=bool(
                budget_search.get("validation_stop_after_first_failure", True)
            ),
        ),
        shaping_calibration=L1ShapingCalibrationConfig(
            seeds=tuple(int(x) for x in cal["seeds"]),
            stage_iterations=_tuple4_positive(cal["stage_iterations"], "shaping_calibration.stage_iterations"),
            parallel_envs=_positive_int(cal["parallel_envs"], "shaping_calibration.parallel_envs"),
            rollout_events=_positive_int(cal["rollout_events"], "shaping_calibration.rollout_events"),
            ppo_epochs=_positive_int(cal["ppo_epochs"], "shaping_calibration.ppo_epochs"),
            normalization_episodes=_positive_int(cal["normalization_episodes"], "shaping_calibration.normalization_episodes"),
            normalization_max_graphs=_positive_int(cal["normalization_max_graphs"], "shaping_calibration.normalization_max_graphs"),
            validation_root=str(cal["validation_root"]),
            validation_base_seed=int(cal["validation_base_seed"]),
            validation_instances_per_combination=_positive_int(cal["validation_instances_per_combination"], "shaping_calibration.validation_instances_per_combination"),
            validation_scales=tuple(str(x) for x in cal["validation_scales"]),
            validation_scenarios=tuple(str(x) for x in cal["validation_scenarios"]),
            validation_load_ratios=tuple(float(x) for x in cal["validation_load_ratios"]),
            validation_due_tightness=tuple(str(x) for x in cal["validation_due_tightness"]),
            validation_every_iterations=_positive_int(cal["validation_every_iterations"], "shaping_calibration.validation_every_iterations"),
            patience_validations=_positive_int(cal["patience_validations"], "shaping_calibration.patience_validations"),
            output_root=str(cal["output_root"]),
            resume_existing_runs=bool(cal.get("resume_existing_runs", True)),
            strict_twt_control_gate=bool(cal.get("strict_twt_control_gate", True)),
            validation_stop_after_first_failure=bool(
                cal.get("validation_stop_after_first_failure", True)
            ),
            stop_candidate_after_first_invalid_seed=bool(
                cal.get("stop_candidate_after_first_invalid_seed", True)
            ),
        ),
        shaping_confirmation=L1ShapingConfirmationConfig(
            validation_root=str(confirm["validation_root"]),
            validation_base_seed=int(confirm["validation_base_seed"]),
            validation_instances_per_combination=_positive_int(confirm["validation_instances_per_combination"], "shaping_confirmation.validation_instances_per_combination"),
            validation_scales=tuple(str(x) for x in confirm["validation_scales"]),
            validation_scenarios=tuple(str(x) for x in confirm["validation_scenarios"]),
            validation_load_ratios=tuple(float(x) for x in confirm["validation_load_ratios"]),
            validation_due_tightness=tuple(str(x) for x in confirm["validation_due_tightness"]),
            output_root=str(confirm["output_root"]),
        ),
        resource_decoder_implementation=str(decoder["implementation"]),
        fixed_source_order_exists=bool(decoder["fixed_source_order_exists"]),
        hardware_pilot_stop_after=_positive_int(
            pilot["stage1_stop_after_global_iterations"], "hardware_pilot.stage1_stop_after_global_iterations"
        ),
        stress_start_stage_index=int(stress["start_stage_index"]),
        stress_scale_pool=tuple(str(x) for x in stress["scale_pool"]),
        stress_scenario_pool=tuple(str(x) for x in stress["scenario_pool"]),
        stress_load_ratio_pool=tuple(float(x) for x in stress["load_ratio_pool"]),
        stress_due_tightness_pool=tuple(str(x) for x in stress["due_tightness_pool"]),
        stress_additional_iterations=_positive_int(stress["additional_iterations"], "hardware_pilot.full_stress.additional_iterations"),
    )
    validate_phase_l1_config(cfg, project_cfg=project_cfg)
    return cfg


def validate_phase_l1_config(l1: PhaseL1Config, *, project_cfg=None) -> None:
    if l1.status not in {"alignment_pending_shaping_selection", "frozen_for_formal_training"}:
        raise ValueError("unknown Phase L1 status")
    if l1.formal_training_ready != (l1.status == "frozen_for_formal_training"):
        raise ValueError("formal_training_ready must match the frozen status")
    if tuple(l1.training_seeds) != (0,):
        raise ValueError("Advisor Scheme-2 requires exactly training seed 0")
    if set(l1.validation.scales) != {"S", "M", "L"}:
        raise ValueError("formal validation must cover S/M/L")
    if set(l1.validation.scenarios) != {"D1", "D2", "D3", "D4", "D5"}:
        raise ValueError("formal validation must cover D1-D5")
    if set(round(x, 2) for x in l1.validation.load_ratios) != {0.65, 0.80, 0.95}:
        raise ValueError("formal validation must cover rho={0.65,0.80,0.95}")
    if set(l1.validation.due_tightness) != {"tight", "medium", "loose"}:
        raise ValueError("formal validation must cover all due-tightness levels")
    if l1.validation.root == l1.test_root:
        raise ValueError("validation and test roots must be different")
    if not l1.forbid_test_access_during_training:
        raise ValueError("Phase L1 must forbid fixed-test access during training")
    if l1.minimum_test_instances_per_combination != 1:
        raise ValueError("Advisor Scheme-2 requires one test instance per combination")
    if l1.memory_rollout_storage_mode != "compact_replay_state":
        raise ValueError("Phase L1.2 formal runtime must use compact_replay_state")
    if not 0 < l1.memory_replay_microbatch_size <= l1.minibatch_size:
        raise ValueError("replay microbatch must be within the logical PPO minibatch")
    if not l1.potential_shaping_implemented:
        raise ValueError("Phase L1.1 requires the paper Eq. (16)-(17) shaping implementation")
    if 0.0 not in l1.shaping_calibration_candidates or any(x < 0.0 for x in l1.shaping_calibration_candidates):
        raise ValueError("shaping calibration must include eta=0 and contain no negative eta")
    if l1.formal_training_ready:
        if l1.selected_reward_variant not in {"strict_twt_integral", "potential_shaping"}:
            raise ValueError("frozen formal run must name the selected reward variant")
        if l1.selected_reward_variant == "strict_twt_integral":
            if l1.main_training_reward != "twt_integral":
                raise ValueError("strict TWT frozen protocol must name twt_integral as the training reward")
            if l1.potential_shaping_enabled or l1.potential_eta not in {None, 0.0}:
                raise ValueError("strict TWT selection cannot enable shaping")
        else:
            if l1.main_training_reward != "potential_shaped_twt_integral":
                raise ValueError("potential-shaping frozen protocol must name potential_shaped_twt_integral")
            if not l1.potential_shaping_enabled or l1.potential_eta is None or l1.potential_eta <= 0.0:
                raise ValueError("potential-shaping selection requires a positive eta")
            if float(l1.potential_eta) not in l1.shaping_calibration_candidates:
                raise ValueError("frozen potential eta must come from the predeclared calibration grid")
    else:
        if l1.main_training_reward != "pending_validation_selection":
            raise ValueError("alignment-pending L1.1 must keep reward selection pending")
    cal = l1.shaping_calibration
    if len(cal.seeds) < 2 or len(set(cal.seeds)) != len(cal.seeds):
        raise ValueError("shaping calibration requires at least two distinct diagnostic seeds")
    if not set(cal.validation_scales).issubset({"S", "M", "L"}):
        raise ValueError("invalid shaping-calibration validation scales")
    if not set(cal.validation_scenarios).issubset({"D1", "D2", "D3", "D4", "D5"}):
        raise ValueError("invalid shaping-calibration validation scenarios")
    if not set(cal.validation_due_tightness).issubset({"tight", "medium", "loose"}):
        raise ValueError("invalid shaping-calibration due tightness")
    if not cal.resume_existing_runs:
        raise ValueError("Phase L1.3d requires non-destructive resumable shaping calibration")
    if not cal.strict_twt_control_gate:
        raise ValueError("Phase L1.3d requires the strict-TWT stability gate before positive eta runs")
    if not cal.validation_stop_after_first_failure:
        raise ValueError("Phase L1.3d.1 compact validation must stop after the first policy-level failure")
    if not cal.stop_candidate_after_first_invalid_seed:
        raise ValueError("Phase L1.3d.1 must stop new/resumed training for an eta after its first invalid diagnostic seed")

    budget = l1.shaping_budget_search
    if len(budget.candidates) < 2:
        raise ValueError("Phase L1.3d budget search requires at least two ascending candidates")
    labels = [item.label for item in budget.candidates]
    if len(labels) != len(set(labels)):
        raise ValueError("shaping budget labels must be unique")
    totals = [sum(item.stage_iterations) for item in budget.candidates]
    if any(b <= a for a, b in zip(totals, totals[1:])):
        raise ValueError("shaping budget candidates must be strictly increasing in total iterations")
    for item in budget.candidates:
        if any(int(x) > int(y) for x, y in zip(item.stage_iterations, l1.stage_iterations)):
            raise ValueError("diagnostic shaping budget cannot exceed the formal stage budget")
    first = budget.candidates[0]
    if first.stage_iterations != cal.stage_iterations or first.run_root != cal.output_root:
        raise ValueError("Budget A must reuse the existing L1.3b/L1.3c compact control directory")
    if not budget.stop_after_first_stable or not budget.stop_budget_after_first_invalid_seed:
        raise ValueError("Phase L1.3d requires early budget stopping once stability/failure is known")
    if not budget.validation_stop_after_first_failure:
        raise ValueError("Phase L1.3d budget search requires failure-first validation")

    confirm = l1.shaping_confirmation
    if set(confirm.validation_scales) != {"S", "M", "L"}:
        raise ValueError("shaping confirmation must cover S/M/L")
    if set(confirm.validation_scenarios) != {"D1", "D2", "D3", "D4", "D5"}:
        raise ValueError("shaping confirmation must cover D1-D5")
    if set(round(x, 2) for x in confirm.validation_load_ratios) != {0.65, 0.80, 0.95}:
        raise ValueError("shaping confirmation must cover all paper load ratios")
    if set(confirm.validation_due_tightness) != {"tight", "medium", "loose"}:
        raise ValueError("shaping confirmation must cover all due-tightness levels")
    if confirm.validation_root in {l1.test_root, l1.validation.root}:
        raise ValueError("shaping confirmation must use a separate fixed validation root")
    if l1.resource_decoder_implementation != "global_masked_autoregressive_edge_set":
        raise ValueError("Phase L1 must preserve the K3-tested global edge-set decoder")
    if l1.fixed_source_order_exists:
        raise ValueError("global edge-set decoder must not claim a fixed source order")
    if l1.stress_start_stage_index != 3:
        raise ValueError("L1.1 full stress pilot must start at Stage IV")
    if l1.stress_scale_pool != ("L",) or l1.stress_scenario_pool != ("D4",):
        raise ValueError("L1.1 full stress pilot must force L/D4")
    if l1.stress_load_ratio_pool != (0.95,) or l1.stress_due_tightness_pool != ("tight",):
        raise ValueError("L1.1 full stress pilot must force rho=0.95/tight")

    if project_cfg is not None:
        algo = project_cfg.algo
        if tuple(int(x) for x in algo.training_seeds) != l1.training_seeds:
            raise ValueError("L1 training seeds differ from configs/algo.yaml")
        expected = {
            "parallel_envs": int(algo.parallel_envs),
            "rollout_events": int(algo.rollout_events),
            "minibatch_size": int(algo.minibatch_size),
            "ppo_epochs": int(algo.ppo_epochs),
        }
        actual = {
            "parallel_envs": l1.parallel_envs,
            "rollout_events": l1.rollout_events,
            "minibatch_size": l1.minibatch_size,
            "ppo_epochs": l1.ppo_epochs,
        }
        if actual != expected:
            raise ValueError(f"L1 paper-table runtime mismatch: expected={expected}, actual={actual}")
        if str(project_cfg.env.reward.base) != "twt_integral":
            raise ValueError("project base reward is no longer the exact TWT integral")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_formal_project(l1: PhaseL1Config):
    """Build the effective project config used by a formal run.

    configs/env.yaml intentionally remains strict-TWT by default so diagnostic
    experiments and baselines are not silently changed globally.  Once L1 is
    frozen, only the formal runner receives the validation-selected reward
    override recorded in configs/phase_l1.yaml.
    """
    if not l1.formal_training_ready:
        return load_config(PROJECT_CONFIGS)
    if l1.selected_reward_variant == "strict_twt_integral":
        eta = 0.0
    elif l1.selected_reward_variant == "potential_shaping":
        if l1.potential_eta is None or float(l1.potential_eta) <= 0.0:
            raise ValueError("frozen potential-shaping protocol requires positive eta")
        eta = float(l1.potential_eta)
    else:
        raise ValueError(f"unknown frozen reward variant: {l1.selected_reward_variant!r}")
    return load_config(
        PROJECT_CONFIGS,
        overrides=[
            f"env.reward.optional_potential_shaping={'true' if eta > 0.0 else 'false'}",
            f"env.reward.potential_eta={eta if eta > 0.0 else 'null'}",
        ],
    )


def _replace_unique_yaml_scalar(text: str, pattern: str, replacement: str, *, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label} field in configs/phase_l1.yaml, found {count}")
    return updated


def _write_reward_freeze_yaml(path: str | Path, *, variant: str, eta: float) -> None:
    path = Path(path)
    original = path.read_text(encoding="utf-8")
    enabled = variant == "potential_shaping"
    main_reward = "potential_shaped_twt_integral" if enabled else "twt_integral"
    eta_text = f"{float(eta):.12g}" if enabled else "null"
    text = original
    text = _replace_unique_yaml_scalar(text, r"^  status: .*?$", "  status: frozen_for_formal_training", label="phase status")
    text = _replace_unique_yaml_scalar(text, r"^  formal_training_ready: .*?$", "  formal_training_ready: true", label="formal_training_ready")
    text = _replace_unique_yaml_scalar(text, r"^    main_training_reward: .*?$", f"    main_training_reward: {main_reward}", label="main_training_reward")
    text = _replace_unique_yaml_scalar(text, r"^    selected_variant: .*?$", f"    selected_variant: {variant}", label="selected_variant")
    text = _replace_unique_yaml_scalar(text, r"^    potential_shaping_enabled: .*?$", f"    potential_shaping_enabled: {'true' if enabled else 'false'}", label="potential_shaping_enabled")
    text = _replace_unique_yaml_scalar(text, r"^    potential_eta: .*?$", f"    potential_eta: {eta_text}", label="potential_eta")
    path.write_text(text, encoding="utf-8")


def freeze_reward_protocol(
    *,
    l1_path: str | Path = L1_CONFIG,
    freeze_record_root: str | Path = "result/phase_l1",
    confirmation_recommendation_path: str | Path | None = None,
    budget_selection_path: str | Path | None = None,
) -> dict:
    """Freeze the Tier-2-selected reward protocol after strict evidence checks.

    This is deliberately an explicit user-invoked mutation. It never reads the
    fixed test set and it refuses to freeze if the broad-validation diagnostic
    seeds do not unanimously support the same candidate. The operation is
    idempotent when the already-frozen config matches the saved recommendation.
    """
    l1_path = Path(l1_path)
    base_project = load_config(PROJECT_CONFIGS)
    l1 = load_phase_l1_config(l1_path, project_cfg=None)
    validate_phase_l1_config(
        l1, project_cfg=(build_formal_project(l1) if l1.formal_training_ready else base_project)
    )

    if budget_selection_path is None:
        _, default_budget_json = _budget_selection_paths(l1)
        budget_selection_path = default_budget_json
    budget_selection_path = Path(budget_selection_path)
    if not budget_selection_path.is_file():
        raise FileNotFoundError(f"missing stable-budget selection: {budget_selection_path}")
    budget = json.loads(budget_selection_path.read_text(encoding="utf-8"))
    if budget.get("budget_search_status") != "stable_budget_found":
        raise RuntimeError("reward freeze requires a stable compact calibration budget")
    budget_label = str(budget.get("selected_budget_label"))
    matches = [x for x in l1.shaping_budget_search.candidates if x.label == budget_label]
    if len(matches) != 1:
        raise RuntimeError("saved budget label is absent or ambiguous in configs/phase_l1.yaml")
    budget_cfg = matches[0]
    if tuple(int(x) for x in budget.get("selected_stage_iterations", [])) != budget_cfg.stage_iterations:
        raise RuntimeError("saved budget stage iterations disagree with configs/phase_l1.yaml")
    if str(budget.get("selected_run_root")) != budget_cfg.run_root:
        raise RuntimeError("saved budget run root disagrees with configs/phase_l1.yaml")

    if confirmation_recommendation_path is None:
        confirmation_recommendation_path = (
            Path(l1.shaping_confirmation.output_root)
            / f"budget_{budget_label}"
            / "shaping_confirmation_recommendation.json"
        )
    confirmation_recommendation_path = Path(confirmation_recommendation_path)
    if not confirmation_recommendation_path.is_file():
        raise FileNotFoundError(f"missing Tier-2 confirmation recommendation: {confirmation_recommendation_path}")
    confirmation = json.loads(confirmation_recommendation_path.read_text(encoding="utf-8"))
    if confirmation.get("confirmation_status") != "stable_candidate":
        raise RuntimeError("Tier-2 confirmation is not a stable_candidate; reward protocol remains blocked")
    variant = str(confirmation.get("recommended_variant"))
    eta_raw = confirmation.get("recommended_eta")
    if eta_raw is None:
        raise RuntimeError("stable Tier-2 recommendation has no eta")
    eta = float(eta_raw)
    expected_variant = "strict_twt_integral" if eta == 0.0 else "potential_shaping"
    if variant != expected_variant:
        raise RuntimeError("Tier-2 recommended variant is inconsistent with recommended eta")
    if eta not in l1.shaping_calibration_candidates:
        raise RuntimeError("Tier-2 recommended eta was not in the predeclared calibration candidate grid")
    winners = {str(k): (None if v is None else float(v)) for k, v in dict(confirmation.get("seed_winners", {})).items()}
    for seed in l1.shaping_calibration.seeds:
        if winners.get(str(seed)) != eta:
            raise RuntimeError(
                f"Tier-2 diagnostic seed {seed} does not independently select eta={eta:g}; refuse freeze"
            )

    old_text = l1_path.read_text(encoding="utf-8")
    already_frozen = bool(l1.formal_training_ready)
    if already_frozen:
        if l1.selected_reward_variant != variant or (float(l1.potential_eta or 0.0) != eta):
            raise RuntimeError("config is already frozen to a reward protocol that disagrees with Tier-2")
    else:
        try:
            _write_reward_freeze_yaml(l1_path, variant=variant, eta=eta)
            frozen_l1 = load_phase_l1_config(l1_path, project_cfg=None)
            formal_project = build_formal_project(frozen_l1)
            validate_phase_l1_config(frozen_l1, project_cfg=formal_project)
        except Exception:
            l1_path.write_text(old_text, encoding="utf-8")
            raise

    frozen_l1 = load_phase_l1_config(l1_path, project_cfg=None)
    formal_project = build_formal_project(frozen_l1)
    validate_phase_l1_config(frozen_l1, project_cfg=formal_project)
    effective_enabled = bool(formal_project.env.reward.optional_potential_shaping)
    effective_eta = formal_project.env.reward.potential_eta
    if variant == "potential_shaping":
        if not effective_enabled or not math.isclose(float(effective_eta), eta, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError("effective formal project does not carry the frozen potential eta")
    elif effective_enabled or effective_eta is not None:
        raise RuntimeError("effective strict-TWT formal project unexpectedly enables shaping")

    record_root = Path(freeze_record_root)
    record_root.mkdir(parents=True, exist_ok=True)
    record_path = record_root / "reward_protocol_freeze.json"
    record = {
        "phase": "L1.5",
        "status": "formal_reward_protocol_frozen",
        "selected_variant": variant,
        "selected_eta": eta,
        "production_objective": "total_weighted_tardiness",
        "selection_source": "Tier-2 broad fixed validation; paired TWT ranks; equal S/M/L macro weight",
        "selected_calibration_budget_label": budget_label,
        "selected_calibration_stage_iterations": list(budget_cfg.stage_iterations),
        "diagnostic_seed_winners": winners,
        "confirmation_recommendation_json": str(confirmation_recommendation_path),
        "confirmation_recommendation_sha256": sha256_file(confirmation_recommendation_path),
        "budget_selection_json": str(budget_selection_path),
        "budget_selection_sha256": sha256_file(budget_selection_path),
        "phase_l1_config": str(l1_path),
        "phase_l1_config_sha256": sha256_file(l1_path),
        "effective_formal_optional_potential_shaping": effective_enabled,
        "effective_formal_potential_eta": effective_eta,
        "fixed_test_used_for_selection": False,
        "formal_training_ready": True,
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**record, "freeze_record_json": str(record_path), "already_frozen": already_frozen}


def validate_reward_freeze_record(
    l1: PhaseL1Config,
    *,
    l1_path: str | Path = L1_CONFIG,
    record_path: str | Path = "result/phase_l1/reward_protocol_freeze.json",
) -> dict:
    """Verify that a frozen config is backed by the audited Tier-2 freeze record."""
    if not l1.formal_training_ready:
        raise RuntimeError("reward freeze record is only valid for a formal-ready config")
    record_path = Path(record_path)
    if not record_path.is_file():
        raise FileNotFoundError(
            f"formal reward config is frozen but audit record is missing: {record_path}; "
            "run python phase_l1.py --freeze-reward"
        )
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if payload.get("status") != "formal_reward_protocol_frozen":
        raise RuntimeError("reward freeze record has an invalid status")
    if str(payload.get("selected_variant")) != str(l1.selected_reward_variant):
        raise RuntimeError("reward freeze record variant disagrees with configs/phase_l1.yaml")
    eta_record = float(payload.get("selected_eta", 0.0))
    eta_config = float(l1.potential_eta or 0.0)
    if not math.isclose(eta_record, eta_config, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("reward freeze record eta disagrees with configs/phase_l1.yaml")
    if payload.get("fixed_test_used_for_selection") is not False:
        raise RuntimeError("reward freeze record does not attest that fixed test data were excluded")
    if str(payload.get("phase_l1_config_sha256")) != sha256_file(l1_path):
        raise RuntimeError("configs/phase_l1.yaml changed after the reward protocol was frozen")
    return payload


def _load_eval_minimum(path: str | Path = EVAL_CONFIG) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return int(raw["phase_k"]["formal"]["instances_per_combination"])


def preflight_report(
    *,
    l1_path: str | Path = L1_CONFIG,
    freeze_record_path: str | Path = "result/phase_l1/reward_protocol_freeze.json",
) -> dict:
    base_project = load_config(PROJECT_CONFIGS)
    l1 = load_phase_l1_config(l1_path, project_cfg=None)
    project = build_formal_project(l1) if l1.formal_training_ready else base_project
    validate_phase_l1_config(l1, project_cfg=project)
    phase_j = load_phase_j_config(TRAIN_CONFIG, project_cfg=project)
    throughput = load_throughput_profile(THROUGHPUT_CONFIG)
    if throughput.rollout_action_batch_size > l1.parallel_envs:
        raise ValueError("throughput rollout_action_batch_size cannot exceed formal parallel_envs")
    if throughput.replay_microbatch_size > l1.minibatch_size:
        raise ValueError("throughput replay_microbatch_size cannot exceed the 512 logical minibatch")
    freeze_record = (
        validate_reward_freeze_record(l1, l1_path=l1_path, record_path=freeze_record_path)
        if l1.formal_training_ready else None
    )
    test_min = _load_eval_minimum()
    if test_min < l1.minimum_test_instances_per_combination:
        raise ValueError("configs/eval.yaml formal test count is below the L1 frozen minimum")

    # The formal trainer uses online samplers and receives only a validation root.
    # No test-root argument exists in PhaseJRunSettings; preserve that separation.
    run_fields = set(PhaseJRunSettings.__dataclass_fields__)
    if "test_root" in run_fields or "test_records" in run_fields:
        raise ValueError("formal trainer unexpectedly exposes fixed-test data")

    # Training-stage distributions are inherited from the already validated Phase-J
    # paper curriculum, while L1 freezes only previously-unspecified runtime choices.
    if tuple(phase_j.normalization.scale_pool) != ("S", "M", "L"):
        raise ValueError("training normalizer must be fitted on S/M/L only")

    stages = project.curriculum.stages
    if tuple(stages[0].allowed_scales) != ("S",):
        raise ValueError("Stage I must remain S-only")
    if tuple(stages[1].allowed_scales) != ("S",):
        raise ValueError("paper Stage II is small-scale and must remain S-only")
    if tuple(stages[2].allowed_scales) != ("M",):
        raise ValueError("L1.1 Stage III representative-scale choice must be M-only")
    if tuple(stages[3].allowed_scales) != ("S", "M", "L"):
        raise ValueError("paper Stage IV must be the explicit S/M/L joint stage")
    if tuple(phase_j.stage_runtime["full_set_reconfiguration"]["scenario_pool"]) != ("D1", "D2", "D3", "D4"):
        raise ValueError("Stage III must introduce D2-D4 burst/mix dynamics without D5")

    paths = [*PROJECT_CONFIGS, TRAIN_CONFIG, str(l1_path), EVAL_CONFIG, THROUGHPUT_CONFIG]
    checksums = {str(p): sha256_file(p) for p in paths}
    return {
        "phase": "L1.4",
        "status": (
            "FORMAL_REWARD_FROZEN_PREFLIGHT_PASSED"
            if l1.formal_training_ready else
            "REWARD_SELECTION_PENDING_PREFLIGHT_PASSED"
        ),
        "formal_training_ready": l1.formal_training_ready,
        "stage_iterations_upper_bounds": list(l1.stage_iterations),
        "training_seeds": list(l1.training_seeds),
        "parallel_envs": l1.parallel_envs,
        "rollout_events_per_iteration": l1.rollout_events,
        "ppo_epochs": l1.ppo_epochs,
        "minibatch_size": l1.minibatch_size,
        "planned_iterations_per_seed": l1.planned_iterations_per_seed,
        "planned_event_interactions_per_seed": l1.planned_event_interactions_per_seed,
        "planned_event_interactions_all_seeds": l1.planned_event_interactions_all_seeds,
        "validation_combinations": l1.validation_combinations,
        "validation_instances_per_combination": l1.validation.instances_per_combination,
        "validation_total_instances": l1.validation_instances,
        "validation_every_iterations": l1.validation.every_iterations,
        "validation_patience": l1.validation.patience_validations,
        "formal_test_minimum_per_combination": test_min,
        "main_reward": l1.main_training_reward,
        "potential_shaping": {
            "implemented": l1.potential_shaping_implemented,
            "selected_variant": l1.selected_reward_variant,
            "enabled": l1.potential_shaping_enabled,
            "eta": l1.potential_eta,
            "calibration_candidates": list(l1.shaping_calibration_candidates),
        },
        "effective_formal_reward": {
            "base": str(project.env.reward.base),
            "optional_potential_shaping": bool(project.env.reward.optional_potential_shaping),
            "potential_eta": project.env.reward.potential_eta,
        },
        "reward_freeze_record": freeze_record,
        "curriculum_alignment": {
            "stage_I": "S/D1 fixed configuration",
            "stage_II": "S/D1 rho=0.65/0.80 single-resource",
            "stage_III": "M/D1-D4 full-set reconfiguration (implementation choice: representative M)",
            "stage_IV": "S/M/L D1-D5 joint finetuning",
        },
        "shaping_budget_search": {
            "mode": "strict_twt_control_only_minimum_stable_budget",
            "output_root": l1.shaping_budget_search.output_root,
            "candidates": [
                {
                    "label": item.label,
                    "stage_iterations": list(item.stage_iterations),
                    "total_iterations": int(sum(item.stage_iterations)),
                    "events_per_seed": int(sum(item.stage_iterations) * l1.shaping_calibration.rollout_events),
                    "run_root": item.run_root,
                }
                for item in l1.shaping_budget_search.candidates
            ],
            "stop_after_first_stable": l1.shaping_budget_search.stop_after_first_stable,
            "stop_budget_after_first_invalid_seed": l1.shaping_budget_search.stop_budget_after_first_invalid_seed,
            "validation_stop_after_first_failure": l1.shaping_budget_search.validation_stop_after_first_failure,
        },
        "shaping_calibration": {
            "tier": "budget_selected_compact_screen_only",
            "seeds": list(l1.shaping_calibration.seeds),
            "stage_iterations": list(l1.shaping_calibration.stage_iterations),
            "parallel_envs": l1.shaping_calibration.parallel_envs,
            "rollout_events": l1.shaping_calibration.rollout_events,
            "ppo_epochs": l1.shaping_calibration.ppo_epochs,
            "validation_root": l1.shaping_calibration.validation_root,
            "selection_metric": "scale_macro_paired_rank",
            "resume_existing_runs": l1.shaping_calibration.resume_existing_runs,
            "strict_twt_control_gate": l1.shaping_calibration.strict_twt_control_gate,
            "validation_stop_after_first_failure": l1.shaping_calibration.validation_stop_after_first_failure,
            "stop_candidate_after_first_invalid_seed": l1.shaping_calibration.stop_candidate_after_first_invalid_seed,
        },
        "shaping_confirmation": {
            "mode": "evaluation_only_no_gradient_updates",
            "validation_root": l1.shaping_confirmation.validation_root,
            "validation_instances_per_combination": l1.shaping_confirmation.validation_instances_per_combination,
            "validation_combinations": (
                len(l1.shaping_confirmation.validation_scales)
                * len(l1.shaping_confirmation.validation_scenarios)
                * len(l1.shaping_confirmation.validation_load_ratios)
                * len(l1.shaping_confirmation.validation_due_tightness)
            ),
            "selection_metric": "scale_macro_paired_rank",
        },
        "memory_safe_rollout_storage": l1.memory_rollout_storage_mode,
        "replay_microbatch_size": throughput.replay_microbatch_size,
        "legacy_safe_replay_microbatch_size": l1.memory_replay_microbatch_size,
        "logical_minibatch_size": l1.minibatch_size,
        "throughput_execution": {
            "rollout_action_batch_size": throughput.rollout_action_batch_size,
            "replay_microbatch_size": throughput.replay_microbatch_size,
            "replay_microbatch_min_size": throughput.replay_microbatch_min_size,
            "cuda_oom_fallback": throughput.cuda_oom_fallback,
            "progress_every_events": throughput.progress_every_events,
            "diagnostics": throughput.diagnostics,
            "semantic_context_device": "cpu",
            "learned_graph_encoder_device": "accelerator",
            "changes_training_semantics": False,
        },
        "resource_decode_order": "global_edge_set_no_fixed_source_sequence",
        "test_data_visible_to_training": False,
        "config_sha256": checksums,
    }


def write_frozen_manifest(report: dict, *, root: str | Path = "result/phase_l1") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("frozen_training_plan.json" if report.get("formal_training_ready") else "alignment_training_plan.json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return path


def prepare_validation_suite(l1: PhaseL1Config, project_cfg) -> tuple:
    return ensure_fixed_validation_suite(
        project_cfg,
        root=l1.validation.root,
        base_seed=l1.validation.base_seed,
        scales=l1.validation.scales,
        scenarios=l1.validation.scenarios,
        load_ratios=l1.validation.load_ratios,
        due_tightness=l1.validation.due_tightness,
        instances_per_combination=l1.validation.instances_per_combination,
    )


def build_run_settings(
    l1: PhaseL1Config,
    phase_j,
    *,
    seed: int,
    device: str | None = None,
    pilot: bool = False,
    stress_pilot: bool = False,
    resume_checkpoint: str | None = None,
) -> PhaseJRunSettings:
    seed = int(seed)
    if seed not in l1.training_seeds:
        raise ValueError(f"seed {seed} is not in frozen L1 seeds {l1.training_seeds}")
    if pilot and stress_pilot:
        raise ValueError("pilot and stress_pilot are mutually exclusive")
    throughput = load_throughput_profile(THROUGHPUT_CONFIG)
    if stress_pilot:
        run_name = f"l1_full_stress_pilot_seed{seed}"
    elif pilot:
        run_name = f"l1_hardware_pilot_seed{seed}"
    else:
        run_name = f"{l1.run_name_prefix}{seed}"
    return PhaseJRunSettings(
        stage_iterations=l1.stage_iterations,
        parallel_envs=l1.parallel_envs,
        rollout_events=l1.rollout_events,
        training_seed=seed,
        normalization_episodes=l1.normalization_episodes,
        normalization_max_graphs=l1.normalization_max_graphs,
        max_episode_decisions=phase_j.runtime.max_episode_decisions,
        device=phase_j.runtime.device if device is None else str(device),
        run_root=l1.run_root,
        run_name=run_name,
        validation_root=l1.validation.root,
        validation_base_seed=l1.validation.base_seed,
        validation_instances_per_combination=l1.validation.instances_per_combination,
        validation_scales=l1.validation.scales,
        validation_scenarios=l1.validation.scenarios,
        validation_load_ratios=l1.validation.load_ratios,
        validation_due_tightness=l1.validation.due_tightness,
        validation_every_iterations=l1.validation.every_iterations,
        patience_validations=l1.validation.patience_validations,
        validation_min_delta=l1.validation.min_delta,
        checkpoint_every_iterations=phase_j.runtime.checkpoint_every_iterations,
        restore_best_before_next_stage=phase_j.runtime.restore_best_before_next_stage,
        validate_at_stage_end=True,
        ppo_epochs_override=None,
        stop_after_global_iterations=(
            l1.hardware_pilot_stop_after if pilot else
            (sum(l1.stage_iterations[:l1.stress_start_stage_index]) + l1.stress_additional_iterations if stress_pilot else None)
        ),
        resume_checkpoint=resume_checkpoint,
        start_stage_index=(l1.stress_start_stage_index if stress_pilot else 0),
        forced_scale_pool=(l1.stress_scale_pool if stress_pilot else None),
        forced_scenario_pool=(l1.stress_scenario_pool if stress_pilot else None),
        forced_load_ratio_pool=(l1.stress_load_ratio_pool if stress_pilot else None),
        forced_due_tightness_pool=(l1.stress_due_tightness_pool if stress_pilot else None),
        progress_every_events=(512 if stress_pilot else throughput.progress_every_events),
        normalization_progress_every_graphs=(128 if stress_pilot else 0),
        rollout_storage_mode_override=l1.memory_rollout_storage_mode,
        replay_microbatch_size_override=throughput.replay_microbatch_size,
        replay_microbatch_min_size_override=throughput.replay_microbatch_min_size,
        replay_microbatch_oom_fallback=throughput.cuda_oom_fallback,
        rollout_action_batch_size=min(throughput.rollout_action_batch_size, l1.parallel_envs),
        throughput_diagnostics=throughput.diagnostics,
        use_multiprocess_rollout=bool(
            throughput.formal_strategy1.enabled and not (pilot or stress_pilot)
        ),
        multiprocess_worker_processes=int(throughput.formal_strategy1.worker_processes),
        multiprocess_worker_torch_threads=int(throughput.formal_strategy1.worker_torch_threads),
        multiprocess_start_method=str(throughput.formal_strategy1.start_method),
        replay_static_processing_cache=bool(throughput.formal_strategy1.static_processing_cache),
        trusted_replay_normalization=bool(throughput.formal_strategy1.trusted_replay_normalization),
        cross_epoch_materialize_cache=bool(
            throughput.formal_strategy1.enabled
            and throughput.formal_strategy1.cross_epoch_materialize_cache
            and not (pilot or stress_pilot)
        ),
        materialize_cache_max_mib=int(
            throughput.formal_strategy1.materialize_cache_max_mib
            if not (pilot or stress_pilot) else 0
        ),
    )


def run_formal_seed(
    *,
    seed: int,
    device: str | None = None,
    pilot: bool = False,
    stress_pilot: bool = False,
    resume: bool = False,
):
    base_project = load_config(PROJECT_CONFIGS)
    l1 = load_phase_l1_config(L1_CONFIG, project_cfg=None)
    if not l1.formal_training_ready:
        validate_phase_l1_config(l1, project_cfg=base_project)
        project = base_project
    else:
        project = build_formal_project(l1)
        validate_phase_l1_config(l1, project_cfg=project)
    phase_j = load_phase_j_config(TRAIN_CONFIG, project_cfg=project)
    if l1.formal_training_ready:
        validate_reward_freeze_record(l1, l1_path=L1_CONFIG)
    if not (pilot or stress_pilot) and not l1.formal_training_ready:
        raise RuntimeError(
            "Phase L1 blocks formal --run-seed/--resume-seed until the Tier-2-selected "
            "reward variant/eta is explicitly frozen."
        )
    resume_path = None
    if resume:
        run_name = f"{l1.run_name_prefix}{int(seed)}"
        resume_candidate = Path(l1.run_root) / run_name / "checkpoints" / "latest.pt"
        if not resume_candidate.is_file():
            raise FileNotFoundError(resume_candidate)
        resume_path = str(resume_candidate)
    settings = build_run_settings(
        l1, phase_j, seed=int(seed), device=device, pilot=pilot, stress_pilot=stress_pilot,
        resume_checkpoint=resume_path,
    )
    return PhaseJTrainer(
        project,
        phase_j,
        settings,
        config_paths=PROJECT_CONFIGS + (L1_CONFIG, THROUGHPUT_CONFIG),
    ).run()


def run_throughput_pilot(*, seed: int, device: str | None = None):
    """Run one isolated optimized iteration from a formal latest.pt.

    The source formal checkpoint is read-only. The diagnostic continuation writes
    to a separate run directory, so benchmarking cannot corrupt Seed 0.
    """
    base_project = load_config(PROJECT_CONFIGS)
    l1 = load_phase_l1_config(L1_CONFIG, project_cfg=None)
    if not l1.formal_training_ready:
        raise RuntimeError("throughput pilot requires a frozen formal reward protocol")
    project = build_formal_project(l1)
    validate_phase_l1_config(l1, project_cfg=project)
    validate_reward_freeze_record(l1, l1_path=L1_CONFIG)
    phase_j = load_phase_j_config(TRAIN_CONFIG, project_cfg=project)
    throughput = load_throughput_profile(THROUGHPUT_CONFIG)
    seed = int(seed)
    source_run = Path(l1.run_root) / f"{l1.run_name_prefix}{seed}"
    latest = source_run / "checkpoints" / "latest.pt"
    if not latest.is_file():
        raise FileNotFoundError(latest)
    payload = load_training_checkpoint(latest, map_location="cpu")
    global_iteration = int(payload["curriculum_state"]["state"]["global_iteration"])
    run_name = f"{throughput.pilot.run_name_prefix}{seed}_from_g{global_iteration}"
    settings = build_run_settings(
        l1, phase_j, seed=seed, device=device, pilot=False, stress_pilot=False,
        resume_checkpoint=str(latest),
    )
    settings = replace(
        settings,
        run_name=run_name,
        stop_after_global_iterations=global_iteration + throughput.pilot.additional_iterations,
        validation_every_iterations=10**9,
        validate_at_stage_end=False,
        restore_best_before_next_stage=False,
        throughput_diagnostics=True,
    )
    result = PhaseJTrainer(
        project, phase_j, settings,
        config_paths=PROJECT_CONFIGS + (L1_CONFIG, THROUGHPUT_CONFIG),
    ).run()
    log_path = Path(result.run_dir) / "train_log.csv"
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8", newline="")))
    latest_row = rows[-1] if rows else {}
    return {
        "result": result,
        "source_checkpoint": str(latest),
        "source_global_iteration": global_iteration,
        "pilot_global_iteration": result.global_iterations,
        "wall_seconds": (None if not latest_row else float(latest_row["wall_seconds"])),
        "rollout_action_batch_size": throughput.rollout_action_batch_size,
        "replay_microbatch_size": throughput.replay_microbatch_size,
        "logical_minibatch_size": l1.minibatch_size,
        "ppo_epochs": l1.ppo_epochs,
    }


def _eta_slug(eta: float) -> str:
    return ("0" if float(eta) == 0.0 else f"{float(eta):.4g}").replace(".", "p")


def _project_for_eta(eta: float):
    eta = float(eta)
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    overrides = [
        f"env.reward.optional_potential_shaping={'true' if eta > 0.0 else 'false'}",
        f"env.reward.potential_eta={eta if eta > 0.0 else 'null'}",
    ]
    return load_config(PROJECT_CONFIGS, overrides=overrides)


def _average_ranks(values: dict[float, float]) -> dict[float, float]:
    """Return average ranks (1=best) with deterministic tie handling."""
    ordered = sorted((float(v), float(k)) for k, v in values.items())
    out: dict[float, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(ordered[j][0], ordered[i][0], rel_tol=0.0, abs_tol=1e-12):
            j += 1
        avg_rank = ((i + 1) + j) / 2.0
        for _, eta in ordered[i:j]:
            out[eta] = float(avg_rank)
        i = j
    return out


def _macro_rank_analysis(instance_rows: list[dict], *, candidates, seeds) -> tuple[list[dict], dict]:
    """Paired, equal-scale ranking for eta selection.

    Raw TWT magnitudes grow strongly with problem scale, so L1.3d does *not* pool
    S/M/L TWT values into one selection mean.  Instead, on each fixed instance we
    rank candidates using paired TWT, average ranks within each scale, then give
    S/M/L equal weight.  Failed/non-terminating candidate runs are ineligible and
    never receive an invented finite TWT penalty.
    """
    candidates = tuple(float(x) for x in candidates)
    seeds = tuple(int(x) for x in seeds)
    rows_by_key = {
        (float(r["eta"]), int(r["seed"]), str(r["instance_id"])): r
        for r in instance_rows
    }

    # The fixed suite must be identical across eta for a given seed. Use the union
    # as the audit target, then require every eligible candidate to cover it fully.
    ids_by_seed: dict[int, set[str]] = {}
    meta_by_seed_id: dict[tuple[int, str], dict] = {}
    for r in instance_rows:
        seed = int(r["seed"]); iid = str(r["instance_id"])
        ids_by_seed.setdefault(seed, set()).add(iid)
        meta_by_seed_id[(seed, iid)] = r

    eligible: dict[float, bool] = {}
    for eta in candidates:
        ok = True
        for seed in seeds:
            ids = ids_by_seed.get(seed, set())
            if not ids:
                ok = False; break
            for iid in ids:
                r = rows_by_key.get((eta, seed, iid))
                if r is None or str(r["status"]) != "completed" or r.get("twt") in {None, "", "None"}:
                    ok = False; break
                try:
                    if not math.isfinite(float(r["twt"])):
                        ok = False; break
                except (TypeError, ValueError):
                    ok = False; break
            if not ok:
                break
        eligible[eta] = ok

    eligible_etas = tuple(eta for eta in candidates if eligible[eta])
    per_seed_candidate: dict[tuple[int, float], dict] = {}
    seed_winners: dict[int, float | None] = {}

    for seed in seeds:
        ids = sorted(ids_by_seed.get(seed, set()))
        scale_ranks: dict[float, dict[str, list[float]]] = {
            eta: {} for eta in eligible_etas
        }
        raw_twts: dict[float, list[float]] = {eta: [] for eta in eligible_etas}
        baseline = 0.0 if 0.0 in eligible_etas else None
        pair_counts: dict[float, list[int]] = {eta: [0, 0, 0] for eta in eligible_etas}  # win/tie/loss vs eta0

        for iid in ids:
            values = {
                eta: float(rows_by_key[(eta, seed, iid)]["twt"])
                for eta in eligible_etas
            }
            ranks = _average_ranks(values)
            scale = str(meta_by_seed_id[(seed, iid)]["scale"])
            for eta in eligible_etas:
                scale_ranks[eta].setdefault(scale, []).append(float(ranks[eta]))
                raw_twts[eta].append(values[eta])
                if baseline is not None and eta != 0.0:
                    base_value = values[0.0]
                    if values[eta] < base_value - 1e-12:
                        pair_counts[eta][0] += 1
                    elif values[eta] > base_value + 1e-12:
                        pair_counts[eta][2] += 1
                    else:
                        pair_counts[eta][1] += 1

        for eta in eligible_etas:
            scale_means = {
                scale: float(sum(vals) / len(vals))
                for scale, vals in scale_ranks[eta].items() if vals
            }
            macro_rank = (
                float(sum(scale_means.values()) / len(scale_means))
                if scale_means else float("inf")
            )
            wins, ties, losses = pair_counts[eta]
            per_seed_candidate[(seed, eta)] = {
                "macro_mean_rank": macro_rank,
                "mean_raw_twt_diagnostic": float(sum(raw_twts[eta]) / len(raw_twts[eta])) if raw_twts[eta] else None,
                "scale_S_mean_rank": scale_means.get("S"),
                "scale_M_mean_rank": scale_means.get("M"),
                "scale_L_mean_rank": scale_means.get("L"),
                "wins_vs_eta0": wins,
                "ties_vs_eta0": ties,
                "losses_vs_eta0": losses,
            }

        if eligible_etas:
            values = {eta: per_seed_candidate[(seed, eta)]["macro_mean_rank"] for eta in eligible_etas}
            best_value = min(values.values())
            best_etas = [eta for eta, value in values.items() if math.isclose(value, best_value, abs_tol=1e-12)]
            seed_winners[seed] = best_etas[0] if len(best_etas) == 1 else None
        else:
            seed_winners[seed] = None

    summary: list[dict] = []
    for eta in candidates:
        metrics = [per_seed_candidate[(seed, eta)] for seed in seeds if (seed, eta) in per_seed_candidate]
        mean_macro = (
            float(sum(float(m["macro_mean_rank"]) for m in metrics) / len(metrics))
            if len(metrics) == len(seeds) else float("inf")
        )
        summary.append({
            "eta": eta,
            "eligible": bool(eligible[eta]),
            "aggregate_macro_mean_rank": mean_macro,
            "mean_raw_twt_diagnostic": (
                None if not metrics else float(sum(float(m["mean_raw_twt_diagnostic"]) for m in metrics) / len(metrics))
            ),
            **{
                f"seed_{seed}_macro_mean_rank": (
                    per_seed_candidate[(seed, eta)]["macro_mean_rank"] if (seed, eta) in per_seed_candidate else None
                ) for seed in seeds
            },
            **{
                f"seed_{seed}_wins_vs_eta0": (
                    per_seed_candidate[(seed, eta)]["wins_vs_eta0"] if (seed, eta) in per_seed_candidate else None
                ) for seed in seeds
            },
        })

    return summary, {
        "eligible_etas": list(eligible_etas),
        "seed_winners": {str(k): v for k, v in seed_winners.items()},
    }


def _compact_shortlist(instance_rows: list[dict], *, candidates, seeds) -> tuple[list[dict], dict]:
    summary, analysis = _macro_rank_analysis(instance_rows, candidates=candidates, seeds=seeds)
    eligible_rows = [r for r in summary if bool(r["eligible"]) and math.isfinite(float(r["aggregate_macro_mean_rank"]))]
    eta0_ok = any(float(r["eta"]) == 0.0 for r in eligible_rows)
    positive = sorted(
        [r for r in eligible_rows if float(r["eta"]) > 0.0],
        key=lambda r: (float(r["aggregate_macro_mean_rank"]), float(r["eta"])),
    )
    if not eta0_ok or not positive:
        status = "inconclusive"
        shortlist: list[float] = []
        note = (
            "Compact screen cannot form a strict-TWT-versus-positive-shaping shortlist: "
            "eta=0 or all positive eta candidates failed the fixed validation screen."
        )
    else:
        # Keep strict TWT as the scientific control plus up to the two strongest
        # positive candidates. Tier-2 confirmation is evaluation-only, so this
        # modest shortlist does not multiply training cost.
        shortlist = [0.0] + [float(r["eta"]) for r in positive[:2]]
        status = "ready_for_confirmation"
        note = (
            "Tier-1 is a shortlist only. Run --shaping-confirmation on the broader "
            "fixed validation suite before freezing any reward variant."
        )

    leading_eta = None
    if eligible_rows:
        best_value = min(float(r["aggregate_macro_mean_rank"]) for r in eligible_rows)
        best = [float(r["eta"]) for r in eligible_rows if math.isclose(float(r["aggregate_macro_mean_rank"]), best_value, abs_tol=1e-12)]
        if len(best) == 1:
            leading_eta = best[0]

    recommendation = {
        "compact_screen_status": status,
        "screen_leading_eta": leading_eta,
        "confirmation_candidates": shortlist,
        "recommended_eta": None,
        "recommended_variant": None,
        "seed_winners": analysis["seed_winners"],
        "selection_basis": "paired per-instance TWT ranks, macro-averaged with equal S/M/L weight",
        "formal_training_ready": False,
        "confirmation_required": True,
        "note": note,
    }
    return summary, recommendation


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _stage4_checkpoint_evaluation(project, *, checkpoint: str | Path, records, device: str | None, max_episode_decisions: int):
    """Evaluate a saved final Stage-IV checkpoint on fixed validation records."""
    from data.io import load_instance_csv
    from agent.training.trainer import resolve_device

    records = tuple(records)
    if not records:
        raise ValueError("fixed validation records are empty")
    dev = resolve_device("auto" if device is None else str(device))
    payload = load_training_checkpoint(checkpoint, map_location="cpu")
    normalizer = payload["normalizer"]
    probe_env = AssemblyEnv(project)
    probe_env.reset(load_instance_csv(records[0].path))
    reference_graph = normalizer.transform(probe_env.graph())
    policy = EGDMCompositePolicy(project, reference_graph).to(dev)
    policy.load_state_dict(payload["policy_state"])
    policy.eval()
    return evaluate_fixed_validation(
        project,
        policy=policy,
        normalizer=normalizer,
        records=records,
        stage_index=3,
        deterministic=True,
        device=dev,
        max_episode_decisions=int(max_episode_decisions),
        return_instance_results=True,
    )


def _instance_result_rows(*, eta: float, seed: int, tier: str, results) -> list[dict]:
    out = []
    for r in results:
        row = asdict(r)
        row.update({"eta": float(eta), "seed": int(seed), "tier": str(tier)})
        # Put identifiers first for human-readable CSVs.
        out.append({
            "tier": row.pop("tier"), "eta": row.pop("eta"), "seed": row.pop("seed"), **row
        })
    return out


def _checkpoint_progress(path: str | Path) -> dict:
    """Read only the semantic progress needed by the resumable calibration driver."""
    payload = load_training_checkpoint(path, map_location="cpu")
    curriculum = payload.get("curriculum_state", {})
    state = dict(curriculum.get("state", {}))
    run_settings = dict(payload.get("run_settings", {}))
    return {
        "finished": bool(state.get("finished", False)),
        "stage_index": int(state.get("stage_index", 0)),
        "stage_iteration": int(state.get("stage_iteration", 0)),
        "global_iteration": int(state.get("global_iteration", 0)),
        "run_settings": run_settings,
    }


def _assert_calibration_resume_compatible(progress: dict, *, cal, seed: int) -> None:
    """Refuse unsafe resume when the checkpoint belongs to a different run budget."""
    settings = dict(progress.get("run_settings", {}))
    checks = {
        "training_seed": int(seed),
        "parallel_envs": int(cal.parallel_envs),
        "rollout_events": int(cal.rollout_events),
        "stage_iterations": tuple(int(x) for x in cal.stage_iterations),
        "ppo_epochs_override": int(cal.ppo_epochs),
    }
    for key, expected in checks.items():
        if key not in settings:
            raise RuntimeError(f"resume checkpoint is missing run setting {key!r}")
        actual = settings[key]
        if key == "stage_iterations":
            actual = tuple(int(x) for x in actual)
        elif key in {"training_seed", "parallel_envs", "rollout_events", "ppo_epochs_override"}:
            actual = int(actual)
        if actual != expected:
            raise RuntimeError(
                f"resume checkpoint setting mismatch for {key}: expected={expected!r}, actual={actual!r}"
            )


def _typed_validation_instance_row(raw: dict, *, eta: float, seed: int, tier: str) -> dict:
    def opt_float(value):
        if value in {None, "", "None", "null"}:
            return None
        return float(value)

    return {
        "tier": str(tier),
        "eta": float(eta),
        "seed": int(seed),
        "instance_id": str(raw["instance_id"]),
        "scale": str(raw["scale"]),
        "scenario": str(raw["scenario"]),
        "target_load_ratio": float(raw["target_load_ratio"]),
        "due_tightness": str(raw["due_tightness"]),
        "status": str(raw["status"]),
        "twt": opt_float(raw.get("twt")),
        "decisions": int(raw.get("decisions", 0)),
        "reconfigurations": int(raw.get("reconfigurations", 0)),
        "simulation_time": float(raw.get("simulation_time", 0.0)),
        "completed_orders": int(raw.get("completed_orders", 0)),
        "total_orders": int(raw.get("total_orders", 0)),
        "completed_operations": int(raw.get("completed_operations", 0)),
        "total_operations": int(raw.get("total_operations", 0)),
        "reward_identity_error": opt_float(raw.get("reward_identity_error")),
        "failure_reason": str(raw.get("failure_reason", "")),
    }


def _final_stage4_from_training_logs(
    run_dir: str | Path, *, eta: float, seed: int, checkpoint: str | Path
) -> tuple[dict, list[dict]] | None:
    """Recover the already-computed final Stage-IV validation without re-running it.

    Phase L1.3b used validation_every_iterations=999 with only 8 Stage-IV updates,
    so the Stage-IV validation written by the trainer is the final fixed-suite
    validation for that completed checkpoint. Reusing those CSV rows avoids hours
    of duplicate non-termination evaluation after an interruption.
    """
    run_dir = Path(run_dir)
    validation_path = run_dir / "validation_log.csv"
    instance_path = run_dir / "validation_instance_log.csv"
    if not validation_path.is_file() or not instance_path.is_file():
        return None
    validation_rows = _read_csv(validation_path)
    stage4 = [r for r in validation_rows if int(r.get("stage_index", -1)) == 3]
    if not stage4:
        return None
    final_summary = max(stage4, key=lambda r: int(r["global_iteration"]))
    global_iteration = int(final_summary["global_iteration"])

    raw_instances = [
        r for r in _read_csv(instance_path)
        if int(r.get("stage_index", -1)) == 3
        and int(r.get("global_iteration", -1)) == global_iteration
    ]
    # A crash/retry around a validation boundary can append duplicate rows. Keep
    # the last row for each fixed instance rather than double-counting it.
    dedup: dict[str, dict] = {}
    for row in raw_instances:
        dedup[str(row["instance_id"])] = row
    expected = int(final_summary.get("instances", len(dedup)))
    if len(dedup) != expected:
        return None
    instance_rows = [
        _typed_validation_instance_row(row, eta=eta, seed=seed, tier="compact")
        for row in dedup.values()
    ]
    failed = int(final_summary.get("failed_instances", 0))
    try:
        mean_twt = float(final_summary.get("mean_twt", "inf"))
    except (TypeError, ValueError):
        mean_twt = float("inf")
    run_status = "valid" if failed == 0 and math.isfinite(mean_twt) else "invalid_policy"
    run_row = {
        "eta": float(eta),
        "seed": int(seed),
        "run_status": run_status,
        "stage4_validation_mean_twt_diagnostic": mean_twt,
        "stage4_completed_instances": int(final_summary.get("completed_instances", 0)),
        "stage4_failed_instances": failed,
        "stage4_nonterminating_instances": int(final_summary.get("nonterminating_instances", 0)),
        "stage4_deadlocked_instances": int(final_summary.get("deadlocked_instances", 0)),
        "stage4_failure_rate": float(final_summary.get("failure_rate", 0.0)),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
    }
    return run_row, instance_rows


def _run_cache_paths(run_dir: str | Path) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    return run_dir / "shaping_final_run.json", run_dir / "shaping_final_instance_results.csv"


def _save_run_cache(run_dir: str | Path, run_row: dict, instance_rows: list[dict]) -> None:
    meta_path, instance_path = _run_cache_paths(run_dir)
    meta_path.write_text(
        json.dumps(_json_safe(run_row), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(instance_path, instance_rows)


def _load_run_cache(run_dir: str | Path) -> tuple[dict, list[dict]] | None:
    meta_path, instance_path = _run_cache_paths(run_dir)
    if not meta_path.is_file() or not instance_path.is_file():
        return None
    row = json.loads(meta_path.read_text(encoding="utf-8"))
    if row.get("stage4_validation_mean_twt_diagnostic") is None and row.get("run_status") != "valid":
        row["stage4_validation_mean_twt_diagnostic"] = float("inf")
    instances = [
        _typed_validation_instance_row(r, eta=float(row["eta"]), seed=int(row["seed"]), tier="compact")
        for r in _read_csv(instance_path)
    ]
    return row, instances


def _calibration_settings(*, l1, cal, phase_j, seed: int, run_name: str, device: str | None, resume_checkpoint: str | None):
    return PhaseJRunSettings(
        stage_iterations=cal.stage_iterations,
        parallel_envs=cal.parallel_envs,
        rollout_events=cal.rollout_events,
        training_seed=int(seed),
        normalization_episodes=cal.normalization_episodes,
        normalization_max_graphs=cal.normalization_max_graphs,
        max_episode_decisions=phase_j.runtime.max_episode_decisions,
        device=phase_j.runtime.device if device is None else str(device),
        run_root=str(cal.output_root), run_name=run_name,
        validation_root=cal.validation_root,
        validation_base_seed=cal.validation_base_seed,
        validation_instances_per_combination=cal.validation_instances_per_combination,
        validation_scales=cal.validation_scales,
        validation_scenarios=cal.validation_scenarios,
        validation_load_ratios=cal.validation_load_ratios,
        validation_due_tightness=cal.validation_due_tightness,
        validation_every_iterations=cal.validation_every_iterations,
        patience_validations=cal.patience_validations,
        validation_min_delta=0.0,
        checkpoint_every_iterations=phase_j.runtime.checkpoint_every_iterations,
        restore_best_before_next_stage=True,
        validate_at_stage_end=True,
        validation_stop_after_first_failure=cal.validation_stop_after_first_failure,
        ppo_epochs_override=cal.ppo_epochs,
        resume_checkpoint=resume_checkpoint,
        rollout_storage_mode_override=l1.memory_rollout_storage_mode,
        replay_microbatch_size_override=l1.memory_replay_microbatch_size,
    )


def _run_or_resume_calibration_candidate(
    *, l1, cal, eta: float, seed: int, device: str | None
) -> tuple[dict, list[dict], str]:
    project = _project_for_eta(eta)
    phase_j = load_phase_j_config(TRAIN_CONFIG, project_cfg=project)
    run_name = f"eta_{_eta_slug(eta)}_seed{seed}"
    run_dir = Path(cal.output_root) / run_name
    latest = run_dir / "checkpoints" / "latest.pt"
    action = "NEW"
    resume_path: str | None = None

    if latest.is_file():
        progress = _checkpoint_progress(latest)
        _assert_calibration_resume_compatible(progress, cal=cal, seed=int(seed))
        if progress["finished"]:
            cached = _load_run_cache(run_dir)
            if cached is None:
                cached = _final_stage4_from_training_logs(
                    run_dir, eta=float(eta), seed=int(seed), checkpoint=latest
                )
                if cached is None:
                    # Fallback for unusual legacy artifacts only. Ordinary L1.3b
                    # completed runs are recovered from their existing CSVs and
                    # do not spend hours re-evaluating non-terminating policies.
                    summary, final_results = _stage4_checkpoint_evaluation(
                        project, checkpoint=latest, records=ensure_fixed_validation_suite(
                            project,
                            root=cal.validation_root,
                            base_seed=cal.validation_base_seed,
                            scales=cal.validation_scales,
                            scenarios=cal.validation_scenarios,
                            load_ratios=cal.validation_load_ratios,
                            due_tightness=cal.validation_due_tightness,
                            instances_per_combination=cal.validation_instances_per_combination,
                        ),
                        device=device,
                        max_episode_decisions=phase_j.runtime.max_episode_decisions,
                    )
                    run_status = "valid" if summary.failed_instances == 0 and math.isfinite(summary.mean_twt) else "invalid_policy"
                    run_row = {
                        "eta": float(eta), "seed": int(seed), "run_status": run_status,
                        "stage4_validation_mean_twt_diagnostic": float(summary.mean_twt),
                        "stage4_completed_instances": int(summary.completed_instances),
                        "stage4_failed_instances": int(summary.failed_instances),
                        "stage4_nonterminating_instances": int(summary.nonterminating_instances),
                        "stage4_deadlocked_instances": int(summary.deadlocked_instances),
                        "stage4_failure_rate": float(summary.failure_rate),
                        "run_dir": str(run_dir), "checkpoint": str(latest),
                    }
                    cached = (
                        run_row,
                        _instance_result_rows(eta=eta, seed=seed, tier="compact", results=final_results),
                    )
                _save_run_cache(run_dir, cached[0], cached[1])
            return cached[0], cached[1], "SKIP_COMPLETED"
        if not cal.resume_existing_runs:
            raise RuntimeError(
                f"incomplete calibration checkpoint exists at {latest}; "
                "resume_existing_runs=false forbids destructive restart"
            )
        resume_path = str(latest)
        action = "RESUME"
        print(
            f"[shaping-calibration] eta={float(eta):g}, seed={int(seed)}: "
            f"RESUME from global_iteration={progress['global_iteration']}, "
            f"stage={progress['stage_index'] + 1}, stage_iteration={progress['stage_iteration']}",
            flush=True,
        )
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"non-empty calibration run directory has no latest checkpoint: {run_dir}. "
            "L1.3d will not delete it automatically; back it up and inspect it explicitly."
        )
    else:
        print(
            f"[shaping-calibration] eta={float(eta):g}, seed={int(seed)}: NEW "
            f"({cal.stage_iterations[0]}/{cal.stage_iterations[1]}/"
            f"{cal.stage_iterations[2]}/{cal.stage_iterations[3]} iterations)",
            flush=True,
        )

    settings = _calibration_settings(
        l1=l1, cal=cal, phase_j=phase_j, seed=int(seed), run_name=run_name,
        device=device, resume_checkpoint=resume_path,
    )
    result = PhaseJTrainer(
        project, phase_j, settings, config_paths=PROJECT_CONFIGS + (L1_CONFIG,)
    ).run()
    if not result.finished:
        raise RuntimeError(f"calibration run returned before curriculum completion: {run_name}")

    recovered = _final_stage4_from_training_logs(
        result.run_dir, eta=float(eta), seed=int(seed), checkpoint=result.latest_checkpoint
    )
    if recovered is None:
        # Safety fallback; under the frozen compact protocol a final Stage-IV
        # trainer validation must exist before a finished checkpoint is saved.
        summary, final_results = _stage4_checkpoint_evaluation(
            project,
            checkpoint=result.latest_checkpoint,
            records=ensure_fixed_validation_suite(
                project,
                root=cal.validation_root,
                base_seed=cal.validation_base_seed,
                scales=cal.validation_scales,
                scenarios=cal.validation_scenarios,
                load_ratios=cal.validation_load_ratios,
                due_tightness=cal.validation_due_tightness,
                instances_per_combination=cal.validation_instances_per_combination,
            ),
            device=device,
            max_episode_decisions=phase_j.runtime.max_episode_decisions,
        )
        run_status = "valid" if summary.failed_instances == 0 and math.isfinite(summary.mean_twt) else "invalid_policy"
        run_row = {
            "eta": float(eta), "seed": int(seed), "run_status": run_status,
            "stage4_validation_mean_twt_diagnostic": float(summary.mean_twt),
            "stage4_completed_instances": int(summary.completed_instances),
            "stage4_failed_instances": int(summary.failed_instances),
            "stage4_nonterminating_instances": int(summary.nonterminating_instances),
            "stage4_deadlocked_instances": int(summary.deadlocked_instances),
            "stage4_failure_rate": float(summary.failure_rate),
            "run_dir": result.run_dir, "checkpoint": result.latest_checkpoint,
        }
        recovered = (
            run_row,
            _instance_result_rows(eta=eta, seed=seed, tier="compact", results=final_results),
        )
    _save_run_cache(result.run_dir, recovered[0], recovered[1])
    return recovered[0], recovered[1], action


def _write_calibration_progress(out_root: Path, run_rows: list[dict], instance_rows: list[dict]) -> None:
    _write_csv(out_root / "shaping_calibration_runs.csv", run_rows)
    _write_csv(out_root / "shaping_calibration_instance_results.csv", instance_rows)


def _strict_control_gate(run_rows: list[dict], *, seeds) -> dict:
    by_seed = {
        int(r["seed"]): r for r in run_rows if float(r["eta"]) == 0.0
    }
    missing = [int(seed) for seed in seeds if int(seed) not in by_seed]
    invalid = [
        int(seed) for seed in seeds
        if int(seed) in by_seed and str(by_seed[int(seed)].get("run_status")) != "valid"
    ]
    passed = not missing and not invalid
    return {
        "passed": bool(passed),
        "missing_seeds": missing,
        "invalid_seeds": invalid,
        "status_by_seed": {
            str(seed): (None if int(seed) not in by_seed else str(by_seed[int(seed)].get("run_status")))
            for seed in seeds
        },
    }



def _budget_selection_paths(l1: PhaseL1Config) -> tuple[Path, Path]:
    root = Path(l1.shaping_budget_search.output_root)
    return root / "shaping_budget_search.csv", root / "shaping_budget_selection.json"


def _write_budget_search_state(l1: PhaseL1Config, rows: list[dict], payload: dict) -> None:
    csv_path, json_path = _budget_selection_paths(l1)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _load_stable_budget_selection(l1: PhaseL1Config, *, required: bool = True) -> dict | None:
    _, json_path = _budget_selection_paths(l1)
    if not json_path.is_file():
        if required:
            raise FileNotFoundError(
                "run python phase_l1.py --shaping-budget-search before --shaping-calibration"
            )
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("budget_search_status") != "stable_budget_found":
        if required:
            raise RuntimeError(
                "no stable compact calibration budget has been found; do not compare eta yet"
            )
        return payload
    label = str(payload["selected_budget_label"])
    matches = [item for item in l1.shaping_budget_search.candidates if item.label == label]
    if len(matches) != 1:
        raise RuntimeError(f"selected budget label {label!r} is not present uniquely in config")
    item = matches[0]
    if tuple(int(x) for x in payload["selected_stage_iterations"]) != item.stage_iterations:
        raise RuntimeError("saved shaping budget selection disagrees with configs/phase_l1.yaml")
    if str(payload["selected_run_root"]) != item.run_root:
        raise RuntimeError("saved shaping budget run root disagrees with configs/phase_l1.yaml")
    return payload


def _selected_calibration_config(l1: PhaseL1Config) -> tuple[L1ShapingCalibrationConfig, dict]:
    selection = _load_stable_budget_selection(l1, required=True)
    assert selection is not None
    return replace(
        l1.shaping_calibration,
        stage_iterations=tuple(int(x) for x in selection["selected_stage_iterations"]),
        output_root=str(selection["selected_run_root"]),
        validation_stop_after_first_failure=l1.shaping_budget_search.validation_stop_after_first_failure,
    ), selection


def _budget_row(*, candidate: L1BudgetCandidate, run_rows: list[dict], seeds, gate: dict) -> dict:
    by_seed = {int(r["seed"]): r for r in run_rows if float(r["eta"]) == 0.0}
    return {
        "budget_label": candidate.label,
        "stage_iterations": "/".join(str(int(x)) for x in candidate.stage_iterations),
        "total_iterations": int(sum(candidate.stage_iterations)),
        "attempted_seeds": ",".join(str(x) for x in sorted(by_seed)),
        "valid_seeds": ",".join(
            str(seed) for seed in sorted(by_seed)
            if str(by_seed[seed].get("run_status")) == "valid"
        ),
        "invalid_seeds": ",".join(str(x) for x in gate["invalid_seeds"]),
        "missing_seeds": ",".join(str(x) for x in gate["missing_seeds"]),
        "stable": bool(gate["passed"]),
        "run_root": candidate.run_root,
    }


def run_shaping_budget_search(*, device: str | None = None) -> dict:
    """Find the smallest tested compact budget stable for strict TWT only.

    This is validation-only protocol selection. It does not compare positive eta
    values and never touches the formal test set. Budgets are tried in ascending
    order. Once any strict-control seed is invalid, that budget cannot satisfy the
    all-seed gate, so the remaining seed can be skipped. The seed that failed the
    previous budget is tried first at the next budget to reduce wasted compute.
    """
    base_project = load_config(PROJECT_CONFIGS)
    l1 = load_phase_l1_config(L1_CONFIG, project_cfg=base_project)
    base_cal = l1.shaping_calibration
    search = l1.shaping_budget_search
    ensure_fixed_validation_suite(
        base_project,
        root=base_cal.validation_root,
        base_seed=base_cal.validation_base_seed,
        scales=base_cal.validation_scales,
        scenarios=base_cal.validation_scenarios,
        load_ratios=base_cal.validation_load_ratios,
        due_tightness=base_cal.validation_due_tightness,
        instances_per_combination=base_cal.validation_instances_per_combination,
    )

    budget_rows: list[dict] = []
    selected: L1BudgetCandidate | None = None
    prior_invalid: list[int] = []
    all_run_rows: list[dict] = []

    for candidate in search.candidates:
        cal = replace(
            base_cal,
            stage_iterations=candidate.stage_iterations,
            output_root=candidate.run_root,
            validation_stop_after_first_failure=search.validation_stop_after_first_failure,
        )
        priority = [seed for seed in prior_invalid if seed in cal.seeds]
        seed_order = priority + [seed for seed in cal.seeds if seed not in priority]
        print(
            f"[shaping-budget-search] budget={candidate.label} "
            f"iterations={'/'.join(map(str, candidate.stage_iterations))}, "
            f"seed_order={seed_order}",
            flush=True,
        )
        current_rows: list[dict] = []
        for seed in seed_order:
            row, _rows, action = _run_or_resume_calibration_candidate(
                l1=l1, cal=cal, eta=0.0, seed=int(seed), device=device
            )
            current_rows.append(row)
            all_run_rows.append({"budget_label": candidate.label, **row})
            print(
                f"[shaping-budget-search] budget={candidate.label}, eta=0, seed={seed}: "
                f"{action}; status={row['run_status']}; "
                f"failed_instances={row['stage4_failed_instances']}",
                flush=True,
            )
            if (
                search.stop_budget_after_first_invalid_seed
                and str(row.get("run_status")) != "valid"
            ):
                break

        gate = _strict_control_gate(current_rows, seeds=cal.seeds)
        budget_rows.append(_budget_row(
            candidate=candidate, run_rows=current_rows, seeds=cal.seeds, gate=gate
        ))
        prior_invalid = list(gate["invalid_seeds"])
        if gate["passed"]:
            selected = candidate

        payload = {
            "budget_search_status": (
                "stable_budget_found" if selected is not None else "search_in_progress_or_unstable"
            ),
            "selected_budget_label": None if selected is None else selected.label,
            "selected_stage_iterations": None if selected is None else list(selected.stage_iterations),
            "selected_run_root": None if selected is None else selected.run_root,
            "strict_twt_eta": 0.0,
            "diagnostic_seeds": list(base_cal.seeds),
            "selection_rule": "smallest tested budget for which every strict-TWT diagnostic seed is valid",
            "failure_first_validation": bool(search.validation_stop_after_first_failure),
            "formal_training_ready": False,
            "budget_rows": budget_rows,
        }
        _write_budget_search_state(l1, budget_rows, payload)
        if selected is not None and search.stop_after_first_stable:
            break

    status = "stable_budget_found" if selected is not None else "no_stable_budget_found"
    payload = {
        "budget_search_status": status,
        "selected_budget_label": None if selected is None else selected.label,
        "selected_stage_iterations": None if selected is None else list(selected.stage_iterations),
        "selected_run_root": None if selected is None else selected.run_root,
        "strict_twt_eta": 0.0,
        "diagnostic_seeds": list(base_cal.seeds),
        "selection_rule": "smallest tested budget for which every strict-TWT diagnostic seed is valid",
        "failure_first_validation": bool(search.validation_stop_after_first_failure),
        "formal_training_ready": False,
        "budget_rows": budget_rows,
        "note": (
            "Proceed to the eta screen with this exact compact budget."
            if selected is not None else
            "Even the largest diagnostic budget is unstable; do not compare eta or use test data."
        ),
    }
    _write_budget_search_state(l1, budget_rows, payload)
    csv_path, json_path = _budget_selection_paths(l1)
    return {
        **payload,
        "summary_csv": str(csv_path),
        "selection_json": str(json_path),
        "run_rows": all_run_rows,
    }


def _load_completed_candidate_without_resuming(*, l1, cal, eta: float, seed: int, device: str | None):
    """Load an already-finished eta×seed result, but never resume a partial run.

    Used after another seed has already made the eta ineligible. Completed work is
    still preserved in diagnostics at essentially zero cost; partial checkpoints
    remain untouched instead of consuming more training/validation time.
    """
    run_name = f"eta_{_eta_slug(eta)}_seed{seed}"
    run_dir = Path(cal.output_root) / run_name
    cached = _load_run_cache(run_dir)
    if cached is not None:
        return cached[0], cached[1], "SKIP_COMPLETED"
    latest = run_dir / "checkpoints" / "latest.pt"
    if not latest.is_file():
        return None
    progress = _checkpoint_progress(latest)
    _assert_calibration_resume_compatible(progress, cal=cal, seed=int(seed))
    if not progress["finished"]:
        return None
    return _run_or_resume_calibration_candidate(
        l1=l1, cal=cal, eta=float(eta), seed=int(seed), device=device
    )


def run_shaping_calibration(*, device: str | None = None) -> dict:
    """Tier-1 compact screen with exact resume/skip and a strict-TWT gate.

    L1.3d requires a successful strict-TWT budget-sufficiency search first. The
    eta screen then uses that exact smallest stable tested budget for eta=0 and all
    positive candidates. Completed eta×seed runs are skipped, incomplete runs resume
    from ``latest.pt``, and existing directories are never removed automatically.
    """
    base_project = load_config(PROJECT_CONFIGS)
    l1 = load_phase_l1_config(L1_CONFIG, project_cfg=base_project)
    cal, budget_selection = _selected_calibration_config(l1)
    out_root = Path(cal.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    # Materialize/audit the fixed compact suite once; individual trainer runs
    # share exactly these files and never touch the formal test set.
    ensure_fixed_validation_suite(
        base_project,
        root=cal.validation_root,
        base_seed=cal.validation_base_seed,
        scales=cal.validation_scales,
        scenarios=cal.validation_scenarios,
        load_ratios=cal.validation_load_ratios,
        due_tightness=cal.validation_due_tightness,
        instances_per_combination=cal.validation_instances_per_combination,
    )

    run_rows: list[dict] = []
    instance_rows: list[dict] = []

    # Scientific control first. This is also the inexpensive stability gate that
    # prevents spending a day on positive eta values when the compact budget is
    # already too small for eta=0 to produce a valid policy across both seeds.
    for seed in cal.seeds:
        row, rows, action = _run_or_resume_calibration_candidate(
            l1=l1, cal=cal, eta=0.0, seed=int(seed), device=device
        )
        run_rows.append(row); instance_rows.extend(rows)
        _write_calibration_progress(out_root, run_rows, instance_rows)
        print(
            f"[shaping-calibration] eta=0, seed={int(seed)}: {action}; "
            f"status={row['run_status']}, diagnostic_mean_twt="
            f"{row['stage4_validation_mean_twt_diagnostic']}, "
            f"failed_instances={row['stage4_failed_instances']}",
            flush=True,
        )

    gate = _strict_control_gate(run_rows, seeds=cal.seeds)
    if cal.strict_twt_control_gate and not gate["passed"]:
        control_summary, _ = _macro_rank_analysis(
            instance_rows, candidates=(0.0,), seeds=cal.seeds
        )
        recommendation = {
            "compact_screen_status": "control_unstable",
            "control_gate_status": "failed",
            "control_gate": gate,
            "screen_leading_eta": None,
            "confirmation_candidates": [],
            "recommended_eta": None,
            "recommended_variant": None,
            "seed_winners": {},
            "selection_basis": "strict-TWT stability gate before positive shaping candidates",
            "formal_training_ready": False,
            "confirmation_required": False,
            "note": (
                "The budget selected by the L1.3d sufficiency search became inconsistent for eta=0 across diagnostic seeds. "
                "Positive eta runs were intentionally not launched/resumed. Increase the diagnostic "
                "training budget on validation only before comparing shaping coefficients. Existing "
                "positive-eta partial checkpoints are preserved untouched."
            ),
        }
        _write_csv(out_root / "shaping_calibration_summary.csv", control_summary)
        (out_root / "shaping_calibration_recommendation.json").write_text(
            json.dumps(_json_safe({"summary": control_summary, **recommendation}), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return {
            **recommendation,
            "summary": control_summary,
            "detail_csv": str(out_root / "shaping_calibration_runs.csv"),
            "instance_csv": str(out_root / "shaping_calibration_instance_results.csv"),
            "summary_csv": str(out_root / "shaping_calibration_summary.csv"),
            "recommendation_json": str(out_root / "shaping_calibration_recommendation.json"),
        }

    for eta in l1.shaping_calibration_candidates:
        eta = float(eta)
        if eta == 0.0:
            continue
        candidate_ineligible = False
        invalid_seed: int | None = None
        for seed in cal.seeds:
            seed = int(seed)
            if candidate_ineligible and cal.stop_candidate_after_first_invalid_seed:
                completed = _load_completed_candidate_without_resuming(
                    l1=l1, cal=cal, eta=eta, seed=seed, device=device
                )
                if completed is None:
                    print(
                        f"[shaping-calibration] eta={eta:g}, seed={seed}: SKIP_INELIGIBLE; "
                        f"eta already invalid after seed={invalid_seed}; any partial checkpoint is preserved",
                        flush=True,
                    )
                    continue
                row, rows, action = completed
            else:
                row, rows, action = _run_or_resume_calibration_candidate(
                    l1=l1, cal=cal, eta=eta, seed=seed, device=device
                )

            run_rows.append(row); instance_rows.extend(rows)
            _write_calibration_progress(out_root, run_rows, instance_rows)
            print(
                f"[shaping-calibration] eta={eta:g}, seed={seed}: {action}; "
                f"status={row['run_status']}, diagnostic_mean_twt="
                f"{row['stage4_validation_mean_twt_diagnostic']}, "
                f"failed_instances={row['stage4_failed_instances']}",
                flush=True,
            )
            if str(row.get("run_status")) != "valid":
                candidate_ineligible = True
                invalid_seed = seed

    screen_summary, recommendation = _compact_shortlist(
        instance_rows, candidates=l1.shaping_calibration_candidates, seeds=cal.seeds
    )
    recommendation.update({
        "selected_budget_label": budget_selection["selected_budget_label"],
        "selected_stage_iterations": budget_selection["selected_stage_iterations"],
        "selected_run_root": budget_selection["selected_run_root"],
    })
    detail_path = out_root / "shaping_calibration_runs.csv"
    instance_path = out_root / "shaping_calibration_instance_results.csv"
    summary_path = out_root / "shaping_calibration_summary.csv"
    json_path = out_root / "shaping_calibration_recommendation.json"
    _write_calibration_progress(out_root, run_rows, instance_rows)
    _write_csv(summary_path, screen_summary)
    json_path.write_text(
        json.dumps(_json_safe({"summary": screen_summary, "control_gate": gate, **recommendation}), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        **recommendation,
        "control_gate_status": "passed",
        "control_gate": gate,
        "summary": screen_summary,
        "detail_csv": str(detail_path),
        "instance_csv": str(instance_path),
        "summary_csv": str(summary_path),
        "recommendation_json": str(json_path),
    }

def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _confirmation_decision(instance_rows: list[dict], *, candidates, seeds) -> tuple[list[dict], dict]:
    summary, analysis = _macro_rank_analysis(instance_rows, candidates=candidates, seeds=seeds)
    if not summary or not all(bool(r["eligible"]) for r in summary):
        status = "inconclusive"
        eta = None
        note = "At least one shortlisted checkpoint failed the broad fixed-validation confirmation."
    else:
        winners = [analysis["seed_winners"].get(str(seed)) for seed in seeds]
        if all(w is not None for w in winners) and len(set(winners)) == 1:
            status = "stable_candidate"
            eta = float(winners[0])
            note = (
                "Both diagnostic seeds independently select the same eta on the broad fixed "
                "validation suite using equal-scale paired ranks. Freeze only after manual review."
            )
        else:
            status = "inconclusive"
            eta = None
            note = "Broad confirmation seed winners disagree or tie; do not freeze the reward protocol."
    return summary, {
        "confirmation_status": status,
        "recommended_eta": eta,
        "recommended_variant": (
            None if eta is None else ("strict_twt_integral" if eta == 0.0 else "potential_shaping")
        ),
        "seed_winners": analysis["seed_winners"],
        "selection_basis": "broad fixed validation; paired TWT ranks; equal S/M/L macro weight",
        "formal_training_ready": False,
        "note": note,
    }


def run_shaping_confirmation(*, device: str | None = None) -> dict:
    """Tier-2 broad validation confirmation with no additional training."""
    base_project = load_config(PROJECT_CONFIGS)
    l1 = load_phase_l1_config(L1_CONFIG, project_cfg=base_project)
    cal, budget_selection = _selected_calibration_config(l1)
    confirm = l1.shaping_confirmation
    cal_root = Path(cal.output_root)
    recommendation_path = cal_root / "shaping_calibration_recommendation.json"
    run_path = cal_root / "shaping_calibration_runs.csv"
    if not recommendation_path.is_file() or not run_path.is_file():
        raise FileNotFoundError("run --shaping-calibration successfully before --shaping-confirmation")
    screen = json.loads(recommendation_path.read_text(encoding="utf-8"))
    if screen.get("compact_screen_status") != "ready_for_confirmation":
        raise RuntimeError("compact shaping screen is not ready for confirmation")
    candidates = tuple(float(x) for x in screen.get("confirmation_candidates", []))
    if 0.0 not in candidates or len(candidates) < 2:
        raise RuntimeError("confirmation shortlist must contain eta=0 and at least one positive eta")
    run_rows = _read_csv(run_path)
    checkpoints = {
        (float(r["eta"]), int(r["seed"])): str(r["checkpoint"])
        for r in run_rows if str(r.get("run_status")) == "valid"
    }

    records = ensure_fixed_validation_suite(
        base_project,
        root=confirm.validation_root,
        base_seed=confirm.validation_base_seed,
        scales=confirm.validation_scales,
        scenarios=confirm.validation_scenarios,
        load_ratios=confirm.validation_load_ratios,
        due_tightness=confirm.validation_due_tightness,
        instances_per_combination=confirm.validation_instances_per_combination,
    )
    out_root = Path(confirm.output_root) / f"budget_{budget_selection['selected_budget_label']}"
    out_root.mkdir(parents=True, exist_ok=True)
    instance_rows: list[dict] = []
    confirm_runs: list[dict] = []
    total = len(candidates) * len(cal.seeds)
    idx = 0
    for eta in candidates:
        project = _project_for_eta(eta)
        phase_j = load_phase_j_config(TRAIN_CONFIG, project_cfg=project)
        for seed in cal.seeds:
            idx += 1
            checkpoint = checkpoints.get((eta, int(seed)))
            if checkpoint is None or not Path(checkpoint).is_file():
                raise FileNotFoundError(f"missing compact checkpoint for eta={eta}, seed={seed}: {checkpoint}")
            print(
                f"[shaping-confirmation] {idx}/{total}: eta={eta:g}, seed={seed}, "
                f"fixed_validation_instances={len(records)}", flush=True,
            )
            summary, results = _stage4_checkpoint_evaluation(
                project, checkpoint=checkpoint, records=records, device=device,
                max_episode_decisions=phase_j.runtime.max_episode_decisions,
            )
            instance_rows.extend(_instance_result_rows(eta=eta, seed=seed, tier="confirmation", results=results))
            confirm_runs.append({
                "eta": eta, "seed": int(seed),
                "run_status": "valid" if summary.failed_instances == 0 else "invalid_policy",
                "diagnostic_mean_twt": float(summary.mean_twt),
                "completed_instances": int(summary.completed_instances),
                "failed_instances": int(summary.failed_instances),
                "nonterminating_instances": int(summary.nonterminating_instances),
                "deadlocked_instances": int(summary.deadlocked_instances),
                "checkpoint": checkpoint,
            })

    summary, recommendation = _confirmation_decision(
        instance_rows, candidates=candidates, seeds=cal.seeds
    )
    run_csv = out_root / "shaping_confirmation_runs.csv"
    instance_csv = out_root / "shaping_confirmation_instance_results.csv"
    summary_csv = out_root / "shaping_confirmation_summary.csv"
    json_path = out_root / "shaping_confirmation_recommendation.json"
    _write_csv(run_csv, confirm_runs); _write_csv(instance_csv, instance_rows); _write_csv(summary_csv, summary)
    json_path.write_text(
        json.dumps(_json_safe({"summary": summary, **recommendation}), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        **recommendation,
        "summary": summary,
        "run_csv": str(run_csv),
        "instance_csv": str(instance_csv),
        "summary_csv": str(summary_csv),
        "recommendation_json": str(json_path),
    }

def write_seed_command_files(l1: PhaseL1Config, *, root: str | Path = "result/phase_l1") -> list[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for seed in l1.training_seeds:
        path = root / f"run_seed{seed}_windows.bat"
        if l1.formal_training_ready:
            text = "@echo off\n" + f"python phase_l1.py --run-seed {seed}\n" + "pause\n"
        else:
            text = (
                "@echo off\n"
                "echo BLOCKED: Phase L1.4 reward protocol is not frozen yet.\n"
                "echo Run shaping budget search, Tier-1 shaping calibration, and Tier-2 confirmation first.\n"
                "exit /b 2\n"
            )
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    notes = root / "FORMAL_RUN_ORDER.txt"
    notes.write_text(
        "Phase L1.5 formal run order\n"
        "============================\n"
        "1) python phase_l1.py --preflight\n"
        "2) python phase_l1.py --hardware-pilot-full --seed 0\n"
        "3) python phase_l1.py --shaping-budget-search\n"
        "4) python phase_l1.py --shaping-calibration\n"
        "5) python phase_l1.py --shaping-confirmation\n"
        "6) python phase_l1.py --freeze-reward\n"
        "7) python phase_l1.py --preflight && python phase_l1.py --prepare\n"
        "8) Before long formal training, benchmark the current checkpoint with: python phase_l1.py --throughput-pilot --seed 0\n"
        "9) Run/Resume Seed 0 first; inspect it before Seeds 1..4. Use the same frozen throughput profile for all formal seeds.\n"
        "Fixed test data must never be used for model/hyperparameter selection.\n",
        encoding="utf-8",
    )
    paths.append(notes)
    return paths


__all__ = [
    "L1ValidationConfig", "L1BudgetCandidate", "L1ShapingBudgetSearchConfig",
    "L1ShapingCalibrationConfig", "PhaseL1Config", "PROJECT_CONFIGS",
    "build_formal_project", "build_run_settings", "freeze_reward_protocol",
    "load_phase_l1_config", "preflight_report",
    "prepare_validation_suite", "run_formal_seed", "run_throughput_pilot", "run_shaping_budget_search",
    "run_shaping_calibration", "run_shaping_confirmation", "sha256_file",
    "validate_reward_freeze_record",
    "validate_phase_l1_config", "write_frozen_manifest", "write_seed_command_files",
]
