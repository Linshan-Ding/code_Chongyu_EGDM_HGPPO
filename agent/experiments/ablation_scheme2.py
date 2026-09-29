"""Scheme-2 component-ablation orchestration.

The runner intentionally reuses ``RandomJointTrainer`` so every variant has an
independent run directory/checkpoint and identical online data/PPO budgets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
import yaml

from agent.ablation_variants import ABLATION_VARIANTS, build_ablation_policy, get_ablation_spec
from agent.constraints import PolicyActionConstraints
from agent.ppo import PPOComponentMask
from agent.baselines.learned_variants import build_learned_baseline_policy
from configs.config import load_config
from agent.experiments.l1 import L1_CONFIG, PROJECT_CONFIGS, THROUGHPUT_CONFIG, build_formal_project, load_phase_l1_config, validate_phase_l1_config
from agent.experiments.scheme2 import SCHEME2_CONFIG, TRAIN_CONFIG, _settings, load_scheme2_config
from agent.training.config import load_phase_j_config
from agent.training.trainer_random_joint import RandomJointTrainer


ABLATION_CONFIG = "configs/ablation.yaml"


@dataclass(frozen=True, slots=True)
class AblationSettings:
    variants: tuple[str, ...]
    run_root: str
    eval_csv: str
    periodic_gate_period: int


def load_ablation_settings(path: str | Path = ABLATION_CONFIG) -> AblationSettings:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    block = raw.get("ablation") or {}
    variant_rows = block.get("variants") or []
    names = tuple(str(row["name"]) for row in variant_rows)
    if not names:
        raise ValueError("configs/ablation.yaml declares no variants")
    if len(names) != len(set(names)):
        raise ValueError("configs/ablation.yaml contains duplicate variants")
    unknown = set(names).difference(ABLATION_VARIANTS)
    missing = set(ABLATION_VARIANTS).difference(names)
    if unknown or missing:
        raise ValueError(
            f"ablation matrix disagrees with Table 10; unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}"
        )
    period = int(block.get("periodic_gate_period", 0))
    if period <= 0:
        raise ValueError("ablation.periodic_gate_period must be positive")
    return AblationSettings(
        variants=names,
        run_root=str(block["run_root"]),
        eval_csv=str(block["eval_csv"]),
        periodic_gate_period=period,
    )


def configure_ablation_joint_spec(joint_spec, spec):
    """Apply exactly the action/PPO control owned by one ablation."""
    if spec.sequential_dispatch:
        joint_spec = replace(
            joint_spec,
            constraints=PolicyActionConstraints(max_schedule_assignments=1),
        )
    elif spec.flat_reconfiguration:
        joint_spec = replace(
            joint_spec,
            constraints=PolicyActionConstraints(
                max_worker_moves=1, max_robot_moves=1
            ),
        )
    if spec.without_event_gate or spec.periodic_gate:
        joint_spec = replace(
            joint_spec,
            components=PPOComponentMask(
                gate_policy=False,
                resource_policy=True,
                schedule_policy=True,
                gate_value=False,
                rec_value=True,
                sch_value=True,
            ),
        )
    return joint_spec


class AblationRandomJointTrainer(RandomJointTrainer):
    def __init__(self, *args, variant: str, periodic_gate_period: int = 4, **kwargs):
        self.variant = str(variant)
        self.periodic_gate_period = max(1, int(periodic_gate_period))
        self.spec = get_ablation_spec(self.variant, periodic_gate_period=self.periodic_gate_period)
        super().__init__(*args, **kwargs)
        self.joint_spec = configure_ablation_joint_spec(self.joint_spec, self.spec)

    def _build_policy(self, reference_graph):
        if self.spec.homogeneous_gat:
            return build_learned_baseline_policy("GAT-PPO", self.cfg, reference_graph)
        return build_ablation_policy(
            self.variant, self.cfg, reference_graph,
            periodic_gate_period=self.periodic_gate_period,
        )

    def _configure_agent(self, agent) -> None:
        agent.duration_aware_gae = bool(self.spec.duration_aware_gae)

    def _write_scheme2_protocol_audit(self):
        super()._write_scheme2_protocol_audit()
        path = self.run_dir / "scheme2_protocol.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update({"training_protocol": "teacher_scheme2_component_ablation", "ablation_variant": self.variant, "ablation_description": self.spec.description, "same_online_distribution_and_ppo_budget": True, "potential_shaping_enabled": bool(self.spec.potential_shaping and getattr(self.cfg.env.reward, "optional_potential_shaping", False)), "duration_aware_gae": bool(self.spec.duration_aware_gae)})
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_ablation(*, variant: str, iterations: int | None = None, device: str | None = None, budget_profile: str | None = None, resume: bool = False):
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"unknown ablation variant {variant!r}")
    s2 = load_scheme2_config(budget_profile=budget_profile)
    iterations = int(s2.formal_iterations if iterations is None else iterations)
    l1 = load_phase_l1_config(L1_CONFIG, project_cfg=None)
    project = build_formal_project(l1)
    validate_phase_l1_config(l1, project_cfg=project)
    spec = get_ablation_spec(variant)
    ablation_cfg = load_ablation_settings()
    if variant not in ablation_cfg.variants:
        raise ValueError(f"ablation variant is disabled by config: {variant}")
    periodic_gate_period = ablation_cfg.periodic_gate_period
    # Potential shaping is disabled only through a private config flag; the
    # frozen phase_l1.yaml remains untouched and all other reward terms persist.
    if not spec.potential_shaping:
        project.env.reward._data["ablation_disable_potential_shaping"] = True
    phase_j = load_phase_j_config(TRAIN_CONFIG, project_cfg=project)
    run_name = f"ablation_{variant}_seed{s2.training_seed}_n{int(iterations)}"
    run_root = ablation_cfg.run_root
    candidate = Path(run_root) / run_name / "checkpoints" / "latest.pt"
    resume_path = str(candidate) if resume and candidate.is_file() else None
    settings = _settings(s2, phase_j, total_iterations=iterations, device=device, run_name=run_name, resume_checkpoint=resume_path, validation_every_iterations=s2.validation_every_iterations)
    # The CPU worker factory reconstructs the exact ablation variant.  Keep the
    # measured formal multiprocess collector instead of silently making nine
    # ablation runs serial.  Deterministic gate controls currently use the
    # ordinary decoder path; other variants retain tensorized scoring.
    fixed_gate_variant = bool(spec.without_event_gate or spec.periodic_gate)
    settings = replace(
        settings, run_root=run_root, visdom_enabled=False,
        tensorized_decoder_scoring=(
            False if fixed_gate_variant else settings.tensorized_decoder_scoring
        ),
    )
    trainer = AblationRandomJointTrainer(project, phase_j, settings, total_iterations=int(iterations), joint_scale_pool=s2.scales, joint_scenario_pool=s2.scenarios, joint_load_ratio_pool=s2.load_ratios, joint_due_tightness_pool=s2.due_tightness, config_paths=PROJECT_CONFIGS + (L1_CONFIG, THROUGHPUT_CONFIG, TRAIN_CONFIG, SCHEME2_CONFIG, ABLATION_CONFIG), validate_at_end=True, fresh_instances_each_iteration=True, validation_monitor_subset="parameter_cases", validation_progress_every_instances=0, instance_parameter_table_path=s2.instance_parameter_table_path, training_instance_parameter_table_path=None, variant=variant, periodic_gate_period=periodic_gate_period)
    return trainer.run(), trainer


__all__ = [
    "ABLATION_VARIANTS",
    "AblationRandomJointTrainer",
    "AblationSettings",
    "configure_ablation_joint_spec",
    "load_ablation_settings",
    "run_ablation",
]
