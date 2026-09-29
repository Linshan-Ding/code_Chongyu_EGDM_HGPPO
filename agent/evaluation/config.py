"""Phase K evaluation settings loaded from configs/eval.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

from agent.baselines.base import RuleBaselineConfig
from agent.baselines.optimization import OptimizationBaselineConfig
from agent.baselines.alns import ALNSBaselineConfig


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    test_root: str
    output_csv: str
    max_episode_decisions: int
    deterministic_learned_policy: bool
    learned_policy_batch_size: int
    formal_instances_per_combination: int
    formal_scales: tuple[str, ...]
    formal_scenarios: tuple[str, ...]
    formal_load_ratios: tuple[float, ...]
    formal_due_tightness: tuple[str, ...]
    formal_instance_parameter_table_path: str | None
    rule_config: RuleBaselineConfig
    smoke_root: str
    smoke_output_csv: str
    smoke_seed: int
    smoke_scale: str
    smoke_scenario: str
    smoke_load_ratio: float
    smoke_due_tightness: str
    optimization_config: OptimizationBaselineConfig
    alns_config: ALNSBaselineConfig


def load_eval_settings(path: str | Path = "configs/eval.yaml") -> EvaluationSettings:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = raw.get("phase_k", {})
    formal = root.get("formal", {})
    rules = root.get("rule_baselines", {})
    smoke = root.get("smoke", {})
    k2 = root.get("k2", {})
    opt = k2.get("optimization", {})
    alns = k2.get("alns", {})
    optimization_config = OptimizationBaselineConfig(
        offline_time_limit_seconds=float(opt.get("offline_time_limit_seconds", 3600.0)),
        offline_mip_rel_gap=float(opt.get("offline_mip_rel_gap", 0.0)),
        offline_max_operations=int(opt.get("offline_max_operations", 140)),
        rolling_time_limit_seconds=float(opt.get("rolling_time_limit_seconds", 0.20)),
        rolling_max_reconfiguration_moves=int(opt.get("rolling_max_reconfiguration_moves", 3)),
        rolling_relocation_penalty=float(opt.get("rolling_relocation_penalty", 0.05)),
        rolling_tardiness_pressure=float(opt.get("rolling_tardiness_pressure", 2.0)),
        rolling_processing_penalty=float(opt.get("rolling_processing_penalty", 0.02)),
    )
    optimization_config.validate()
    alns_config = ALNSBaselineConfig(
        iterations_per_event=int(alns.get("iterations_per_event", 40)),
        max_reconfiguration_moves=int(alns.get("max_reconfiguration_moves", 4)),
        reaction_factor=float(alns.get("reaction_factor", 0.20)),
        random_accept_temperature=float(alns.get("random_accept_temperature", 0.10)),
        relocation_penalty=float(alns.get("relocation_penalty", 0.05)),
        imbalance_weight=float(alns.get("imbalance_weight", 1.0)),
        schedule_weight=float(alns.get("schedule_weight", 1.0)),
    )
    alns_config.validate()
    cfg = RuleBaselineConfig(
        periodic_interval_minutes=float(rules.get("periodic_interval_minutes", 10.0)),
        atc_k=float(rules.get("atc_k", 2.0)),
        threshold_load_capacity_ratio=float(rules.get("threshold_load_capacity_ratio", 1.0)),
        bottleneck_min_gap=float(rules.get("bottleneck_min_gap", 0.15)),
        relocation_time_penalty=float(rules.get("relocation_time_penalty", 0.05)),
        max_reconfiguration_moves=int(rules.get("max_reconfiguration_moves", 2)),
    )
    cfg.validate()
    return EvaluationSettings(
        test_root=str(root.get("test_root", "data/instances/test")),
        output_csv=str(root.get("output_csv", "result/eval_results.csv")),
        max_episode_decisions=int(root.get("max_episode_decisions", 20000)),
        deterministic_learned_policy=bool(root.get("deterministic_learned_policy", True)),
        learned_policy_batch_size=int(root.get("learned_policy_batch_size", 32)),
        formal_instances_per_combination=int(formal.get("instances_per_combination", 100)),
        formal_scales=tuple(formal.get("scales", ["S", "M", "L"])),
        formal_scenarios=tuple(formal.get("scenarios", ["D1", "D2", "D3", "D4", "D5"])),
        formal_load_ratios=tuple(float(x) for x in formal.get("load_ratios", [0.65, 0.80, 0.95])),
        formal_due_tightness=tuple(formal.get("due_tightness", ["tight", "medium", "loose"])),
        formal_instance_parameter_table_path=(
            str(formal["instance_parameter_table_path"])
            if formal.get("instance_parameter_table_path") is not None else None
        ),
        rule_config=cfg,
        smoke_root=str(smoke.get("root", "result/phase_k_smoke_test")),
        smoke_output_csv=str(smoke.get("output_csv", "result/phase_k_smoke_eval_results.csv")),
        smoke_seed=int(smoke.get("seed", 310000)),
        smoke_scale=str(smoke.get("scale", "S")),
        smoke_scenario=str(smoke.get("scenario", "D1")),
        smoke_load_ratio=float(smoke.get("load_ratio", 0.65)),
        smoke_due_tightness=str(smoke.get("due_tightness", "tight")),
        optimization_config=optimization_config,
        alns_config=alns_config,
    )


__all__ = ["EvaluationSettings", "load_eval_settings"]
