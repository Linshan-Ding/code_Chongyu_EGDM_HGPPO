"""Configuration center for the EGDM-HGPPO project.

Phase B scope:
- merge multiple YAML files from left to right;
- expose a structured, attribute-accessible configuration;
- validate paper-specific invariants;
- support CLI dotted-key overrides.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import yaml


class ConfigNode:
    """Recursive attribute wrapper around a mapping."""

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", {})
        for key, value in data.items():
            self._data[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    @classmethod
    def _unwrap(cls, value: Any) -> Any:
        if isinstance(value, ConfigNode):
            return {k: cls._unwrap(v) for k, v in value._data.items()}
        if isinstance(value, list):
            return [cls._unwrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> dict[str, Any]:
        return self._unwrap(self)

    def __repr__(self) -> str:
        return f"ConfigNode({self.to_dict()!r})"

    def __getstate__(self):
        # Explicit pickle contract used by Phase J exact iteration-boundary
        # checkpoints. Serializing raw ``_data`` through ``__getattr__`` can
        # recurse before the attribute exists during unpickling.
        return self.to_dict()

    def __setstate__(self, state):
        object.__setattr__(self, "_data", {})
        for key, value in state.items():
            self._data[key] = self._wrap(value)


@dataclass(frozen=True)
class ProjectConfig:
    instance: ConfigNode
    env: ConfigNode
    algo: ConfigNode
    curriculum: ConfigNode

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance.to_dict(),
            "env": self.env.to_dict(),
            "algo": self.algo.to_dict(),
            "curriculum": self.curriculum.to_dict(),
        }


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML object must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_dotted(target: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            cursor[part] = {}
        elif not isinstance(existing, dict):
            raise KeyError(f"Cannot set {key!r}: {part!r} is not a mapping.")
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse_overrides(items: Iterable[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must have KEY=VALUE form, got: {item!r}")
        key, raw_value = item.split("=", 1)
        _set_dotted(result, key.strip(), yaml.safe_load(raw_value))
    return result


def load_raw_config(
    paths: Iterable[str | Path],
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged = _deep_merge(merged, _read_yaml(path))
    return _deep_merge(merged, parse_overrides(overrides))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_raw_config(raw: Mapping[str, Any]) -> None:
    required = {"instance", "env", "algo", "curriculum"}
    missing = required.difference(raw)
    _require(not missing, f"Missing config sections: {sorted(missing)}")

    inst = raw["instance"]
    env = raw["env"]
    algo = raw["algo"]
    curriculum = raw["curriculum"]

    # Paper system boundary
    _require(
        inst["parallel_cells_per_stage"] == [1, 3],
        "Paper boundary requires 1-3 parallel dedicated cells per stage.",
    )

    # Paper Table 5
    expected_scales = {
        "S": {
            "num_stages": 4,
            "total_cells": [5, 7],
            "orders": [15, 35],
            "workers": [3, 5],
            "robots": [2, 4],
            "product_types": [3, 3],
        },
        "M": {
            "num_stages": 6,
            "total_cells": [8, 12],
            "orders": [40, 80],
            "workers": [5, 8],
            "robots": [4, 7],
            "product_types": [4, 5],
        },
        "L": {
            "num_stages": 8,
            "total_cells": [12, 18],
            "orders": [90, 160],
            "workers": [8, 12],
            "robots": [6, 10],
            "product_types": [6, 8],
        },
        "XL": {
            "num_stages": 10,
            "total_cells": [18, 24],
            "orders": [180, 300],
            "workers": [12, 16],
            "robots": [9, 13],
            "product_types": [8, 10],
        },
    }
    _require(inst["scales"] == expected_scales, "Scale table differs from paper Table 5.")

    # Paper Table 7
    _require(
        set(inst["dynamic_scenarios"].keys()) == {"D1", "D2", "D3", "D4", "D5"},
        "Dynamic scenarios must be D1-D5.",
    )

    # Phase C implementation choices: paper-unspecified values must stay explicit
    # and internally consistent rather than becoming hidden magic numbers.
    _require("implementation_choices" in inst, "Phase C requires instance.implementation_choices")
    impl = inst["implementation_choices"]
    _require(
        0.0 <= float(impl["initial_configuration_perturbation_rate"]) <= 1.0,
        "initial_configuration_perturbation_rate must be in [0,1]",
    )
    _require(
        0.0 <= float(impl["d2_burst_start_ratio"]) < float(impl["d2_burst_end_ratio"]) <= 1.0,
        "D2 burst ratios must satisfy 0 <= start < end <= 1",
    )
    _require(
        0.0 < float(impl["d3_mix_switch_ratio"]) < 1.0,
        "D3 mix switch ratio must be in (0,1)",
    )
    _require(
        len(impl["d3_dominant_product_sequence"]) == 2,
        "D3 dominant product sequence must contain two entries",
    )
    _require(
        len(impl["d4_change_ratios"]) == 2
        and len(impl["d4_arrival_rate_multipliers"]) == 3
        and len(impl["d4_dominant_product_sequence"]) == 3,
        "D4 requires two change ratios and three segment settings",
    )
    _require(
        0.0 < float(impl["dominant_product_share"]) < 1.0,
        "dominant_product_share must be in (0,1)",
    )
    _require(
        0.0 < float(impl["d5_urgent_batch_fraction"]) <= 1.0
        and 0.0 <= float(impl["d5_urgent_batch_center_ratio"]) <= 1.0,
        "D5 urgent batch fraction/center is invalid",
    )

    # Paper event boundary
    _require(
        env["decision_events"]
        == ["ORDER_ARRIVAL", "OPERATION_FINISH", "RELOCATION_FINISH"],
        "Only the three paper-defined decision events are allowed.",
    )
    _require(
        env["due_date_is_decision_event"] is False,
        "Due dates must not be policy decision events.",
    )
    _require(
        env["reward"]["base"] == "twt_integral",
        "Base reward must be the objective-consistent TWT integral.",
    )
    _require(
        bool(env["reward"].get("paper_allows_optional_potential_shaping", False)),
        "Paper Eq. (16)-(17) optional potential-shaping flag is missing.",
    )
    if bool(env["reward"].get("optional_potential_shaping", False)):
        eta = env["reward"].get("potential_eta")
        _require(eta is not None and float(eta) > 0.0,
                 "Enabled potential shaping requires positive env.reward.potential_eta.")

    # Phase E graph-state contract.
    graph = env.get("graph")
    _require(graph is not None, "Phase E requires env.graph configuration")
    _require(
        graph["node_types"] == ["operation", "stage", "cell", "worker", "robot"],
        "Graph node types must match the paper's five-node heterogeneous graph.",
    )
    max_paper_stages = max(v["num_stages"] for v in expected_scales.values())
    _require(
        int(graph["max_stages"]) >= max_paper_stages,
        "env.graph.max_stages must cover the largest paper scale.",
    )
    _require(
        graph["continuous_normalization"] == "training_distribution_zscore",
        "Paper graph continuous features must use training-distribution normalization.",
    )
    _require(float(graph["normalizer_min_std"]) > 0, "normalizer_min_std must be positive")
    _require(
        graph["due_pressure_definition"] == "normalized_slack_pressure",
        "Phase E due-pressure implementation choice changed unexpectedly.",
    )
    _require(
        graph["load_capacity_definition"] == "remaining_workload_over_current_service_cells",
        "Phase E load-capacity implementation choice changed unexpectedly.",
    )

    # Paper Table 9
    _require(algo["gamma"] == 0.995, "Paper Table 9 specifies gamma=0.995.")
    _require(algo["tau_0_minutes"] == 10.0, "Paper Table 9 specifies tau_0=10 min.")
    _require(algo["gae_lambda"] == 0.95, "Paper Table 9 specifies GAE lambda=0.95.")
    _require(algo["clip_eps"] == 0.20, "Paper Table 9 specifies PPO clip=0.20.")
    _require(algo["embed_dim"] == 128, "Paper Table 9 specifies embed_dim=128.")
    _require(algo["hgt_layers"] == 3, "Paper Table 9 specifies 3 HGT layers.")
    _require(algo["attention_heads"] == 4, "Paper Table 9 specifies 4 attention heads.")
    _require(
        int(algo["embed_dim"]) % int(algo["attention_heads"]) == 0,
        "embed_dim must be divisible by attention_heads.",
    )
    _require(
        int(algo["relation_type_embed_dim"]) == 32,
        "Paper Table 9 specifies relation/event type embedding dimension 32.",
    )
    matching = algo.get("matching_decoder")
    _require(matching is not None, "Phase G requires algo.matching_decoder configuration")
    _require(
        matching["strategy"] == "masked_autoregressive_edge_selection",
        "Phase G requires masked autoregressive edge-set decoding.",
    )
    _require(bool(matching["stop_token"]), "All Phase G set decoders require STOP.")
    _require(bool(matching["worker_before_robot"]), "Worker matching must precede robot matching.")
    _require(
        bool(matching["update_cell_embedding_after_worker_matching"]),
        "Robot matcher must condition on worker-updated cell embeddings.",
    )
    _require(
        bool(matching["operation_dispatch_is_parallel_set_matching"]),
        "Scheduling must be a set matcher rather than fixed-order dispatch.",
    )
    _require(
        float(matching["scalar_time_reference_minutes"]) > 0,
        "matching scalar_time_reference_minutes must be positive.",
    )

    rep = algo.get("representation_network")
    _require(rep is not None, "Phase F requires algo.representation_network configuration")
    _require(
        rep["pooling"] == "type_aware_attention",
        "Paper Phase F requires type-aware attention pooling.",
    )
    _require(
        bool(rep["relation_attention_uses_edge_features"]),
        "Paper relation-aware attention must consume edge features where available.",
    )
    _require(
        int(rep["event_type_embed_dim"]) == int(algo["relation_type_embed_dim"]),
        "Phase F event-type embedding must use the paper's 32-d relation/event embedding size.",
    )
    _require(
        int(rep["node_categorical_embed_dim"]) > 0,
        "node_categorical_embed_dim must be positive.",
    )
    _require(
        float(rep["dropout"]) == 0.0,
        "Phase F keeps dropout at 0.0 so individual/batched graph equivalence is exact in smoke tests.",
    )
    _require(algo["rollout_events"] == 8192, "Paper Table 9 specifies 8192 events/iteration.")
    _require(algo["minibatch_size"] == 512, "Paper Table 9 specifies minibatch 512.")
    _require(algo["ppo_epochs"] == 4, "Paper Table 9 specifies 4 PPO epochs.")
    _require(algo["max_grad_norm"] == 0.5, "Paper Table 9 specifies grad clip 0.5.")
    # Advisor Scheme-2 overrides the historical paper replication protocol:
    # the formal experiment is exactly one run with seed 0.
    seeds = list(algo["training_seeds"])
    _require(
        int(algo["num_training_seeds"]) == len(seeds),
        "num_training_seeds must match training_seeds length.",
    )
    _require(
        seeds == [0],
        "Advisor-approved Scheme-2 formal protocol requires exactly seed 0.",
    )
    _require(
        algo.get("training_stabilization", {}).get("validation_early_stopping") is False,
        "Scheme-2 uses a fixed iteration budget; validation cannot early-stop training.",
    )
    ppo_impl = algo.get("ppo_implementation", {})
    _require(
        ppo_impl.get("policy_advantage_fusion") == "mean_active_heads",
        "Phase H requires the documented mean_active_heads advantage fusion.",
    )
    _require(
        float(ppo_impl.get("value_clip_eps", 0.0)) > 0.0,
        "Phase H value_clip_eps must be positive.",
    )
    _require(
        ppo_impl.get("rollout_storage_device") in {"cpu"},
        "Phase H currently fixes rollout storage to CPU for bounded accelerator memory.",
    )
    evaluation_protocol = inst.get("evaluation_protocol", {})
    _require(
        int(evaluation_protocol.get("test_instances_per_combination_min", 0)) == 1,
        "Advisor Scheme-2 requires exactly one fixed test instance per combination.",
    )

    # Paper Section 6.9
    stages = curriculum["stages"]
    _require(len(stages) == 4, "Curriculum must contain exactly four stages.")
    expected_names = [
        "fixed_configuration_warmup",
        "single_resource_reconfiguration",
        "full_set_reconfiguration",
        "multi_scale_joint_finetuning",
    ]
    _require(
        [stage["name"] for stage in stages] == expected_names,
        "Curriculum stage order differs from the paper.",
    )


def load_config(
    paths: Iterable[str | Path],
    overrides: Iterable[str] | None = None,
) -> ProjectConfig:
    raw = load_raw_config(paths, overrides)
    validate_raw_config(raw)
    return ProjectConfig(
        instance=ConfigNode(raw["instance"]),
        env=ConfigNode(raw["env"]),
        algo=ConfigNode(raw["algo"]),
        curriculum=ConfigNode(raw["curriculum"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load, merge, override, and validate EGDM-HGPPO YAML configs."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="YAML files merged from left to right.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Dotted-key override, e.g. --set algo.parallel_envs=2",
    )
    parser.add_argument(
        "--print",
        dest="print_config",
        action="store_true",
        help="Print the effective validated configuration as JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.print_config:
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
    else:
        print("Configuration validation passed.")


if __name__ == "__main__":
    main()
