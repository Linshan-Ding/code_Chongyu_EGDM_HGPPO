import hashlib
import json
from pathlib import Path

import yaml

from configs.config import ConfigNode, ProjectConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "paper-sm100-2"
EPOCH_SEMANTICS = "multi_instance_update_v2"
LEARNERS = ("full", "mlp", "shared_relation", "periodic_gate", "no_rule_prior", "fixed_dispatch", "gat", "dan")
ALIASES = {"no_graph": "mlp", "MLP-PPO": "mlp", "GAT-PPO": "gat", "Periodic-DAN": "dan"}
RULES = ("Fixed-EDD", "Fixed-ATC", "Periodic-ATC", "RH-MILP", "RH-ALNS", "Opportunity-ATC")
# Execution keys never enter the scientific identity. ``device`` is execution:
# the protocol does not promise bitwise identity across hardware anyway.
EXECUTION_KEYS = {"result_root", "data_manifest", "workers", "update_threads", "visdom_port", "monitor_seconds",
                  "run_id", "device"}
CHOICES = dict(advantage_mode=("gae", "self_critical"), critic=("none", "auxiliary"),
               potential_mode=("legacy", "future_only"), candidate_version=("v1", "v2"),
               gate_trigger_mode=("every_event", "trigger_events"),
               train_instance_perturbation=("none", "arrivals"), validation_split=("validation",))
LEGACY_KEYS = ("complete_epochs_per_update", "pilot_epochs", "formal_epochs", "evaluation_epochs",
               "pilot_epochs_per_instance", "formal_epochs_per_instance")


def scientific_options(options):
    return {k: v for k, v in options.items() if k not in EXECUTION_KEYS}


def settings(path=None, **overrides):
    value = yaml.safe_load(Path(path or ROOT / "configs/paper.yaml").read_text("utf-8"))
    value.update(overrides)
    value.setdefault("save_full_rollouts", False)
    if not isinstance(value["save_full_rollouts"], bool):
        raise ValueError("save_full_rollouts must be a boolean")
    if any(k in value for k in LEGACY_KEYS) or value.get("epoch_semantics") != EPOCH_SEMANTICS:
        raise ValueError("Incompatible epoch semantics; use multi-instance update batches (paper-sm100-2)")
    if value["device"] not in ("cpu", "cuda") or value["gamma"] != 1 or value["dispatch_beta"] != 1:
        raise ValueError("This registered protocol requires device cpu|cuda, gamma=1 and ATC beta=1")
    for key in ("instances_per_update", "episodes_per_instance", "validation_every_updates", "pilot_updates",
                "formal_updates", "tail_epochs", "workers", "ppo_update_epochs", "minibatch", "microbatch",
                "update_threads", "kl_anneal_updates", "candidate_max_moves", "replays"):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(value["lr_warmup_updates"], int) or value["lr_warmup_updates"] < 0:
        raise ValueError("lr_warmup_updates must be a non-negative integer")
    # One update = instances_per_update cases x episodes_per_instance complete
    # episodes. ``environments`` is the derived episode count per update.
    derived = value["instances_per_update"] * value["episodes_per_instance"]
    if value.setdefault("environments", derived) != derived:
        raise ValueError("environments must equal instances_per_update * episodes_per_instance")
    if not 1 <= value["workers"] <= value["environments"]:
        raise ValueError("Require 1 <= workers <= environments")
    for key, allowed in CHOICES.items():
        if value[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
    if value["advantage_mode"] == "gae" and value["critic"] == "none":
        raise ValueError("GAE advantages require a trained critic")
    for key in ("kl_prior_coef_start", "kl_prior_coef_end", "entropy", "entropy_dan", "potential_eta", "travel_penalty"):
        if not isinstance(value[key], (int, float)) or value[key] < 0:
            raise ValueError(f"{key} must be a non-negative number")
    if not 0 < value["lr_final_ratio"] <= 1:
        raise ValueError("lr_final_ratio must lie in (0, 1]")
    if value["target_kl"] is not None and not value["target_kl"] > 0:
        raise ValueError("target_kl must be null or positive")
    for key in ("anchor_prior", "dispatch_prior"):
        if not 0 < value[key] < 1:
            raise ValueError(f"{key} must lie in (0, 1)")
    for key in ("group_center", "value_clip", "transit_aware_features", "allow_unconfigure", "learned_dispatch"):
        if not isinstance(value[key], bool):
            raise ValueError(f"{key} must be a boolean")
    budget = value["reconfiguration_budget"]
    if budget is not None and (not isinstance(budget, dict) or not budget or
                               any(not isinstance(v, int) or v < 0 for v in budget.values())):
        raise ValueError("reconfiguration_budget must be null or a scale -> non-negative integer mapping")
    if value["max_candidates"] < 2 + value["single_candidates"] + value["pair_candidates"] + 1:
        raise ValueError("Candidate budget cannot silently truncate a registered family")
    return value


def project_config(options):
    cfg = load_config([ROOT / "configs" / name for name in
                       ("instance.yaml", "env.yaml", "algo.yaml", "curriculum.yaml")]).to_dict()
    cfg["algo"].update(gamma=1.0, embed_dim=options["hidden"], hgt_layers=options["layers"])
    cfg["env"]["graph"].update(drop_completed=True, sparse_competition=True, continuous_normalization="signed_log1p",
                               transit_features=bool(options.get("transit_aware_features", False)))
    cfg["env"]["reward"].update(optional_potential_shaping=True, potential_eta=options["potential_eta"],
                                potential_mode=options.get("potential_mode", "legacy"))
    return ProjectConfig(**{k: ConfigNode(v) for k, v in cfg.items()})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    files = set()
    for folder in ("agent", "environment", "data", "configs"):
        files.update((ROOT / folder).rglob("*.py"))
    files.update((ROOT / "configs").glob("*.yaml"))
    files.add(ROOT / "scripts/run_31_paper.py")
    files.update(p for p in (ROOT / "vendor/dan_official").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files) if p.exists()}
    # Execution locations and worker counts may differ between desktops.
    paper = yaml.safe_load((ROOT / "configs/paper.yaml").read_text("utf-8"))
    hashes["configs/paper.yaml"] = hashlib.sha256(json.dumps(scientific_options(paper), sort_keys=True).encode()).hexdigest()
    return dict(files=hashes, digest=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())
