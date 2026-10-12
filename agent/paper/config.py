import hashlib
import json
from pathlib import Path

import yaml

from configs.config import ConfigNode, ProjectConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "paper-sm100-1"
EPOCH_SEMANTICS = "same_instance_complete_batch_v1"
LEARNERS = ("full", "mlp", "shared_relation", "periodic_gate", "no_rule_prior", "gat", "dan")
ALIASES = {"no_graph": "mlp", "MLP-PPO": "mlp", "GAT-PPO": "gat", "Periodic-DAN": "dan"}
RULES = ("Fixed-EDD", "Fixed-ATC", "Periodic-ATC", "RH-MILP", "RH-ALNS", "Opportunity-ATC")
EXECUTION_KEYS = {"result_root", "data_manifest", "workers", "update_threads", "visdom_port", "monitor_seconds", "run_id"}


def scientific_options(options):
    return {k: v for k, v in options.items() if k not in EXECUTION_KEYS}


def settings(path=None, **overrides):
    value = yaml.safe_load(Path(path or ROOT / "configs/paper.yaml").read_text("utf-8"))
    value.update(overrides)
    value.setdefault("save_full_rollouts", False)
    if not isinstance(value["save_full_rollouts"], bool):
        raise ValueError("save_full_rollouts must be a boolean")
    if "complete_epochs_per_update" in value or value.get("epoch_semantics") != EPOCH_SEMANTICS:
        raise ValueError("Incompatible epoch semantics; use same-instance complete batches and remove complete_epochs_per_update")
    if value["device"] != "cpu" or value["gamma"] != 1 or value["dispatch_beta"] != 1:
        raise ValueError("This registered protocol requires CPU, gamma=1 and ATC beta=1")
    if not 1 <= value["workers"] <= value["environments"]:
        raise ValueError("Require 1 <= workers <= environments")
    if any(k in value for k in ("pilot_epochs", "formal_epochs", "evaluation_epochs")):
        raise ValueError("Old total-epoch/periodic-validation options are incompatible with SM100")
    for key in ("pilot_epochs_per_instance", "formal_epochs_per_instance", "tail_epochs", "environments", "workers", "ppo_update_epochs", "minibatch", "microbatch", "update_threads"):
        if not isinstance(value[key], int) or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if value["max_candidates"] < 2 + value["single_candidates"] + value["pair_candidates"] + 1:
        raise ValueError("Candidate budget cannot silently truncate a registered family")
    return value


def project_config(options):
    cfg = load_config([ROOT / "configs" / name for name in
                       ("instance.yaml", "env.yaml", "algo.yaml", "curriculum.yaml")]).to_dict()
    cfg["algo"].update(gamma=1.0, embed_dim=options["hidden"], hgt_layers=options["layers"])
    cfg["env"]["graph"].update(drop_completed=True, sparse_competition=True, continuous_normalization="signed_log1p")
    cfg["env"]["reward"].update(optional_potential_shaping=True, potential_eta=options["potential_eta"])
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
