"""Phase J run directories, configuration snapshots, and exact iteration-boundary checkpoints."""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import sys
import hashlib
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_VERSION = 1


def _source_manifest(root: Path, config_paths) -> dict[str, Any]:
    """Return a lightweight, deterministic source/config fingerprint for a run."""
    root = Path(root).resolve()
    files: list[dict[str, str]] = []
    seen: set[str] = set()
    candidates = [root / str(p) for p in config_paths]
    candidates.extend(root / name for name in (
        "scripts/scheme2.py", "agent/experiments/scheme2.py", "agent/training/trainer_random_joint.py",
        "agent/training/trainer_j.py", "agent/training/multiprocess_rollout.py",
        "agent/ppo.py", "agent/policy.py", "agent/pooling.py",
    ))
    for path in candidates:
        try:
            resolved = path.resolve()
            key = str(resolved)
            if key in seen or not resolved.is_file() or root not in resolved.parents:
                continue
            seen.add(key)
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            files.append({"path": str(resolved.relative_to(root).as_posix()), "sha256": digest})
        except OSError:
            continue
    files.sort(key=lambda item: item["path"])
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(item["path"].encode("utf-8"))
        aggregate.update(item["sha256"].encode("ascii"))
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {
        "root": str(root),
        "git_commit": git_commit,
        "files": files,
        "aggregate_sha256": aggregate.hexdigest(),
    }


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def create_run_directory(root: str | Path, run_name: str) -> Path:
    path = Path(root) / str(run_name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    return path


def _environment_metadata() -> dict[str, Any]:
    """Capture interpreter and accelerator facts without making snapshots fragile."""
    cuda_available = bool(torch.cuda.is_available())
    meta: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": 0,
        "cuda_device_index": None,
        "cuda_device_name": None,
        "cuda_total_vram_gb": None,
        "cuda_bf16_supported": False,
        "pid": os.getpid(),
    }
    if not cuda_available:
        return meta

    try:
        device_count = int(torch.cuda.device_count())
        meta["cuda_device_count"] = device_count
        if device_count <= 0:
            return meta
        device_index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device_index)
        meta["cuda_device_index"] = device_index
        meta["cuda_device_name"] = str(properties.name)
        meta["cuda_total_vram_gb"] = round(float(properties.total_memory) / (1024 ** 3), 3)
    except (AssertionError, RuntimeError):
        # A driver query failure must not discard an otherwise recoverable run.
        pass

    try:
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        meta["cuda_bf16_supported"] = bool(is_bf16_supported()) if is_bf16_supported else None
    except (AssertionError, RuntimeError):
        meta["cuda_bf16_supported"] = None
    return meta


def write_run_snapshot(
    run_dir: str | Path,
    *,
    project_config,
    runtime_settings,
    config_paths,
) -> None:
    run_dir = Path(run_dir)
    snapshot = run_dir / "config_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for config_path in config_paths:
        source = Path(config_path)
        if source.is_file():
            shutil.copy2(source, snapshot / source.name)
    with (run_dir / "effective_project_config.json").open("w", encoding="utf-8") as handle:
        json.dump(project_config.to_dict(), handle, ensure_ascii=False, indent=2)
    settings = asdict(runtime_settings) if is_dataclass(runtime_settings) else dict(runtime_settings)
    with (run_dir / "run_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)
    meta = _environment_metadata()
    with (run_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    project_root = Path.cwd().resolve()
    for candidate in config_paths:
        source = Path(candidate)
        if not source.is_absolute():
            source = (Path.cwd() / source).resolve()
        if source.is_file() and source.parent.name == "configs":
            project_root = source.parent.parent
            break
    manifest = _source_manifest(project_root, config_paths)
    with (run_dir / "source_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    (run_dir / "git_commit.txt").write_text(
        str(manifest.get("git_commit") or "unavailable: project is not a git worktree") + "\n",
        encoding="utf-8",
    )


def save_training_checkpoint(
    path: str | Path,
    *,
    policy,
    optimizer,
    normalizer,
    vector_env,
    collector_next_env: int,
    curriculum_state: dict,
    metrics_rows: list[dict],
    validation_rows: list[dict],
    run_settings,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "normalizer": normalizer,
        "vector_env": vector_env,
        "collector_next_env": int(collector_next_env),
        "curriculum_state": curriculum_state,
        "metrics_rows": list(metrics_rows),
        "validation_rows": list(validation_rows),
        "run_settings": asdict(run_settings) if is_dataclass(run_settings) else dict(run_settings),
        "rng_state": capture_rng_state(),
    }
    torch.save(payload, path)
    return path


def load_training_checkpoint(path: str | Path, *, map_location="cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("unsupported Phase J checkpoint version")
    return payload


def save_model_optimizer_snapshot(
    path: str | Path, *, policy, optimizer, metric: float, stage_index: int,
    normalizer=None, global_iteration: int | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_version": CHECKPOINT_VERSION,
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "validation_metric": float(metric),
        "stage_index": int(stage_index),
        "normalizer": normalizer,
        "global_iteration": (None if global_iteration is None else int(global_iteration)),
    }, path)
    return path


def load_model_optimizer_snapshot(path: str | Path, *, map_location="cpu") -> dict:
    """Load a lightweight model/optimizer snapshot without mutating modules."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("unsupported model snapshot checkpoint version")
    if "policy_state" not in payload or "optimizer_state" not in payload:
        raise ValueError("model snapshot is missing policy/optimizer state")
    return payload


def restore_model_optimizer_snapshot(path: str | Path, *, policy, optimizer, map_location="cpu") -> dict:
    payload = load_model_optimizer_snapshot(path, map_location=map_location)
    policy.load_state_dict(payload["policy_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return payload


__all__ = [
    "capture_rng_state", "create_run_directory", "load_model_optimizer_snapshot",
    "load_training_checkpoint", "restore_model_optimizer_snapshot", "restore_rng_state",
    "save_model_optimizer_snapshot", "save_training_checkpoint",
    "write_run_snapshot",
]
