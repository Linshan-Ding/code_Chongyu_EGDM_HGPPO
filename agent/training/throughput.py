"""Execution-only throughput profile for L1.5.

This module deliberately lives outside ``configs/phase_l1.yaml`` so changing
hardware execution granularity does not invalidate the already frozen reward
protocol.  Logical PPO semantics remain controlled by the paper-aligned L1 file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class ThroughputPilotConfig:
    additional_iterations: int
    run_name_prefix: str


@dataclass(frozen=True, slots=True)
class FormalStrategy1Config:
    enabled: bool
    worker_processes: int
    worker_torch_threads: int
    start_method: str
    static_processing_cache: bool
    trusted_replay_normalization: bool
    cross_epoch_materialize_cache: bool
    materialize_cache_max_mib: int


@dataclass(frozen=True, slots=True)
class ThroughputProfile:
    rollout_action_batch_size: int
    replay_microbatch_size: int
    replay_microbatch_min_size: int
    cuda_oom_fallback: bool
    progress_every_events: int
    diagnostics: bool
    formal_strategy1: FormalStrategy1Config
    pilot: ThroughputPilotConfig


def load_throughput_profile(path: str | Path = "configs/throughput.yaml") -> ThroughputProfile:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = raw.get("throughput")
    if not isinstance(root, dict):
        raise ValueError("configs/throughput.yaml must contain throughput")
    pilot = root.get("pilot") or {}
    formal = root.get("formal_strategy1") or {}
    profile = ThroughputProfile(
        rollout_action_batch_size=int(root["rollout_action_batch_size"]),
        replay_microbatch_size=int(root["replay_microbatch_size"]),
        replay_microbatch_min_size=int(root.get("replay_microbatch_min_size", 1)),
        cuda_oom_fallback=bool(root.get("cuda_oom_fallback", True)),
        progress_every_events=int(root.get("progress_every_events", 0)),
        diagnostics=bool(root.get("diagnostics", True)),
        formal_strategy1=FormalStrategy1Config(
            enabled=bool(formal.get("enabled", False)),
            worker_processes=int(formal.get("worker_processes", 4)),
            worker_torch_threads=int(formal.get("worker_torch_threads", 2)),
            start_method=str(formal.get("start_method", "spawn")),
            static_processing_cache=bool(formal.get("static_processing_cache", False)),
            trusted_replay_normalization=bool(formal.get("trusted_replay_normalization", False)),
            cross_epoch_materialize_cache=bool(formal.get("cross_epoch_materialize_cache", False)),
            materialize_cache_max_mib=int(formal.get("materialize_cache_max_mib", 0)),
        ),
        pilot=ThroughputPilotConfig(
            additional_iterations=int(pilot.get("additional_iterations", 1)),
            run_name_prefix=str(pilot.get("run_name_prefix", "l1_5_throughput_pilot_seed")),
        ),
    )
    if profile.rollout_action_batch_size <= 0:
        raise ValueError("rollout_action_batch_size must be positive")
    if profile.replay_microbatch_size <= 0:
        raise ValueError("replay_microbatch_size must be positive")
    if not 0 < profile.replay_microbatch_min_size <= profile.replay_microbatch_size:
        raise ValueError("replay_microbatch_min_size must be within replay microbatch size")
    if profile.progress_every_events < 0:
        raise ValueError("progress_every_events must be non-negative")
    if profile.pilot.additional_iterations <= 0:
        raise ValueError("pilot.additional_iterations must be positive")
    formal = profile.formal_strategy1
    if formal.enabled:
        if formal.worker_processes <= 1:
            raise ValueError("formal_strategy1.worker_processes must be at least 2")
        if formal.worker_torch_threads <= 0:
            raise ValueError("formal_strategy1.worker_torch_threads must be positive")
        if formal.start_method not in {"spawn", "forkserver", "fork"}:
            raise ValueError("formal_strategy1.start_method is unsupported")
        if not formal.static_processing_cache:
            raise ValueError("formal Strategy-1 promotion requires the validated static processing cache")
        if not formal.trusted_replay_normalization:
            raise ValueError("formal Strategy-1 promotion requires the validated trusted normalization fast path")
        if formal.cross_epoch_materialize_cache and formal.materialize_cache_max_mib <= 0:
            raise ValueError("formal cross-epoch materialize cache requires a positive materialize_cache_max_mib")
    return profile


__all__ = [
    "FormalStrategy1Config", "ThroughputPilotConfig", "ThroughputProfile",
    "load_throughput_profile",
]
