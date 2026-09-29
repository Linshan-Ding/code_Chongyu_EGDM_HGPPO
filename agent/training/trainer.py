"""Phase I real fixed-configuration PPO trainer."""

from __future__ import annotations

import csv
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from agent.policy import EGDMCompositePolicy
from agent.ppo import PPOAgent
from agent.training.curriculum import (
    configure_fixed_configuration_warmup,
    fixed_configuration_component_mask,
)
from agent.training.normalization import fit_training_normalizer
from agent.training.rollout import RolloutCollector
from agent.training.sampler import OnlineInstanceSampler
from agent.training.vector_env import EventVectorEnv


@dataclass(frozen=True, slots=True)
class PhaseIRunSettings:
    iterations: int
    parallel_envs: int
    rollout_events: int
    training_seed: int
    scale_pool: tuple[str, ...]
    scenario_pool: tuple[str, ...]
    normalization_episodes: int
    normalization_max_graphs: int
    max_episode_decisions: int
    device: str
    log_csv: str


@dataclass(frozen=True, slots=True)
class IterationMetrics:
    iteration: int
    events: int
    completed_episodes: int
    mean_completed_twt: float | None
    reward_sum: float
    reward_mean: float
    mean_delta_t: float
    mean_schedule_edges: float
    reconfiguration_fraction: float
    mean_raw_reconfigure_probability: float
    policy_loss: float
    value_loss: float
    value_sch_loss: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    preclip_grad_norm: float
    learning_rate: float
    encoder_learning_rate: float
    schedule_entropy: float
    matching_entropy_coef: float
    optimizer_steps: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class PhaseITrainingResult:
    settings: PhaseIRunSettings
    device: str
    normalization_graphs: int
    normalization_episodes_started: int
    trainable_parameters: int
    frozen_parameters: int
    metrics: tuple[IterationMetrics, ...]
    total_parameter_l1_change: float


def resolve_device(name: str) -> torch.device:
    name = str(name)
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _child_seeds(seed: int, count: int) -> list[int]:
    sequence = np.random.SeedSequence(int(seed))
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in sequence.spawn(count)]


def _parameter_vector(policy) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in policy.parameters()])


def _write_metrics(path: str | Path, metrics: list[IterationMetrics]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in metrics]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class PhaseITrainer:
    """Train only curriculum Stage I on online S/M/L generator instances."""

    def __init__(self, cfg, settings: PhaseIRunSettings) -> None:
        self.cfg = cfg
        self.settings = settings
        if settings.iterations <= 0 or settings.parallel_envs <= 0 or settings.rollout_events <= 0:
            raise ValueError("iterations/parallel_envs/rollout_events must be positive")
        self.device = resolve_device(settings.device)

    def run(self) -> PhaseITrainingResult:
        _seed_everything(self.settings.training_seed)
        # Dedicated, deterministic training-only RNG streams: one for normalization
        # and one per logical rollout environment.
        seeds = _child_seeds(self.settings.training_seed, self.settings.parallel_envs + 1)
        normalization_sampler = OnlineInstanceSampler(
            self.cfg,
            seed=seeds[0],
            scale_pool=self.settings.scale_pool,
            scenario_pool=self.settings.scenario_pool,
        )
        normalizer, norm_summary = fit_training_normalizer(
            self.cfg,
            normalization_sampler,
            episodes=self.settings.normalization_episodes,
            max_graphs=self.settings.normalization_max_graphs,
            max_episode_decisions=self.settings.max_episode_decisions,
        )

        samplers = [
            OnlineInstanceSampler(
                self.cfg,
                seed=seeds[i + 1],
                scale_pool=self.settings.scale_pool,
                scenario_pool=self.settings.scenario_pool,
            )
            for i in range(self.settings.parallel_envs)
        ]
        vector_env = EventVectorEnv(
            self.cfg,
            samplers,
            max_episode_decisions=self.settings.max_episode_decisions,
        )
        reference_graph = normalizer.transform(vector_env.graph(0))
        policy = EGDMCompositePolicy(self.cfg, reference_graph).to(self.device)
        trainability = configure_fixed_configuration_warmup(policy)
        agent = PPOAgent(self.cfg, policy)
        components = fixed_configuration_component_mask()
        collector = RolloutCollector(
            vector_env=vector_env,
            agent=agent,
            normalizer=normalizer,
            storage_device=self.cfg.algo.ppo_implementation.rollout_storage_device,
            force_gate=False,
        )

        before_all = _parameter_vector(policy)
        metrics: list[IterationMetrics] = []
        for iteration in range(self.settings.iterations):
            started = time.perf_counter()
            rollout = collector.collect(self.settings.rollout_events)
            # Phase I does not yet know the full four-stage training horizon.
            # Keep paper LR/entropy schedules at their start values; Phase J will
            # own global curriculum progress and annealing.
            update = agent.update(
                rollout.buffer,
                bootstrap_values=rollout.bootstrap_values,
                progress=0.0,
                components=components,
            )
            twts = [summary.twt for summary in rollout.episode_summaries]
            item = IterationMetrics(
                iteration=iteration,
                events=rollout.stats.events,
                completed_episodes=rollout.stats.completed_episodes,
                mean_completed_twt=None if not twts else mean(twts),
                reward_sum=rollout.stats.reward_sum,
                reward_mean=rollout.stats.reward_mean,
                mean_delta_t=rollout.stats.mean_delta_t,
                mean_schedule_edges=rollout.stats.mean_schedule_edges,
                reconfiguration_fraction=rollout.stats.reconfiguration_fraction,
                mean_raw_reconfigure_probability=rollout.stats.mean_raw_reconfigure_probability,
                policy_loss=update.policy_loss,
                value_loss=update.value_loss,
                value_sch_loss=update.value_sch_loss,
                total_loss=update.total_loss,
                approx_kl=update.approx_kl,
                clip_fraction=update.clip_fraction,
                preclip_grad_norm=update.preclip_grad_norm,
                learning_rate=update.learning_rate,
                encoder_learning_rate=update.encoder_learning_rate,
                schedule_entropy=update.schedule_entropy,
                matching_entropy_coef=update.matching_entropy_coef,
                optimizer_steps=update.optimizer_steps,
                wall_seconds=time.perf_counter() - started,
            )
            floats = [
                item.reward_sum, item.reward_mean, item.mean_delta_t,
                item.mean_schedule_edges, item.reconfiguration_fraction,
                item.mean_raw_reconfigure_probability, item.policy_loss,
                item.value_loss, item.total_loss, item.approx_kl,
                item.clip_fraction, item.preclip_grad_norm,
            ]
            if not all(np.isfinite(x) for x in floats):
                raise FloatingPointError("Phase I iteration produced a non-finite metric")
            if item.reconfiguration_fraction != 0.0:
                raise RuntimeError("Stage I must not execute resource reconfiguration")
            metrics.append(item)
            _write_metrics(self.settings.log_csv, metrics)

        after_all = _parameter_vector(policy)
        parameter_change = float((after_all - before_all).abs().sum())
        if parameter_change <= 0.0:
            raise RuntimeError("real Phase I training did not change any policy parameter")
        return PhaseITrainingResult(
            settings=self.settings,
            device=str(self.device),
            normalization_graphs=norm_summary.graphs,
            normalization_episodes_started=norm_summary.episodes_started,
            trainable_parameters=trainability.trainable_parameters,
            frozen_parameters=trainability.frozen_parameters,
            metrics=tuple(metrics),
            total_parameter_l1_change=parameter_change,
        )


__all__ = [
    "IterationMetrics",
    "PhaseIRunSettings",
    "PhaseITrainer",
    "PhaseITrainingResult",
    "resolve_device",
]
