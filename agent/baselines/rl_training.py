"""Equal-budget PPO training harness for Phase-K2 learned baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import random
import time

import numpy as np
import torch

from agent.constraints import PolicyActionConstraints
from agent.ppo import PPOAgent, PPOComponentMask
from agent.baselines.learned_variants import LEARNED_BASELINE_METHODS, build_learned_baseline_policy
from agent.training.normalization import fit_training_normalizer
from agent.training.rollout import RolloutCollector
from agent.training.sampler import OnlineInstanceSampler
from agent.training.trainer import resolve_device
from agent.training.vector_env import EventVectorEnv


@dataclass(frozen=True, slots=True)
class LearnedBaselineRunSettings:
    method: str
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
    output_dir: str
    ppo_epochs_override: int | None = None


@dataclass(frozen=True, slots=True)
class LearnedBaselineTrainingResult:
    method: str
    checkpoint: str
    interactions: int
    iterations: int
    parameter_change_l1: float
    train_log: str


def _seed(seed: int):
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _child(seed: int, k: int) -> int:
    return int(np.random.SeedSequence([seed, k]).generate_state(1, dtype=np.uint32)[0])


def baseline_constraints(method: str) -> PolicyActionConstraints:
    if method in {"MLP-PPO", "HGT-PPO-Flat"}:
        return PolicyActionConstraints.flat_single_edge()
    return PolicyActionConstraints.unconstrained()


class LearnedBaselineTrainer:
    def __init__(self, cfg, settings: LearnedBaselineRunSettings) -> None:
        if settings.method not in LEARNED_BASELINE_METHODS:
            raise ValueError(f"unsupported baseline method={settings.method}")
        if min(settings.iterations, settings.parallel_envs, settings.rollout_events) <= 0:
            raise ValueError("iterations/parallel_envs/rollout_events must be positive")
        self.cfg = cfg
        self.settings = settings
        self.device = resolve_device(settings.device)

    def run(self) -> LearnedBaselineTrainingResult:
        s = self.settings
        _seed(s.training_seed)
        out = Path(s.output_dir); out.mkdir(parents=True, exist_ok=True)
        norm_sampler = OnlineInstanceSampler(
            self.cfg, seed=_child(s.training_seed, 0),
            scale_pool=s.scale_pool, scenario_pool=s.scenario_pool,
        )
        normalizer, _ = fit_training_normalizer(
            self.cfg, norm_sampler,
            episodes=s.normalization_episodes,
            max_graphs=s.normalization_max_graphs,
            max_episode_decisions=s.max_episode_decisions,
        )
        samplers = [
            OnlineInstanceSampler(
                self.cfg, seed=_child(s.training_seed, i + 1),
                scale_pool=s.scale_pool, scenario_pool=s.scenario_pool,
            ) for i in range(s.parallel_envs)
        ]
        vec = EventVectorEnv(self.cfg, samplers, max_episode_decisions=s.max_episode_decisions)
        reference = normalizer.transform(vec.graph(0))
        policy = build_learned_baseline_policy(s.method, self.cfg, reference).to(self.device)
        agent = PPOAgent(self.cfg, policy)
        if s.ppo_epochs_override is not None:
            agent.ppo_epochs = int(s.ppo_epochs_override)
        constraints = baseline_constraints(s.method)
        collector = RolloutCollector(
            vector_env=vec, agent=agent, normalizer=normalizer,
            storage_device=self.cfg.algo.ppo_implementation.rollout_storage_device,
            constraints=constraints,
        )
        before = torch.cat([p.detach().flatten().cpu() for p in policy.parameters()])
        rows = []
        for it in range(s.iterations):
            t0 = time.perf_counter()
            rollout = collector.collect(s.rollout_events)
            progress = 0.0 if s.iterations <= 1 else it / (s.iterations - 1)
            update = agent.update(
                rollout.buffer, bootstrap_values=rollout.bootstrap_values,
                progress=progress, components=PPOComponentMask.full(),
            )
            rows.append({
                "iteration": it,
                "events": rollout.stats.events,
                "completed_episodes": rollout.stats.completed_episodes,
                "reward_mean": rollout.stats.reward_mean,
                "reconfiguration_fraction": rollout.stats.reconfiguration_fraction,
                "policy_loss": update.policy_loss,
                "value_loss": update.value_loss,
                "approx_kl": update.approx_kl,
                "clip_fraction": update.clip_fraction,
                "learning_rate": update.learning_rate,
                "encoder_learning_rate": update.encoder_learning_rate,
                "wall_seconds": time.perf_counter() - t0,
            })
        after = torch.cat([p.detach().flatten().cpu() for p in policy.parameters()])
        change = float((after - before).abs().sum())
        if not np.isfinite(change) or change <= 0:
            raise RuntimeError(f"{s.method} baseline training did not update parameters")
        log_path = out / "train_log.csv"
        with log_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
        checkpoint = out / "best_model.pt"
        torch.save({
            "method": s.method,
            "policy_state": policy.state_dict(),
            "normalizer": normalizer,
            "training_seed": int(s.training_seed),
            "interactions": int(s.iterations * s.rollout_events),
            "settings": asdict(s),
        }, checkpoint)
        return LearnedBaselineTrainingResult(
            method=s.method, checkpoint=str(checkpoint),
            interactions=s.iterations*s.rollout_events,
            iterations=s.iterations, parameter_change_l1=change,
            train_log=str(log_path),
        )


__all__ = [
    "LearnedBaselineRunSettings", "LearnedBaselineTrainer",
    "LearnedBaselineTrainingResult", "baseline_constraints",
]
