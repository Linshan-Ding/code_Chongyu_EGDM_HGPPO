"""Logical parallel event environments for Phase I.

This is a single-process round-robin vector wrapper.  It proves the multi-env
rollout/GAE semantics before Phase J adds performance-oriented orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass

from environment.env import AssemblyEnv
from environment.replay_state import capture_compact_snapshot
from agent.training.numerics import assert_reward_identity


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    env_id: int
    episode_id: int
    instance_id: str
    decisions: int
    total_reward: float
    base_twt_reward: float
    twt: float
    reward_identity_error: float
    final_time: float


@dataclass(slots=True)
class _Slot:
    env: AssemblyEnv
    sampler: object
    episode_id: int = -1
    episode_reward: float = 0.0
    episode_base_twt_reward: float = 0.0
    episode_decisions: int = 0
    instance_id: str = ""


class EventVectorEnv:
    def __init__(self, cfg, samplers, *, max_episode_decisions: int) -> None:
        samplers = list(samplers)
        if not samplers:
            raise ValueError("at least one environment sampler is required")
        self.cfg = cfg
        self.max_episode_decisions = int(max_episode_decisions)
        if self.max_episode_decisions <= 0:
            raise ValueError("max_episode_decisions must be positive")
        self.slots = [_Slot(AssemblyEnv(cfg), sampler) for sampler in samplers]
        self.reset_all()

    def __len__(self) -> int:
        return len(self.slots)

    def reset_all(self) -> None:
        for env_id in range(len(self.slots)):
            self.reset_slot(env_id)

    def reset_slot(self, env_id: int) -> None:
        slot = self.slots[int(env_id)]
        instance = slot.sampler.sample()
        slot.env.reset(instance)
        slot.episode_id += 1
        slot.episode_reward = 0.0
        slot.episode_base_twt_reward = 0.0
        slot.episode_decisions = 0
        slot.instance_id = instance.instance_id

    def current_key(self, env_id: int) -> tuple[int, int]:
        slot = self.slots[int(env_id)]
        return int(env_id), int(slot.episode_id)

    def graph(self, env_id: int):
        return self.slots[int(env_id)].env.graph()

    def action_context(self, env_id: int):
        return self.slots[int(env_id)].env.action_context()

    def replay_snapshot(self, env_id: int):
        """Capture a compact pre-action state for exact PPO replay."""
        return capture_compact_snapshot(self.slots[int(env_id)].env.sim)

    def step(self, env_id: int, action):
        slot = self.slots[int(env_id)]
        state, reward, done, info = slot.env.step(action)
        slot.episode_reward += float(reward)
        slot.episode_base_twt_reward += float(info["reward_twt"])
        slot.episode_decisions += 1
        if slot.episode_decisions > self.max_episode_decisions:
            raise RuntimeError(
                f"env {env_id} episode exceeded {self.max_episode_decisions} decisions"
            )

        summary = None
        if done:
            twt = float(info["twt"])
            error = assert_reward_identity(
                slot.episode_base_twt_reward,
                twt,
                context=f"runtime reward identity failed in env {env_id}",
            )
            summary = EpisodeSummary(
                env_id=int(env_id),
                episode_id=int(slot.episode_id),
                instance_id=slot.instance_id,
                decisions=int(slot.episode_decisions),
                total_reward=float(slot.episode_reward),
                base_twt_reward=float(slot.episode_base_twt_reward),
                twt=twt,
                reward_identity_error=error,
                final_time=float(state.time),
            )
        return state, float(reward), bool(done), info, summary


__all__ = ["EpisodeSummary", "EventVectorEnv"]
