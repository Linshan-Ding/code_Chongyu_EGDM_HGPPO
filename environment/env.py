"""Standard reset/step environment wrapper for EGDM-HGPPO through Phase G."""

from __future__ import annotations

from dataclasses import dataclass

from data.schema import AssemblyInstance
from environment.action_context import build_action_context
from environment.graph_builder import DynamicHeteroGraphBuilder
from environment.masks import (
    InvalidActionError,
    build_action_candidates,
    validate_composite_action,
)
from environment.reward import duration_discount, tardiness_potential
from environment.simulator import AssemblySimulator, SimulatorError
from environment.state import ActionCandidates, CompositeAction, DecisionState


class DeadlockError(SimulatorError):
    pass


@dataclass(frozen=True, slots=True)
class StepInfo:
    delta_t: float
    gamma_effective: float
    twt: float | None
    reward_twt: float
    reward_shaping: float
    reconfiguration_count: int
    decision_event_types: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "delta_t": self.delta_t,
            "gamma_effective": self.gamma_effective,
            "twt": self.twt,
            "reward_twt": self.reward_twt,
            "reward_shaping": self.reward_shaping,
            "reconfiguration_count": self.reconfiguration_count,
            "decision_event_types": self.decision_event_types,
        }


class AssemblyEnv:
    """Paper-specific SMDP environment.

    reset/step retain the validated Phase D DecisionState contract; Phase E adds
    graph() as a read-only five-node heterogeneous representation of that state.
    """

    def __init__(self, cfg, *, minimum_dwell_time: float | None = None) -> None:
        self.cfg = cfg
        self.sim = AssemblySimulator(cfg, minimum_dwell_time=minimum_dwell_time)
        self.graph_builder = DynamicHeteroGraphBuilder(cfg)
        self._last_state: DecisionState | None = None

    def reset(self, instance: AssemblyInstance) -> DecisionState:
        state = self.sim.reset(instance)
        self._last_state = state
        return state

    def candidates(self) -> ActionCandidates:
        if self.sim.instance is None:
            raise RuntimeError("call reset() before candidates()")
        return build_action_candidates(self.sim)

    def action_context(self):
        """Return exact read-only feasibility tensors for Phase G decoders.

        The context contains only already-arrived operations plus static instance
        facts needed to update autoregressive masks.  It does not advance time or
        expose future orders.
        """
        if self.sim.instance is None:
            raise RuntimeError("call reset() before action_context()")
        return build_action_context(self.sim)

    def graph(self):
        """Return the current raw Phase E heterogeneous graph.

        Continuous features are intentionally raw here.  Training-distribution
        z-score statistics are fitted separately by GraphContinuousNormalizer so
        validation/test information can never leak into normalization statistics.
        """
        if self.sim.instance is None:
            raise RuntimeError("call reset() before graph()")
        return self.graph_builder.build(self.sim)

    def step(self, action: CompositeAction):
        if self.sim.instance is None:
            raise RuntimeError("call reset() before step()")
        if self.sim.is_done:
            raise RuntimeError("episode is already done")

        shaping_enabled = bool(self.cfg.env.reward.optional_potential_shaping)
        # Controlled ablation switch. It is injected only in an ablation run's
        # private config and never changes the frozen formal Scheme-2 config.
        shaping_enabled = shaping_enabled and not bool(
            getattr(self.cfg.env.reward, "ablation_disable_potential_shaping", False)
        )
        eta_raw = self.cfg.env.reward.potential_eta
        if shaping_enabled and eta_raw is None:
            raise ValueError("potential shaping is enabled but env.reward.potential_eta is null")
        phi_before = (
            tardiness_potential(
                self.sim.instance, self.sim.orders, self.sim.operations, self.sim.time, float(eta_raw)
            )
            if shaping_enabled else 0.0
        )

        plan = validate_composite_action(self.sim, action)
        self.sim.apply_validated_action(plan)

        # If nothing was started/moved and no external event remains, the episode is
        # physically deadlocked rather than a valid terminal state.
        if not self.sim.is_done and len(self.sim.events) == 0:
            raise DeadlockError(
                "unfinished work remains, no processing/relocation is active, and no future arrival exists"
            )

        advance = self.sim.advance()
        done = self.sim.is_done
        state = self.sim.snapshot()
        self._last_state = state

        gamma_eff = duration_discount(
            float(self.cfg.algo.gamma),
            float(advance.delta_t),
            float(self.cfg.algo.tau_0_minutes),
        )

        phi_after = (
            tardiness_potential(
                self.sim.instance, self.sim.orders, self.sim.operations, self.sim.time, float(eta_raw)
            )
            if shaping_enabled else 0.0
        )
        shaping = float(gamma_eff * phi_after - phi_before) if shaping_enabled else 0.0
        reward = float(advance.base_reward) + shaping
        twt = self.sim.twt() if done else None
        info = StepInfo(
            delta_t=float(advance.delta_t),
            gamma_effective=gamma_eff,
            twt=twt,
            reward_twt=float(advance.base_reward),
            reward_shaping=shaping,
            reconfiguration_count=int(self.sim.reconfiguration_count),
            decision_event_types=tuple(t.value for t in advance.event_types),
        ).to_dict()
        return state, reward, done, info


__all__ = [
    "AssemblyEnv",
    "CompositeAction",
    "DeadlockError",
    "InvalidActionError",
]
