"""Shared decision-policy interface for Phase K evaluation baselines.

All online baselines receive the same :class:`AssemblyEnv` at the current event
and must return exactly one paper-compatible ``CompositeAction``.  They may use
only policy-visible/currently-arrived information; future arrivals are never
passed to the baseline interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from environment.env import AssemblyEnv
from environment.state import CompositeAction


class DecisionPolicy(ABC):
    method_name: str

    def reset(self, env: AssemblyEnv) -> None:
        """Optional per-instance hook. Stateless policies can ignore it."""

    @abstractmethod
    def act(self, env: AssemblyEnv) -> CompositeAction:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RuleBaselineConfig:
    periodic_interval_minutes: float = 10.0
    atc_k: float = 2.0
    threshold_load_capacity_ratio: float = 1.0
    bottleneck_min_gap: float = 0.15
    relocation_time_penalty: float = 0.05
    max_reconfiguration_moves: int = 2

    def validate(self) -> None:
        if self.periodic_interval_minutes < 0:
            raise ValueError("periodic_interval_minutes must be non-negative")
        if self.atc_k <= 0:
            raise ValueError("atc_k must be positive")
        if self.threshold_load_capacity_ratio < 0:
            raise ValueError("threshold_load_capacity_ratio must be non-negative")
        if self.bottleneck_min_gap < 0:
            raise ValueError("bottleneck_min_gap must be non-negative")
        if self.relocation_time_penalty < 0:
            raise ValueError("relocation_time_penalty must be non-negative")
        if self.max_reconfiguration_moves < 0:
            raise ValueError("max_reconfiguration_moves must be non-negative")


__all__ = ["DecisionPolicy", "RuleBaselineConfig"]
