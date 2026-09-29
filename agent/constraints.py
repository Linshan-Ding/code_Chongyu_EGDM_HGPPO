"""Curriculum-time action constraints for the composite policy.

These constraints are *algorithm/curriculum* restrictions, not physical facts.
They stay separate from :class:`environment.action_context.ActionContext`, whose
job is to expose only current simulator feasibility information.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PolicyActionConstraints:
    """Restrictions applied while decoding one composite event action.

    ``force_gate`` is used by curriculum Stage I to force KEEP.  Stage II keeps
    the learned gate but caps each resource matcher at one selected edge before
    STOP.  ``None`` move limits mean full set decoding.
    """

    force_gate: bool | None = None
    max_worker_moves: int | None = None
    max_robot_moves: int | None = None
    max_schedule_assignments: int | None = None

    def validate(self) -> None:
        for name, value in (
            ("max_worker_moves", self.max_worker_moves),
            ("max_robot_moves", self.max_robot_moves),
            ("max_schedule_assignments", self.max_schedule_assignments),
        ):
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} must be non-negative or None")

    @classmethod
    def unconstrained(cls) -> "PolicyActionConstraints":
        return cls()

    @classmethod
    def fixed_configuration(cls) -> "PolicyActionConstraints":
        return cls(force_gate=False)

    @classmethod
    def single_resource_reconfiguration(cls) -> "PolicyActionConstraints":
        return cls(force_gate=None, max_worker_moves=1, max_robot_moves=1)

    @classmethod
    def flat_single_edge(cls) -> "PolicyActionConstraints":
        """Baseline restriction: at most one non-stay resource change and one schedule edge."""
        return cls(
            force_gate=None,
            max_worker_moves=1,
            max_robot_moves=1,
            max_schedule_assignments=1,
        )


__all__ = ["PolicyActionConstraints"]
