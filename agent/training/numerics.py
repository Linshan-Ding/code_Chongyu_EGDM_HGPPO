"""Numerical guards shared by training and fixed validation.

The TWT reward identity is exact mathematically, but two independently accumulated
IEEE-754 floating-point paths can differ by tiny roundoff on long event traces.
The guard below keeps the identity strict at the scale that matters while avoiding
false failures at O(1e-4) absolute error on large validation episodes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

REWARD_IDENTITY_ABS_TOL = 1e-4
REWARD_IDENTITY_REL_TOL = 1e-9


def stable_sum(values: Iterable[float]) -> float:
    return float(math.fsum(float(v) for v in values))


def reward_identity_error_and_tolerance(base_twt_reward: float, twt: float) -> tuple[float, float]:
    base = float(base_twt_reward)
    target = float(twt)
    error = abs(base + target)
    scale = max(1.0, abs(base), abs(target))
    tolerance = max(REWARD_IDENTITY_ABS_TOL, REWARD_IDENTITY_REL_TOL * scale)
    return float(error), float(tolerance)


def assert_reward_identity(base_twt_reward: float, twt: float, *, context: str) -> float:
    error, tolerance = reward_identity_error_and_tolerance(base_twt_reward, twt)
    if error > tolerance:
        raise RuntimeError(
            f"{context}: error={error:.17g}, tolerance={tolerance:.17g}, "
            f"base_twt_reward={float(base_twt_reward):.17g}, twt={float(twt):.17g}"
        )
    return error


__all__ = [
    "REWARD_IDENTITY_ABS_TOL", "REWARD_IDENTITY_REL_TOL",
    "assert_reward_identity", "reward_identity_error_and_tolerance", "stable_sum",
]
