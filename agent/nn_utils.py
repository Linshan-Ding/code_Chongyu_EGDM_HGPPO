"""Small neural-network construction utilities shared by Phase F heads."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn


def build_gelu_layernorm_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = int(input_dim)
    for hidden in hidden_dims:
        hidden = int(hidden)
        layers.extend([nn.Linear(width, hidden), nn.GELU(), nn.LayerNorm(hidden)])
        width = hidden
    layers.append(nn.Linear(width, int(output_dim)))
    return nn.Sequential(*layers)


__all__ = ["build_gelu_layernorm_mlp"]
