"""Learning-based event gate head from paper Section 6.3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from agent.nn_utils import build_gelu_layernorm_mlp


@dataclass(frozen=True, slots=True)
class GateOutput:
    raw_reconfigure_logit: torch.Tensor
    masked_logits: torch.Tensor
    probabilities: torch.Tensor


class EventGateHead(nn.Module):
    """Score KEEP vs RECONFIGURE using [h_G, five explicit imbalance features]."""

    def __init__(self, cfg, *, gate_feature_dim: int) -> None:
        super().__init__()
        self.embed_dim = int(cfg.algo.embed_dim)
        hidden = [int(x) for x in cfg.algo.policy_value_mlp]
        self.mlp = build_gelu_layernorm_mlp(
            self.embed_dim + int(gate_feature_dim), hidden, 1
        )

    def forward(
        self,
        global_embedding: torch.Tensor,
        gate_continuous: torch.Tensor,
        reconfigure_feasible: torch.Tensor,
    ) -> GateOutput:
        if global_embedding.ndim != 2 or gate_continuous.ndim != 2:
            raise ValueError("gate inputs must be batched matrices")
        if global_embedding.shape[0] != gate_continuous.shape[0]:
            raise ValueError("gate batch size mismatch")
        if reconfigure_feasible.shape != (global_embedding.shape[0],):
            raise ValueError("reconfigure_feasible must have shape [B]")

        raw = self.mlp(torch.cat([global_embedding, gate_continuous], dim=-1)).squeeze(-1)
        masked_reconfigure = torch.where(
            reconfigure_feasible,
            raw,
            torch.full_like(raw, -1.0e9),
        )
        # sigmoid(raw) is exactly the reconfiguration probability when feasible;
        # [KEEP logit=0, RECONFIGURE logit=raw] gives the same probability and is
        # convenient for the later categorical policy interface.
        logits = torch.stack([torch.zeros_like(masked_reconfigure), masked_reconfigure], dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        return GateOutput(
            raw_reconfigure_logit=raw,
            masked_logits=logits,
            probabilities=probabilities,
        )


__all__ = ["EventGateHead", "GateOutput"]
