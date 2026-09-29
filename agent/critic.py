"""Event-type-aware three-head value network from paper Section 6.7."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from agent.nn_utils import build_gelu_layernorm_mlp


class EventTypeEncoder(nn.Module):
    """Learnable 32-d embedding for the three possibly simultaneous event types."""

    def __init__(self, event_count: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(event_count), int(embedding_dim))

    def forward(self, event_type_multihot: torch.Tensor) -> torch.Tensor:
        if event_type_multihot.ndim != 2:
            raise ValueError("event_type_multihot must have shape [B,E]")
        if event_type_multihot.shape[1] != self.embedding.num_embeddings:
            raise ValueError("event-type width mismatch")
        weights = event_type_multihot.to(dtype=self.embedding.weight.dtype)
        summed = weights @ self.embedding.weight
        count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return summed / count


@dataclass(frozen=True, slots=True)
class CriticOutput:
    v_gate: torch.Tensor
    v_rec: torch.Tensor
    v_sch: torch.Tensor
    event_embedding: torch.Tensor


class EventTypeAwareCritic(nn.Module):
    """Three independent value heads sharing the graph/event representation."""

    def __init__(self, cfg, *, event_count: int = 3) -> None:
        super().__init__()
        event_dim = int(cfg.algo.representation_network.event_type_embed_dim)
        embed_dim = int(cfg.algo.embed_dim)
        hidden = [int(x) for x in cfg.algo.policy_value_mlp]
        self.event_encoder = EventTypeEncoder(event_count, event_dim)
        input_dim = embed_dim + event_dim
        self.v_gate = build_gelu_layernorm_mlp(input_dim, hidden, 1)
        self.v_rec = build_gelu_layernorm_mlp(input_dim, hidden, 1)
        self.v_sch = build_gelu_layernorm_mlp(input_dim, hidden, 1)

    def forward(
        self,
        global_embedding: torch.Tensor,
        event_type_multihot: torch.Tensor,
    ) -> CriticOutput:
        event_embedding = self.event_encoder(event_type_multihot)
        context = torch.cat([global_embedding, event_embedding], dim=-1)
        return CriticOutput(
            v_gate=self.v_gate(context).squeeze(-1),
            v_rec=self.v_rec(context).squeeze(-1),
            v_sch=self.v_sch(context).squeeze(-1),
            event_embedding=event_embedding,
        )


__all__ = ["CriticOutput", "EventTypeAwareCritic", "EventTypeEncoder"]
