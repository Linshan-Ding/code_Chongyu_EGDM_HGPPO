"""Shared Phase F representation backbone reused by the Phase G actor.

This module remains independent from matching decoders and PPO update logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from agent.critic import CriticOutput, EventTypeAwareCritic
from agent.encoder import HeteroGraphEncoder
from agent.gate import EventGateHead, GateOutput
from agent.pooling import PoolingOutput, TypeAwareAttentionPooling
from environment.graph_types import HeteroGraph


@dataclass(frozen=True, slots=True)
class RepresentationOutput:
    node_embeddings: Mapping[str, torch.Tensor]
    pooling: PoolingOutput
    gate: GateOutput
    critics: CriticOutput


class EGDMRepresentationNetwork(nn.Module):
    """Shared HGT/pooling/gate/critic representation backbone."""

    def __init__(self, cfg, reference_graph: HeteroGraph) -> None:
        super().__init__()
        reference_graph.validate()
        self.encoder = HeteroGraphEncoder(cfg, reference_graph)
        self.pooling = TypeAwareAttentionPooling(
            node_types=self.encoder.node_types,
            embed_dim=int(cfg.algo.embed_dim),
        )
        self.gate = EventGateHead(
            cfg, gate_feature_dim=int(reference_graph.gate_continuous.shape[1])
        )
        self.critics = EventTypeAwareCritic(
            cfg, event_count=int(reference_graph.event_type_multihot.shape[1])
        )

    def forward(
        self, graph: HeteroGraph, *, validate_graph: bool = True
    ) -> RepresentationOutput:
        node_embeddings = self.encoder(graph, validate_graph=validate_graph)
        pooling = self.pooling(node_embeddings, graph)
        gate = self.gate(
            pooling.global_embedding,
            graph.gate_continuous,
            graph.reconfigure_feasible,
        )
        critics = self.critics(
            pooling.global_embedding,
            graph.event_type_multihot,
        )
        return RepresentationOutput(
            node_embeddings=node_embeddings,
            pooling=pooling,
            gate=gate,
            critics=critics,
        )


__all__ = ["EGDMRepresentationNetwork", "RepresentationOutput"]
