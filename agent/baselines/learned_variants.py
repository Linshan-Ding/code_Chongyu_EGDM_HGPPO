"""Learned comparison architectures for Phase K2.

All variants keep the same environment, hard masks, event step, reward and PPO
budget.  Only the representation/action-coupling component named in Table 8 is
changed.  This isolates the ablated algorithmic factor instead of giving a
baseline a different simulator.

Operational definitions used by this repository (paper Table 8 does not provide
implementation-level equations for the baselines):

* MLP-PPO: type-specific local MLP node encoders with **no graph message passing**;
  action decoding is restricted to one non-stay worker move, one non-stay robot
  move and one schedule edge per event (flat/single-edge action).
* GAT-PPO: one homogeneous shared-parameter GAT over all relation edges; relation
  labels/edge features are ignored. The EGDM set decoder is retained so the
  experiment isolates relation-aware HGT encoding.
* HGT-PPO-Flat: the paper HGT representation but the same single-edge action cap
  as MLP-PPO, removing full set reconfiguration/parallel set dispatch.
* HGT-MAPPO: shared HGT representation and shared reward, but worker and robot
  actors are decoupled: the robot actor sees the *pre-worker* cell representation
  rather than the worker-updated cell embedding. This is the repository's
  centralized-training/decentralized-resource operationalization of Table 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import torch
from torch import nn

from agent.base import RepresentationOutput
from agent.critic import EventTypeAwareCritic
from agent.encoder import (
    EdgeFeatureSchema,
    NodeFeatureEncoder,
    NodeFeatureSchema,
    _segment_softmax,
)
from agent.gate import EventGateHead
from agent.policy import EGDMCompositePolicy
from agent.pooling import TypeAwareAttentionPooling
from environment.graph_types import HeteroGraph


class _ResidualBudgetBlock(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class IndependentNodeMLPEncoder(nn.Module):
    """Typed node MLPs, no edge/message passing."""

    def __init__(self, cfg, reference_graph: HeteroGraph) -> None:
        super().__init__()
        reference_graph.validate()
        self.embed_dim = int(cfg.algo.embed_dim)
        self.node_types = tuple(reference_graph.nodes.keys())
        cat_dim = int(cfg.algo.representation_network.node_categorical_embed_dim)
        self.schemas = {
            nt: NodeFeatureSchema(
                continuous_names=tuple(store.continuous_names),
                binary_names=tuple(store.binary_names),
                categorical_names=tuple(store.categorical.keys()),
            ) for nt, store in reference_graph.nodes.items()
        }
        self.input_encoders = nn.ModuleDict({
            nt: NodeFeatureEncoder(
                cfg=cfg, node_type=nt, schema=self.schemas[nt],
                embed_dim=self.embed_dim, categorical_embed_dim=cat_dim,
            ) for nt in self.node_types
        })
        pm = cfg.algo.baselines.parameter_match
        hidden = int(pm.hidden_dim)
        blocks = int(pm.residual_blocks_per_type)
        if hidden <= 0 or blocks < 0:
            raise ValueError("baseline parameter-match hidden/blocks must be valid")
        self.post = nn.ModuleDict({
            nt: nn.Sequential(*[_ResidualBudgetBlock(self.embed_dim, hidden) for _ in range(blocks)])
            for nt in self.node_types
        })

    def forward(self, graph: HeteroGraph) -> dict[str, torch.Tensor]:
        graph.validate()
        return {
            nt: self.post[nt](self.input_encoders[nt](graph.nodes[nt]))
            for nt in self.node_types
        }


class HomogeneousGATLayer(nn.Module):
    """Shared-parameter GAT ignoring relation labels and edge attributes."""

    def __init__(self, *, node_types: tuple[str, ...], embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.node_types = node_types
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.ModuleDict({nt: nn.LayerNorm(embed_dim) for nt in node_types})
        self.act = nn.GELU()

    def forward(self, node_embeddings: Mapping[str, torch.Tensor], graph: HeteroGraph):
        out = {}
        for dst_type in self.node_types:
            dst_h = node_embeddings[dst_type]
            if dst_h.shape[0] == 0:
                out[dst_type] = dst_h
                continue
            dst_parts = []
            k_parts = []
            v_parts = []
            for (src_type, _rel, target_type), store in graph.edges.items():
                if target_type != dst_type or store.num_edges == 0:
                    continue
                src_idx, dst_idx = store.edge_index
                src_h = node_embeddings[src_type]
                dst_parts.append(dst_idx)
                k_parts.append(self.k(src_h).view(-1, self.num_heads, self.head_dim).index_select(0, src_idx))
                v_parts.append(self.v(src_h).view(-1, self.num_heads, self.head_dim).index_select(0, src_idx))
            if not dst_parts:
                out[dst_type] = self.norm[dst_type](dst_h)
                continue
            dst_idx = torch.cat(dst_parts, dim=0)
            k = torch.cat(k_parts, dim=0)
            v = torch.cat(v_parts, dim=0)
            q_all = self.q(dst_h).view(-1, self.num_heads, self.head_dim)
            q = q_all.index_select(0, dst_idx)
            logits = (q * k).sum(dim=-1) / sqrt(self.head_dim)
            alpha = _segment_softmax(logits, dst_idx, dst_h.shape[0])
            weighted = alpha.unsqueeze(-1) * v
            msg = torch.zeros(
                (dst_h.shape[0], self.num_heads, self.head_dim),
                dtype=dst_h.dtype, device=dst_h.device,
            )
            msg.index_add_(0, dst_idx, weighted)
            msg = self.o(msg.reshape(dst_h.shape[0], self.embed_dim))
            out[dst_type] = self.norm[dst_type](dst_h + self.act(msg))
        return out


class HomogeneousGATEncoder(nn.Module):
    def __init__(self, cfg, reference_graph: HeteroGraph) -> None:
        super().__init__()
        self.base = IndependentNodeMLPEncoder(cfg, reference_graph)
        self.node_types = self.base.node_types
        self.embed_dim = int(cfg.algo.embed_dim)
        self.layers = nn.ModuleList([
            HomogeneousGATLayer(
                node_types=self.node_types,
                embed_dim=self.embed_dim,
                num_heads=int(cfg.algo.attention_heads),
            ) for _ in range(int(cfg.algo.hgt_layers))
        ])

    @property
    def input_encoders(self):
        return self.base.input_encoders

    def forward(self, graph: HeteroGraph):
        h = self.base(graph)
        for layer in self.layers:
            h = layer(h, graph)
        return h


class _Representation(nn.Module):
    def __init__(self, cfg, reference_graph: HeteroGraph, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = TypeAwareAttentionPooling(
            node_types=tuple(reference_graph.nodes.keys()),
            embed_dim=int(cfg.algo.embed_dim),
        )
        self.gate = EventGateHead(cfg, gate_feature_dim=int(reference_graph.gate_continuous.shape[1]))
        self.critics = EventTypeAwareCritic(
            cfg, event_count=int(reference_graph.event_type_multihot.shape[1])
        )

    def forward(self, graph: HeteroGraph) -> RepresentationOutput:
        node_embeddings = self.encoder(graph)
        pooling = self.pooling(node_embeddings, graph)
        gate = self.gate(pooling.global_embedding, graph.gate_continuous, graph.reconfigure_feasible)
        critics = self.critics(pooling.global_embedding, graph.event_type_multihot)
        return RepresentationOutput(node_embeddings=node_embeddings, pooling=pooling, gate=gate, critics=critics)


class MLPRepresentationNetwork(_Representation):
    def __init__(self, cfg, reference_graph: HeteroGraph) -> None:
        super().__init__(cfg, reference_graph, IndependentNodeMLPEncoder(cfg, reference_graph))


class GATRepresentationNetwork(_Representation):
    def __init__(self, cfg, reference_graph: HeteroGraph) -> None:
        super().__init__(cfg, reference_graph, HomogeneousGATEncoder(cfg, reference_graph))


class MLPCompositePolicy(EGDMCompositePolicy):
    baseline_name = "MLP-PPO"
    def __init__(self, cfg, reference_graph: HeteroGraph) -> None:
        super().__init__(cfg, reference_graph)
        self.representation = MLPRepresentationNetwork(cfg, reference_graph)


class GATCompositePolicy(EGDMCompositePolicy):
    baseline_name = "GAT-PPO"
    def __init__(self, cfg, reference_graph: HeteroGraph) -> None:
        super().__init__(cfg, reference_graph)
        self.representation = GATRepresentationNetwork(cfg, reference_graph)


class HGTFlatCompositePolicy(EGDMCompositePolicy):
    baseline_name = "HGT-PPO-Flat"


class HGTMAPPOCompositePolicy(EGDMCompositePolicy):
    baseline_name = "HGT-MAPPO"

    def _worker_updated_cells(self, *, representation, graph, planner):
        # Decentralized resource actor operationalization: the robot actor does not
        # observe the worker actor's just-selected action in the same event.
        return representation.node_embeddings["cell"].clone()


LEARNED_BASELINE_METHODS = ("MLP-PPO", "GAT-PPO", "HGT-PPO-Flat", "HGT-MAPPO")


def build_learned_baseline_policy(method: str, cfg, reference_graph: HeteroGraph):
    table = {
        "MLP-PPO": MLPCompositePolicy,
        "GAT-PPO": GATCompositePolicy,
        "HGT-PPO-Flat": HGTFlatCompositePolicy,
        "HGT-MAPPO": HGTMAPPOCompositePolicy,
    }
    try:
        return table[str(method)](cfg, reference_graph)
    except KeyError as exc:
        raise KeyError(f"unknown learned baseline: {method}") from exc


__all__ = [
    "GATCompositePolicy", "GATRepresentationNetwork", "HGTFlatCompositePolicy",
    "HGTMAPPOCompositePolicy", "LEARNED_BASELINE_METHODS", "MLPCompositePolicy",
    "MLPRepresentationNetwork", "build_learned_baseline_policy",
]
