"""Phase F relation-aware heterogeneous graph encoder.

This module implements the paper's shared HGT/GAT-style representation backbone
without depending on PyTorch Geometric.  It consumes the explicit Phase E
``HeteroGraph`` tensor contract and follows Eqs. (18)-(20): relation-specific
Q/K/V/O projections, edge-feature-aware attention, residual update and LayerNorm.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import torch
from torch import nn

from environment.entities import ExecutionMode, OperationStatus
from environment.graph_types import EdgeType, HeteroGraph


def _relation_key(edge_type: EdgeType) -> str:
    return "__".join(edge_type)


def _segment_softmax(
    logits: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Vectorized softmax over rows sharing the same segment id.

    ``logits`` is ``[E,H]`` and ``index`` is the destination-node id of each
    edge.  The normalization is therefore performed over incoming neighbors of
    one relation type exactly as required by Eq. (18).
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [E,H]")
    if index.ndim != 1 or index.shape[0] != logits.shape[0]:
        raise ValueError("index must have shape [E]")
    if logits.shape[0] == 0:
        return logits

    # Keep the segmented reduction in a single dtype.  Under CUDA BF16
    # autocast, ``exp`` can be promoted to FP32 while the input logits remain
    # BF16; mixed dtypes are rejected by ``scatter_add_``.  FP32 also gives a
    # more stable normalization for attention weights.
    reduce_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    work_logits = logits.to(dtype=reduce_dtype)
    heads = logits.shape[1]
    expanded = index[:, None].expand(-1, heads)
    max_per_segment = torch.full(
        (num_segments, heads),
        -torch.inf,
        dtype=reduce_dtype,
        device=logits.device,
    )
    max_per_segment.scatter_reduce_(
        0, expanded, work_logits, reduce="amax", include_self=True
    )
    stabilized = work_logits - max_per_segment.index_select(0, index)
    exp_logits = stabilized.exp()
    denominator = torch.zeros(
        (num_segments, heads), dtype=exp_logits.dtype, device=logits.device
    )
    denominator.scatter_add_(0, expanded, exp_logits)
    return exp_logits / denominator.index_select(0, index).clamp_min(eps)


def _max_scale_value(cfg, field: str) -> int:
    scales = cfg.instance.scales.to_dict()
    values = []
    for spec in scales.values():
        value = spec[field]
        values.append(int(value[-1] if isinstance(value, list) else value))
    return max(values)


def categorical_cardinality(cfg, node_type: str, feature_name: str) -> int:
    """Return paper-wide cardinality so an S-created model also supports XL.

    Cardinalities are derived from the configured largest paper scale rather
    than inferred from one small reference graph.  Zero is reserved as the
    explicit "none/unconfigured" token where Phase E uses one-based IDs.
    """

    max_stages = int(cfg.env.graph.max_stages)
    max_cells = _max_scale_value(cfg, "total_cells")
    max_workers = _max_scale_value(cfg, "workers")
    max_robots = _max_scale_value(cfg, "robots")
    max_products = _max_scale_value(cfg, "product_types")

    table = {
        ("operation", "product_type"): max_products,
        ("operation", "stage"): max_stages,
        ("operation", "status"): len(list(OperationStatus)),
        ("stage", "stage"): max_stages,
        ("cell", "stage"): max_stages,
        ("cell", "worker_slot"): max_workers + 1,
        ("cell", "robot_slot"): max_robots + 1,
        ("cell", "mode"): len(list(ExecutionMode)),
        ("worker", "configured_cell"): max_cells + 1,
        ("worker", "physical_cell"): max_cells + 1,
        ("worker", "configured_stage"): max_stages + 1,
        ("worker", "physical_stage"): max_stages + 1,
        ("robot", "configured_cell"): max_cells + 1,
        ("robot", "physical_cell"): max_cells + 1,
        ("robot", "configured_stage"): max_stages + 1,
        ("robot", "physical_stage"): max_stages + 1,
    }
    try:
        return int(table[(node_type, feature_name)])
    except KeyError as exc:
        raise KeyError(
            f"No configured categorical cardinality for {node_type}.{feature_name}"
        ) from exc


@dataclass(frozen=True, slots=True)
class NodeFeatureSchema:
    continuous_names: tuple[str, ...]
    binary_names: tuple[str, ...]
    categorical_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EdgeFeatureSchema:
    continuous_names: tuple[str, ...]
    binary_names: tuple[str, ...]


class NodeFeatureEncoder(nn.Module):
    """Project raw/normalized typed features into the common graph dimension."""

    def __init__(
        self,
        *,
        cfg,
        node_type: str,
        schema: NodeFeatureSchema,
        embed_dim: int,
        categorical_embed_dim: int,
    ) -> None:
        super().__init__()
        self.node_type = node_type
        self.schema = schema
        self.categorical_names = schema.categorical_names
        self.embeddings = nn.ModuleDict()
        for name in self.categorical_names:
            self.embeddings[name] = nn.Embedding(
                categorical_cardinality(cfg, node_type, name),
                categorical_embed_dim,
            )

        input_dim = (
            len(schema.continuous_names)
            + len(schema.binary_names)
            + len(schema.categorical_names) * categorical_embed_dim
        )
        if input_dim <= 0:
            raise ValueError(f"{node_type}: empty node feature vector")
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, store) -> torch.Tensor:
        if tuple(store.continuous_names) != self.schema.continuous_names:
            raise ValueError(f"{self.node_type}: continuous feature schema changed")
        if tuple(store.binary_names) != self.schema.binary_names:
            raise ValueError(f"{self.node_type}: binary feature schema changed")
        if tuple(store.categorical.keys()) != self.schema.categorical_names:
            raise ValueError(f"{self.node_type}: categorical feature schema changed")

        parts = [store.continuous, store.binary]
        for name in self.categorical_names:
            values = store.categorical[name]
            embedding = self.embeddings[name]
            if values.numel() and int(values.max()) >= embedding.num_embeddings:
                raise ValueError(
                    f"{self.node_type}.{name} value exceeds configured paper-scale cardinality"
                )
            parts.append(embedding(values))
        return self.projection(torch.cat(parts, dim=-1))


class RelationAwareHGTLayer(nn.Module):
    """One relation-aware multi-head message-passing layer (Eqs. 18-20)."""

    def __init__(
        self,
        *,
        node_types: tuple[str, ...],
        edge_schemas: Mapping[EdgeType, EdgeFeatureSchema],
        embed_dim: int,
        num_heads: int,
        relation_embed_dim: int,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.node_types = tuple(node_types)
        self.edge_types = tuple(edge_schemas.keys())
        self.edge_schemas = dict(edge_schemas)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads

        self.relation_embedding = nn.Embedding(len(self.edge_types), relation_embed_dim)
        self.q = nn.ModuleDict()
        self.k = nn.ModuleDict()
        self.v = nn.ModuleDict()
        self.o = nn.ModuleDict()
        self.edge_projection = nn.ModuleDict()
        self.relation_index: dict[EdgeType, int] = {}

        for idx, edge_type in enumerate(self.edge_types):
            key = _relation_key(edge_type)
            self.relation_index[edge_type] = idx
            self.q[key] = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k[key] = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v[key] = nn.Linear(embed_dim, embed_dim, bias=False)
            self.o[key] = nn.Linear(embed_dim, embed_dim, bias=False)
            schema = self.edge_schemas[edge_type]
            edge_width = len(schema.continuous_names) + len(schema.binary_names)
            self.edge_projection[key] = nn.Linear(
                edge_width + relation_embed_dim, embed_dim, bias=False
            )

        self.norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(embed_dim) for node_type in self.node_types}
        )
        self.activation = nn.GELU()

    def _validate_edge_schema(self, edge_type: EdgeType, store) -> None:
        schema = self.edge_schemas[edge_type]
        if tuple(store.continuous_names) != schema.continuous_names:
            raise ValueError(f"{edge_type}: continuous edge schema changed")
        if tuple(store.binary_names) != schema.binary_names:
            raise ValueError(f"{edge_type}: binary edge schema changed")

    def forward(
        self,
        node_embeddings: Mapping[str, torch.Tensor],
        graph: HeteroGraph,
        *,
        disabled_node_types: frozenset[str] = frozenset(),
    ) -> dict[str, torch.Tensor]:
        unknown = disabled_node_types.difference(self.node_types)
        if unknown:
            raise ValueError(f"unknown disabled node types: {sorted(unknown)}")
        messages = {
            node_type: torch.zeros_like(node_embeddings[node_type])
            for node_type in self.node_types
        }

        for edge_type in self.edge_types:
            src_type, _, dst_type = edge_type
            # A strict node-type ablation must remove both incoming and
            # outgoing information paths.  Merely zeroing the final embedding
            # would still let this node type influence its neighbours inside
            # earlier message-passing layers.
            if src_type in disabled_node_types or dst_type in disabled_node_types:
                continue
            store = graph.edges[edge_type]
            self._validate_edge_schema(edge_type, store)
            if store.num_edges == 0:
                continue

            key = _relation_key(edge_type)
            src_index, dst_index = store.edge_index
            src_h = node_embeddings[src_type]
            dst_h = node_embeddings[dst_type]
            num_dst = dst_h.shape[0]

            q = self.q[key](dst_h).view(-1, self.num_heads, self.head_dim)
            k = self.k[key](src_h).view(-1, self.num_heads, self.head_dim)
            v = self.v[key](src_h).view(-1, self.num_heads, self.head_dim)

            edge_features = torch.cat([store.continuous, store.binary], dim=-1)
            relation_id = torch.full(
                (store.num_edges,),
                self.relation_index[edge_type],
                dtype=torch.long,
                device=src_h.device,
            )
            relation_features = self.relation_embedding(relation_id)
            edge_context = self.edge_projection[key](
                torch.cat([edge_features, relation_features], dim=-1)
            ).view(-1, self.num_heads, self.head_dim)

            q_edge = q.index_select(0, dst_index)
            k_edge = k.index_select(0, src_index) + edge_context
            logits = (q_edge * k_edge).sum(dim=-1) / sqrt(self.head_dim)
            attention = _segment_softmax(logits, dst_index, num_dst)

            weighted_values = attention.unsqueeze(-1) * v.index_select(0, src_index)
            relation_message = torch.zeros(
                (num_dst, self.num_heads, self.head_dim),
                dtype=weighted_values.dtype,
                device=weighted_values.device,
            )
            relation_message.index_add_(0, dst_index, weighted_values)
            relation_message = relation_message.reshape(num_dst, self.embed_dim)
            messages[dst_type] = messages[dst_type] + self.o[key](relation_message)

        output = {}
        for node_type in self.node_types:
            if node_type in disabled_node_types:
                # Keep the public heterogeneous schema stable for the shared
                # decoders while making this type a constant zero tensor with
                # no trainable information channel.
                output[node_type] = torch.zeros_like(node_embeddings[node_type])
            else:
                output[node_type] = self.norm[node_type](
                    node_embeddings[node_type] + self.activation(messages[node_type])
                )
        return output


class HeteroGraphEncoder(nn.Module):
    """Shared node projection + stacked relation-aware HGT layers."""

    def __init__(self, cfg, reference_graph: HeteroGraph) -> None:
        super().__init__()
        reference_graph.validate()
        self.embed_dim = int(cfg.algo.embed_dim)
        self.node_types = tuple(reference_graph.nodes.keys())
        self.edge_types = tuple(reference_graph.edges.keys())
        # Controlled ablations may disable a complete node type without
        # changing graph shapes/checkpoint plumbing.  The formal model leaves
        # this empty.
        self.disabled_node_types: frozenset[str] = frozenset()
        categorical_embed_dim = int(cfg.algo.representation_network.node_categorical_embed_dim)

        self.node_schemas = {
            node_type: NodeFeatureSchema(
                continuous_names=tuple(store.continuous_names),
                binary_names=tuple(store.binary_names),
                categorical_names=tuple(store.categorical.keys()),
            )
            for node_type, store in reference_graph.nodes.items()
        }
        self.edge_schemas = {
            edge_type: EdgeFeatureSchema(
                continuous_names=tuple(store.continuous_names),
                binary_names=tuple(store.binary_names),
            )
            for edge_type, store in reference_graph.edges.items()
        }

        self.input_encoders = nn.ModuleDict(
            {
                node_type: NodeFeatureEncoder(
                    cfg=cfg,
                    node_type=node_type,
                    schema=self.node_schemas[node_type],
                    embed_dim=self.embed_dim,
                    categorical_embed_dim=categorical_embed_dim,
                )
                for node_type in self.node_types
            }
        )
        self.layers = nn.ModuleList(
            [
                RelationAwareHGTLayer(
                    node_types=self.node_types,
                    edge_schemas=self.edge_schemas,
                    embed_dim=self.embed_dim,
                    num_heads=int(cfg.algo.attention_heads),
                    relation_embed_dim=int(cfg.algo.relation_type_embed_dim),
                )
                for _ in range(int(cfg.algo.hgt_layers))
            ]
        )

    def _validate_graph_schema(self, graph: HeteroGraph, *, validate_graph: bool = True) -> None:
        if validate_graph:
            graph.validate()
        if tuple(graph.nodes.keys()) != self.node_types:
            raise ValueError("node-type schema changed")
        if tuple(graph.edges.keys()) != self.edge_types:
            raise ValueError("edge-type schema changed")

    def forward(
        self, graph: HeteroGraph, *, validate_graph: bool = True
    ) -> dict[str, torch.Tensor]:
        self._validate_graph_schema(graph, validate_graph=validate_graph)
        disabled = frozenset(self.disabled_node_types)
        unknown = disabled.difference(self.node_types)
        if unknown:
            raise ValueError(f"unknown disabled node types: {sorted(unknown)}")
        h = {}
        for node_type in self.node_types:
            store = graph.nodes[node_type]
            if node_type in disabled:
                h[node_type] = store.continuous.new_zeros(
                    (store.num_nodes, self.embed_dim)
                )
            else:
                h[node_type] = self.input_encoders[node_type](store)
        for layer in self.layers:
            h = layer(h, graph, disabled_node_types=disabled)
        return h


__all__ = [
    "EdgeFeatureSchema",
    "HeteroGraphEncoder",
    "NodeFeatureEncoder",
    "NodeFeatureSchema",
    "RelationAwareHGTLayer",
    "categorical_cardinality",
]
