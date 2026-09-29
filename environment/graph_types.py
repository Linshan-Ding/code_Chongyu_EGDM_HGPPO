"""Tensor containers for the paper's dynamic heterogeneous graph.

Phase E deliberately does not depend on PyTorch Geometric.  The graph is kept in
small, explicit PyTorch tensor stores so that:
- the environment remains independent from the future agent implementation;
- unit tests can verify node/edge semantics and batching without a GNN;
- Phase F can either consume these stores directly or adapt them to PyG HeteroData.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


EdgeType = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class NodeStore:
    ids: torch.Tensor
    continuous: torch.Tensor
    binary: torch.Tensor
    categorical: Mapping[str, torch.Tensor]
    continuous_names: tuple[str, ...]
    binary_names: tuple[str, ...]
    batch: torch.Tensor
    ptr: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.ids.numel())

    def to(self, device: torch.device | str) -> "NodeStore":
        return NodeStore(
            ids=self.ids.to(device),
            continuous=self.continuous.to(device),
            binary=self.binary.to(device),
            categorical={k: v.to(device) for k, v in self.categorical.items()},
            continuous_names=self.continuous_names,
            binary_names=self.binary_names,
            batch=self.batch.to(device),
            ptr=self.ptr.to(device),
        )


@dataclass(frozen=True, slots=True)
class EdgeStore:
    edge_index: torch.Tensor
    continuous: torch.Tensor
    binary: torch.Tensor
    continuous_names: tuple[str, ...]
    binary_names: tuple[str, ...]
    batch: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: torch.device | str) -> "EdgeStore":
        return EdgeStore(
            edge_index=self.edge_index.to(device),
            continuous=self.continuous.to(device),
            binary=self.binary.to(device),
            continuous_names=self.continuous_names,
            binary_names=self.binary_names,
            batch=self.batch.to(device),
        )


@dataclass(frozen=True, slots=True)
class HeteroGraph:
    """Single or batched heterogeneous graph.

    For a single graph, ``batch_size == 1`` and every node/edge batch id is zero.
    ``gate_continuous`` is always shaped ``[B, F]`` so batching never changes the
    public interface.
    """

    nodes: Mapping[str, NodeStore]
    edges: Mapping[EdgeType, EdgeStore]
    gate_continuous: torch.Tensor
    gate_continuous_names: tuple[str, ...]
    event_type_multihot: torch.Tensor
    reconfigure_feasible: torch.Tensor
    decision_time: torch.Tensor
    decision_index: torch.Tensor
    instance_ids: tuple[str, ...]
    batch_size: int

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gate_continuous.ndim != 2 or self.gate_continuous.shape[0] != self.batch_size:
            raise ValueError("gate_continuous must have shape [B,F]")
        if self.event_type_multihot.shape != (self.batch_size, 3):
            raise ValueError("event_type_multihot must have shape [B,3]")
        if self.reconfigure_feasible.shape != (self.batch_size,):
            raise ValueError("reconfigure_feasible must have shape [B]")
        if self.decision_time.shape != (self.batch_size,):
            raise ValueError("decision_time must have shape [B]")
        if self.decision_index.shape != (self.batch_size,):
            raise ValueError("decision_index must have shape [B]")
        if len(self.instance_ids) != self.batch_size:
            raise ValueError("instance_ids length must equal batch_size")

        for node_type, store in self.nodes.items():
            n = store.num_nodes
            if store.ids.dtype != torch.long or store.ids.ndim != 1:
                raise ValueError(f"{node_type}: ids must be 1D long")
            if store.continuous.ndim != 2 or store.continuous.shape[0] != n:
                raise ValueError(f"{node_type}: invalid continuous feature shape")
            if store.binary.ndim != 2 or store.binary.shape[0] != n:
                raise ValueError(f"{node_type}: invalid binary feature shape")
            if store.continuous.shape[1] != len(store.continuous_names):
                raise ValueError(f"{node_type}: continuous names mismatch")
            if store.binary.shape[1] != len(store.binary_names):
                raise ValueError(f"{node_type}: binary names mismatch")
            if store.batch.shape != (n,) or store.batch.dtype != torch.long:
                raise ValueError(f"{node_type}: invalid batch vector")
            if store.ptr.shape != (self.batch_size + 1,) or store.ptr.dtype != torch.long:
                raise ValueError(f"{node_type}: invalid ptr")
            if int(store.ptr[0]) != 0 or int(store.ptr[-1]) != n:
                raise ValueError(f"{node_type}: ptr endpoints are invalid")
            if n and (int(store.batch.min()) < 0 or int(store.batch.max()) >= self.batch_size):
                raise ValueError(f"{node_type}: batch ids out of range")
            for name, values in store.categorical.items():
                if values.dtype != torch.long or values.shape != (n,):
                    raise ValueError(f"{node_type}.{name}: categorical feature must be [N] long")
                if n and int(values.min()) < 0:
                    raise ValueError(f"{node_type}.{name}: categorical values must be non-negative")

        for edge_type, store in self.edges.items():
            src_type, _, dst_type = edge_type
            if src_type not in self.nodes or dst_type not in self.nodes:
                raise ValueError(f"{edge_type}: unknown node type")
            if store.edge_index.dtype != torch.long or store.edge_index.ndim != 2 or store.edge_index.shape[0] != 2:
                raise ValueError(f"{edge_type}: edge_index must have shape [2,E]")
            e = store.num_edges
            if store.continuous.ndim != 2 or store.continuous.shape[0] != e:
                raise ValueError(f"{edge_type}: invalid continuous edge features")
            if store.binary.ndim != 2 or store.binary.shape[0] != e:
                raise ValueError(f"{edge_type}: invalid binary edge features")
            if store.continuous.shape[1] != len(store.continuous_names):
                raise ValueError(f"{edge_type}: continuous edge names mismatch")
            if store.binary.shape[1] != len(store.binary_names):
                raise ValueError(f"{edge_type}: binary edge names mismatch")
            if store.batch.shape != (e,) or store.batch.dtype != torch.long:
                raise ValueError(f"{edge_type}: invalid edge batch vector")
            if e:
                src, dst = store.edge_index
                if int(src.min()) < 0 or int(src.max()) >= self.nodes[src_type].num_nodes:
                    raise ValueError(f"{edge_type}: source index out of range")
                if int(dst.min()) < 0 or int(dst.max()) >= self.nodes[dst_type].num_nodes:
                    raise ValueError(f"{edge_type}: target index out of range")
                src_batch = self.nodes[src_type].batch[src]
                dst_batch = self.nodes[dst_type].batch[dst]
                if not torch.equal(src_batch, dst_batch) or not torch.equal(src_batch, store.batch):
                    raise ValueError(f"{edge_type}: cross-graph edge detected")

        tensors = [self.gate_continuous, self.event_type_multihot, self.decision_time]
        tensors.extend(s.continuous for s in self.nodes.values())
        tensors.extend(s.binary for s in self.nodes.values())
        tensors.extend(s.continuous for s in self.edges.values())
        tensors.extend(s.binary for s in self.edges.values())
        for tensor in tensors:
            if tensor.numel() and not torch.isfinite(tensor).all():
                raise ValueError("graph contains NaN/Inf")

    def to(
        self,
        device: torch.device | str,
        *,
        validate: bool = True,
    ) -> "HeteroGraph":
        """Move graph tensors to ``device``.

        Normal public calls validate the result.  Replay already receives graphs
        produced by the validated materializer, so its hot path can explicitly
        skip the second full edge/index/finite-value scan.
        """
        graph = HeteroGraph(
            nodes={k: v.to(device) for k, v in self.nodes.items()},
            edges={k: v.to(device) for k, v in self.edges.items()},
            gate_continuous=self.gate_continuous.to(device),
            gate_continuous_names=self.gate_continuous_names,
            event_type_multihot=self.event_type_multihot.to(device),
            reconfigure_feasible=self.reconfigure_feasible.to(device),
            decision_time=self.decision_time.to(device),
            decision_index=self.decision_index.to(device),
            instance_ids=self.instance_ids,
            batch_size=self.batch_size,
        )
        if validate:
            graph.validate()
        return graph


def batch_heterographs(
    graphs: list[HeteroGraph] | tuple[HeteroGraph, ...],
    *,
    validate: bool = True,
) -> HeteroGraph:
    """Batch graphs with explicit node/edge batch vectors and type-wise offsets."""

    graphs = list(graphs)
    if not graphs:
        raise ValueError("at least one graph is required")
    for graph in graphs:
        if validate:
            graph.validate()
        if graph.batch_size != 1:
            raise ValueError("batch_heterographs expects unbatched input graphs")

    node_types = tuple(graphs[0].nodes.keys())
    edge_types = tuple(graphs[0].edges.keys())
    if any(tuple(g.nodes.keys()) != node_types for g in graphs):
        raise ValueError("all graphs must use the same node schema")
    if any(tuple(g.edges.keys()) != edge_types for g in graphs):
        raise ValueError("all graphs must use the same edge schema")

    batched_nodes: dict[str, NodeStore] = {}
    node_offsets_per_graph: list[dict[str, int]] = [dict() for _ in graphs]

    for node_type in node_types:
        template = graphs[0].nodes[node_type]
        ids_parts: list[torch.Tensor] = []
        cont_parts: list[torch.Tensor] = []
        bin_parts: list[torch.Tensor] = []
        cat_parts: dict[str, list[torch.Tensor]] = {k: [] for k in template.categorical}
        batch_parts: list[torch.Tensor] = []
        ptr = [0]
        offset = 0

        for b, graph in enumerate(graphs):
            store = graph.nodes[node_type]
            if store.continuous_names != template.continuous_names or store.binary_names != template.binary_names:
                raise ValueError(f"{node_type}: feature schema mismatch")
            if tuple(store.categorical.keys()) != tuple(template.categorical.keys()):
                raise ValueError(f"{node_type}: categorical schema mismatch")
            node_offsets_per_graph[b][node_type] = offset
            ids_parts.append(store.ids)
            cont_parts.append(store.continuous)
            bin_parts.append(store.binary)
            for name in cat_parts:
                cat_parts[name].append(store.categorical[name])
            batch_parts.append(torch.full((store.num_nodes,), b, dtype=torch.long, device=store.ids.device))
            offset += store.num_nodes
            ptr.append(offset)

        batched_nodes[node_type] = NodeStore(
            ids=torch.cat(ids_parts, dim=0),
            continuous=torch.cat(cont_parts, dim=0),
            binary=torch.cat(bin_parts, dim=0),
            categorical={k: torch.cat(v, dim=0) for k, v in cat_parts.items()},
            continuous_names=template.continuous_names,
            binary_names=template.binary_names,
            batch=torch.cat(batch_parts, dim=0),
            ptr=torch.tensor(ptr, dtype=torch.long, device=template.ids.device),
        )

    batched_edges: dict[EdgeType, EdgeStore] = {}
    for edge_type in edge_types:
        src_type, _, dst_type = edge_type
        template = graphs[0].edges[edge_type]
        idx_parts: list[torch.Tensor] = []
        cont_parts: list[torch.Tensor] = []
        bin_parts: list[torch.Tensor] = []
        batch_parts: list[torch.Tensor] = []
        for b, graph in enumerate(graphs):
            store = graph.edges[edge_type]
            if store.continuous_names != template.continuous_names or store.binary_names != template.binary_names:
                raise ValueError(f"{edge_type}: feature schema mismatch")
            if store.num_edges:
                offset = torch.tensor(
                    [[node_offsets_per_graph[b][src_type]], [node_offsets_per_graph[b][dst_type]]],
                    dtype=torch.long,
                    device=store.edge_index.device,
                )
                idx_parts.append(store.edge_index + offset)
            else:
                idx_parts.append(store.edge_index)
            cont_parts.append(store.continuous)
            bin_parts.append(store.binary)
            batch_parts.append(torch.full((store.num_edges,), b, dtype=torch.long, device=store.edge_index.device))

        batched_edges[edge_type] = EdgeStore(
            edge_index=torch.cat(idx_parts, dim=1),
            continuous=torch.cat(cont_parts, dim=0),
            binary=torch.cat(bin_parts, dim=0),
            continuous_names=template.continuous_names,
            binary_names=template.binary_names,
            batch=torch.cat(batch_parts, dim=0),
        )

    out = HeteroGraph(
        nodes=batched_nodes,
        edges=batched_edges,
        gate_continuous=torch.cat([g.gate_continuous for g in graphs], dim=0),
        gate_continuous_names=graphs[0].gate_continuous_names,
        event_type_multihot=torch.cat([g.event_type_multihot for g in graphs], dim=0),
        reconfigure_feasible=torch.cat([g.reconfigure_feasible for g in graphs], dim=0),
        decision_time=torch.cat([g.decision_time for g in graphs], dim=0),
        decision_index=torch.cat([g.decision_index for g in graphs], dim=0),
        instance_ids=tuple(g.instance_ids[0] for g in graphs),
        batch_size=len(graphs),
    )
    if validate:
        out.validate()
    return out


__all__ = [
    "EdgeStore",
    "EdgeType",
    "HeteroGraph",
    "NodeStore",
    "batch_heterographs",
]
