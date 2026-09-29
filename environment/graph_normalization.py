"""Training-distribution z-score normalization for graph continuous features.

The paper asks continuous graph features to be normalized using training-distribution
statistics.  Phase E keeps graph construction raw and deterministic, then provides
this separate fitter/transformer so statistics can later be collected only from
training rollouts (never validation/test data).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from environment.graph_types import EdgeStore, EdgeType, HeteroGraph, NodeStore


@dataclass(frozen=True, slots=True)
class MomentStats:
    mean: torch.Tensor
    std: torch.Tensor
    count: int


@dataclass(frozen=True, slots=True)
class GraphNormalizationStats:
    node: dict[str, MomentStats]
    edge: dict[EdgeType, MomentStats]
    gate: MomentStats


class _Accumulator:
    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.count = 0
        self.sum = torch.zeros((width,), dtype=torch.float64)
        self.sumsq = torch.zeros((width,), dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.width:
            raise ValueError("feature width mismatch while fitting normalizer")
        if values.shape[0] == 0 or self.width == 0:
            return
        x = values.detach().to(dtype=torch.float64, device="cpu")
        self.count += int(x.shape[0])
        self.sum += x.sum(dim=0)
        self.sumsq += (x * x).sum(dim=0)

    def finalize(self, min_std: float) -> MomentStats:
        if self.width == 0:
            empty = torch.empty((0,), dtype=torch.float32)
            return MomentStats(empty, empty, self.count)
        if self.count == 0:
            raise ValueError("cannot fit normalization for a feature store with zero observations")
        mean = self.sum / self.count
        var = (self.sumsq / self.count) - mean * mean
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)
        std = torch.clamp(std, min=float(min_std))
        return MomentStats(mean.float(), std.float(), self.count)


class GraphContinuousNormalizer:
    def __init__(self, *, min_std: float = 1e-6) -> None:
        if min_std <= 0:
            raise ValueError("min_std must be positive")
        self.min_std = float(min_std)
        self.stats: GraphNormalizationStats | None = None

    def fit(self, graphs: Iterable[HeteroGraph]) -> GraphNormalizationStats:
        # Phase L1.2 intentionally consumes the iterable online.  The previous
        # implementation called ``list(graphs)`` and therefore retained every
        # L-scale heterogeneous graph (including dense same-stage competition
        # edges) until fitting finished.  Streaming accumulation is mathematically
        # identical because the sufficient statistics are count/sum/sumsq.
        iterator = iter(graphs)
        try:
            template = next(iterator)
        except StopIteration as exc:
            raise ValueError("at least one graph is required to fit normalization") from exc
        template.validate()
        node_acc = {
            node_type: _Accumulator(store.continuous.shape[1])
            for node_type, store in template.nodes.items()
        }
        edge_acc = {
            edge_type: _Accumulator(store.continuous.shape[1])
            for edge_type, store in template.edges.items()
        }
        gate_acc = _Accumulator(template.gate_continuous.shape[1])

        def update(graph: HeteroGraph) -> None:
            graph.validate()
            if tuple(graph.nodes) != tuple(template.nodes) or tuple(graph.edges) != tuple(template.edges):
                raise ValueError("graph schema changed during normalizer fitting")
            for node_type, store in graph.nodes.items():
                node_acc[node_type].update(store.continuous)
            for edge_type, store in graph.edges.items():
                if store.continuous.shape[1] > 0:
                    edge_acc[edge_type].update(store.continuous)
            gate_acc.update(graph.gate_continuous)

        update(template)
        for graph in iterator:
            update(graph)

        node_stats: dict[str, MomentStats] = {}
        for node_type, acc in node_acc.items():
            # Node types always exist in this paper's graph, including empty operation
            # only before any arrival; reset advances to an arrival before decisions.
            node_stats[node_type] = acc.finalize(self.min_std)

        edge_stats: dict[EdgeType, MomentStats] = {}
        for edge_type, acc in edge_acc.items():
            if acc.width == 0:
                edge_stats[edge_type] = acc.finalize(self.min_std)
            elif acc.count == 0:
                # Some relation may be absent in a small training sample.  Keep an
                # identity transform until the training-statistics collection sees it.
                edge_stats[edge_type] = MomentStats(
                    mean=torch.zeros((acc.width,), dtype=torch.float32),
                    std=torch.ones((acc.width,), dtype=torch.float32),
                    count=0,
                )
            else:
                edge_stats[edge_type] = acc.finalize(self.min_std)

        stats = GraphNormalizationStats(
            node=node_stats,
            edge=edge_stats,
            gate=gate_acc.finalize(self.min_std),
        )
        self.stats = stats
        return stats

    @staticmethod
    def _zscore(values: torch.Tensor, stats: MomentStats) -> torch.Tensor:
        if values.shape[1] == 0:
            return values.clone()
        mean = stats.mean.to(values.device)
        std = stats.std.to(values.device)
        return (values - mean) / std

    def _transform_impl(self, graph: HeteroGraph, *, validate: bool) -> HeteroGraph:
        if self.stats is None:
            raise RuntimeError("fit() must be called before transform()")
        if validate:
            graph.validate()
        stats = self.stats

        nodes = {
            node_type: NodeStore(
                ids=store.ids,
                continuous=self._zscore(store.continuous, stats.node[node_type]),
                binary=store.binary,
                categorical=store.categorical,
                continuous_names=store.continuous_names,
                binary_names=store.binary_names,
                batch=store.batch,
                ptr=store.ptr,
            )
            for node_type, store in graph.nodes.items()
        }
        edges = {
            edge_type: EdgeStore(
                edge_index=store.edge_index,
                continuous=self._zscore(store.continuous, stats.edge[edge_type]),
                binary=store.binary,
                continuous_names=store.continuous_names,
                binary_names=store.binary_names,
                batch=store.batch,
            )
            for edge_type, store in graph.edges.items()
        }
        out = HeteroGraph(
            nodes=nodes,
            edges=edges,
            gate_continuous=self._zscore(graph.gate_continuous, stats.gate),
            gate_continuous_names=graph.gate_continuous_names,
            event_type_multihot=graph.event_type_multihot,
            reconfigure_feasible=graph.reconfigure_feasible,
            decision_time=graph.decision_time,
            decision_index=graph.decision_index,
            instance_ids=graph.instance_ids,
            batch_size=graph.batch_size,
        )
        if validate:
            out.validate()
        return out

    def transform(self, graph: HeteroGraph) -> HeteroGraph:
        return self._transform_impl(graph, validate=True)

    def transform_replay_trusted(self, graph: HeteroGraph) -> HeteroGraph:
        """Exact z-score transform without duplicate validation in PPO replay.

        This is an opt-in execution fast path for graphs produced immediately by
        ``DynamicHeteroGraphBuilder.build``.  That builder already validates the raw
        graph, while ``batch_heterographs`` validates every normalized graph again
        before HGT batching.  Skipping only the two redundant checks here changes no
        tensor values, feature schema, masks, or learning objective.
        """
        return self._transform_impl(graph, validate=False)


__all__ = [
    "GraphContinuousNormalizer",
    "GraphNormalizationStats",
    "MomentStats",
]
