"""Type-aware attention pooling for Phase F graph representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from environment.graph_types import HeteroGraph


def _batch_softmax(scores: torch.Tensor, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
    if scores.ndim != 1 or batch.shape != scores.shape:
        raise ValueError("scores/batch must both have shape [N]")
    if scores.numel() == 0:
        return scores
    # CUDA autocast may promote ``exp`` to FP32 even when ``scores`` are BF16.
    # Keep the whole reduction in one stable dtype so scatter operations never
    # receive mixed source/target dtypes.  The caller casts the resulting
    # weights back to the node embedding dtype before index_add_ pooling.
    reduce_dtype = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
    work_scores = scores.to(dtype=reduce_dtype)
    maxima = torch.full(
        (batch_size,), -torch.inf, dtype=reduce_dtype, device=scores.device
    )
    maxima.scatter_reduce_(0, batch, work_scores, reduce="amax", include_self=True)
    exp_scores = (work_scores - maxima.index_select(0, batch)).exp()
    denom = torch.zeros((batch_size,), dtype=exp_scores.dtype, device=scores.device)
    denom.scatter_add_(0, batch, exp_scores)
    return exp_scores / denom.index_select(0, batch).clamp_min(1e-12)


@dataclass(frozen=True, slots=True)
class PoolingOutput:
    global_embedding: torch.Tensor
    typed_embeddings: Mapping[str, torch.Tensor]
    operation_embedding: torch.Tensor
    stage_embedding: torch.Tensor
    cell_embedding: torch.Tensor
    resource_embedding: torch.Tensor


class TypeAwareAttentionPooling(nn.Module):
    """Attention-pool each node type, then fuse the five typed summaries."""

    def __init__(self, *, node_types: tuple[str, ...], embed_dim: int) -> None:
        super().__init__()
        self.node_types = tuple(node_types)
        self.embed_dim = int(embed_dim)
        self.type_scorers = nn.ModuleDict(
            {node_type: nn.Linear(embed_dim, 1, bias=False) for node_type in self.node_types}
        )
        self.global_projection = nn.Sequential(
            nn.Linear(len(self.node_types) * embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.resource_projection = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def _pool_one_type(
        self,
        node_type: str,
        h: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if h.ndim != 2 or h.shape[1] != self.embed_dim:
            raise ValueError(f"{node_type}: expected [N,{self.embed_dim}] embedding")
        if h.shape[0] != batch.shape[0]:
            raise ValueError(f"{node_type}: batch vector length mismatch")
        if h.shape[0] == 0:
            return torch.zeros((batch_size, self.embed_dim), dtype=h.dtype, device=h.device)
        scores = self.type_scorers[node_type](h).squeeze(-1)
        weights = _batch_softmax(scores, batch, batch_size)
        # ``weights`` is intentionally computed in FP32 for numerical
        # stability under AMP; match ``h`` for the in-place index_add_.
        weights = weights.to(dtype=h.dtype)
        pooled = torch.zeros(
            (batch_size, self.embed_dim), dtype=h.dtype, device=h.device
        )
        pooled.index_add_(0, batch, weights.unsqueeze(-1) * h)
        return pooled

    def forward(
        self,
        node_embeddings: Mapping[str, torch.Tensor],
        graph: HeteroGraph,
    ) -> PoolingOutput:
        typed = {
            node_type: self._pool_one_type(
                node_type,
                node_embeddings[node_type],
                graph.nodes[node_type].batch,
                graph.batch_size,
            )
            for node_type in self.node_types
        }
        global_embedding = self.global_projection(
            torch.cat([typed[node_type] for node_type in self.node_types], dim=-1)
        )
        resource_embedding = self.resource_projection(
            torch.cat([typed["worker"], typed["robot"]], dim=-1)
        )
        return PoolingOutput(
            global_embedding=global_embedding,
            typed_embeddings=typed,
            operation_embedding=typed["operation"],
            stage_embedding=typed["stage"],
            cell_embedding=typed["cell"],
            resource_embedding=resource_embedding,
        )


__all__ = ["PoolingOutput", "TypeAwareAttentionPooling"]
