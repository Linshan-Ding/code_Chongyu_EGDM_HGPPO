"""Typed node features and segmented attention used by current graph encoders."""
from dataclasses import dataclass
import torch
from torch import nn
from environment.entities import ExecutionMode, OperationStatus


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
