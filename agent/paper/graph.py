"""Batched edge-aware relation attention and a genuinely homogeneous GAT control."""
from dataclasses import replace
import torch
from torch import nn
from .graph_features import NodeFeatureEncoder, NodeFeatureSchema, _segment_softmax
from environment.graph_types import batch_heterographs, EdgeStore


def squash(x):
    return x.sign() * x.abs().log1p()


def with_reservations(graph, state=None):
    """Explicit transit-to-reserved-cell relations, using only current state."""
    if ("worker", "reserved_at", "cell") in graph.edges:
        return graph
    edges = dict(graph.edges)
    for kind in ("worker", "robot"):
        store = graph.nodes[kind]
        # Transit resources have configured_cell=None. The physical cell
        # reservation is the authoritative current destination, not a guess from
        # configured_at edges. A schema-only reference has empty reservation edges.
        pairs = [] if state is None else [(getattr(c, "reserved_" + kind), c.cell_id)
            for c in state.cells if getattr(c, "reserved_" + kind) is not None]
        indices = torch.tensor(pairs, dtype=torch.long).T.contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
        values = torch.ones((indices.shape[1], 1))
        batch = store.batch[indices[0]]
        remaining = store.continuous[indices[0], :1]
        names = ("remaining_migration_time_minutes",)
        edges[(kind, "reserved_at", "cell")] = EdgeStore(indices, remaining, values, names, ("migration_reservation",), batch)
        edges[("cell", "reserved_for_" + kind, kind)] = EdgeStore(indices.flip(0), remaining, values, names, ("migration_reservation",), batch)
    return replace(graph, edges=edges)


class GraphEncoder(nn.Module):
    def __init__(self, cfg, reference, options, method):
        super().__init__()
        reference = with_reservations(reference)
        self.method, self.dim = method, options["hidden"]
        self.types, self.relations = tuple(reference.nodes), tuple(reference.edges)
        d = self.dim
        self.edge_names = tuple(sorted({"c:" + n for e in reference.edges.values() for n in e.continuous_names}
                                      | {"b:" + n for e in reference.edges.values() for n in e.binary_names}))
        self.node_names = tuple(sorted({"c:" + n for s in reference.nodes.values() for n in s.continuous_names}
                                      | {"b:" + n for s in reference.nodes.values() for n in s.binary_names}
                                      | {"k:" + n for s in reference.nodes.values() for n in s.categorical}))
        if method == "gat":
            self.input = nn.Sequential(nn.Linear(len(self.node_names), d), nn.Tanh(), nn.LayerNorm(d))
        else:
            self.inputs = nn.ModuleDict({t: NodeFeatureEncoder(cfg=cfg, node_type=t,
                schema=NodeFeatureSchema(s.continuous_names, s.binary_names, tuple(s.categorical)),
                embed_dim=d, categorical_embed_dim=8) for t, s in reference.nodes.items()})
        count = len(self.relations) if method not in ("shared_relation", "gat") else 1
        self.messages = nn.ModuleList([nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(count)]) for _ in range(options["layers"])])
        self.edge_values = nn.ModuleList([nn.ModuleList([nn.Linear(len(self.edge_names), d, bias=False) for _ in range(count)]) for _ in range(options["layers"])])
        self.edge_scores = nn.ModuleList([nn.ModuleList([nn.Linear(len(self.edge_names), 1, bias=False) for _ in range(count)]) for _ in range(options["layers"])])
        self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(d) for t in self.types}) for _ in range(options["layers"])])

    def node_matrix(self, store):
        x = torch.zeros((store.num_nodes, len(self.node_names)))
        for prefix, names, values in (("c:", store.continuous_names, squash(store.continuous)), ("b:", store.binary_names, store.binary)):
            for j, n in enumerate(names):
                x[:, self.node_names.index(prefix + n)] = values[:, j]
        for n, v in store.categorical.items():
            x[:, self.node_names.index("k:" + n)] = squash(v.float())
        return x

    def edge_matrix(self, store):
        x = torch.zeros((store.num_edges, len(self.edge_names)))
        for prefix, names, values in (("c:", store.continuous_names, squash(store.continuous)), ("b:", store.binary_names, store.binary)):
            for j, n in enumerate(names):
                x[:, self.edge_names.index(prefix + n)] = values[:, j]
        return x

    def forward(self, graphs):
        g = batch_heterographs([with_reservations(g) for g in graphs], validate=False)
        nodes = {t: self.input(self.node_matrix(s)) if self.method == "gat" else
                 self.inputs[t](replace(s, continuous=squash(s.continuous))) for t, s in g.nodes.items()}
        edges = {r: self.edge_matrix(s) for r, s in g.edges.items()}
        for layer, edge_values, edge_scores, norms in zip(self.messages, self.edge_values, self.edge_scores, self.norms):
            offsets, offset = {}, 0
            for t, x in nodes.items():
                offsets[t], offset = offset, offset + len(x)
            joined = torch.cat(list(nodes.values()))
            source_ids, target_ids, messages, logits, relation_ids = [], [], [], [], []
            for rid, (r, e) in enumerate(g.edges.items()):
                if not e.num_edges:
                    continue
                src, dst = e.edge_index
                k = rid if len(layer) > 1 else 0
                m = layer[k](nodes[r[0]][src]) + edge_values[k](edges[r])
                score = (m * nodes[r[2]][dst]).sum(-1, keepdim=True) / self.dim**.5 + edge_scores[k](edges[r])
                source_ids.append(src + offsets[r[0]])
                target_ids.append(dst + offsets[r[2]])
                messages.append(m)
                logits.append(score)
                relation_ids.append(torch.full((len(src),), rid, dtype=torch.long))
            aggregate = torch.zeros_like(joined)
            counts = torch.zeros((len(joined), 1))
            if messages:
                dst, rid = torch.cat(target_ids), torch.cat(relation_ids)
                # Shared-relation retains typed neighborhoods; GAT normalizes over
                # all neighbors of the single homogeneous node index space.
                segments = dst if self.method == "gat" else dst + rid * len(joined)
                size = len(joined) if self.method == "gat" else len(joined) * len(self.relations)
                w = _segment_softmax(torch.cat(logits).float(), segments, size)
                aggregate.index_add_(0, dst, torch.cat(messages) * w)
                counts.index_add_(0, dst, w)
            nodes = {t: norms[t](x + torch.tanh(aggregate[offsets[t]:offsets[t] + len(x)] /
                       counts[offsets[t]:offsets[t] + len(x)].clamp_min(1))) for t, x in nodes.items()}
        pooled = []
        for t, x in nodes.items():
            sums = torch.zeros((len(graphs), self.dim)).index_add_(0, g.nodes[t].batch, x)
            pooled.append(sums / (g.nodes[t].ptr[1:] - g.nodes[t].ptr[:-1]).clamp_min(1)[:, None])
        return torch.cat(pooled, -1), nodes, g
