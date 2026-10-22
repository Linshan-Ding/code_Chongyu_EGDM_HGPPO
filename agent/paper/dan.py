"""Pinned official DAN dispatch network with the SAME periodic resource manager.

Only the state/action interface, shared feasibility, TWT and PopArt are adapted.
The dispatch sequence has a summed joint log probability, never an average.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from types import SimpleNamespace

from .dan_official import official_classes
from .graph import squash
import time
from .popart import PopArt
from agent.baselines.rules import _schedule_options
from environment.state import CompositeAction
from .candidates import Builder


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 8
        c = SimpleNamespace(fea_j_input_dim=10, fea_m_input_dim=8, layer_fea_output_dim=[32, 8],
                            num_heads_OAB=[4, 4], num_heads_MAB=[4, 4], dropout_prob=0.)
        self.dan = official_classes()["DualAttentionNetwork"](c)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, graphs):
        b = len(graphs)
        n = max(1, max(g.nodes["operation"].num_nodes for g in graphs))
        m = max(g.nodes["cell"].num_nodes for g in graphs)
        j = max(1, max(int(g.nodes["operation"].binary[:, 0].sum()) for g in graphs))
        fj = torch.zeros((b, n, 10))
        fm = torch.zeros((b, m, 8))
        opmask = torch.ones((b, n, 3))
        opmask[:, :, 1] = 0
        candidate = torch.zeros((b, j), dtype=torch.long)
        competition = torch.zeros((b, m, m, j))
        machine_mask = torch.eye(m).repeat(b, 1, 1)
        for i, g in enumerate(graphs):
            ops, cells = g.nodes["operation"], g.nodes["cell"]
            no, nm = ops.num_nodes, cells.num_nodes
            fj[i, :no] = torch.cat((squash(ops.continuous), ops.binary[:, :2]), -1)
            fm[i, :nm] = torch.cat((squash(cells.continuous), cells.binary[:, :5]), -1)
            edges = g.edges[("operation", "precedes", "operation")].edge_index
            if edges.numel():
                src, dst = edges
                if not torch.all(dst == src + 1):
                    raise ValueError("DAN route adjacency must remain consecutive")
                opmask[i, src, 2] = 0
                opmask[i, dst, 0] = 0
            ready = torch.where(ops.binary[:, 0] > 0)[0]
            candidate[i, : len(ready)] = ready
            oc = g.edges[("operation", "compatible_with", "cell")]
            legal = torch.zeros((no, nm))
            if oc.num_edges:
                legal[oc.edge_index[0], oc.edge_index[1]] = oc.binary[:, 1]
            if len(ready):
                eligible = legal[ready].T
                competition[i, :nm, :nm, : len(ready)] = (
                    eligible[:, None, :] * eligible[None, :, :]
                )
                machine_mask[i, :nm, :nm] = (
                    (competition[i, :nm, :nm].sum(-1) > 0)
                    | torch.eye(nm, dtype=torch.bool)
                ).float()
        transfer_start = time.perf_counter()
        fj, fm, opmask, candidate, machine_mask, competition = [
            x.to(self.device)
            for x in (fj, fm, opmask, candidate, machine_mask, competition)
        ]
        self.transfer_seconds = (
            getattr(self, "transfer_seconds", 0.0)
            + time.perf_counter()
            - transfer_start
        )
        ops, cells, _, _ = self.dan(
            fj, opmask, candidate, fm, machine_mask, competition
        )
        result = []
        for i, g in enumerate(graphs):
            operation = ops[i, : g.nodes["operation"].num_nodes]
            cell = cells[i, : g.nodes["cell"].num_nodes]
            op_global = (
                operation.mean(0)
                if len(operation)
                else torch.zeros(self.dim, device=self.device)
            )
            result.append(
                {
                    "operation": operation,
                    "cell": cell,
                    "global_op": op_global,
                    "global_cell": cell.mean(0),
                }
            )
        return result


@dataclass
class Token:
    sources: list
    targets: list
    pairs: np.ndarray
    identifiers: list


@dataclass
class Observation:
    graph: object
    tokens: list


class Network(nn.Module):
    def __init__(self, options, graph):
        super().__init__()
        self.method = "dan"
        self.actor_encoder = Encoder()
        self.critic_encoder = Encoder()
        self.actor = official_classes()["Actor"](3, 40, 64, 1)
        self.critic = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.value = PopArt(64, options["popart_beta"])
        self.use_critic = options.get("advantage_mode", "gae") == "gae" or options.get("critic", "auxiliary") != "none"

    def actor_parameters(self):
        return list(self.actor_encoder.parameters()) + list(self.actor.parameters())

    def critic_parameters(self):
        return list(self.critic_encoder.parameters()) + list(self.critic.parameters()) + list(self.value.parameters())

    def distribution(self, e, token):
        n = len(token.sources)
        x = torch.cat((e["operation"][token.sources], e["cell"][token.targets],
                       e["global_op"].expand(n, -1), e["global_cell"].expand(n, -1), torch.from_numpy(token.pairs)), -1)
        return Categorical(logits=self.actor(x).flatten())

    def normalized_values(self, observations):
        es = self.critic_encoder([o.graph for o in observations])
        x = torch.stack([torch.cat((e["global_op"], e["global_cell"])) for e in es])
        return self.value(self.critic(x))

    def evaluate_actions(self, observations, actions, need_values=True):
        es = self.actor_encoder([o.graph for o in observations])
        logs, entropies = [], []
        for e, obs, sequence in zip(es, observations, actions):
            lp = e["global_cell"].sum() * 0
            entropy = lp
            for token, choice in zip(obs.tokens, sequence):
                dist = self.distribution(e, token)
                lp = lp + dist.log_prob(torch.tensor(choice))
                entropy = entropy + dist.entropy()
            if len(sequence) != len(obs.tokens):
                raise ValueError("DAN prefix replay length mismatch")
            logs.append(lp)
            entropies.append(entropy)
        # DAN has no candidate prior: its divergence term is identically zero.
        values = self.normalized_values(observations) if need_values else None
        return torch.stack(logs), torch.stack(entropies), torch.zeros(len(observations)), values

    @torch.no_grad()
    def act(self, env, options, generator=None, deterministic=False):
        builder = Builder(env, options)
        graph = builder.env.graph()
        e = self.actor_encoder([graph])[0]
        opids = {int(oid): i for i, oid in enumerate(graph.nodes["operation"].ids)}
        ws, rs = builder.resources(0)
        prefix, tokens, sequence = (), [], []
        logp = 0.
        while True:
            legal = _schedule_options(builder.env, reconfigure=bool(ws or rs), workers=ws, robots=rs, prefix=prefix)
            if not legal:
                break
            pairs = []
            for a, p in legal:
                op = builder.operations[a.operation_id]
                order = builder.orders[op.order_id]
                pairs.append([p, order.due_date - env.sim.time - p, order.weight,
                              builder.after[a.operation_id], len(prefix), len(ws), len(rs), 1.])
            x = np.asarray(pairs, dtype=np.float32)
            token = Token([opids[a.operation_id] for a, _ in legal], [a.cell_id for a, _ in legal],
                          np.sign(x) * np.log1p(np.abs(x)), [(a.operation_id, a.cell_id) for a, _ in legal])
            dist = self.distribution(e, token)
            a = int(dist.probs.argmax()) if deterministic else int(torch.multinomial(dist.probs, 1, generator=generator)[0])
            logp += float(dist.log_prob(torch.tensor(a)))
            tokens.append(token)
            sequence.append(a)
            prefix += (legal[a][0],)
        obs = Observation(graph, tokens)
        value = float(self.value.denormalize(self.normalized_values([obs]))[0]) if self.use_critic else 0.0
        details = dict(gate=int(bool(ws or rs)), gate_logp=0., conditional_logp=logp, joint_logp=logp,
                       candidate="Periodic-DAN", anchor=False, choices=max([len(t.sources) for t in tokens] + [1]))
        return obs, sequence, CompositeAction(bool(ws or rs), ws, rs, prefix), logp, value, details
