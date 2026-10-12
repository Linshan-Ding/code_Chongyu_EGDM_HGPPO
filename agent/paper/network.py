"""Independent actor/critic encoders; explicit gate and conditional plan likelihood."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .resource_features import ACTION_DIM, GLOBAL_DIM, STAGE_DIM
from .popart import PopArt
from .graph import GraphEncoder
from .config import project_config


class Encoder(nn.Module):
    def __init__(self, options, graph, method):
        super().__init__()
        self.method, self.dim = method, options["hidden"]
        if method == "mlp":
            self.stage = nn.Sequential(nn.Linear(STAGE_DIM, self.dim), nn.Tanh(), nn.Linear(self.dim, self.dim), nn.Tanh())
        else:
            self.graph = GraphEncoder(project_config(options), graph, options, method)
        state_dim = GLOBAL_DIM + len(graph.gate_continuous_names) + 3
        self.output_dim = (2 if method == "mlp" else 5) * self.dim + state_dim

    def forward(self, observations):
        state = torch.from_numpy(np.stack([o.state for o in observations]))
        if self.method == "mlp":
            lengths = [len(o.stages) for o in observations]
            h = self.stage(torch.from_numpy(np.concatenate([o.stages for o in observations])))
            parts = h.split(lengths)
            pooled = torch.stack([torch.cat((x.mean(0), x.amax(0))) for x in parts])
            return torch.cat((pooled, state), -1), None, None
        pooled, nodes, graph = self.graph([o.graph for o in observations])
        return torch.cat((pooled, state), -1), nodes, graph


class Network(nn.Module):
    def __init__(self, options, graph, method="full"):
        super().__init__()
        self.method, self.options = method, options
        h = options["hidden"]
        self.actor_encoder = Encoder(options, graph, method)
        self.critic_encoder = Encoder(options, graph, method)
        sd = self.actor_encoder.output_dim
        self.gate = nn.Sequential(nn.Linear(sd, h), nn.Tanh(), nn.Linear(h, 2))
        self.plan = nn.Sequential(nn.Linear(sd + ACTION_DIM + 3 * h, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        for head in (self.gate[-1], self.plan[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.critic = nn.Sequential(nn.Linear(sd, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.value = PopArt(h, options["popart_beta"])

    def actor_parameters(self):
        return list(self.actor_encoder.parameters()) + list(self.gate.parameters()) + list(self.plan.parameters())

    def critic_parameters(self):
        return list(self.critic_encoder.parameters()) + list(self.critic.parameters()) + list(self.value.parameters())

    def distributions(self, observations):
        state, nodes, graph = self.actor_encoder(observations)
        gate_residual = self.gate(state)
        distributions, decompositions = [], []
        for i, o in enumerate(observations):
            h = self.options["hidden"]
            links = []
            for moves in o.moves:
                embeddings = []
                if nodes is not None:
                    for kind, rid, source, target in moves:
                        resource = nodes[kind][graph.nodes[kind].ptr[i] + rid]
                        sc = nodes["cell"][graph.nodes["cell"].ptr[i] + source]
                        tc = nodes["cell"][graph.nodes["cell"].ptr[i] + target]
                        embeddings.append(torch.cat((resource, sc, tc)))
                links.append(torch.stack(embeddings).mean(0) if embeddings else torch.zeros(3 * h))
            features = torch.from_numpy(o.features)
            residual = self.plan(torch.cat((state[i].expand(len(o.prior), -1), features, torch.stack(links)), -1)).squeeze(-1)
            prior = torch.from_numpy(o.prior)
            groups = torch.from_numpy(o.groups).long()
            gp = torch.zeros(2).scatter_add(0, groups, prior)
            gate_log = torch.log_softmax(gp.clamp_min(1e-30).log().masked_fill(gp == 0, -torch.inf) + gate_residual[i], -1)
            conditional = torch.empty_like(prior)
            for group in (0, 1):
                mask = groups == group
                if mask.any():
                    conditional[mask] = torch.log_softmax(prior[mask].log() - gp[group].log() + residual[mask], -1)
            joint = gate_log[groups] + conditional
            distributions.append(Categorical(logits=joint))
            decompositions.append((gate_log[groups], conditional))
        return distributions, decompositions

    def normalized_values(self, observations):
        return self.value(self.critic(self.critic_encoder(observations)[0]))

    def evaluate_actions(self, observations, actions):
        distributions, _ = self.distributions(observations)
        logp = torch.stack([d.log_prob(torch.tensor(a)) for d, a in zip(distributions, actions)])
        entropy = torch.stack([d.entropy() for d in distributions])
        return logp, entropy, self.normalized_values(observations)

    @torch.no_grad()
    def select(self, observation, generator=None, deterministic=False):
        ds, parts = self.distributions([observation])
        d = ds[0]
        a = int(d.probs.argmax()) if deterministic else int(torch.multinomial(d.probs, 1, generator=generator)[0])
        value = float(self.value.denormalize(self.normalized_values([observation]))[0])
        return a, float(d.log_prob(torch.tensor(a))), value, {
            "gate": int(observation.groups[a]), "conditional_action": a,
            "gate_logp": float(parts[0][0][a]), "conditional_logp": float(parts[0][1][a]),
            "joint_logp": float(d.log_prob(torch.tensor(a))), "candidate": observation.identifiers[a],
            "mask": observation.mask.tolist(), "candidate_prior": observation.prior.tolist(),
            "anchor": a == observation.anchor, "choices": len(observation.prior),
        }


def create_network(options, graph, method):
    if method == "dan":
        from .dan import Network as DAN
        return DAN(options, graph)
    return Network(options, graph, method)
