"""Independent actor/critic encoders; explicit gate and conditional plan likelihood, optional learned dispatch."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .resource_features import ACTION_DIM, GLOBAL_DIM, STAGE_DIM
from .dispatch import DISPATCH_DIM, dispatch_step
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

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, observations):
        state = torch.from_numpy(np.stack([o.state for o in observations])).to(self.device)
        if self.method == "mlp":
            lengths = [len(o.stages) for o in observations]
            h = self.stage(torch.from_numpy(np.concatenate([o.stages for o in observations])).to(self.device))
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
        # ``fixed_dispatch`` is the ablation of the learned dispatcher: same graph,
        # gate and plans, but the schedule stays the ATC(beta=1) rule.
        self.learned_dispatch = bool(options.get("learned_dispatch", False)) and method != "fixed_dispatch"
        self.dispatcher = nn.Sequential(nn.Linear(sd + DISPATCH_DIM + 2 * h, h), nn.Tanh(), nn.Linear(h, 1))
        for head in (self.gate[-1], self.plan[-1], self.dispatcher[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.critic = nn.Sequential(nn.Linear(sd, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.value = PopArt(h, options["popart_beta"])
        # Without a trained critic the value head is never queried during sampling.
        self.use_critic = options.get("advantage_mode", "gae") == "gae" or options.get("critic", "auxiliary") != "none"
        self._encoding = None

    @property
    def device(self):
        return next(self.parameters()).device

    def actor_parameters(self):
        params = list(self.actor_encoder.parameters()) + list(self.gate.parameters()) + list(self.plan.parameters())
        return params + (list(self.dispatcher.parameters()) if self.learned_dispatch else [])

    def critic_parameters(self):
        return list(self.critic_encoder.parameters()) + list(self.critic.parameters()) + list(self.value.parameters())

    def encode(self, observations):
        state, nodes, graph = self.actor_encoder(observations)
        return state, nodes, graph

    def plan_distributions(self, observations, encoding):
        state, nodes, graph = encoding
        device = state.device
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
                        tc = nodes["cell"][graph.nodes["cell"].ptr[i] + max(target, 0)]
                        if target < 0:
                            tc = torch.zeros_like(tc)
                        embeddings.append(torch.cat((resource, sc, tc)))
                links.append(torch.stack(embeddings).mean(0) if embeddings else torch.zeros(3 * h, device=device))
            features = torch.from_numpy(o.features).to(device)
            residual = self.plan(torch.cat((state[i].expand(len(o.prior), -1), features, torch.stack(links)), -1)).squeeze(-1)
            prior = torch.from_numpy(o.prior).to(device)
            groups = torch.from_numpy(o.groups).long().to(device)
            gp = torch.zeros(2, device=device).scatter_add(0, groups, prior)
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

    def distributions(self, observations):
        encoding = self.encode(observations)
        self._encoding = (tuple(id(o) for o in observations), encoding)
        return self.plan_distributions(observations, encoding)

    def dispatch_distribution(self, encoding, i, step):
        state, nodes, graph = encoding
        device = state.device
        h = self.options["hidden"]
        if nodes is not None:
            base = graph.nodes["operation"].ptr[i]
            ops = torch.stack([nodes["operation"][base + k] if k >= 0 else torch.zeros(h, device=device) for k in step.operations])
            cells = torch.stack([nodes["cell"][graph.nodes["cell"].ptr[i] + c] for c in step.cells])
        else:
            ops = torch.zeros((len(step.cells), h), device=device)
            cells = torch.zeros((len(step.cells), h), device=device)
        features = torch.from_numpy(step.features).to(device)
        residual = self.dispatcher(torch.cat((state[i].expand(len(step.prior), -1), features, ops, cells), -1)).squeeze(-1)
        return Categorical(logits=torch.from_numpy(step.prior).to(device).clamp_min(1e-30).log() + residual)

    @staticmethod
    def divergence(distribution, prior):
        log_prior = torch.from_numpy(prior).to(distribution.logits.device).clamp_min(1e-30).log()
        return (distribution.probs * (distribution.logits - log_prior)).sum()

    def normalized_values(self, observations):
        return self.value(self.critic(self.critic_encoder(observations)[0]))

    def evaluate_actions(self, observations, actions, need_values=True):
        encoding = self.encode(observations)
        distributions, _ = self.plan_distributions(observations, encoding)
        logps, entropies, divergences = [], [], []
        for i, (d, o, a) in enumerate(zip(distributions, observations, actions)):
            plan_index, choices = (a if isinstance(a, (tuple, list)) else (a, ()))
            lp = d.log_prob(torch.tensor(int(plan_index), device=d.logits.device))
            ent = d.entropy()
            kl = self.divergence(d, o.prior)
            steps = getattr(o, "dispatch_steps", ())
            if len(choices) != len(steps):
                raise ValueError("Dispatch replay length mismatch")
            for step, choice in zip(steps, choices):
                dist = self.dispatch_distribution(encoding, i, step)
                lp = lp + dist.log_prob(torch.tensor(int(choice), device=dist.logits.device))
                ent = ent + dist.entropy()
                kl = kl + self.divergence(dist, step.prior)
            logps.append(lp)
            entropies.append(ent)
            divergences.append(kl)
        values = self.normalized_values(observations) if need_values else None
        return torch.stack(logps), torch.stack(entropies), torch.stack(divergences), values

    @torch.no_grad()
    def select(self, observation, generator=None, deterministic=False):
        ds, parts = self.distributions([observation])
        d = ds[0]
        a = int(d.probs.argmax()) if deterministic else int(torch.multinomial(d.probs.cpu(), 1, generator=generator)[0])
        value = float(self.value.denormalize(self.normalized_values([observation]))[0]) if self.use_critic else 0.0
        return a, float(d.log_prob(torch.tensor(a, device=d.logits.device))), value, {
            "gate": int(observation.groups[a]), "conditional_action": a,
            "gate_logp": float(parts[0][0][a]), "conditional_logp": float(parts[0][1][a]),
            "joint_logp": float(d.log_prob(torch.tensor(a))), "candidate": observation.identifiers[a],
            "mask": observation.mask.tolist(), "candidate_prior": observation.prior.tolist(),
            "anchor": a == observation.anchor, "choices": len(observation.prior),
        }

    @torch.no_grad()
    def dispatch(self, observation, plan_index, builder, generator=None, deterministic=False):
        """Sequential learned dispatch for the chosen plan; returns steps, schedule and summed log-probability."""
        cached = self._encoding
        encoding = cached[1] if cached and cached[0] == (id(observation),) else self.encode([observation])
        plan = observation.actions[plan_index]
        ws, rs = plan.worker_assignments, plan.robot_assignments
        prefix, steps, total, atc_steps = (), [], 0.0, 0
        while True:
            step = dispatch_step(builder, ws, rs, prefix, self.options, observation.graph)
            if step is None:
                break
            dist = self.dispatch_distribution(encoding, 0, step)
            choice = int(dist.probs.argmax()) if deterministic else int(torch.multinomial(dist.probs.cpu(), 1, generator=generator)[0])
            total += float(dist.log_prob(torch.tensor(choice, device=dist.logits.device)))
            step.choice = choice
            atc_steps += int(choice == step.atc)
            steps.append(step)
            prefix += (step.options[choice][0],)
        return steps, prefix, total, atc_steps


def create_network(options, graph, method):
    if method == "dan":
        from .dan import Network as DAN
        return DAN(options, graph)
    return Network(options, graph, method)
