"""Strict Table-10 component-ablation policy knobs.

Each variant keeps the Scheme-2 environment, sampling, reward accounting and
PPO budget fixed.  The knobs are deliberately small and serialisable so a
checkpoint records exactly which component was disabled.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from agent.critic import CriticOutput, EventTypeEncoder
from agent.nn_utils import build_gelu_layernorm_mlp
from agent.policy import EGDMCompositePolicy


ABLATION_VARIANTS = (
    "without_event_gate",
    "periodic_gate",
    "without_stage_nodes",
    "homogeneous_gat",
    "flat_reconfiguration",
    "sequential_dispatch",
    "without_duration_aware_gae",
    "without_potential_shaping",
    "single_critic",
)

# The reference policy is the already-trained formal Scheme-2 run; it is not
# retrained as an ablation.  ``full`` remains accepted by ``get_ablation_spec``
# for checkpoint/evaluation plumbing, but is deliberately absent from the
# formal ablation matrix above.


@dataclass(frozen=True, slots=True)
class AblationSpec:
    name: str
    description: str
    disable_stage_nodes: bool = False
    periodic_gate: bool = False
    sequential_dispatch: bool = False
    duration_aware_gae: bool = True
    potential_shaping: bool = True
    single_critic: bool = False
    without_event_gate: bool = False
    homogeneous_gat: bool = False
    flat_reconfiguration: bool = False
    periodic_gate_period: int = 4


def get_ablation_spec(name: str, *, periodic_gate_period: int = 4) -> AblationSpec:
    key = str(name)
    if key not in ABLATION_VARIANTS and key != "full":
        raise ValueError(f"unknown ablation variant: {name!r}; expected one of {ABLATION_VARIANTS}")
    period = max(1, int(periodic_gate_period))
    return {
        "full": AblationSpec("full", "Teacher Scheme-2", periodic_gate_period=period),
        "without_stage_nodes": AblationSpec("without_stage_nodes", "w/o Stage Nodes", disable_stage_nodes=True),
        "periodic_gate": AblationSpec("periodic_gate", "Periodic Gate", periodic_gate=True, periodic_gate_period=period),
        "sequential_dispatch": AblationSpec("sequential_dispatch", "Sequential Dispatch", sequential_dispatch=True),
        "without_duration_aware_gae": AblationSpec("without_duration_aware_gae", "w/o Duration-aware GAE", duration_aware_gae=False),
        "without_potential_shaping": AblationSpec("without_potential_shaping", "w/o Potential Shaping", potential_shaping=False),
        "single_critic": AblationSpec("single_critic", "Single Critic", single_critic=True),
        "without_event_gate": AblationSpec("without_event_gate", "w/o Event Gate", without_event_gate=True),
        "homogeneous_gat": AblationSpec("homogeneous_gat", "Homogeneous GAT", homogeneous_gat=True),
        "flat_reconfiguration": AblationSpec("flat_reconfiguration", "Flat Reconfiguration", flat_reconfiguration=True),
    }[key]


class SingleCritic(nn.Module):
    """One value head shared by gate/resource/schedule targets."""

    def __init__(self, cfg, *, event_count: int, embed_dim: int):
        super().__init__()
        event_dim = int(cfg.algo.representation_network.event_type_embed_dim)
        hidden = [int(x) for x in cfg.algo.policy_value_mlp]
        self.event_encoder = EventTypeEncoder(event_count, event_dim)
        self.head = build_gelu_layernorm_mlp(embed_dim + event_dim, hidden, 1)

    def forward(self, global_embedding, event_type_multihot):
        event_embedding = self.event_encoder(event_type_multihot)
        value = self.head(torch.cat([global_embedding, event_embedding], dim=-1)).squeeze(-1)
        return CriticOutput(value, value, value, event_embedding)


class AblationCompositePolicy(EGDMCompositePolicy):
    """Composite policy with one isolated Table-10 component switch."""

    def __init__(self, cfg, reference_graph, *, spec: AblationSpec):
        super().__init__(cfg, reference_graph)
        self.ablation_spec = spec
        # Exposed as plain serialisable metadata so CPU rollout workers can
        # reconstruct the exact variant before loading the state dict.
        self.ablation_variant = spec.name
        self.ablation_periodic_gate_period = int(spec.periodic_gate_period)
        if spec.disable_stage_nodes:
            self.representation.encoder.disabled_node_types = frozenset({"stage"})
        if spec.single_critic:
            self.representation.critics = SingleCritic(
                cfg, event_count=int(reference_graph.event_type_multihot.shape[1]),
                embed_dim=int(cfg.algo.embed_dim),
            )

    def _decode_from_representation(self, graph, context, representation, **kwargs):
        # ``without_stage_nodes`` is enforced inside every encoder layer: its
        # raw features are never encoded, all incident relations are skipped,
        # and its representation remains zero.  No post-hoc masking is used.
        # The two gate ablations are deterministic controls.  They still obey
        # the environment's hard feasibility bit, but never use the learned
        # gate logit.  ``periodic_gate`` uses the graph's simulator decision
        # index, so the schedule is reproducible across batched and serial
        # collectors (a mutable Python counter would diverge in workers).
        if kwargs.get("force_gate") is None:
            feasible = bool(graph.reconfigure_feasible.reshape(-1)[0].item())
            if self.ablation_spec.without_event_gate:
                kwargs["force_gate"] = feasible
            elif self.ablation_spec.periodic_gate:
                decision_index = int(graph.decision_index.reshape(-1)[0].item())
                period = max(1, int(self.ablation_spec.periodic_gate_period))
                kwargs["force_gate"] = bool(feasible and decision_index % period == 0)
        return super()._decode_from_representation(graph, context, representation, **kwargs)


def build_ablation_policy(name: str, cfg, reference_graph, *, periodic_gate_period: int = 4):
    return AblationCompositePolicy(
        cfg, reference_graph,
        spec=get_ablation_spec(name, periodic_gate_period=periodic_gate_period),
    )


__all__ = ["ABLATION_VARIANTS", "AblationSpec", "AblationCompositePolicy", "build_ablation_policy", "get_ablation_spec"]
