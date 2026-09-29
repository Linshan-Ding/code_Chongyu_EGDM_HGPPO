"""EGDM-HGPPO agent modules implemented through Phase H."""

from agent.base import EGDMRepresentationNetwork, RepresentationOutput
from agent.buffer import GAEOutput, HeadValues, RolloutBuffer, RolloutTransition
from agent.critic import CriticOutput, EventTypeAwareCritic, EventTypeEncoder
from agent.encoder import HeteroGraphEncoder, RelationAwareHGTLayer
from agent.gate import EventGateHead, GateOutput
from agent.matching import (
    DecoderFeasibilityError,
    MatchingChoice,
    MatchingResult,
    MatchingTrace,
    OperationCellAutoregressiveMatcher,
    PlannedConfiguration,
    ResourceCellAutoregressiveMatcher,
)
from agent.policy import CompositePolicyOutput, EGDMCompositePolicy, PolicyTrace
from agent.pooling import PoolingOutput, TypeAwareAttentionPooling
from agent.ppo import PPOAgent, PPOUpdateStats

__all__ = [
    "CompositePolicyOutput",
    "CriticOutput",
    "DecoderFeasibilityError",
    "EGDMCompositePolicy",
    "EGDMRepresentationNetwork",
    "EventGateHead",
    "EventTypeAwareCritic",
    "EventTypeEncoder",
    "GAEOutput",
    "GateOutput",
    "HeadValues",
    "HeteroGraphEncoder",
    "MatchingChoice",
    "MatchingResult",
    "MatchingTrace",
    "OperationCellAutoregressiveMatcher",
    "PPOAgent",
    "PPOUpdateStats",
    "PlannedConfiguration",
    "PolicyTrace",
    "PoolingOutput",
    "RelationAwareHGTLayer",
    "RepresentationOutput",
    "ResourceCellAutoregressiveMatcher",
    "RolloutBuffer",
    "RolloutTransition",
    "TypeAwareAttentionPooling",
]
