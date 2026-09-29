"""Phase H rollout storage and duration-aware GAE for event-driven SMDP PPO.

The buffer stores the exact Phase G semantic action trace, not sampled tensor
indices.  This lets PPO reconstruct the candidate sets/masks and recompute the
same composite action probability under the current policy.

Transitions may be interleaved across parallel environments.  GAE is therefore
computed independently for each ``(env_id, episode_id)`` trajectory before the
results are written back to the original insertion order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from agent.constraints import PolicyActionConstraints
from agent.policy import CompositePolicyOutput, PolicyTrace
from environment.action_context import ActionContext
from environment.graph_types import HeteroGraph
from environment.replay_state import CompactDecisionSnapshot, CompactReplayMaterializer


@dataclass(frozen=True, slots=True)
class HeadValues:
    v_gate: float
    v_rec: float
    v_sch: float

    @classmethod
    def zeros(cls) -> "HeadValues":
        return cls(0.0, 0.0, 0.0)




@dataclass(frozen=True, slots=True)
class PolicyStorageScalars:
    """CPU scalars packed once per rollout action batch.

    L1.5 uses this to avoid one CUDA synchronization per scalar per event when
    storing old log-probabilities/critic values. The values are identical to the
    legacy ``RolloutBuffer._scalar`` path; only the transfer granularity changes.
    """

    total_log_prob: float
    gate_log_prob: float
    resource_log_prob: float
    schedule_log_prob: float
    values: HeadValues

@dataclass(frozen=True, slots=True)
class RolloutTransition:
    graph: HeteroGraph | None
    context: ActionContext | None
    trace: PolicyTrace
    reward: float
    done: bool
    delta_t: float
    env_id: int
    episode_id: int
    old_total_log_prob: float
    old_gate_log_prob: float
    old_resource_log_prob: float
    old_schedule_log_prob: float
    old_values: HeadValues
    constraints: PolicyActionConstraints = field(
        default_factory=PolicyActionConstraints.unconstrained
    )
    replay_state: CompactDecisionSnapshot | None = None

    @property
    def reconfiguration_active(self) -> bool:
        return bool(self.trace.reconfigure)


@dataclass(frozen=True, slots=True)
class GAEOutput:
    gamma_effective: torch.Tensor
    advantage_gate: torch.Tensor
    advantage_rec: torch.Tensor
    advantage_sch: torch.Tensor
    return_gate: torch.Tensor
    return_rec: torch.Tensor
    return_sch: torch.Tensor

    def composite_advantage(self, reconfigure_mask: torch.Tensor) -> torch.Tensor:
        """Fuse the three critic advantages for the paper's *single* composite PPO ratio.

        The paper specifies three value heads but writes one composite-action PPO
        surrogate.  It does not prescribe the exact scalar fusion.  Phase H uses
        the mean over active decision heads: gate + schedule at every event, plus
        reconfiguration when the event gate selected RECONFIGURE.
        """

        mask = reconfigure_mask.to(dtype=self.advantage_gate.dtype)
        return (
            self.advantage_gate
            + self.advantage_sch
            + mask * self.advantage_rec
        ) / (2.0 + mask)


class RolloutBuffer:
    """In-memory on-policy event buffer with CPU storage by default."""

    def __init__(
        self,
        *,
        storage_device: str | torch.device = "cpu",
        storage_mode: str = "full_graph_context",
        replay_materializer: CompactReplayMaterializer | None = None,
    ) -> None:
        self.storage_device = torch.device(storage_device)
        self.storage_mode = str(storage_mode)
        if self.storage_mode not in {"full_graph_context", "compact_replay_state"}:
            raise ValueError(f"unsupported rollout storage_mode={self.storage_mode!r}")
        if self.storage_mode == "compact_replay_state" and replay_materializer is None:
            raise ValueError("compact_replay_state requires a replay_materializer")
        self.replay_materializer = replay_materializer
        self.transitions: list[RolloutTransition] = []

    @property
    def compact_storage(self) -> bool:
        return self.storage_mode == "compact_replay_state"

    @property
    def estimated_replay_bytes(self) -> int:
        return int(sum(
            0 if t.replay_state is None else t.replay_state.estimated_bytes
            for t in self.transitions
        ))

    def materialize(self, index: int) -> tuple[HeteroGraph, ActionContext]:
        transition = self.transitions[int(index)]
        if transition.graph is not None and transition.context is not None:
            return transition.graph, transition.context
        if transition.replay_state is None or self.replay_materializer is None:
            raise RuntimeError("transition has no materializable graph/context payload")
        return self.replay_materializer.materialize(transition.replay_state)

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions.clear()

    @staticmethod
    def _scalar(value: torch.Tensor) -> float:
        if value.numel() != 1:
            raise ValueError("expected scalar tensor")
        return float(value.detach().cpu().item())

    def add(
        self,
        *,
        graph: HeteroGraph | None,
        context: ActionContext | None,
        policy_output: CompositePolicyOutput,
        replay_state: CompactDecisionSnapshot | None = None,
        constraints: PolicyActionConstraints | None = None,
        policy_scalars: PolicyStorageScalars | None = None,
        reward: float,
        done: bool,
        delta_t: float,
        env_id: int = 0,
        episode_id: int = 0,
    ) -> None:
        if graph is not None and graph.batch_size != 1:
            raise ValueError("RolloutBuffer stores one event graph per transition")
        if self.compact_storage:
            if replay_state is None:
                raise ValueError("compact rollout storage requires replay_state")
            stored_graph = None
            stored_context = None
        else:
            if graph is None or context is None:
                raise ValueError("full rollout storage requires graph and context")
            stored_graph = graph.to(self.storage_device)
            stored_context = context.to(self.storage_device)
        if delta_t < 0.0:
            raise ValueError("delta_t must be non-negative")
        critics = policy_output.representation.critics
        if policy_scalars is None:
            stored_total = self._scalar(policy_output.total_log_prob)
            stored_gate = self._scalar(policy_output.gate_log_prob)
            stored_resource = self._scalar(policy_output.worker_log_prob + policy_output.robot_log_prob)
            stored_schedule = self._scalar(policy_output.schedule_log_prob)
            stored_values = HeadValues(
                v_gate=self._scalar(critics.v_gate),
                v_rec=self._scalar(critics.v_rec),
                v_sch=self._scalar(critics.v_sch),
            )
        else:
            stored_total = float(policy_scalars.total_log_prob)
            stored_gate = float(policy_scalars.gate_log_prob)
            stored_resource = float(policy_scalars.resource_log_prob)
            stored_schedule = float(policy_scalars.schedule_log_prob)
            stored_values = policy_scalars.values
        transition = RolloutTransition(
            graph=stored_graph,
            context=stored_context,
            trace=policy_output.trace,
            constraints=constraints or PolicyActionConstraints.unconstrained(),
            reward=float(reward),
            done=bool(done),
            delta_t=float(delta_t),
            env_id=int(env_id),
            episode_id=int(episode_id),
            old_total_log_prob=stored_total,
            old_gate_log_prob=stored_gate,
            old_resource_log_prob=stored_resource,
            old_schedule_log_prob=stored_schedule,
            old_values=stored_values,
            replay_state=replay_state if self.compact_storage else None,
        )
        self.transitions.append(transition)

    def _trajectory_groups(self) -> dict[tuple[int, int], list[int]]:
        groups: dict[tuple[int, int], list[int]] = {}
        for idx, transition in enumerate(self.transitions):
            key = (transition.env_id, transition.episode_id)
            groups.setdefault(key, []).append(idx)
        return groups

    def compute_gae(
        self,
        *,
        gamma: float,
        tau_0_minutes: float,
        gae_lambda: float,
        duration_aware: bool = True,
        bootstrap_values: Mapping[tuple[int, int], HeadValues] | None = None,
    ) -> GAEOutput:
        """Compute paper Eq. duration-aware GAE independently per trajectory.

        ``bootstrap_values`` is required only for trajectory fragments whose final
        stored transition is not terminal.  A full episode needs no bootstrap.
        """

        if not self.transitions:
            raise ValueError("cannot compute GAE from an empty buffer")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0,1]")
        if tau_0_minutes <= 0.0:
            raise ValueError("tau_0_minutes must be positive")
        if not (0.0 <= gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in [0,1]")

        bootstrap_values = dict(bootstrap_values or {})
        n = len(self.transitions)
        dtype = torch.float64
        gamma_eff = torch.empty(n, dtype=dtype)
        adv_gate = torch.zeros(n, dtype=dtype)
        adv_rec = torch.zeros(n, dtype=dtype)
        adv_sch = torch.zeros(n, dtype=dtype)

        for i, tr in enumerate(self.transitions):
            gamma_eff[i] = float(
                gamma ** (tr.delta_t / tau_0_minutes)
                if duration_aware else gamma
            )

        def head_value(values: HeadValues, head: str) -> float:
            return float(getattr(values, head))

        for key, indices in self._trajectory_groups().items():
            last = self.transitions[indices[-1]]
            if last.done:
                bootstrap = HeadValues.zeros()
            else:
                if key not in bootstrap_values:
                    raise ValueError(
                        f"missing bootstrap value for truncated trajectory {key}"
                    )
                bootstrap = bootstrap_values[key]

            next_adv = {"v_gate": 0.0, "v_rec": 0.0, "v_sch": 0.0}
            for pos in range(len(indices) - 1, -1, -1):
                idx = indices[pos]
                tr = self.transitions[idx]
                discount = float(gamma_eff[idx].item())
                not_done = 0.0 if tr.done else 1.0

                if pos + 1 < len(indices):
                    next_values = self.transitions[indices[pos + 1]].old_values
                else:
                    next_values = bootstrap

                for head, target in (
                    ("v_gate", adv_gate),
                    ("v_rec", adv_rec),
                    ("v_sch", adv_sch),
                ):
                    value = head_value(tr.old_values, head)
                    next_value = head_value(next_values, head)
                    delta = tr.reward + discount * not_done * next_value - value
                    advantage = delta + discount * gae_lambda * not_done * next_adv[head]
                    target[idx] = advantage
                    next_adv[head] = advantage

        values_gate = torch.tensor(
            [tr.old_values.v_gate for tr in self.transitions], dtype=dtype
        )
        values_rec = torch.tensor(
            [tr.old_values.v_rec for tr in self.transitions], dtype=dtype
        )
        values_sch = torch.tensor(
            [tr.old_values.v_sch for tr in self.transitions], dtype=dtype
        )
        return GAEOutput(
            gamma_effective=gamma_eff.to(torch.float32),
            advantage_gate=adv_gate.to(torch.float32),
            advantage_rec=adv_rec.to(torch.float32),
            advantage_sch=adv_sch.to(torch.float32),
            return_gate=(adv_gate + values_gate).to(torch.float32),
            return_rec=(adv_rec + values_rec).to(torch.float32),
            return_sch=(adv_sch + values_sch).to(torch.float32),
        )


__all__ = [
    "GAEOutput",
    "HeadValues",
    "PolicyStorageScalars",
    "RolloutBuffer",
    "RolloutTransition",
]
