"""Checkpoint-backed evaluators for independently trained Table-10 variants."""
from __future__ import annotations

from pathlib import Path

import torch

from agent.baselines.base import DecisionPolicy
from agent.baselines.learned_variants import build_learned_baseline_policy
from agent.ablation_variants import build_ablation_policy
from agent.constraints import PolicyActionConstraints


class AblationEvaluationPolicy(DecisionPolicy):
    """Load one ablation checkpoint without changing the fixed-test protocol."""

    def __init__(self, cfg, *, variant: str, checkpoint: str | Path,
                 device: torch.device | str = "cpu", deterministic: bool = True,
                 periodic_gate_period: int = 4):
        self.cfg = cfg
        self.variant = str(variant)
        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.periodic_gate_period = max(1, int(periodic_gate_period))
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "policy_state" not in payload or payload.get("normalizer") is None:
            raise ValueError(f"ablation checkpoint is missing policy_state/normalizer: {self.checkpoint_path}")
        self.policy_state = payload["policy_state"]
        self.normalizer = payload["normalizer"]
        self.policy = None
        self.constraints = (
            PolicyActionConstraints(max_schedule_assignments=1)
            if self.variant == "sequential_dispatch" else
            PolicyActionConstraints(max_worker_moves=1, max_robot_moves=1)
            if self.variant == "flat_reconfiguration" else
            PolicyActionConstraints.unconstrained()
        )

    @property
    def method_name(self) -> str:
        return f"Ablation-{self.variant}"

    def _ensure_policy(self, env) -> None:
        if self.policy is not None:
            return
        graph = self.normalizer.transform(env.graph()).to(self.device)
        if self.variant == "homogeneous_gat":
            policy = build_learned_baseline_policy("GAT-PPO", self.cfg, graph)
        else:
            policy = build_ablation_policy(
                self.variant, self.cfg, graph,
                periodic_gate_period=self.periodic_gate_period,
            )
        policy = policy.to(self.device)
        policy.load_state_dict(self.policy_state)
        policy.eval()
        self.policy = policy

    @torch.no_grad()
    def act(self, env):
        return self.act_batch((env,))[0]

    @torch.no_grad()
    def act_batch(self, envs):
        envs = tuple(envs)
        if not envs:
            return ()
        self._ensure_policy(envs[0])
        graphs = [self.normalizer.transform(env.graph()) for env in envs]
        contexts = [env.action_context() for env in envs]
        outputs = self.policy.act_batch(
            graphs, contexts, deterministic=self.deterministic,
            constraints=self.constraints,
        )
        return tuple(output.action for output in outputs)


__all__ = ["AblationEvaluationPolicy"]
