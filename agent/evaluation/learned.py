"""Checkpoint-backed EGDM-HGPPO evaluator for Phase K."""

from __future__ import annotations

from pathlib import Path

import torch

from agent.policy import EGDMCompositePolicy
from agent.baselines.base import DecisionPolicy
from agent.training.checkpoint import load_training_checkpoint


class LearnedPolicyCheckpointError(RuntimeError):
    pass


class EGDMHGPPOEvaluationPolicy(DecisionPolicy):
    method_name = "EGDM-HGPPO"

    def __init__(
        self,
        cfg,
        *,
        checkpoint: str | Path,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
        normalizer_checkpoint: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.checkpoint_path = Path(checkpoint)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "policy_state" not in payload:
            raise LearnedPolicyCheckpointError("checkpoint does not contain policy_state")
        self.policy_state = payload["policy_state"]
        self.normalizer = payload.get("normalizer")
        if self.normalizer is None and normalizer_checkpoint is not None:
            source = load_training_checkpoint(normalizer_checkpoint, map_location="cpu")
            self.normalizer = source.get("normalizer")
        if self.normalizer is None:
            raise LearnedPolicyCheckpointError(
                "checkpoint has no normalizer; pass --normalizer-checkpoint pointing to a Phase J latest.pt"
            )
        self.policy: EGDMCompositePolicy | None = None

    def _ensure_policy(self, env) -> None:
        if self.policy is not None:
            return
        graph = self.normalizer.transform(env.graph()).to(self.device)
        policy = EGDMCompositePolicy(self.cfg, graph).to(self.device)
        policy.load_state_dict(self.policy_state)
        policy.eval()
        self.policy = policy

    @torch.no_grad()
    def act(self, env):
        return self.act_batch((env,))[0]

    @torch.no_grad()
    def act_batch(self, envs):
        """Choose one exact feasible action for every independent environment.

        Only the heterogeneous-graph encoder is batched. The policy keeps its
        established per-environment autoregressive decoder, feasibility masks,
        and action semantics unchanged.
        """
        envs = tuple(envs)
        if not envs:
            return ()
        self._ensure_policy(envs[0])
        graphs = [self.normalizer.transform(env.graph()) for env in envs]
        contexts = [env.action_context() for env in envs]
        outputs = self.policy.act_batch(
            graphs, contexts, deterministic=self.deterministic,
        )
        return tuple(output.action for output in outputs)


__all__ = ["EGDMHGPPOEvaluationPolicy", "LearnedPolicyCheckpointError"]
