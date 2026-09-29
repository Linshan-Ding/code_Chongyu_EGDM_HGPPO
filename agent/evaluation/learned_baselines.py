"""Checkpoint adapters for the four Phase-K2 learned comparison methods."""

from __future__ import annotations

from pathlib import Path
import torch

from agent.constraints import PolicyActionConstraints
from agent.baselines.base import DecisionPolicy
from agent.baselines.learned_variants import (
    LEARNED_BASELINE_METHODS, build_learned_baseline_policy,
)


class LearnedBaselineEvaluationPolicy(DecisionPolicy):
    def __init__(
        self, cfg, *, method: str, checkpoint: str | Path,
        device: torch.device | str = "cpu", deterministic: bool = True,
    ) -> None:
        if method not in LEARNED_BASELINE_METHODS:
            raise ValueError(f"unsupported learned baseline {method}")
        self.method_name = method
        self.cfg = cfg
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if "policy_state" not in payload or "normalizer" not in payload:
            raise ValueError("baseline checkpoint must contain policy_state and normalizer")
        ck_method = payload.get("method")
        if ck_method is not None and str(ck_method) != method:
            raise ValueError(f"checkpoint method={ck_method!r} does not match requested {method!r}")
        self.policy_state = payload["policy_state"]
        self.normalizer = payload["normalizer"]
        self.policy = None

    def reset(self, env) -> None:
        # Model is shape-generic after creation; no per-instance hidden state.
        pass

    def _constraints(self):
        if self.method_name in {"MLP-PPO", "HGT-PPO-Flat"}:
            return PolicyActionConstraints.flat_single_edge()
        return PolicyActionConstraints.unconstrained()

    def _ensure(self, env):
        if self.policy is not None:
            return
        graph = self.normalizer.transform(env.graph()).to(self.device)
        policy = build_learned_baseline_policy(self.method_name, self.cfg, graph).to(self.device)
        policy.load_state_dict(self.policy_state)
        policy.eval()
        self.policy = policy

    @torch.no_grad()
    def act(self, env):
        return self.act_batch((env,))[0]

    @torch.no_grad()
    def act_batch(self, envs):
        """Batch encoder inference while preserving each method's constraints."""
        envs = tuple(envs)
        if not envs:
            return ()
        self._ensure(envs[0])
        graphs = [self.normalizer.transform(env.graph()) for env in envs]
        contexts = [env.action_context() for env in envs]
        outputs = self.policy.act_batch(
            graphs,
            contexts,
            deterministic=self.deterministic,
            constraints=self._constraints(),
        )
        return tuple(output.action for output in outputs)


__all__ = ["LearnedBaselineEvaluationPolicy"]
