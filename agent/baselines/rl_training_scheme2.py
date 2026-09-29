"""Scheme-2 formal trainer for learned comparison policies.

This module changes only the comparison-policy architecture/action constraint.
It reuses the accepted Scheme-2 random-joint training distribution, fresh
instances per PPO iteration, L1.7.6 execution stack, fixed validation monitor,
best-checkpoint selection and resume behavior.
"""

from __future__ import annotations

from dataclasses import replace
import json

from agent.baselines.learned_variants import (
    LEARNED_BASELINE_METHODS,
    build_learned_baseline_policy,
)
from agent.baselines.rl_training import baseline_constraints
from agent.training.trainer_random_joint import RandomJointTrainer


class LearnedBaselineRandomJointTrainer(RandomJointTrainer):
    def __init__(self, *args, method: str, **kwargs) -> None:
        method = str(method)
        if method not in LEARNED_BASELINE_METHODS:
            raise ValueError(f"unsupported learned baseline: {method}")
        self.method = method
        super().__init__(*args, **kwargs)
        # The only action-space difference is the one defined by the baseline.
        # Distribution/components/event semantics remain the same as Scheme 2.
        self.joint_spec = replace(
            self.joint_spec,
            constraints=baseline_constraints(self.method),
        )

    def _diag(self, message: str) -> None:
        if self.diagnostics_enabled:
            print(f"[B-S2:{self.method}] {message}", flush=True)

    def _build_policy(self, reference_graph):
        return build_learned_baseline_policy(self.method, self.cfg, reference_graph)

    def _write_scheme2_protocol_audit(self) -> None:
        super()._write_scheme2_protocol_audit()
        path = self.run_dir / "scheme2_protocol.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update({
            "training_protocol": "teacher_scheme2_learned_baseline",
            "method": self.method,
            "comparison_policy_only": True,
            "same_environment_reward_event_step_and_ppo_budget": True,
            "action_constraints": (
                "flat_single_edge"
                if self.method in {"MLP-PPO", "HGT-PPO-Flat"}
                else "unconstrained_physical_feasibility_only"
            ),
        })
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["LearnedBaselineRandomJointTrainer"]
