"""Phase H duration-aware PPO update for EGDM-HGPPO.

Implemented paper components:
- exact replay of the Phase G semantic composite-action trace;
- SMDP discount ``Gamma_e = gamma ** (delta_t / tau_0)``;
- duration-aware GAE;
- one PPO ratio for the exact composite action log-probability;
- three critic heads;
- value clipping, advantage normalization, entropy annealing;
- AdamW with a lower encoder learning rate and global gradient clipping.

Rollout orchestration and validation/checkpoint logging remain trainer-level
responsibilities. Historical curriculum masks are retained for compatibility;
formal Scheme-2 always uses the full component mask from iteration 1.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from math import isfinite
import time
from contextlib import nullcontext
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from agent.buffer import GAEOutput, HeadValues, RolloutBuffer
from agent.constraints import PolicyActionConstraints
from agent.policy import CompositePolicyOutput, EGDMCompositePolicy
from environment.action_context import ActionContext
from environment.graph_types import HeteroGraph


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    value_gate_loss: float
    value_rec_loss: float
    value_sch_loss: float
    gate_entropy: float
    resource_entropy: float
    schedule_entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    mean_ratio: float
    preclip_grad_norm: float
    learning_rate: float
    encoder_learning_rate: float
    gate_entropy_coef: float
    matching_entropy_coef: float
    transitions: int
    optimizer_steps: int


@dataclass(frozen=True, slots=True)
class PPOComponentMask:
    """Select which policy/value components participate in one PPO update.

    The default is the full EGDM-HGPPO objective from Phase H.  Phase I uses the
    scheduling-only mask for the paper's fixed-configuration warm-up curriculum:
    the gate/resource policy heads and their critics are excluded from both the
    PPO ratio and value loss rather than merely being force-sampled.
    """

    gate_policy: bool = True
    resource_policy: bool = True
    schedule_policy: bool = True
    gate_value: bool = True
    rec_value: bool = True
    sch_value: bool = True

    @classmethod
    def full(cls) -> "PPOComponentMask":
        return cls()

    @classmethod
    def scheduling_warmup(cls) -> "PPOComponentMask":
        return cls(
            gate_policy=False,
            resource_policy=False,
            schedule_policy=True,
            gate_value=False,
            rec_value=False,
            sch_value=True,
        )

    def validate(self) -> None:
        if not (self.gate_policy or self.resource_policy or self.schedule_policy):
            raise ValueError("at least one policy component must be active")
        if not (self.gate_value or self.rec_value or self.sch_value):
            raise ValueError("at least one value head must be active")



class _CrossEpochMaterializeCache:
    """Bounded CPU cache for exact compact-replay graph/context materializations.

    L1.7.5 diagnostic only.  Entries contain CPU semantic tensors with no autograd
    history.  The byte estimate is intentionally conservative: shared tensor
    storages referenced by multiple ActionContexts are counted again for each
    cached transition, so the configured budget cannot be exceeded by the
    estimate even when Python objects share immutable static tensors.
    """

    def __init__(self, *, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.entries: dict[int, tuple[HeteroGraph, ActionContext]] = {}
        self.estimated_bytes = 0
        self.admission_closed = self.max_bytes <= 0
        self.hits = 0
        self.misses = 0
        self.admissions = 0
        self.rejections = 0

    @staticmethod
    def _validate_parameter_independent_cpu_item(item) -> None:
        """Reject cached tensors that could retain a stale trainable graph."""
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], HeteroGraph)
            or not isinstance(item[1], ActionContext)
        ):
            raise TypeError(
                "materialize cache entries must be (HeteroGraph, ActionContext)"
            )

        seen: set[int] = set()

        def walk(value) -> None:
            oid = id(value)
            if oid in seen:
                return
            seen.add(oid)
            if torch.is_tensor(value):
                if value.device.type != "cpu":
                    raise ValueError("materialize cache may only retain CPU tensors")
                if value.requires_grad or value.grad_fn is not None:
                    raise ValueError(
                        "materialize cache may not retain autograd-dependent tensors"
                    )
                return
            if isinstance(value, Mapping):
                for child in value.values():
                    walk(child)
                return
            if isinstance(value, (tuple, list)):
                for child in value:
                    walk(child)
                return
            if is_dataclass(value):
                for field in fields(value):
                    walk(getattr(value, field.name))

        walk(item)

    @staticmethod
    def _tensor_bytes(obj) -> int:
        seen_objects: set[int] = set()
        seen_tensors: set[int] = set()

        def walk(value) -> int:
            oid = id(value)
            if oid in seen_objects:
                return 0
            seen_objects.add(oid)
            if torch.is_tensor(value):
                tid = id(value)
                if tid in seen_tensors:
                    return 0
                seen_tensors.add(tid)
                return int(value.numel() * value.element_size())
            if isinstance(value, Mapping):
                return sum(walk(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return sum(walk(v) for v in value)
            if is_dataclass(value):
                return sum(walk(getattr(value, f.name)) for f in fields(value))
            return 0

        return int(walk(obj))

    def get(self, index: int, loader) -> tuple[HeteroGraph, ActionContext]:
        index = int(index)
        cached = self.entries.get(index)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        item = loader(index)
        if not self.admission_closed:
            self._validate_parameter_independent_cpu_item(item)
            size = self._tensor_bytes(item)
            if self.estimated_bytes + size <= self.max_bytes:
                self.entries[index] = item
                self.estimated_bytes += int(size)
                self.admissions += 1
            else:
                # Freeze the admitted subset once the conservative budget is
                # reached.  This guarantees stable cross-epoch hits rather than
                # turning the cache into an LRU whose entries are evicted before
                # later shuffled epochs can reuse them.
                self.rejections += 1
                self.admission_closed = True
        elif index not in self.entries:
            self.rejections += 1
        return item

    def snapshot(self) -> dict[str, int | float | bool]:
        total = self.hits + self.misses
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "admissions": int(self.admissions),
            "rejections": int(self.rejections),
            "cached_transitions": int(len(self.entries)),
            "estimated_bytes": int(self.estimated_bytes),
            "estimated_mib": float(self.estimated_bytes / (1024.0 * 1024.0)),
            "max_bytes": int(self.max_bytes),
            "max_mib": float(self.max_bytes / (1024.0 * 1024.0)),
            "hit_rate": float(self.hits / total) if total else 0.0,
            "admission_closed": bool(self.admission_closed),
        }


class PPOAgent:
    """PPO optimizer around the Phase G composite policy."""

    def __init__(self, cfg, policy: EGDMCompositePolicy) -> None:
        self.cfg = cfg
        self.policy = policy
        self.clip_eps = float(cfg.algo.clip_eps)
        self.value_loss_coef = float(cfg.algo.value_loss_coef)
        self.ppo_epochs = int(cfg.algo.ppo_epochs)
        self.minibatch_size = int(cfg.algo.minibatch_size)
        self.max_grad_norm = float(cfg.algo.max_grad_norm)
        self.gamma = float(cfg.algo.gamma)
        self.tau_0_minutes = float(cfg.algo.tau_0_minutes)
        self.gae_lambda = float(cfg.algo.gae_lambda)
        # Ablation hook: the formal policy uses duration-aware SMDP GAE;
        # controlled ablations may disable the elapsed-time exponent while
        # keeping the same rollout/reward/PPO budget.
        self.duration_aware_gae = True

        impl = cfg.algo.ppo_implementation
        self.value_clip_eps = float(impl.value_clip_eps)
        self.advantage_fusion = str(impl.policy_advantage_fusion)
        # Phase L1.2 implementation choice: keep the paper's logical minibatch
        # size (512) and optimizer-step count unchanged, but split each logical
        # minibatch into smaller replay microbatches. Gradients are accumulated
        # with exact sample-count weighting before one optimizer.step().
        self.replay_microbatch_size = int(
            getattr(impl, "replay_microbatch_size", self.minibatch_size)
        )
        if self.replay_microbatch_size <= 0:
            raise ValueError("replay_microbatch_size must be positive")
        self.replay_microbatch_min_size = int(
            getattr(impl, "replay_microbatch_min_size", 1)
        )
        if not 0 < self.replay_microbatch_min_size <= self.replay_microbatch_size:
            raise ValueError("replay_microbatch_min_size must be within replay microbatch size")
        self.replay_microbatch_oom_fallback = bool(
            getattr(impl, "replay_microbatch_oom_fallback", False)
        )
        # Historical behavior also mutates replay_microbatch_size after OOM.
        # Keep that compatibility flag; formal Scheme-2 leaves the configured
        # target immutable and uses safe_replay_microbatch_cap across updates.
        self.replay_microbatch_persist_oom_fallback = True
        # Run-local execution cap.  This is deliberately separate from the
        # configured target: a repeated OOM on a fixed GPU should not make
        # every later update retry an unsafe microbatch, while the scientific
        # logical minibatch and optimizer-step count remain unchanged.
        self.safe_replay_microbatch_cap = int(self.replay_microbatch_size)
        self._safe_replay_microbatch_source_target = int(self.replay_microbatch_size)
        self.last_effective_replay_microbatch_size = int(self.replay_microbatch_size)
        self.last_oom_fallback_count = 0
        self.value_clipping = bool(cfg.algo.training_stabilization.value_target_clipping)
        self.advantage_normalization = bool(
            cfg.algo.training_stabilization.advantage_normalization
        )
        amp_cfg = getattr(impl, "mixed_precision", None)
        self.amp_enabled = bool(
            amp_cfg is not None
            and getattr(amp_cfg, "enabled", False)
            and torch.cuda.is_available()
        )
        self.amp_dtype = torch.bfloat16 if str(
            getattr(amp_cfg, "dtype", "bf16")
        ).lower() in {"bf16", "bfloat16"} else torch.float16

        self.base_lr = float(cfg.algo.learning_rate)
        self.base_encoder_lr = float(cfg.algo.encoder_learning_rate)
        self.final_lr_ratio = float(cfg.algo.lr_schedule.final_ratio)
        self.weight_decay = float(cfg.algo.weight_decay)

        # Keep every parameter in the optimizer from the start. Formal Scheme-2
        # trains all of them from iteration 1; legacy Phase-J compatibility may
        # still use ``requires_grad`` to freeze a subset.
        encoder_params = list(policy.representation.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [p for p in policy.parameters() if id(p) not in encoder_ids]
        if not encoder_params or not other_params:
            raise ValueError("PPO optimizer requires both encoder and non-encoder parameters")
        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.base_encoder_lr, "name": "encoder"},
                {"params": other_params, "lr": self.base_lr, "name": "policy_value"},
            ],
            weight_decay=self.weight_decay,
        )
        self.progress = 0.0
        self.gate_entropy_coef = float(cfg.algo.entropy.gate.start)
        self.matching_entropy_coef = float(cfg.algo.entropy.matching.start)
        self.set_progress(0.0)
        # L1.7.1 diagnostic-only timing switch.  The formal executor leaves this
        # False, so the paper objective and ordinary training path are unchanged.
        self.profile_update_timing = False
        self.last_update_profile: dict | None = None
        self.operator_profiler_step: Callable[[], None] | None = None
        # L1.7.5 diagnostic-only bounded CPU reuse of compact replay
        # materializations across PPO epochs. Formal training leaves this off.
        self.cross_epoch_materialize_cache = False
        self.materialize_cache_max_bytes = 0
        self.last_materialize_cache_stats: dict | None = None
        # Replay materialization has already passed graph/context validation.
        # Formal executors may opt out of repeating those scans per microbatch.
        self.trusted_replay_inputs = False

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    @staticmethod
    def _lerp(start: float, end: float, progress: float) -> float:
        return float(start + (end - start) * progress)

    def set_progress(self, progress: float) -> None:
        """Apply paper linear LR decay and per-head entropy annealing."""
        progress = float(max(0.0, min(1.0, progress)))
        self.progress = progress
        lr_ratio = self._lerp(1.0, self.final_lr_ratio, progress)
        for group in self.optimizer.param_groups:
            if group.get("name") == "encoder":
                group["lr"] = self.base_encoder_lr * lr_ratio
            else:
                group["lr"] = self.base_lr * lr_ratio
        self.gate_entropy_coef = self._lerp(
            float(self.cfg.algo.entropy.gate.start),
            float(self.cfg.algo.entropy.gate.end),
            progress,
        )
        self.matching_entropy_coef = self._lerp(
            float(self.cfg.algo.entropy.matching.start),
            float(self.cfg.algo.entropy.matching.end),
            progress,
        )

    def _amp_context(self):
        """Return CUDA autocast for inference/update forwards when enabled."""
        if self.amp_enabled:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    @torch.no_grad()
    def act(
        self,
        graph: HeteroGraph,
        context: ActionContext,
        *,
        deterministic: bool = False,
        force_gate: bool | None = None,
        constraints: PolicyActionConstraints | None = None,
    ) -> CompositePolicyOutput:
        self.policy.eval()
        with self._amp_context():
            return self.policy.act_batch(
                [graph], [context],
                deterministic=deterministic,
                force_gate=force_gate,
                constraints=constraints,
            )[0]

    @torch.no_grad()
    def act_batch(
        self,
        graphs: list[HeteroGraph] | tuple[HeteroGraph, ...],
        contexts: list[ActionContext] | tuple[ActionContext, ...],
        *,
        deterministic: bool = False,
        force_gate: bool | None = None,
        constraints: PolicyActionConstraints | list[PolicyActionConstraints] | tuple[PolicyActionConstraints, ...] | None = None,
        validate_inputs: bool = True,
    ) -> tuple[CompositePolicyOutput, ...]:
        self.policy.eval()
        with self._amp_context():
            return self.policy.act_batch(
                graphs, contexts,
                deterministic=deterministic,
                force_gate=force_gate,
                constraints=constraints,
                validate_inputs=validate_inputs,
            )

    @torch.no_grad()
    def value(self, graph: HeteroGraph) -> HeadValues:
        self.policy.eval()
        with self._amp_context():
            representation = self.policy.representation(graph.to(self.device))
        critics = representation.critics
        return HeadValues(
            float(critics.v_gate.squeeze().cpu()),
            float(critics.v_rec.squeeze().cpu()),
            float(critics.v_sch.squeeze().cpu()),
        )

    def _policy_advantage(
        self,
        gae: GAEOutput,
        reconfigure_mask: torch.Tensor,
        components: PPOComponentMask,
    ) -> torch.Tensor:
        if self.advantage_fusion != "mean_active_heads":
            raise ValueError(
                f"unsupported policy_advantage_fusion={self.advantage_fusion!r}"
            )
        components.validate()
        dtype = gae.advantage_gate.dtype
        numerator = torch.zeros_like(gae.advantage_gate)
        denominator = torch.zeros_like(gae.advantage_gate)
        if components.gate_policy:
            numerator = numerator + gae.advantage_gate
            denominator = denominator + 1.0
        if components.schedule_policy:
            numerator = numerator + gae.advantage_sch
            denominator = denominator + 1.0
        if components.resource_policy:
            active = reconfigure_mask.to(dtype=dtype)
            numerator = numerator + active * gae.advantage_rec
            denominator = denominator + active
        if torch.any(denominator <= 0):
            raise ValueError("policy component mask leaves an event with no active advantage")
        advantage = numerator / denominator
        if self.advantage_normalization and advantage.numel() > 1:
            mean = advantage.mean()
            std = advantage.std(unbiased=False)
            advantage = (advantage - mean) / std.clamp_min(1e-8)
        return advantage

    @staticmethod
    def _old_policy_log_prob(
        transitions,
        components: PPOComponentMask,
    ) -> torch.Tensor:
        values: list[float] = []
        for transition in transitions:
            total = 0.0
            if components.gate_policy:
                total += transition.old_gate_log_prob
            if components.resource_policy and transition.reconfiguration_active:
                total += transition.old_resource_log_prob
            if components.schedule_policy:
                total += transition.old_schedule_log_prob
            values.append(total)
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _new_policy_log_prob(
        output: CompositePolicyOutput,
        components: PPOComponentMask,
    ) -> torch.Tensor:
        total = output.total_log_prob.new_zeros(())
        if components.gate_policy:
            total = total + output.gate_log_prob
        if components.resource_policy and output.trace.reconfigure:
            total = total + output.worker_log_prob + output.robot_log_prob
        if components.schedule_policy:
            total = total + output.schedule_log_prob
        return total

    def _value_loss(
        self,
        new_value: torch.Tensor,
        old_value: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        plain = (new_value - target).square()
        if not self.value_clipping:
            return 0.5 * plain.mean()
        clipped_value = old_value + (new_value - old_value).clamp(
            -self.value_clip_eps, self.value_clip_eps
        )
        clipped = (clipped_value - target).square()
        return 0.5 * torch.maximum(plain, clipped).mean()

    def update(
        self,
        buffer: RolloutBuffer,
        *,
        bootstrap_values: Mapping[tuple[int, int], HeadValues] | None = None,
        progress: float | None = None,
        components: PPOComponentMask | None = None,
    ) -> PPOUpdateStats:
        if len(buffer) == 0:
            raise ValueError("cannot update PPO from an empty buffer")
        components = components or PPOComponentMask.full()
        components.validate()
        if progress is not None:
            self.set_progress(progress)

        # L1.7.1 uses the *same* update implementation as formal training and
        # merely enables timing around existing sections.  No alternate loss,
        # minibatch order, replay path, or optimizer path is introduced.
        profile_enabled = bool(getattr(self, "profile_update_timing", False))
        profile_timing: dict[str, float] = {}
        profile_counts: dict[str, int] = {
            "epochs": 0,
            "logical_minibatches": 0,
            "microbatches": 0,
            "materialized_transitions": 0,
        }
        epoch_profiles: list[dict] = []
        current_epoch_timing: dict[str, float] | None = None
        current_epoch_counts: dict[str, int] | None = None

        def _profile_sync() -> None:
            if profile_enabled and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        def _profile_start(*, sync: bool = False):
            if not profile_enabled:
                return None
            if sync:
                _profile_sync()
            return time.perf_counter()

        def _profile_end(name: str, started, *, sync: bool = False) -> None:
            if not profile_enabled or started is None:
                return
            if sync:
                _profile_sync()
            elapsed = time.perf_counter() - started
            profile_timing[name] = profile_timing.get(name, 0.0) + elapsed
            if current_epoch_timing is not None:
                current_epoch_timing[name] = current_epoch_timing.get(name, 0.0) + elapsed

        def _profile_count(name: str, amount: int = 1) -> None:
            if not profile_enabled:
                return
            profile_counts[name] = profile_counts.get(name, 0) + int(amount)
            if current_epoch_counts is not None:
                current_epoch_counts[name] = current_epoch_counts.get(name, 0) + int(amount)

        update_started = _profile_start(sync=True)
        setup_started = _profile_start()
        gae = buffer.compute_gae(
            gamma=self.gamma,
            tau_0_minutes=self.tau_0_minutes,
            gae_lambda=self.gae_lambda,
            duration_aware=bool(self.duration_aware_gae),
            bootstrap_values=bootstrap_values,
        )
        transitions = buffer.transitions
        n = len(transitions)
        reconfigure_mask = torch.tensor(
            [t.reconfiguration_active for t in transitions], dtype=torch.bool
        )
        advantage = self._policy_advantage(gae, reconfigure_mask, components)

        old_total = self._old_policy_log_prob(transitions, components)
        old_v_gate = torch.tensor([t.old_values.v_gate for t in transitions], dtype=torch.float32)
        old_v_rec = torch.tensor([t.old_values.v_rec for t in transitions], dtype=torch.float32)
        old_v_sch = torch.tensor([t.old_values.v_sch for t in transitions], dtype=torch.float32)
        # These arrays are tiny compared with the replay graphs, but are used
        # by every PPO epoch.  Move them once per update instead of issuing a
        # host-to-device copy for every logical minibatch.
        old_total_device = old_total.to(self.device)
        advantage_device = advantage.to(self.device)
        old_v_gate_device = old_v_gate.to(self.device)
        old_v_rec_device = old_v_rec.to(self.device)
        old_v_sch_device = old_v_sch.to(self.device)
        return_gate_device = gae.return_gate.to(self.device)
        return_rec_device = gae.return_rec.to(self.device)
        return_sch_device = gae.return_sch.to(self.device)
        _profile_end("gae_and_update_setup_seconds", setup_started)

        metric_names = [
            "policy_loss", "value_loss", "value_gate_loss", "value_rec_loss",
            "value_sch_loss", "gate_entropy", "resource_entropy",
            "schedule_entropy", "total_loss", "approx_kl", "clip_fraction",
            "mean_ratio", "preclip_grad_norm",
        ]
        metric_sum = {name: 0.0 for name in metric_names}
        metric_device_accumulators = {
            name: torch.zeros((), dtype=torch.float32, device=self.device)
            for name in metric_names
        }
        optimizer_steps = 0
        # Run-local OOM adaptation. A Scheme-2 update may fall back from an
        # unsafe execution microbatch and later updates start at the learned
        # safe cap. Legacy runs may additionally mutate the historical target.
        configured_replay_microbatch = int(self.replay_microbatch_size)
        if (
            configured_replay_microbatch != self._safe_replay_microbatch_source_target
            and self.safe_replay_microbatch_cap
            == self._safe_replay_microbatch_source_target
        ):
            # Honour an explicit pre-run execution override. Once an actual OOM
            # has lowered the safe cap, later updates do not reset it merely
            # because the immutable configured target is still larger.
            self.safe_replay_microbatch_cap = configured_replay_microbatch
        self._safe_replay_microbatch_source_target = configured_replay_microbatch
        self.safe_replay_microbatch_cap = min(
            int(self.safe_replay_microbatch_cap), int(self.replay_microbatch_size)
        )
        update_replay_microbatch_size = int(self.safe_replay_microbatch_cap)
        oom_fallback_count = 0
        self.policy.train()

        cache_enabled = bool(getattr(self, "cross_epoch_materialize_cache", False))
        if cache_enabled and not buffer.compact_storage:
            raise ValueError("cross-epoch materialize cache requires compact replay storage")
        materialize_cache = (
            _CrossEpochMaterializeCache(max_bytes=int(self.materialize_cache_max_bytes))
            if cache_enabled else None
        )
        if hasattr(self.policy, "set_replay_plan_cache"):
            self.policy.set_replay_plan_cache(cache_enabled)
        cache_epoch_stats: list[dict] = []
        cache_epoch_started: float | None = None
        cache_epoch_before: dict | None = None

        def accumulate_logical_minibatch(idx: torch.Tensor, logical_size: int, microbatch_size: int):
            """Accumulate the exact logical-minibatch objective without stepping."""
            zero_started = _profile_start(sync=True)
            self.optimizer.zero_grad(set_to_none=True)
            _profile_end("zero_grad_seconds", zero_started, sync=True)
            diagnostic_metric_names = tuple(
                name for name in metric_names if name != "preclip_grad_norm"
            )
            logical_metric_accumulators = {
                name: torch.zeros((), dtype=torch.float32, device=self.device)
                for name in diagnostic_metric_names
            }

            for micro_start in range(0, logical_size, microbatch_size):
                _profile_count("microbatches")
                micro_idx = idx[micro_start : micro_start + microbatch_size]
                indices = micro_idx.tolist()
                _profile_count("materialized_transitions", len(indices))

                materialize_started = _profile_start()
                if materialize_cache is None:
                    materialized = [buffer.materialize(i) for i in indices]
                else:
                    materialized = [materialize_cache.get(i, buffer.materialize) for i in indices]
                _profile_end("materialize_seconds", materialize_started)

                pack_started = _profile_start()
                # L1.5 keeps semantic graph IDs and exact feasibility contexts
                # on CPU. ``evaluate_traces_batch`` transfers only one batched
                # learnable graph to the accelerator, avoiding thousands of
                # implicit CUDA synchronizations from Python scalar mask reads.
                graphs = [item[0] for item in materialized]
                contexts = [item[1] for item in materialized]
                traces = [transitions[i].trace for i in indices]
                constraints = [transitions[i].constraints for i in indices]
                _profile_end("replay_python_pack_seconds", pack_started)

                evaluate_started = _profile_start(sync=True)
                with self._amp_context():
                    outputs = self.policy.evaluate_traces_batch(
                        graphs,
                        contexts,
                        traces,
                        constraints=constraints,
                        validate_inputs=not self.trusted_replay_inputs,
                    )
                _profile_end("evaluate_traces_seconds", evaluate_started, sync=True)

                loss_started = _profile_start(sync=True)
                new_total = torch.stack([
                    self._new_policy_log_prob(out, components) for out in outputs
                ])
                micro_idx_device = micro_idx.to(self.device)
                old_total_micro = old_total_device.index_select(0, micro_idx_device)
                advantage_micro = advantage_device.index_select(0, micro_idx_device)
                log_ratio = new_total - old_total_micro
                ratio = torch.exp(log_ratio)
                surrogate_1 = ratio * advantage_micro
                surrogate_2 = ratio.clamp(
                    1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * advantage_micro
                policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()

                v_gate = torch.cat([
                    out.representation.critics.v_gate for out in outputs
                ], dim=0)
                v_rec = torch.cat([
                    out.representation.critics.v_rec for out in outputs
                ], dim=0)
                v_sch = torch.cat([
                    out.representation.critics.v_sch for out in outputs
                ], dim=0)
                zero = v_sch.new_zeros(())
                value_gate_loss = (
                    self._value_loss(
                        v_gate,
                        old_v_gate_device.index_select(0, micro_idx_device),
                        return_gate_device.index_select(0, micro_idx_device),
                    )
                    if components.gate_value else zero
                )
                value_rec_loss = (
                    self._value_loss(
                        v_rec,
                        old_v_rec_device.index_select(0, micro_idx_device),
                        return_rec_device.index_select(0, micro_idx_device),
                    )
                    if components.rec_value else zero
                )
                value_sch_loss = (
                    self._value_loss(
                        v_sch,
                        old_v_sch_device.index_select(0, micro_idx_device),
                        return_sch_device.index_select(0, micro_idx_device),
                    )
                    if components.sch_value else zero
                )
                value_loss = value_gate_loss + value_rec_loss + value_sch_loss

                gate_entropy = (
                    torch.stack([out.gate_entropy for out in outputs]).mean()
                    if components.gate_policy else zero
                )
                resource_entropy = (
                    torch.stack([
                        out.worker_entropy + out.robot_entropy for out in outputs
                    ]).mean()
                    if components.resource_policy else zero
                )
                schedule_entropy = (
                    torch.stack([out.schedule_entropy for out in outputs]).mean()
                    if components.schedule_policy else zero
                )
                entropy_bonus = (
                    self.gate_entropy_coef * gate_entropy
                    + self.matching_entropy_coef * (resource_entropy + schedule_entropy)
                )
                total_loss = (
                    policy_loss + self.value_loss_coef * value_loss - entropy_bonus
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("PPO total loss became NaN/Inf")
                micro_weight = float(micro_idx.numel()) / float(logical_size)
                _profile_end("loss_tensor_build_seconds", loss_started, sync=True)

                backward_started = _profile_start(sync=True)
                (total_loss * micro_weight).backward()
                _profile_end("backward_seconds", backward_started, sync=True)

                metric_started = _profile_start(sync=True)
                with torch.no_grad():
                    approx_kl = (old_total_micro - new_total).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.clip_eps
                    ).float().mean()
                    values = {
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "value_gate_loss": value_gate_loss,
                        "value_rec_loss": value_rec_loss,
                        "value_sch_loss": value_sch_loss,
                        "gate_entropy": gate_entropy,
                        "resource_entropy": resource_entropy,
                        "schedule_entropy": schedule_entropy,
                        "total_loss": total_loss,
                        "approx_kl": approx_kl,
                        "clip_fraction": clip_fraction,
                        "mean_ratio": ratio.mean(),
                    }
                    for name, value in values.items():
                        weighted = value.detach().to(dtype=torch.float32) * micro_weight
                        logical_metric_accumulators[name].add_(weighted)
                _profile_end("metric_scalar_sync_seconds", metric_started)

                # Drop reconstructed O(n^2) graph edges and processing tensors
                # before materializing the next microbatch.
                cleanup_started = _profile_start()
                del outputs, graphs, contexts, materialized
                _profile_end("microbatch_cleanup_seconds", cleanup_started)

            # Keep the logical-minibatch average on the device.  The caller
            # accumulates one tensor per optimizer step and performs a single
            # D2H transfer after all epochs have completed.
            logical_metrics = {
                name: logical_metric_accumulators[name]
                for name in diagnostic_metric_names
            }
            return logical_metrics

        for epoch_index in range(self.ppo_epochs):
            if materialize_cache is not None:
                cache_epoch_started = time.perf_counter()
                cache_epoch_before = materialize_cache.snapshot()
            if profile_enabled:
                current_epoch_timing = {}
                current_epoch_counts = {
                    "logical_minibatches": 0,
                    "microbatches": 0,
                    "materialized_transitions": 0,
                }
                epoch_started = _profile_start(sync=True)
                profile_counts["epochs"] += 1
            permutation = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                _profile_count("logical_minibatches")
                idx = permutation[start : start + self.minibatch_size]
                logical_size = int(idx.numel())
                if logical_size == 0:
                    continue

                # Logical minibatch = paper minibatch (normally 512). A larger
                # replay microbatch only reduces execution overhead. If the new
                # GPU profile is too aggressive for a later L-scale batch, retry
                # the *same* logical minibatch at half size before optimizer.step.
                # This preserves the exact logical objective and optimizer-step count.
                microbatch_size = min(int(update_replay_microbatch_size), logical_size)
                while True:
                    try:
                        logical_metrics = accumulate_logical_minibatch(
                            idx, logical_size, microbatch_size
                        )
                        break
                    except RuntimeError as exc:
                        is_oom = (
                            self.device.type == "cuda"
                            and "out of memory" in str(exc).lower()
                        )
                        can_fallback = (
                            is_oom
                            and self.replay_microbatch_oom_fallback
                            and microbatch_size > self.replay_microbatch_min_size
                        )
                        if not can_fallback:
                            raise
                        self.optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        next_size = max(
                            int(self.replay_microbatch_min_size), microbatch_size // 2
                        )
                        if next_size >= microbatch_size:
                            raise
                        print(
                            f"[throughput] CUDA OOM at replay_microbatch={microbatch_size}; "
                            f"retrying same logical minibatch with {next_size}",
                            flush=True,
                        )
                        microbatch_size = next_size
                        update_replay_microbatch_size = min(
                            int(update_replay_microbatch_size), int(next_size)
                        )
                        self.safe_replay_microbatch_cap = min(
                            int(self.safe_replay_microbatch_cap), int(next_size)
                        )
                        oom_fallback_count += 1
                        if self.replay_microbatch_persist_oom_fallback:
                            self.replay_microbatch_size = int(next_size)

                grad_started = _profile_start(sync=True)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError("PPO gradient norm became NaN/Inf")
                _profile_end("grad_clip_seconds", grad_started, sync=True)

                step_started = _profile_start(sync=True)
                self.optimizer.step()
                _profile_end("optimizer_step_seconds", step_started, sync=True)

                post_started = _profile_start(sync=True)
                metric_device_accumulators["preclip_grad_norm"].add_(
                    torch.as_tensor(grad_norm, dtype=torch.float32, device=self.device)
                )
                for name, scalar in logical_metrics.items():
                    metric_device_accumulators[name].add_(scalar)
                optimizer_steps += 1
                _profile_end("post_step_metric_accumulation_seconds", post_started, sync=True)
                if self.operator_profiler_step is not None:
                    self.operator_profiler_step()

            if profile_enabled:
                _profile_sync()
                epoch_wall = time.perf_counter() - epoch_started
                epoch_profiles.append({
                    "epoch_index": int(epoch_index),
                    "wall_seconds": float(epoch_wall),
                    "timing_seconds": dict(current_epoch_timing or {}),
                    "counts": dict(current_epoch_counts or {}),
                })
                current_epoch_timing = None
                current_epoch_counts = None
            if materialize_cache is not None:
                after = materialize_cache.snapshot()
                before = cache_epoch_before or {"hits": 0, "misses": 0, "admissions": 0, "rejections": 0}
                cache_epoch_stats.append({
                    "epoch_index": int(epoch_index),
                    "wall_seconds": float(time.perf_counter() - (cache_epoch_started or time.perf_counter())),
                    "hits": int(after["hits"] - int(before.get("hits", 0))),
                    "misses": int(after["misses"] - int(before.get("misses", 0))),
                    "admissions": int(after["admissions"] - int(before.get("admissions", 0))),
                    "rejections": int(after["rejections"] - int(before.get("rejections", 0))),
                    "cached_transitions_end": int(after["cached_transitions"]),
                    "estimated_mib_end": float(after["estimated_mib"]),
                })

        self.last_effective_replay_microbatch_size = int(update_replay_microbatch_size)
        self.last_oom_fallback_count = int(oom_fallback_count)
        replay_plan_stats = {
            "enabled": bool(cache_enabled),
            "entries": int(len(getattr(self.policy, "_replay_plan_cache", {}))),
            "hits": int(getattr(self.policy, "replay_plan_cache_hits", 0)),
            "misses": int(getattr(self.policy, "replay_plan_cache_misses", 0)),
        }

        if materialize_cache is not None:
            final_cache = materialize_cache.snapshot()
            final_cache["epochs"] = cache_epoch_stats
            final_cache["rollout_transitions"] = int(n)
            final_cache["coverage"] = float(final_cache["cached_transitions"] / max(n, 1))
            final_cache["replay_plan_cache"] = replay_plan_stats
            self.last_materialize_cache_stats = final_cache
        else:
            self.last_materialize_cache_stats = None

        metric_values = torch.stack([
            metric_device_accumulators[name] for name in metric_names
        ]).detach().cpu().tolist()
        for name, value in zip(metric_names, metric_values, strict=True):
            value = float(value) / max(optimizer_steps, 1)
            if not isfinite(value):
                raise FloatingPointError(f"non-finite PPO metric: {name}")
            metric_sum[name] = value
        averaged = metric_sum
        encoder_lr = next(
            float(group["lr"]) for group in self.optimizer.param_groups
            if group.get("name") == "encoder"
        )
        main_lr = next(
            float(group["lr"]) for group in self.optimizer.param_groups
            if group.get("name") == "policy_value"
        )

        if profile_enabled:
            _profile_sync()
            total_seconds = time.perf_counter() - update_started
            profile_timing["update_total_seconds"] = float(total_seconds)
            nonoverlap_keys = (
                "gae_and_update_setup_seconds",
                "zero_grad_seconds",
                "materialize_seconds",
                "replay_python_pack_seconds",
                "evaluate_traces_seconds",
                "loss_tensor_build_seconds",
                "backward_seconds",
                "metric_scalar_sync_seconds",
                "microbatch_cleanup_seconds",
                "grad_clip_seconds",
                "optimizer_step_seconds",
                "post_step_metric_accumulation_seconds",
            )
            accounted = sum(float(profile_timing.get(k, 0.0)) for k in nonoverlap_keys)
            profile_timing["loop_shuffle_and_other_seconds"] = max(0.0, total_seconds - accounted)
            self.last_update_profile = {
                "timing_seconds": dict(profile_timing),
                "counts": dict(profile_counts),
                "epochs": epoch_profiles,
                "logical_minibatch_size": int(self.minibatch_size),
                "effective_replay_microbatch_size": int(update_replay_microbatch_size),
                "oom_fallback_count": int(oom_fallback_count),
                "optimizer_steps": int(optimizer_steps),
            }
        else:
            self.last_update_profile = None

        if hasattr(self.policy, "set_replay_plan_cache"):
            self.policy.set_replay_plan_cache(False)

        return PPOUpdateStats(
            **averaged,
            learning_rate=main_lr,
            encoder_learning_rate=encoder_lr,
            gate_entropy_coef=self.gate_entropy_coef,
            matching_entropy_coef=self.matching_entropy_coef,
            transitions=n,
            optimizer_steps=optimizer_steps,
        )


__all__ = ["PPOAgent", "PPOComponentMask", "PPOUpdateStats"]
