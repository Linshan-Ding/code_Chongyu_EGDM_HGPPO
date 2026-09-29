"""Phase G masked autoregressive set decoders.

The paper uses three sequential set-matching policies:
1) worker -> cell;
2) update cell representation with the planned worker assignment;
3) robot -> cell;
4) operation -> cell scheduling.

Each decoder repeatedly scores currently feasible edges plus an explicit STOP token.
Selected sources/capacities are masked immediately.  The trace stores only semantic
choices, so Phase H PPO can recompute exactly the same log-probability later.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal

import torch
from torch import nn
from torch.distributions import Categorical

from agent.nn_utils import build_gelu_layernorm_mlp
from environment.action_context import ActionContext
from environment.graph_types import HeteroGraph
from environment.state import ResourceAssignment, ScheduleAssignment


STOP_TARGET = -2
UNCONFIGURED_TARGET = -1


class DecoderFeasibilityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MatchingChoice:
    source_id: int | None
    target_id: int

    @property
    def is_stop(self) -> bool:
        return self.source_id is None and self.target_id == STOP_TARGET

    @classmethod
    def stop(cls) -> "MatchingChoice":
        return cls(source_id=None, target_id=STOP_TARGET)


@dataclass(frozen=True, slots=True)
class MatchingTrace:
    decoder: str
    choices: tuple[MatchingChoice, ...]

    def validate(self) -> None:
        if not self.choices or not self.choices[-1].is_stop:
            raise ValueError(f"{self.decoder}: trace must terminate with STOP")
        if any(choice.is_stop for choice in self.choices[:-1]):
            raise ValueError(f"{self.decoder}: STOP may only appear at the end")


@dataclass(frozen=True, slots=True)
class MatchingResult:
    trace: MatchingTrace
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True, slots=True)
class ResourceReplayStep:
    candidates: tuple[MatchingChoice, ...]
    selected: int


@dataclass(frozen=True, slots=True)
class ResourceReplayPlan:
    decoder: str
    steps: tuple[ResourceReplayStep, ...]


@dataclass(frozen=True, slots=True)
class ScheduleEdge:
    operation_local: int
    operation_id: int
    cell_id: int
    processing_time: float


@dataclass(frozen=True, slots=True)
class ScheduleReplayStep:
    edges: tuple[ScheduleEdge, ...]
    selected: int
    stop_allowed: bool


@dataclass(frozen=True, slots=True)
class ScheduleReplayPlan:
    steps: tuple[ScheduleReplayStep, ...]


class PlannedConfiguration:
    """Mutable decoder-local copy of the planned resource configuration.

    The environment remains authoritative.  This object mirrors the exact same
    capacity/relocation semantics from the read-only ``ActionContext`` so masks can
    change after each autoregressive choice without mutating the simulator.
    """

    def __init__(self, context: ActionContext, *, validate_context: bool = True) -> None:
        if validate_context:
            context.validate()
        self.context = context
        self.worker_cfg = context.cell_worker.clone()
        self.robot_cfg = context.cell_robot.clone()
        self.worker_res = context.cell_reserved_worker.clone()
        self.robot_res = context.cell_reserved_robot.clone()
        self.worker_selected: set[int] = set()
        self.robot_selected: set[int] = set()
        self.schedule_ops: set[int] = set()
        self.schedule_cells: set[int] = set()
        self.schedule_workers: set[int] = set()
        self.schedule_robots: set[int] = set()
        self.planned_relocation_events = 0

    def clone(self) -> "PlannedConfiguration":
        other = object.__new__(PlannedConfiguration)
        other.context = self.context
        other.worker_cfg = self.worker_cfg.clone()
        other.robot_cfg = self.robot_cfg.clone()
        other.worker_res = self.worker_res.clone()
        other.robot_res = self.robot_res.clone()
        other.worker_selected = set(self.worker_selected)
        other.robot_selected = set(self.robot_selected)
        other.schedule_ops = set(self.schedule_ops)
        other.schedule_cells = set(self.schedule_cells)
        other.schedule_workers = set(self.schedule_workers)
        other.schedule_robots = set(self.schedule_robots)
        other.planned_relocation_events = int(self.planned_relocation_events)
        return other

    def _eventual_worker(self, cell_id: int) -> int:
        w = int(self.worker_cfg[cell_id])
        return w if w >= 0 else int(self.worker_res[cell_id])

    def _eventual_robot(self, cell_id: int) -> int:
        r = int(self.robot_cfg[cell_id])
        return r if r >= 0 else int(self.robot_res[cell_id])

    def eventual_worker_by_cell(self) -> torch.Tensor:
        values = self.worker_cfg.clone()
        missing = values < 0
        values[missing] = self.worker_res[missing]
        return values

    def eventual_robot_by_cell(self) -> torch.Tensor:
        values = self.robot_cfg.clone()
        missing = values < 0
        values[missing] = self.robot_res[missing]
        return values

    def _cell_eventual_hr_legal(self, cell_id: int) -> bool:
        w = self._eventual_worker(cell_id)
        r = self._eventual_robot(cell_id)
        if w < 0 or r < 0:
            return True
        stage = int(self.context.cell_stage[cell_id])
        return bool(self.context.hr_compatibility[w, r, stage])

    def _cell_immediate_executable(self, cell_id: int) -> bool:
        w = int(self.worker_cfg[cell_id])
        r = int(self.robot_cfg[cell_id])
        if w < 0 and r < 0:
            return False
        stage = int(self.context.cell_stage[cell_id])
        if w >= 0 and not bool(self.context.worker_skill[w, stage]):
            return False
        if r >= 0 and not bool(self.context.robot_capability[r, stage]):
            return False
        if w >= 0 and r >= 0 and not bool(self.context.hr_compatibility[w, r, stage]):
            return False
        return True

    def final_reconfiguration_valid(self) -> bool:
        for m in range(self.context.num_cells):
            if not self._cell_eventual_hr_legal(m):
                return False
        covered = [False] * self.context.num_stages
        for m in range(self.context.num_cells):
            if self._cell_immediate_executable(m):
                covered[int(self.context.cell_stage[m])] = True
        return all(covered)

    def _apply_resource(self, kind: Literal["worker", "robot"], rid: int, target: int) -> None:
        ctx = self.context
        if kind == "worker":
            if rid in self.worker_selected or not bool(ctx.worker_idle[rid]):
                raise DecoderFeasibilityError(f"worker {rid} is unavailable/reused")
            cfg, res = self.worker_cfg, self.worker_res
            configured = int(ctx.worker_configured_cell[rid])
            physical = int(ctx.worker_physical_cell[rid])
            compat = ctx.worker_skill[rid]
            relocation = ctx.worker_relocation_time[rid]
            selected = self.worker_selected
        else:
            if rid in self.robot_selected or not bool(ctx.robot_idle[rid]):
                raise DecoderFeasibilityError(f"robot {rid} is unavailable/reused")
            cfg, res = self.robot_cfg, self.robot_res
            configured = int(ctx.robot_configured_cell[rid])
            physical = int(ctx.robot_physical_cell[rid])
            compat = ctx.robot_capability[rid]
            relocation = ctx.robot_relocation_time[rid]
            selected = self.robot_selected

        if configured >= 0:
            if int(cfg[configured]) != rid:
                raise DecoderFeasibilityError("planned/current configuration mismatch")
            cfg[configured] = -1

        if target == UNCONFIGURED_TARGET:
            selected.add(rid)
            return
        if not 0 <= target < ctx.num_cells:
            raise DecoderFeasibilityError(f"invalid target cell {target}")
        if not bool(ctx.cell_idle[target]):
            raise DecoderFeasibilityError("target cell is busy")
        stage = int(ctx.cell_stage[target])
        if not bool(compat[stage]):
            raise DecoderFeasibilityError("resource-stage incompatibility")
        if int(cfg[target]) >= 0 or int(res[target]) >= 0:
            raise DecoderFeasibilityError("target resource capacity unavailable")

        tau = float(relocation[physical, target])
        if physical == target or tau <= 1e-12:
            cfg[target] = rid
        else:
            res[target] = rid
            self.planned_relocation_events += 1
        selected.add(rid)

    def apply_worker(self, rid: int, target: int) -> None:
        self._apply_resource("worker", int(rid), int(target))

    def apply_robot(self, rid: int, target: int) -> None:
        self._apply_resource("robot", int(rid), int(target))

    def resource_candidates(
        self,
        kind: Literal["worker", "robot"],
        *,
        prefix_feasible: bool,
    ) -> list[MatchingChoice]:
        ctx = self.context
        if kind == "worker":
            idle = ctx.worker_idle
            configured = ctx.worker_configured_cell
            physical = ctx.worker_physical_cell
            compat = ctx.worker_skill
            cfg, res = self.worker_cfg, self.worker_res
            selected = self.worker_selected
        else:
            idle = ctx.robot_idle
            configured = ctx.robot_configured_cell
            physical = ctx.robot_physical_cell
            compat = ctx.robot_capability
            cfg, res = self.robot_cfg, self.robot_res
            selected = self.robot_selected

        choices: list[MatchingChoice] = []
        for rid in range(int(idle.numel())):
            if rid in selected or not bool(idle[rid]):
                continue
            # Explicit temporary-unconfigured virtual target is available only for
            # currently configured resources; unconfigured resources may simply be
            # omitted before STOP and therefore remain unconfigured.
            if int(configured[rid]) >= 0:
                choice = MatchingChoice(rid, UNCONFIGURED_TARGET)
                if not prefix_feasible or self._choice_preserves_prefix(kind, choice):
                    choices.append(choice)
            for m in range(ctx.num_cells):
                if not bool(ctx.cell_idle[m]):
                    continue
                stage = int(ctx.cell_stage[m])
                if not bool(compat[rid, stage]):
                    continue
                # Own current slot is allowed as the paper's zero-migration stay edge.
                occupied = int(cfg[m])
                reserved = int(res[m])
                if occupied >= 0 and occupied != rid:
                    continue
                if reserved >= 0 and reserved != rid:
                    continue
                choice = MatchingChoice(rid, m)
                if not prefix_feasible or self._choice_preserves_prefix(kind, choice):
                    choices.append(choice)
        return choices

    def _choice_preserves_prefix(
        self,
        kind: Literal["worker", "robot"],
        choice: MatchingChoice,
    ) -> bool:
        trial = self.clone()
        try:
            if kind == "worker":
                trial.apply_worker(int(choice.source_id), choice.target_id)
            else:
                trial.apply_robot(int(choice.source_id), choice.target_id)
        except DecoderFeasibilityError:
            return False
        return trial.final_reconfiguration_valid()

    def schedule_candidates(self) -> list[ScheduleEdge]:
        ctx = self.context
        out: list[ScheduleEdge] = []
        for local in range(ctx.num_operations):
            op_id = int(ctx.operation_ids[local])
            if op_id in self.schedule_ops or not bool(ctx.operation_ready[local]):
                continue
            stage = int(ctx.operation_stage[local])
            for m in range(ctx.num_cells):
                if m in self.schedule_cells or not bool(ctx.cell_idle[m]):
                    continue
                if int(ctx.cell_stage[m]) != stage:
                    continue
                if int(self.worker_res[m]) >= 0 or int(self.robot_res[m]) >= 0:
                    continue
                w = int(self.worker_cfg[m])
                r = int(self.robot_cfg[m])
                p = float("inf")
                if w >= 0 and r < 0:
                    p = float(ctx.processing_h[local, w])
                elif w < 0 and r >= 0:
                    p = float(ctx.processing_r[local, r])
                elif w >= 0 and r >= 0:
                    p = float(ctx.processing_hr[local, w, r])
                if not isfinite(p):
                    continue
                if w >= 0 and (w in self.schedule_workers or not bool(ctx.worker_idle[w])):
                    continue
                if r >= 0 and (r in self.schedule_robots or not bool(ctx.robot_idle[r])):
                    continue
                out.append(ScheduleEdge(local, op_id, m, p))
        return out

    def schedule_stop_allowed(self) -> bool:
        """Whether ending scheduling now still permits physical time to advance.

        STOP is a policy action, but it is not feasible when no physical event is
        pending and the current composite action has neither scheduled work nor a
        real relocation.  In that situation ``env.step`` would deadlock instead of
        producing the paper's next SMDP event.  Phase I therefore treats this as a
        hard action-mask condition rather than adding a reward penalty.
        """
        if self.schedule_ops:
            return True
        if self.planned_relocation_events > 0:
            return True
        return bool(self.context.pending_physical_event[0])

    def apply_schedule(self, edge: ScheduleEdge) -> None:
        if edge.operation_id in self.schedule_ops or edge.cell_id in self.schedule_cells:
            raise DecoderFeasibilityError("schedule source/cell reused")
        w = int(self.worker_cfg[edge.cell_id])
        r = int(self.robot_cfg[edge.cell_id])
        if w >= 0 and w in self.schedule_workers:
            raise DecoderFeasibilityError("worker reused by schedule set")
        if r >= 0 and r in self.schedule_robots:
            raise DecoderFeasibilityError("robot reused by schedule set")
        self.schedule_ops.add(edge.operation_id)
        self.schedule_cells.add(edge.cell_id)
        if w >= 0:
            self.schedule_workers.add(w)
        if r >= 0:
            self.schedule_robots.add(r)


class _BaseAutoregressiveDecoder(nn.Module):
    def __init__(self, cfg, input_dim: int) -> None:
        super().__init__()
        hidden = [int(x) for x in cfg.algo.policy_value_mlp]
        self.scorer = build_gelu_layernorm_mlp(input_dim, hidden, 1)
        self.stop_scorer = build_gelu_layernorm_mlp(int(cfg.algo.embed_dim), hidden, 1)
        self.time_reference = float(cfg.algo.matching_decoder.scalar_time_reference_minutes)
        if self.time_reference <= 0:
            raise ValueError("matching scalar time reference must be positive")

        # Candidate scoring is kept switchable so archived diagnostic runs can
        # compare the original implementation with the tensorized equivalent.
        # The formal Scheme-2 runner enables this execution-only optimization
        # from its config; it does not alter candidate order or hard masks.
        self.tensorized_scoring_enabled = False

        # L1.7.4 diagnostic candidate. During PPO replay, candidate neural inputs
        # are immutable within one decoder invocation even though feasibility masks
        # shrink/change after each autoregressive choice.  When enabled, cache only
        # the neural score for an already-seen semantic edge (and STOP); candidate
        # generation and all hard feasibility checks still run every step.
        self.memoize_replay_candidate_scores = False

    @staticmethod
    def _select(logits: torch.Tensor, deterministic: bool) -> int:
        if logits.ndim != 1 or logits.numel() == 0:
            raise DecoderFeasibilityError("decoder received empty logits")
        if deterministic:
            return int(torch.argmax(logits).item())
        return int(Categorical(logits=logits).sample().item())

    @staticmethod
    def _step_stats(logits: torch.Tensor, selected: int) -> tuple[torch.Tensor, torch.Tensor]:
        dist = Categorical(logits=logits)
        index = torch.tensor(selected, dtype=torch.long, device=logits.device)
        return dist.log_prob(index), dist.entropy()

    def _sole_stop_zero(self, global_embedding: torch.Tensor) -> torch.Tensor:
        """Return exact zero while preserving AdamW's zero-gradient semantics."""
        zero = global_embedding.reshape(-1)[0] * 0.0
        for parameter in self.stop_scorer.parameters():
            zero = zero + parameter.reshape(-1)[0] * 0.0
        return zero

    @staticmethod
    def _batched_select_and_stats(
        logits_by_item: list[torch.Tensor],
        *,
        deterministic: bool,
    ) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
        """Sample ragged categorical rows with one device synchronization.

        Hard-mask construction remains independent for each environment. Padding
        only combines the resulting variable-width categorical distributions so a
        rollout wave does not synchronize CUDA once per environment and decoder
        step.
        """
        if not logits_by_item:
            raise DecoderFeasibilityError("decoder received an empty logits batch")
        if any(logits.ndim != 1 or logits.numel() == 0 for logits in logits_by_item):
            raise DecoderFeasibilityError("decoder received invalid ragged logits")
        device = logits_by_item[0].device
        if any(logits.device != device for logits in logits_by_item):
            raise ValueError("ragged decoder logits must share one device")
        dtype = logits_by_item[0].dtype
        for logits in logits_by_item[1:]:
            dtype = torch.promote_types(dtype, logits.dtype)
        logits_by_item = [logits.to(dtype=dtype) for logits in logits_by_item]
        widths = [int(logits.numel()) for logits in logits_by_item]
        padded = torch.full(
            (len(logits_by_item), max(widths)),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        for row, logits in enumerate(logits_by_item):
            padded[row, : logits.numel()] = logits
        dist = Categorical(logits=padded)
        selected_tensor = (
            torch.argmax(padded, dim=1)
            if deterministic else dist.sample()
        )
        log_probs = dist.log_prob(selected_tensor)
        entropies = dist.entropy()
        selected = [int(value) for value in selected_tensor.detach().cpu().tolist()]
        return (
            selected,
            [log_probs[index] for index in range(len(logits_by_item))],
            [entropies[index] for index in range(len(logits_by_item))],
        )


class ResourceCellAutoregressiveMatcher(_BaseAutoregressiveDecoder):
    """Worker/robot edge-set decoder with immediate source/capacity masking."""

    def __init__(self, cfg, *, kind: Literal["worker", "robot"]) -> None:
        self.kind = kind
        embed_dim = int(cfg.algo.embed_dim)
        # resource, cell, stage, global, normalized relocation, virtual flag
        super().__init__(cfg, input_dim=4 * embed_dim + 2)
        self.virtual_cell = nn.Parameter(torch.zeros(embed_dim))
        self.virtual_stage = nn.Parameter(torch.zeros(embed_dim))
        self.prefix_feasible = bool(cfg.algo.matching_decoder.prefix_feasible_reconfiguration)

    def _score_choices_legacy(
        self,
        *,
        choices: list[MatchingChoice],
        node_embeddings,
        cell_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        graph: HeteroGraph,
        planner: PlannedConfiguration,
        score_cache: dict[object, torch.Tensor] | None = None,
        index_maps: dict[str, dict[int, int]] | None = None,
    ) -> torch.Tensor:
        if graph.batch_size != 1:
            raise ValueError("Phase G variable-length decoding currently expects one graph")
        resource_h = node_embeddings[self.kind]
        resource_ids = graph.nodes[self.kind].ids.tolist()
        resource_local = {int(rid): i for i, rid in enumerate(resource_ids)}
        cell_ids = graph.nodes["cell"].ids.tolist()
        cell_local = {int(cid): i for i, cid in enumerate(cell_ids)}
        stage_h = node_embeddings["stage"]
        stage_ids = graph.nodes["stage"].ids.tolist()
        stage_local = {int(sid): i for i, sid in enumerate(stage_ids)}
        ctx = planner.context

        def build_row(choice: MatchingChoice) -> torch.Tensor:
            rid = int(choice.source_id)
            rh = resource_h[resource_local[rid]]
            if choice.target_id == UNCONFIGURED_TARGET:
                ch = self.virtual_cell
                sh = self.virtual_stage
                tau = 0.0
                virtual = 1.0
            else:
                m = choice.target_id
                ch = cell_embeddings[cell_local[m]]
                stage = int(ctx.cell_stage[m])
                sh = stage_h[stage_local[stage]]
                physical = int(
                    ctx.worker_physical_cell[rid]
                    if self.kind == "worker"
                    else ctx.robot_physical_cell[rid]
                )
                matrix = (
                    ctx.worker_relocation_time
                    if self.kind == "worker"
                    else ctx.robot_relocation_time
                )
                tau = float(matrix[rid, physical, m]) / self.time_reference
                virtual = 0.0
            scalar = torch.tensor([tau, virtual], dtype=rh.dtype, device=rh.device)
            return torch.cat([rh, ch, sh, global_embedding[0], scalar], dim=-1)

        if score_cache is None:
            rows = [build_row(choice) for choice in choices]
            if rows:
                edge_logits = self.scorer(torch.stack(rows)).squeeze(-1)
            else:
                edge_logits = torch.empty(
                    (0,), dtype=global_embedding.dtype, device=global_embedding.device
                )
            stop = self.stop_scorer(global_embedding).reshape(1)
            return torch.cat([edge_logits, stop], dim=0)

        missing = [choice for choice in choices if choice not in score_cache]
        if missing:
            missing_rows = [build_row(choice) for choice in missing]
            missing_logits = self.scorer(torch.stack(missing_rows)).squeeze(-1)
            for choice, value in zip(missing, missing_logits.unbind(0)):
                score_cache[choice] = value
        if choices:
            edge_logits = torch.stack([score_cache[choice] for choice in choices])
        else:
            edge_logits = torch.empty(
                (0,), dtype=global_embedding.dtype, device=global_embedding.device
            )
        stop_key = ("__stop__", self.kind)
        if stop_key not in score_cache:
            score_cache[stop_key] = self.stop_scorer(global_embedding).reshape(())
        stop = score_cache[stop_key].reshape(1)
        return torch.cat([edge_logits, stop], dim=0)

    @staticmethod
    def _index_maps(graph: HeteroGraph, resource_kind: str) -> dict[str, dict[int, int]]:
        """Build semantic-id to embedding-row maps once per decoder invocation."""
        return {
            "resource_local": {
                int(value): index
                for index, value in enumerate(graph.nodes[resource_kind].ids.tolist())
            },
            "cell_local": {
                int(value): index
                for index, value in enumerate(graph.nodes["cell"].ids.tolist())
            },
            "stage_local": {
                int(value): index
                for index, value in enumerate(graph.nodes["stage"].ids.tolist())
            },
        }

    def _score_choices_tensorized(
        self,
        *,
        choices: list[MatchingChoice],
        node_embeddings,
        cell_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        graph: HeteroGraph,
        planner: PlannedConfiguration,
        score_cache: dict[object, torch.Tensor] | None = None,
        index_maps: dict[str, dict[int, int]] | None = None,
    ) -> torch.Tensor:
        """Score all feasible resource edges with one batched tensor build.

        Hard feasibility and candidate ordering remain in ``PlannedConfiguration``.
        This method only replaces the old per-choice ``torch.tensor``/``cat``
        construction; the scorer and STOP logit are unchanged.
        """
        if graph.batch_size != 1:
            raise ValueError("Phase G variable-length decoding currently expects one graph")
        maps = index_maps or self._index_maps(graph, self.kind)

        def build_rows(rows: list[MatchingChoice]) -> torch.Tensor:
            features = self._choice_features_tensorized(
                choices=rows,
                node_embeddings=node_embeddings,
                cell_embeddings=cell_embeddings,
                global_embedding=global_embedding,
                planner=planner,
                index_maps=maps,
            )
            if not rows:
                return features.new_empty((0,))
            return self.scorer(features).squeeze(-1)

        if score_cache is None:
            edge_logits = build_rows(choices)
            stop = self.stop_scorer(global_embedding).reshape(1)
            return torch.cat([edge_logits, stop], dim=0)

        missing = [choice for choice in choices if choice not in score_cache]
        if missing:
            missing_logits = build_rows(missing)
            for choice, value in zip(missing, missing_logits.unbind(0)):
                score_cache[choice] = value
        if choices:
            edge_logits = torch.stack([score_cache[choice] for choice in choices])
        else:
            edge_logits = torch.empty(
                (0,), dtype=global_embedding.dtype, device=global_embedding.device
            )
        stop_key = ("__stop__", self.kind)
        if stop_key not in score_cache:
            score_cache[stop_key] = self.stop_scorer(global_embedding).reshape(())
        return torch.cat([edge_logits, score_cache[stop_key].reshape(1)], dim=0)

    def _choice_features_tensorized(
        self,
        *,
        choices: list[MatchingChoice],
        node_embeddings,
        cell_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        planner: PlannedConfiguration,
        index_maps: dict[str, dict[int, int]],
    ) -> torch.Tensor:
        resource_h = node_embeddings[self.kind]
        width = int(self.scorer[0].in_features)
        if not choices:
            return resource_h.new_empty((0, width))
        stage_h = node_embeddings["stage"]
        ctx = planner.context
        device = resource_h.device
        resource_indices = torch.tensor(
            [index_maps["resource_local"][int(c.source_id)] for c in choices],
            dtype=torch.long,
            device=device,
        )
        virtual_flags = torch.tensor(
            [c.target_id == UNCONFIGURED_TARGET for c in choices],
            dtype=torch.bool,
            device=device,
        )
        cell_indices = torch.tensor(
            [index_maps["cell_local"].get(int(c.target_id), 0) for c in choices],
            dtype=torch.long,
            device=device,
        )
        stage_indices = torch.tensor(
            [
                index_maps["stage_local"].get(
                    int(ctx.cell_stage[int(c.target_id)]), 0
                ) if c.target_id != UNCONFIGURED_TARGET else 0
                for c in choices
            ],
            dtype=torch.long,
            device=device,
        )
        resource_rows = resource_h.index_select(0, resource_indices)
        cell_rows = cell_embeddings.index_select(0, cell_indices)
        stage_rows = stage_h.index_select(0, stage_indices)
        virtual_cell = self.virtual_cell.to(dtype=resource_rows.dtype)
        virtual_stage = self.virtual_stage.to(dtype=resource_rows.dtype)
        cell_rows = torch.where(
            virtual_flags[:, None], virtual_cell[None, :], cell_rows
        )
        stage_rows = torch.where(
            virtual_flags[:, None], virtual_stage[None, :], stage_rows
        )
        matrix = (
            ctx.worker_relocation_time
            if self.kind == "worker" else ctx.robot_relocation_time
        )
        physical_cells = (
            ctx.worker_physical_cell
            if self.kind == "worker" else ctx.robot_physical_cell
        )
        scalar_values = torch.tensor(
            [
                (0.0, 1.0)
                if c.target_id == UNCONFIGURED_TARGET
                else (
                    float(
                        matrix[
                            int(c.source_id),
                            int(physical_cells[int(c.source_id)]),
                            int(c.target_id),
                        ]
                    ) / self.time_reference,
                    0.0,
                )
                for c in choices
            ],
            dtype=resource_rows.dtype,
            device=device,
        )
        return torch.cat(
            [
                resource_rows,
                cell_rows,
                stage_rows,
                global_embedding.expand(len(choices), -1),
                scalar_values,
            ],
            dim=-1,
        )

    def _score_choices(
        self,
        *,
        choices: list[MatchingChoice],
        node_embeddings,
        cell_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        graph: HeteroGraph,
        planner: PlannedConfiguration,
        score_cache: dict[object, torch.Tensor] | None = None,
        index_maps: dict[str, dict[int, int]] | None = None,
    ) -> torch.Tensor:
        if self.tensorized_scoring_enabled:
            return self._score_choices_tensorized(
                choices=choices,
                node_embeddings=node_embeddings,
                cell_embeddings=cell_embeddings,
                global_embedding=global_embedding,
                graph=graph,
                planner=planner,
                score_cache=score_cache,
                index_maps=index_maps,
            )
        return self._score_choices_legacy(
            choices=choices,
            node_embeddings=node_embeddings,
            cell_embeddings=cell_embeddings,
            global_embedding=global_embedding,
            graph=graph,
            planner=planner,
            score_cache=score_cache,
            index_maps=index_maps,
        )

    def _decode_or_replay(
        self,
        *,
        node_embeddings,
        cell_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        graph: HeteroGraph,
        planner: PlannedConfiguration,
        deterministic: bool,
        replay: MatchingTrace | None,
        max_changes: int | None = None,
    ) -> MatchingResult:
        choices_trace: list[MatchingChoice] = []
        log_prob = global_embedding.new_zeros(())
        entropy = global_embedding.new_zeros(())
        step = 0
        source_bound = (
            planner.context.num_workers if self.kind == "worker" else planner.context.num_robots
        )
        if max_changes is not None:
            max_changes = int(max_changes)
            if max_changes < 0:
                raise ValueError("max_changes must be non-negative or None")
        max_steps = source_bound + 1
        change_count = 0
        score_cache: dict[object, torch.Tensor] | None = (
            {} if replay is not None and bool(self.memoize_replay_candidate_scores) else None
        )
        index_maps = (
            self._index_maps(graph, self.kind)
            if self.tensorized_scoring_enabled else None
        )
        while True:
            candidates = planner.resource_candidates(
                self.kind, prefix_feasible=self.prefix_feasible
            )
            if max_changes is not None and change_count >= max_changes:
                configured = (
                    planner.context.worker_configured_cell
                    if self.kind == "worker" else planner.context.robot_configured_cell
                )
                candidates = [
                    c for c in candidates
                    if c.target_id == int(configured[int(c.source_id)])
                ]
            logits = self._score_choices(
                choices=candidates,
                node_embeddings=node_embeddings,
                cell_embeddings=cell_embeddings,
                global_embedding=global_embedding,
                graph=graph,
                planner=planner,
                score_cache=score_cache,
                index_maps=index_maps,
            )
            all_choices = candidates + [MatchingChoice.stop()]
            if replay is None:
                selected = self._select(logits, deterministic)
                choice = all_choices[selected]
            else:
                if step >= len(replay.choices):
                    raise DecoderFeasibilityError("replay trace ended before STOP")
                choice = replay.choices[step]
                try:
                    selected = all_choices.index(choice)
                except ValueError as exc:
                    raise DecoderFeasibilityError(
                        f"{self.kind} replay choice is no longer feasible: {choice}"
                    ) from exc
            lp, ent = self._step_stats(logits, selected)
            log_prob = log_prob + lp
            entropy = entropy + ent
            choices_trace.append(choice)
            if choice.is_stop:
                break
            configured_before = int(
                planner.context.worker_configured_cell[int(choice.source_id)]
                if self.kind == "worker"
                else planner.context.robot_configured_cell[int(choice.source_id)]
            )
            if choice.target_id != configured_before:
                change_count += 1
            if self.kind == "worker":
                planner.apply_worker(int(choice.source_id), choice.target_id)
            else:
                planner.apply_robot(int(choice.source_id), choice.target_id)
            step += 1
            if step >= max_steps:
                raise DecoderFeasibilityError("resource matcher exceeded finite source bound")

        trace = MatchingTrace(f"{self.kind}_cell", tuple(choices_trace))
        trace.validate()
        if replay is not None and trace != replay:
            raise DecoderFeasibilityError("replayed trace changed")
        return MatchingResult(trace=trace, log_prob=log_prob, entropy=entropy)

    def decode(self, **kwargs) -> MatchingResult:
        return self._decode_or_replay(replay=None, **kwargs)

    def replay(self, *, trace: MatchingTrace, **kwargs) -> MatchingResult:
        trace.validate()
        kwargs.pop("deterministic", None)
        return self._decode_or_replay(replay=trace, deterministic=True, **kwargs)

    def decode_batch(
        self,
        *,
        node_embeddings: list,
        cell_embeddings: list[torch.Tensor],
        global_embeddings: list[torch.Tensor],
        graphs: list[HeteroGraph],
        planners: list[PlannedConfiguration],
        deterministic: bool,
        max_changes: list[int | None],
    ) -> list[MatchingResult]:
        """Decode independent resource matchings in lockstep on one device batch."""
        count = len(planners)
        fields = (
            node_embeddings, cell_embeddings, global_embeddings, graphs,
            max_changes,
        )
        if count == 0 or any(len(value) != count for value in fields):
            raise ValueError("resource decode batch fields must be non-empty and aligned")

        maps = [self._index_maps(graph, self.kind) for graph in graphs]
        traces: list[list[MatchingChoice]] = [[] for _ in range(count)]
        log_probs = [global_h.new_zeros(()) for global_h in global_embeddings]
        entropies = [global_h.new_zeros(()) for global_h in global_embeddings]
        steps = [0] * count
        changes = [0] * count
        finished = [False] * count
        score_caches = [dict() for _ in range(count)] if self.memoize_replay_candidate_scores else None

        while not all(finished):
            active: list[int] = []
            candidates_by_item: dict[int, list[MatchingChoice]] = {}
            missing_by_item: dict[int, list[MatchingChoice]] = {}
            feature_parts: list[torch.Tensor] = []

            for index in range(count):
                if finished[index]:
                    continue
                planner = planners[index]
                candidates = planner.resource_candidates(
                    self.kind, prefix_feasible=self.prefix_feasible
                )
                limit = max_changes[index]
                if limit is not None:
                    limit = int(limit)
                    if limit < 0:
                        raise ValueError("max_changes must be non-negative or None")
                    if changes[index] >= limit:
                        configured = (
                            planner.context.worker_configured_cell
                            if self.kind == "worker"
                            else planner.context.robot_configured_cell
                        )
                        candidates = [
                            candidate for candidate in candidates
                            if candidate.target_id
                            == int(configured[int(candidate.source_id)])
                        ]
                missing = (
                    candidates
                    if score_caches is None
                    else [candidate for candidate in candidates if candidate not in score_caches[index]]
                )
                if missing:
                    feature_parts.append(self._choice_features_tensorized(
                        choices=missing,
                        node_embeddings=node_embeddings[index],
                        cell_embeddings=cell_embeddings[index],
                        global_embedding=global_embeddings[index],
                        planner=planner,
                        index_maps=maps[index],
                    ))
                active.append(index)
                candidates_by_item[index] = candidates
                missing_by_item[index] = missing

            scored = (
                self.scorer(torch.cat(feature_parts, dim=0)).squeeze(-1)
                if feature_parts
                else global_embeddings[active[0]].new_empty((0,))
            )
            cursor = 0
            fresh_scores: dict[int, torch.Tensor] = {}
            for index in active:
                missing = missing_by_item[index]
                part = scored[cursor:cursor + len(missing)]
                cursor += len(missing)
                if score_caches is None:
                    fresh_scores[index] = part
                else:
                    for candidate, value in zip(missing, part.unbind(0)):
                        score_caches[index][candidate] = value

            stop_indices = (
                active
                if score_caches is None
                else [index for index in active if ("__stop__", self.kind) not in score_caches[index]]
            )
            if stop_indices:
                stop_scores = self.stop_scorer(torch.cat(
                    [global_embeddings[index] for index in stop_indices], dim=0
                )).squeeze(-1)
                if score_caches is not None:
                    for index, value in zip(stop_indices, stop_scores.unbind(0)):
                        score_caches[index][("__stop__", self.kind)] = value
            else:
                stop_scores = None
            stop_position = {index: position for position, index in enumerate(stop_indices)}

            logits_by_item: list[torch.Tensor] = []
            for index in active:
                candidates = candidates_by_item[index]
                if score_caches is None:
                    candidate_scores = fresh_scores[index]
                    stop_score = stop_scores[stop_position[index]]
                else:
                    cache = score_caches[index]
                    candidate_scores = (
                        torch.stack([cache[candidate] for candidate in candidates])
                        if candidates else global_embeddings[index].new_empty((0,))
                    )
                    stop_score = cache[("__stop__", self.kind)]
                logits_by_item.append(torch.cat(
                    [candidate_scores, stop_score.reshape(1)], dim=0
                ))

            selected, step_log_probs, step_entropies = self._batched_select_and_stats(
                logits_by_item, deterministic=deterministic
            )
            for position, index in enumerate(active):
                candidates = candidates_by_item[index]
                all_choices = candidates + [MatchingChoice.stop()]
                choice = all_choices[selected[position]]
                log_probs[index] = log_probs[index] + step_log_probs[position]
                entropies[index] = entropies[index] + step_entropies[position]
                traces[index].append(choice)
                if choice.is_stop:
                    finished[index] = True
                    continue
                planner = planners[index]
                configured_before = int(
                    planner.context.worker_configured_cell[int(choice.source_id)]
                    if self.kind == "worker"
                    else planner.context.robot_configured_cell[int(choice.source_id)]
                )
                if choice.target_id != configured_before:
                    changes[index] += 1
                if self.kind == "worker":
                    planner.apply_worker(int(choice.source_id), choice.target_id)
                else:
                    planner.apply_robot(int(choice.source_id), choice.target_id)
                steps[index] += 1
                source_bound = (
                    planner.context.num_workers
                    if self.kind == "worker" else planner.context.num_robots
                )
                if steps[index] >= source_bound + 1:
                    raise DecoderFeasibilityError(
                        "resource matcher exceeded finite source bound"
                    )

        results = []
        for index in range(count):
            trace = MatchingTrace(f"{self.kind}_cell", tuple(traces[index]))
            trace.validate()
            results.append(MatchingResult(
                trace=trace,
                log_prob=log_probs[index],
                entropy=entropies[index],
            ))
        return results

    def prepare_replay_plan(
        self,
        *,
        planner: PlannedConfiguration,
        trace: MatchingTrace,
        max_changes: int | None,
    ) -> ResourceReplayPlan:
        """Precompute parameter-independent hard masks for one stored trace."""
        trace.validate()
        if trace.decoder != f"{self.kind}_cell":
            raise ValueError(f"unexpected trace decoder for {self.kind}: {trace.decoder}")
        trial = planner.clone()
        if max_changes is not None:
            max_changes = int(max_changes)
            if max_changes < 0:
                raise ValueError("max_changes must be non-negative or None")
        steps: list[ResourceReplayStep] = []
        change_count = 0
        for depth, choice in enumerate(trace.choices):
            candidates = trial.resource_candidates(
                self.kind, prefix_feasible=self.prefix_feasible
            )
            if max_changes is not None and change_count >= max_changes:
                configured = (
                    trial.context.worker_configured_cell
                    if self.kind == "worker" else trial.context.robot_configured_cell
                )
                candidates = [
                    candidate for candidate in candidates
                    if candidate.target_id
                    == int(configured[int(candidate.source_id)])
                ]
            all_choices = candidates + [MatchingChoice.stop()]
            try:
                selected = all_choices.index(choice)
            except ValueError as exc:
                raise DecoderFeasibilityError(
                    f"{self.kind} replay choice is no longer feasible: {choice}"
                ) from exc
            steps.append(ResourceReplayStep(tuple(candidates), int(selected)))
            if choice.is_stop:
                if depth != len(trace.choices) - 1:
                    raise DecoderFeasibilityError("resource replay continued after STOP")
                break
            configured_before = int(
                trial.context.worker_configured_cell[int(choice.source_id)]
                if self.kind == "worker"
                else trial.context.robot_configured_cell[int(choice.source_id)]
            )
            if choice.target_id != configured_before:
                change_count += 1
            if self.kind == "worker":
                trial.apply_worker(int(choice.source_id), choice.target_id)
            else:
                trial.apply_robot(int(choice.source_id), choice.target_id)
        return ResourceReplayPlan(
            decoder=f"{self.kind}_cell",
            steps=tuple(steps),
        )

    def _apply_replay_plan(
        self,
        planner: PlannedConfiguration,
        plan: ResourceReplayPlan,
    ) -> None:
        if plan.decoder != f"{self.kind}_cell":
            raise ValueError(f"unexpected prepared plan for {self.kind}")
        for step in plan.steps:
            if not 0 <= int(step.selected) <= len(step.candidates):
                raise DecoderFeasibilityError("prepared resource selection is invalid")
            if int(step.selected) == len(step.candidates):
                break
            choice = step.candidates[int(step.selected)]
            if self.kind == "worker":
                planner.apply_worker(int(choice.source_id), choice.target_id)
            else:
                planner.apply_robot(int(choice.source_id), choice.target_id)

    def replay_prepared_batch(
        self,
        *,
        node_embeddings: list,
        cell_embeddings: list[torch.Tensor],
        global_embeddings: list[torch.Tensor],
        graphs: list[HeteroGraph],
        planners: list[PlannedConfiguration],
        traces: list[MatchingTrace],
        plans: list[ResourceReplayPlan],
    ) -> list[MatchingResult]:
        """Replay cached hard-mask plans while recomputing all neural scores."""
        count = len(plans)
        fields = (
            node_embeddings, cell_embeddings, global_embeddings, graphs,
            planners, traces,
        )
        if count == 0 or any(len(value) != count for value in fields):
            raise ValueError("prepared resource replay fields must be aligned")
        maps = [self._index_maps(graph, self.kind) for graph in graphs]
        log_probs = [global_h.new_zeros(()) for global_h in global_embeddings]
        entropies = [global_h.new_zeros(()) for global_h in global_embeddings]
        score_caches = [dict() for _ in range(count)] if self.memoize_replay_candidate_scores else None
        max_depth = max(len(plan.steps) for plan in plans)

        for depth in range(max_depth):
            active = []
            for index, plan in enumerate(plans):
                if depth >= len(plan.steps):
                    continue
                step = plan.steps[depth]
                if not step.candidates:
                    if int(step.selected) != 0:
                        raise DecoderFeasibilityError(
                            "empty resource candidate set must select the sole STOP action"
                        )
                    # Categorical([STOP]) has log-probability and entropy exactly
                    # zero. Preserve zero-gradient (rather than grad=None) links
                    # so AdamW parameter/weight-decay behavior remains identical.
                    zero = self._sole_stop_zero(global_embeddings[index])
                    log_probs[index] = log_probs[index] + zero
                    entropies[index] = entropies[index] + zero
                    continue
                active.append(index)
            if not active:
                continue
            feature_parts: list[torch.Tensor] = []
            missing_by_item: dict[int, list[MatchingChoice]] = {}
            for index in active:
                candidates = list(plans[index].steps[depth].candidates)
                missing = (
                    candidates
                    if score_caches is None
                    else [candidate for candidate in candidates if candidate not in score_caches[index]]
                )
                if missing:
                    feature_parts.append(self._choice_features_tensorized(
                        choices=missing,
                        node_embeddings=node_embeddings[index],
                        cell_embeddings=cell_embeddings[index],
                        global_embedding=global_embeddings[index],
                        planner=planners[index],
                        index_maps=maps[index],
                    ))
                missing_by_item[index] = missing
            scored = (
                self.scorer(torch.cat(feature_parts, dim=0)).squeeze(-1)
                if feature_parts
                else global_embeddings[active[0]].new_empty((0,))
            )
            cursor = 0
            fresh_scores: dict[int, torch.Tensor] = {}
            for index in active:
                missing = missing_by_item[index]
                part = scored[cursor:cursor + len(missing)]
                cursor += len(missing)
                if score_caches is None:
                    fresh_scores[index] = part
                else:
                    for candidate, value in zip(missing, part.unbind(0)):
                        score_caches[index][candidate] = value
            stop_indices = (
                active
                if score_caches is None
                else [index for index in active if ("__stop__", self.kind) not in score_caches[index]]
            )
            if stop_indices:
                stop_scores = self.stop_scorer(torch.cat(
                    [global_embeddings[index] for index in stop_indices], dim=0
                )).squeeze(-1)
                if score_caches is not None:
                    for index, value in zip(stop_indices, stop_scores.unbind(0)):
                        score_caches[index][("__stop__", self.kind)] = value
            else:
                stop_scores = None
            stop_position = {index: position for position, index in enumerate(stop_indices)}
            for index in active:
                step = plans[index].steps[depth]
                candidates = list(step.candidates)
                if score_caches is None:
                    candidate_scores = fresh_scores[index]
                    stop_score = stop_scores[stop_position[index]]
                else:
                    cache = score_caches[index]
                    candidate_scores = (
                        torch.stack([cache[candidate] for candidate in candidates])
                        if candidates else global_embeddings[index].new_empty((0,))
                    )
                    stop_score = cache[("__stop__", self.kind)]
                logits = torch.cat([candidate_scores, stop_score.reshape(1)], dim=0)
                lp, entropy = self._step_stats(logits, int(step.selected))
                log_probs[index] = log_probs[index] + lp
                entropies[index] = entropies[index] + entropy

        for planner, plan in zip(planners, plans):
            self._apply_replay_plan(planner, plan)
        return [
            MatchingResult(trace=traces[index], log_prob=log_probs[index], entropy=entropies[index])
            for index in range(count)
        ]

    def replay_batch(
        self,
        *,
        node_embeddings: list,
        cell_embeddings: list[torch.Tensor],
        global_embeddings: list[torch.Tensor],
        graphs: list[HeteroGraph],
        planners: list[PlannedConfiguration],
        traces: list[MatchingTrace],
        max_changes: list[int | None],
    ) -> list[MatchingResult]:
        """Replay independent traces in lockstep and batch scorer invocations.

        The planners and hard-mask generation remain CPU-side and exact. Only
        candidate feature rows from the same autoregressive depth are combined
        before the shared neural scorer, avoiding one tiny GPU launch per event.
        """
        count = len(traces)
        fields = (
            node_embeddings, cell_embeddings, global_embeddings, graphs,
            planners, max_changes,
        )
        if count == 0 or any(len(value) != count for value in fields):
            raise ValueError("resource replay batch fields must be non-empty and aligned")
        for trace in traces:
            trace.validate()
            if trace.decoder != f"{self.kind}_cell":
                raise ValueError(f"unexpected trace decoder for {self.kind}: {trace.decoder}")

        maps = [self._index_maps(graph, self.kind) for graph in graphs]
        log_probs = [global_h.new_zeros(()) for global_h in global_embeddings]
        entropies = [global_h.new_zeros(()) for global_h in global_embeddings]
        steps = [0] * count
        changes = [0] * count
        finished = [False] * count
        score_caches = [dict() for _ in range(count)] if self.memoize_replay_candidate_scores else None

        while not all(finished):
            active: list[int] = []
            candidates_by_item: dict[int, list[MatchingChoice]] = {}
            selected_by_item: dict[int, int] = {}
            missing_by_item: dict[int, list[MatchingChoice]] = {}
            feature_parts: list[torch.Tensor] = []

            for index in range(count):
                if finished[index]:
                    continue
                planner = planners[index]
                candidates = planner.resource_candidates(
                    self.kind, prefix_feasible=self.prefix_feasible
                )
                limit = max_changes[index]
                if limit is not None and changes[index] >= int(limit):
                    configured = (
                        planner.context.worker_configured_cell
                        if self.kind == "worker"
                        else planner.context.robot_configured_cell
                    )
                    candidates = [
                        candidate for candidate in candidates
                        if candidate.target_id
                        == int(configured[int(candidate.source_id)])
                    ]
                step = steps[index]
                if step >= len(traces[index].choices):
                    raise DecoderFeasibilityError("replay trace ended before STOP")
                choice = traces[index].choices[step]
                all_choices = candidates + [MatchingChoice.stop()]
                try:
                    selected = all_choices.index(choice)
                except ValueError as exc:
                    raise DecoderFeasibilityError(
                        f"{self.kind} replay choice is no longer feasible: {choice}"
                    ) from exc
                if score_caches is None:
                    missing = candidates
                else:
                    missing = [c for c in candidates if c not in score_caches[index]]
                if missing:
                    feature_parts.append(self._choice_features_tensorized(
                        choices=missing,
                        node_embeddings=node_embeddings[index],
                        cell_embeddings=cell_embeddings[index],
                        global_embedding=global_embeddings[index],
                        planner=planner,
                        index_maps=maps[index],
                    ))
                active.append(index)
                candidates_by_item[index] = candidates
                selected_by_item[index] = selected
                missing_by_item[index] = missing

            scored = (
                self.scorer(torch.cat(feature_parts, dim=0)).squeeze(-1)
                if feature_parts
                else global_embeddings[active[0]].new_empty((0,))
            )
            cursor = 0
            fresh_scores: dict[int, torch.Tensor] = {}
            for index in active:
                missing = missing_by_item[index]
                part = scored[cursor:cursor + len(missing)]
                cursor += len(missing)
                if score_caches is None:
                    fresh_scores[index] = part
                else:
                    for candidate, value in zip(missing, part.unbind(0)):
                        score_caches[index][candidate] = value

            stop_indices = (
                active
                if score_caches is None
                else [i for i in active if ("__stop__", self.kind) not in score_caches[i]]
            )
            if stop_indices:
                stop_scores = self.stop_scorer(torch.cat(
                    [global_embeddings[i] for i in stop_indices], dim=0
                )).squeeze(-1)
                if score_caches is not None:
                    for index, value in zip(stop_indices, stop_scores.unbind(0)):
                        score_caches[index][("__stop__", self.kind)] = value
            else:
                stop_scores = None
            stop_position = {index: pos for pos, index in enumerate(stop_indices)}

            for index in active:
                candidates = candidates_by_item[index]
                if score_caches is None:
                    edge_scores = fresh_scores[index]
                    stop_score = stop_scores[stop_position[index]]
                else:
                    cache = score_caches[index]
                    edge_scores = (
                        torch.stack([cache[candidate] for candidate in candidates])
                        if candidates else global_embeddings[index].new_empty((0,))
                    )
                    stop_score = cache[("__stop__", self.kind)]
                logits = torch.cat([edge_scores, stop_score.reshape(1)], dim=0)
                selected = selected_by_item[index]
                lp, entropy = self._step_stats(logits, selected)
                log_probs[index] = log_probs[index] + lp
                entropies[index] = entropies[index] + entropy
                choice = traces[index].choices[steps[index]]
                if choice.is_stop:
                    finished[index] = True
                    continue
                planner = planners[index]
                configured_before = int(
                    planner.context.worker_configured_cell[int(choice.source_id)]
                    if self.kind == "worker"
                    else planner.context.robot_configured_cell[int(choice.source_id)]
                )
                if choice.target_id != configured_before:
                    changes[index] += 1
                if self.kind == "worker":
                    planner.apply_worker(int(choice.source_id), choice.target_id)
                else:
                    planner.apply_robot(int(choice.source_id), choice.target_id)
                steps[index] += 1
                source_bound = (
                    planner.context.num_workers
                    if self.kind == "worker" else planner.context.num_robots
                )
                if steps[index] >= source_bound + 1:
                    raise DecoderFeasibilityError(
                        "resource matcher exceeded finite source bound"
                    )

        return [
            MatchingResult(trace=trace, log_prob=log_probs[i], entropy=entropies[i])
            for i, trace in enumerate(traces)
        ]


class OperationCellAutoregressiveMatcher(_BaseAutoregressiveDecoder):
    """READY-operation -> idle-cell set decoder with explicit STOP."""

    def __init__(self, cfg) -> None:
        embed_dim = int(cfg.algo.embed_dim)
        # operation, cell, stage, global, p/tau0, slack/tau0
        super().__init__(cfg, input_dim=4 * embed_dim + 2)

    def _score_edges_legacy(
        self,
        *,
        edges: list[ScheduleEdge],
        node_embeddings,
        cell_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        graph: HeteroGraph,
        planner: PlannedConfiguration,
        score_cache: dict[object, torch.Tensor] | None = None,
        index_maps: dict[str, dict[int, int]] | None = None,
    ) -> torch.Tensor:
        op_h = node_embeddings["operation"]
        op_ids = graph.nodes["operation"].ids.tolist()
        op_local_by_id = {int(op_id): i for i, op_id in enumerate(op_ids)}
        cell_ids = graph.nodes["cell"].ids.tolist()
        cell_local = {int(cid): i for i, cid in enumerate(cell_ids)}
        stage_h = node_embeddings["stage"]
        stage_ids = graph.nodes["stage"].ids.tolist()
        stage_local = {int(sid): i for i, sid in enumerate(stage_ids)}
        ctx = planner.context

        def edge_key(edge: ScheduleEdge) -> tuple[int, int]:
            return (int(edge.operation_id), int(edge.cell_id))

        def build_row(edge: ScheduleEdge) -> torch.Tensor:
            ol = op_local_by_id[edge.operation_id]
            m = edge.cell_id
            stage = int(ctx.cell_stage[m])
            scalar = torch.tensor(
                [
                    edge.processing_time / self.time_reference,
                    float(ctx.operation_slack_minutes[edge.operation_local]) / self.time_reference,
                ],
                dtype=op_h.dtype,
                device=op_h.device,
            )
            return torch.cat(
                [
                    op_h[ol],
                    cell_embeddings[cell_local[m]],
                    stage_h[stage_local[stage]],
                    global_embedding[0],
                    scalar,
                ],
                dim=-1,
            )

        if score_cache is None:
            rows = [build_row(edge) for edge in edges]
            if rows:
                edge_logits = self.scorer(torch.stack(rows)).squeeze(-1)
            else:
                edge_logits = torch.empty(
                    (0,), dtype=global_embedding.dtype, device=global_embedding.device
                )
            stop = self.stop_scorer(global_embedding).reshape(1)
            return torch.cat([edge_logits, stop], dim=0)

        missing = [edge for edge in edges if edge_key(edge) not in score_cache]
        if missing:
            missing_rows = [build_row(edge) for edge in missing]
            missing_logits = self.scorer(torch.stack(missing_rows)).squeeze(-1)
            for edge, value in zip(missing, missing_logits.unbind(0)):
                score_cache[edge_key(edge)] = value
        if edges:
            edge_logits = torch.stack([score_cache[edge_key(edge)] for edge in edges])
        else:
            edge_logits = torch.empty(
                (0,), dtype=global_embedding.dtype, device=global_embedding.device
            )
        stop_key = ("__stop__", "schedule")
        if stop_key not in score_cache:
            score_cache[stop_key] = self.stop_scorer(global_embedding).reshape(())
        stop = score_cache[stop_key].reshape(1)
        return torch.cat([edge_logits, stop], dim=0)

    @staticmethod
    def _index_maps(graph: HeteroGraph) -> dict[str, dict[int, int]]:
        return {
            "operation_local": {
                int(value): index
                for index, value in enumerate(graph.nodes["operation"].ids.tolist())
            },
            "cell_local": {
                int(value): index
                for index, value in enumerate(graph.nodes["cell"].ids.tolist())
            },
            "stage_local": {
                int(value): index
                for index, value in enumerate(graph.nodes["stage"].ids.tolist())
            },
        }

    def _edge_features_tensorized(
        self,
        *,
        edges: list[ScheduleEdge],
        node_embeddings,
        cell_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        planner: PlannedConfiguration,
        index_maps: dict[str, dict[int, int]],
    ) -> torch.Tensor:
        op_h = node_embeddings["operation"]
        width = int(self.scorer[0].in_features)
        if not edges:
            return op_h.new_empty((0, width))
        stage_h = node_embeddings["stage"]
        ctx = planner.context
        device = op_h.device
        op_indices = torch.tensor(
            [index_maps["operation_local"][int(edge.operation_id)] for edge in edges],
            dtype=torch.long,
            device=device,
        )
        cell_indices = torch.tensor(
            [index_maps["cell_local"][int(edge.cell_id)] for edge in edges],
            dtype=torch.long,
            device=device,
        )
        stage_indices = torch.tensor(
            [
                index_maps["stage_local"][int(ctx.cell_stage[edge.cell_id])]
                for edge in edges
            ],
            dtype=torch.long,
            device=device,
        )
        scalar_values = torch.tensor(
            [
                (
                    float(edge.processing_time) / self.time_reference,
                    float(ctx.operation_slack_minutes[edge.operation_local])
                    / self.time_reference,
                )
                for edge in edges
            ],
            dtype=op_h.dtype,
            device=device,
        )
        return torch.cat(
            [
                op_h.index_select(0, op_indices),
                cell_embeddings.index_select(0, cell_indices),
                stage_h.index_select(0, stage_indices),
                global_embedding.expand(len(edges), -1),
                scalar_values,
            ],
            dim=-1,
        )

    def _score_edges_tensorized(
        self,
        *,
        edges: list[ScheduleEdge],
        node_embeddings,
        cell_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        graph: HeteroGraph,
        planner: PlannedConfiguration,
        score_cache: dict[object, torch.Tensor] | None = None,
        index_maps: dict[str, dict[int, int]] | None = None,
    ) -> torch.Tensor:
        """Build and score all schedule candidate rows in one tensor operation."""
        maps = index_maps or self._index_maps(graph)
        op_h = node_embeddings["operation"]
        stage_h = node_embeddings["stage"]
        ctx = planner.context

        def edge_key(edge: ScheduleEdge) -> tuple[int, int]:
            return (int(edge.operation_id), int(edge.cell_id))

        def build_rows(rows: list[ScheduleEdge]) -> torch.Tensor:
            features = self._edge_features_tensorized(
                edges=rows,
                node_embeddings=node_embeddings,
                cell_embeddings=cell_embeddings,
                global_embedding=global_embedding,
                planner=planner,
                index_maps=maps,
            )
            if not rows:
                return features.new_empty((0,))
            return self.scorer(features).squeeze(-1)

        if score_cache is None:
            edge_logits = build_rows(edges)
            stop = self.stop_scorer(global_embedding).reshape(1)
            return torch.cat([edge_logits, stop], dim=0)

        missing = [edge for edge in edges if edge_key(edge) not in score_cache]
        if missing:
            missing_logits = build_rows(missing)
            for edge, value in zip(missing, missing_logits.unbind(0)):
                score_cache[edge_key(edge)] = value
        if edges:
            edge_logits = torch.stack([score_cache[edge_key(edge)] for edge in edges])
        else:
            edge_logits = torch.empty(
                (0,), dtype=global_embedding.dtype, device=global_embedding.device
            )
        stop_key = ("__stop__", "schedule")
        if stop_key not in score_cache:
            score_cache[stop_key] = self.stop_scorer(global_embedding).reshape(())
        return torch.cat([edge_logits, score_cache[stop_key].reshape(1)], dim=0)

    def _score_edges(
        self,
        *,
        edges: list[ScheduleEdge],
        node_embeddings,
        cell_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        graph: HeteroGraph,
        planner: PlannedConfiguration,
        score_cache: dict[object, torch.Tensor] | None = None,
        index_maps: dict[str, dict[int, int]] | None = None,
    ) -> torch.Tensor:
        if self.tensorized_scoring_enabled:
            return self._score_edges_tensorized(
                edges=edges,
                node_embeddings=node_embeddings,
                cell_embeddings=cell_embeddings,
                global_embedding=global_embedding,
                graph=graph,
                planner=planner,
                score_cache=score_cache,
                index_maps=index_maps,
            )
        return self._score_edges_legacy(
            edges=edges,
            node_embeddings=node_embeddings,
            cell_embeddings=cell_embeddings,
            global_embedding=global_embedding,
            graph=graph,
            planner=planner,
            score_cache=score_cache,
            index_maps=index_maps,
        )

    def decode_batch(
        self,
        *,
        node_embeddings: list,
        cell_embeddings: list[torch.Tensor],
        global_embeddings: list[torch.Tensor],
        graphs: list[HeteroGraph],
        planners: list[PlannedConfiguration],
        deterministic: bool,
        max_assignments: list[int | None],
    ) -> list[MatchingResult]:
        """Decode independent schedule matchings in lockstep on one device batch."""
        count = len(planners)
        fields = (
            node_embeddings, cell_embeddings, global_embeddings, graphs,
            max_assignments,
        )
        if count == 0 or any(len(value) != count for value in fields):
            raise ValueError("schedule decode batch fields must be non-empty and aligned")

        maps = [self._index_maps(graph) for graph in graphs]
        traces: list[list[MatchingChoice]] = [[] for _ in range(count)]
        log_probs = [global_h.new_zeros(()) for global_h in global_embeddings]
        entropies = [global_h.new_zeros(()) for global_h in global_embeddings]
        steps = [0] * count
        selected_counts = [0] * count
        finished = [False] * count
        score_caches = [dict() for _ in range(count)] if self.memoize_replay_candidate_scores else None

        while not all(finished):
            active: list[int] = []
            edges_by_item: dict[int, list[ScheduleEdge]] = {}
            stop_allowed_by_item: dict[int, bool] = {}
            missing_by_item: dict[int, list[ScheduleEdge]] = {}
            feature_parts: list[torch.Tensor] = []

            for index in range(count):
                if finished[index]:
                    continue
                planner = planners[index]
                edges = planner.schedule_candidates()
                stop_allowed = planner.schedule_stop_allowed()
                limit = max_assignments[index]
                if limit is not None:
                    limit = int(limit)
                    if limit < 0:
                        raise ValueError("max_assignments must be non-negative or None")
                    if selected_counts[index] >= limit:
                        edges = []
                        stop_allowed = True
                if not edges and not stop_allowed:
                    raise DecoderFeasibilityError(
                        "no schedule edge is feasible and STOP would physically deadlock"
                    )
                missing = (
                    edges
                    if score_caches is None
                    else [
                        edge for edge in edges
                        if (int(edge.operation_id), int(edge.cell_id))
                        not in score_caches[index]
                    ]
                )
                if missing:
                    feature_parts.append(self._edge_features_tensorized(
                        edges=missing,
                        node_embeddings=node_embeddings[index],
                        cell_embeddings=cell_embeddings[index],
                        global_embedding=global_embeddings[index],
                        planner=planner,
                        index_maps=maps[index],
                    ))
                active.append(index)
                edges_by_item[index] = edges
                stop_allowed_by_item[index] = stop_allowed
                missing_by_item[index] = missing

            scored = (
                self.scorer(torch.cat(feature_parts, dim=0)).squeeze(-1)
                if feature_parts
                else global_embeddings[active[0]].new_empty((0,))
            )
            cursor = 0
            fresh_scores: dict[int, torch.Tensor] = {}
            for index in active:
                missing = missing_by_item[index]
                part = scored[cursor:cursor + len(missing)]
                cursor += len(missing)
                if score_caches is None:
                    fresh_scores[index] = part
                else:
                    for edge, value in zip(missing, part.unbind(0)):
                        score_caches[index][(int(edge.operation_id), int(edge.cell_id))] = value

            stop_indices = (
                active
                if score_caches is None
                else [
                    index for index in active
                    if ("__stop__", "schedule") not in score_caches[index]
                ]
            )
            if stop_indices:
                stop_scores = self.stop_scorer(torch.cat(
                    [global_embeddings[index] for index in stop_indices], dim=0
                )).squeeze(-1)
                if score_caches is not None:
                    for index, value in zip(stop_indices, stop_scores.unbind(0)):
                        score_caches[index][("__stop__", "schedule")] = value
            else:
                stop_scores = None
            stop_position = {index: position for position, index in enumerate(stop_indices)}

            logits_by_item: list[torch.Tensor] = []
            for index in active:
                edges = edges_by_item[index]
                if score_caches is None:
                    edge_scores = fresh_scores[index]
                    stop_score = stop_scores[stop_position[index]]
                else:
                    cache = score_caches[index]
                    edge_scores = (
                        torch.stack([
                            cache[(int(edge.operation_id), int(edge.cell_id))]
                            for edge in edges
                        ])
                        if edges else global_embeddings[index].new_empty((0,))
                    )
                    stop_score = cache[("__stop__", "schedule")]
                logits = torch.cat([edge_scores, stop_score.reshape(1)], dim=0)
                if not stop_allowed_by_item[index]:
                    logits = logits.clone()
                    logits[-1] = -1.0e9
                logits_by_item.append(logits)

            selected, step_log_probs, step_entropies = self._batched_select_and_stats(
                logits_by_item, deterministic=deterministic
            )
            for position, index in enumerate(active):
                edges = edges_by_item[index]
                all_choices = [
                    MatchingChoice(edge.operation_id, edge.cell_id) for edge in edges
                ] + [MatchingChoice.stop()]
                choice = all_choices[selected[position]]
                if choice.is_stop and not stop_allowed_by_item[index]:
                    raise DecoderFeasibilityError(
                        "schedule STOP is masked because it would deadlock"
                    )
                log_probs[index] = log_probs[index] + step_log_probs[position]
                entropies[index] = entropies[index] + step_entropies[position]
                traces[index].append(choice)
                if choice.is_stop:
                    finished[index] = True
                    continue
                planners[index].apply_schedule(edges[selected[position]])
                selected_counts[index] += 1
                steps[index] += 1
                max_steps = min(
                    planners[index].context.num_operations,
                    planners[index].context.num_cells,
                ) + 1
                if steps[index] >= max_steps:
                    raise DecoderFeasibilityError(
                        "schedule matcher exceeded finite source/cell bound"
                    )

        results = []
        for index in range(count):
            trace = MatchingTrace("operation_cell", tuple(traces[index]))
            trace.validate()
            results.append(MatchingResult(
                trace=trace,
                log_prob=log_probs[index],
                entropy=entropies[index],
            ))
        return results

    def prepare_replay_plan(
        self,
        *,
        planner: PlannedConfiguration,
        trace: MatchingTrace,
        max_assignments: int | None,
    ) -> ScheduleReplayPlan:
        """Precompute parameter-independent scheduling masks for one trace."""
        trace.validate()
        if trace.decoder != "operation_cell":
            raise ValueError(f"unexpected schedule trace decoder: {trace.decoder}")
        trial = planner.clone()
        if max_assignments is not None:
            max_assignments = int(max_assignments)
            if max_assignments < 0:
                raise ValueError("max_assignments must be non-negative or None")
        steps: list[ScheduleReplayStep] = []
        selected_count = 0
        for depth, choice in enumerate(trace.choices):
            edges = trial.schedule_candidates()
            stop_allowed = trial.schedule_stop_allowed()
            if max_assignments is not None and selected_count >= max_assignments:
                edges = []
                stop_allowed = True
            if not edges and not stop_allowed:
                raise DecoderFeasibilityError(
                    "no schedule edge is feasible and STOP would physically deadlock"
                )
            all_choices = [
                MatchingChoice(edge.operation_id, edge.cell_id) for edge in edges
            ] + [MatchingChoice.stop()]
            try:
                selected = all_choices.index(choice)
            except ValueError as exc:
                raise DecoderFeasibilityError(
                    f"schedule replay choice is no longer feasible: {choice}"
                ) from exc
            if choice.is_stop and not stop_allowed:
                raise DecoderFeasibilityError(
                    "schedule STOP is masked because it would deadlock"
                )
            steps.append(ScheduleReplayStep(
                edges=tuple(edges),
                selected=int(selected),
                stop_allowed=bool(stop_allowed),
            ))
            if choice.is_stop:
                if depth != len(trace.choices) - 1:
                    raise DecoderFeasibilityError("schedule replay continued after STOP")
                break
            trial.apply_schedule(edges[selected])
            selected_count += 1
        return ScheduleReplayPlan(steps=tuple(steps))

    @staticmethod
    def _apply_replay_plan(
        planner: PlannedConfiguration,
        plan: ScheduleReplayPlan,
    ) -> None:
        for step in plan.steps:
            if not 0 <= int(step.selected) <= len(step.edges):
                raise DecoderFeasibilityError("prepared schedule selection is invalid")
            if int(step.selected) == len(step.edges):
                break
            planner.apply_schedule(step.edges[int(step.selected)])

    def replay_prepared_batch(
        self,
        *,
        node_embeddings: list,
        cell_embeddings: list[torch.Tensor],
        global_embeddings: list[torch.Tensor],
        graphs: list[HeteroGraph],
        planners: list[PlannedConfiguration],
        traces: list[MatchingTrace],
        plans: list[ScheduleReplayPlan],
    ) -> list[MatchingResult]:
        """Replay cached schedule masks while recomputing current-policy logits."""
        count = len(plans)
        fields = (
            node_embeddings, cell_embeddings, global_embeddings, graphs,
            planners, traces,
        )
        if count == 0 or any(len(value) != count for value in fields):
            raise ValueError("prepared schedule replay fields must be aligned")
        maps = [self._index_maps(graph) for graph in graphs]
        log_probs = [global_h.new_zeros(()) for global_h in global_embeddings]
        entropies = [global_h.new_zeros(()) for global_h in global_embeddings]
        score_caches = [dict() for _ in range(count)] if self.memoize_replay_candidate_scores else None
        max_depth = max(len(plan.steps) for plan in plans)

        for depth in range(max_depth):
            active = []
            for index, plan in enumerate(plans):
                if depth >= len(plan.steps):
                    continue
                step = plan.steps[depth]
                if not step.edges:
                    if not step.stop_allowed or int(step.selected) != 0:
                        raise DecoderFeasibilityError(
                            "empty schedule candidate set must select feasible STOP"
                        )
                    # A sole feasible STOP action contributes zero log-probability
                    # and entropy. Retain zero-gradient links for AdamW equivalence.
                    zero = self._sole_stop_zero(global_embeddings[index])
                    log_probs[index] = log_probs[index] + zero
                    entropies[index] = entropies[index] + zero
                    continue
                active.append(index)
            if not active:
                continue
            feature_parts: list[torch.Tensor] = []
            missing_by_item: dict[int, list[ScheduleEdge]] = {}
            for index in active:
                edges = list(plans[index].steps[depth].edges)
                missing = (
                    edges
                    if score_caches is None
                    else [
                        edge for edge in edges
                        if (int(edge.operation_id), int(edge.cell_id))
                        not in score_caches[index]
                    ]
                )
                if missing:
                    feature_parts.append(self._edge_features_tensorized(
                        edges=missing,
                        node_embeddings=node_embeddings[index],
                        cell_embeddings=cell_embeddings[index],
                        global_embedding=global_embeddings[index],
                        planner=planners[index],
                        index_maps=maps[index],
                    ))
                missing_by_item[index] = missing
            scored = (
                self.scorer(torch.cat(feature_parts, dim=0)).squeeze(-1)
                if feature_parts
                else global_embeddings[active[0]].new_empty((0,))
            )
            cursor = 0
            fresh_scores: dict[int, torch.Tensor] = {}
            for index in active:
                missing = missing_by_item[index]
                part = scored[cursor:cursor + len(missing)]
                cursor += len(missing)
                if score_caches is None:
                    fresh_scores[index] = part
                else:
                    for edge, value in zip(missing, part.unbind(0)):
                        score_caches[index][(int(edge.operation_id), int(edge.cell_id))] = value
            stop_indices = (
                active
                if score_caches is None
                else [
                    index for index in active
                    if ("__stop__", "schedule") not in score_caches[index]
                ]
            )
            if stop_indices:
                stop_scores = self.stop_scorer(torch.cat(
                    [global_embeddings[index] for index in stop_indices], dim=0
                )).squeeze(-1)
                if score_caches is not None:
                    for index, value in zip(stop_indices, stop_scores.unbind(0)):
                        score_caches[index][("__stop__", "schedule")] = value
            else:
                stop_scores = None
            stop_position = {index: position for position, index in enumerate(stop_indices)}
            for index in active:
                step = plans[index].steps[depth]
                edges = list(step.edges)
                if score_caches is None:
                    edge_scores = fresh_scores[index]
                    stop_score = stop_scores[stop_position[index]]
                else:
                    cache = score_caches[index]
                    edge_scores = (
                        torch.stack([
                            cache[(int(edge.operation_id), int(edge.cell_id))]
                            for edge in edges
                        ])
                        if edges else global_embeddings[index].new_empty((0,))
                    )
                    stop_score = cache[("__stop__", "schedule")]
                logits = torch.cat([edge_scores, stop_score.reshape(1)], dim=0)
                if not step.stop_allowed:
                    logits = logits.clone()
                    logits[-1] = -1.0e9
                lp, entropy = self._step_stats(logits, int(step.selected))
                log_probs[index] = log_probs[index] + lp
                entropies[index] = entropies[index] + entropy

        for planner, plan in zip(planners, plans):
            self._apply_replay_plan(planner, plan)
        return [
            MatchingResult(trace=traces[index], log_prob=log_probs[index], entropy=entropies[index])
            for index in range(count)
        ]

    def replay_batch(
        self,
        *,
        node_embeddings: list,
        cell_embeddings: list[torch.Tensor],
        global_embeddings: list[torch.Tensor],
        graphs: list[HeteroGraph],
        planners: list[PlannedConfiguration],
        traces: list[MatchingTrace],
        max_assignments: list[int | None],
    ) -> list[MatchingResult]:
        """Replay schedule traces in lockstep with batched candidate scoring."""
        count = len(traces)
        fields = (
            node_embeddings, cell_embeddings, global_embeddings, graphs,
            planners, max_assignments,
        )
        if count == 0 or any(len(value) != count for value in fields):
            raise ValueError("schedule replay batch fields must be non-empty and aligned")
        for trace in traces:
            trace.validate()
            if trace.decoder != "operation_cell":
                raise ValueError(f"unexpected schedule trace decoder: {trace.decoder}")

        maps = [self._index_maps(graph) for graph in graphs]
        log_probs = [global_h.new_zeros(()) for global_h in global_embeddings]
        entropies = [global_h.new_zeros(()) for global_h in global_embeddings]
        steps = [0] * count
        selected_counts = [0] * count
        finished = [False] * count
        score_caches = [dict() for _ in range(count)] if self.memoize_replay_candidate_scores else None

        while not all(finished):
            active: list[int] = []
            edges_by_item: dict[int, list[ScheduleEdge]] = {}
            selected_by_item: dict[int, int] = {}
            stop_allowed_by_item: dict[int, bool] = {}
            missing_by_item: dict[int, list[ScheduleEdge]] = {}
            feature_parts: list[torch.Tensor] = []

            for index in range(count):
                if finished[index]:
                    continue
                planner = planners[index]
                edges = planner.schedule_candidates()
                stop_allowed = planner.schedule_stop_allowed()
                limit = max_assignments[index]
                if limit is not None and selected_counts[index] >= int(limit):
                    edges = []
                    stop_allowed = True
                if not edges and not stop_allowed:
                    raise DecoderFeasibilityError(
                        "no schedule edge is feasible and STOP would physically deadlock"
                    )
                step = steps[index]
                if step >= len(traces[index].choices):
                    raise DecoderFeasibilityError("schedule replay ended before STOP")
                choice = traces[index].choices[step]
                choices = [MatchingChoice(edge.operation_id, edge.cell_id) for edge in edges]
                all_choices = choices + [MatchingChoice.stop()]
                try:
                    selected = all_choices.index(choice)
                except ValueError as exc:
                    raise DecoderFeasibilityError(
                        f"schedule replay choice is no longer feasible: {choice}"
                    ) from exc
                if choice.is_stop and not stop_allowed:
                    raise DecoderFeasibilityError(
                        "schedule STOP is masked because it would deadlock"
                    )
                missing = (
                    edges
                    if score_caches is None
                    else [
                        edge for edge in edges
                        if (int(edge.operation_id), int(edge.cell_id))
                        not in score_caches[index]
                    ]
                )
                if missing:
                    feature_parts.append(self._edge_features_tensorized(
                        edges=missing,
                        node_embeddings=node_embeddings[index],
                        cell_embeddings=cell_embeddings[index],
                        global_embedding=global_embeddings[index],
                        planner=planner,
                        index_maps=maps[index],
                    ))
                active.append(index)
                edges_by_item[index] = edges
                selected_by_item[index] = selected
                stop_allowed_by_item[index] = stop_allowed
                missing_by_item[index] = missing

            scored = (
                self.scorer(torch.cat(feature_parts, dim=0)).squeeze(-1)
                if feature_parts
                else global_embeddings[active[0]].new_empty((0,))
            )
            cursor = 0
            fresh_scores: dict[int, torch.Tensor] = {}
            for index in active:
                missing = missing_by_item[index]
                part = scored[cursor:cursor + len(missing)]
                cursor += len(missing)
                if score_caches is None:
                    fresh_scores[index] = part
                else:
                    for edge, value in zip(missing, part.unbind(0)):
                        score_caches[index][(int(edge.operation_id), int(edge.cell_id))] = value

            stop_indices = (
                active
                if score_caches is None
                else [i for i in active if ("__stop__", "schedule") not in score_caches[i]]
            )
            if stop_indices:
                stop_scores = self.stop_scorer(torch.cat(
                    [global_embeddings[i] for i in stop_indices], dim=0
                )).squeeze(-1)
                if score_caches is not None:
                    for index, value in zip(stop_indices, stop_scores.unbind(0)):
                        score_caches[index][("__stop__", "schedule")] = value
            else:
                stop_scores = None
            stop_position = {index: pos for pos, index in enumerate(stop_indices)}

            for index in active:
                edges = edges_by_item[index]
                if score_caches is None:
                    edge_scores = fresh_scores[index]
                    stop_score = stop_scores[stop_position[index]]
                else:
                    cache = score_caches[index]
                    edge_scores = (
                        torch.stack([
                            cache[(int(edge.operation_id), int(edge.cell_id))]
                            for edge in edges
                        ])
                        if edges else global_embeddings[index].new_empty((0,))
                    )
                    stop_score = cache[("__stop__", "schedule")]
                logits = torch.cat([edge_scores, stop_score.reshape(1)], dim=0)
                if not stop_allowed_by_item[index]:
                    logits = logits.clone()
                    logits[-1] = -1.0e9
                lp, entropy = self._step_stats(logits, selected_by_item[index])
                log_probs[index] = log_probs[index] + lp
                entropies[index] = entropies[index] + entropy
                choice = traces[index].choices[steps[index]]
                if choice.is_stop:
                    finished[index] = True
                    continue
                selected_edge = edges[selected_by_item[index]]
                planners[index].apply_schedule(selected_edge)
                selected_counts[index] += 1
                steps[index] += 1
                max_steps = min(
                    planners[index].context.num_operations,
                    planners[index].context.num_cells,
                ) + 1
                if steps[index] >= max_steps:
                    raise DecoderFeasibilityError(
                        "schedule matcher exceeded finite source/cell bound"
                    )

        return [
            MatchingResult(trace=trace, log_prob=log_probs[i], entropy=entropies[i])
            for i, trace in enumerate(traces)
        ]

    def _decode_or_replay(
        self,
        *,
        node_embeddings,
        cell_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        graph: HeteroGraph,
        planner: PlannedConfiguration,
        deterministic: bool,
        replay: MatchingTrace | None,
        max_assignments: int | None = None,
    ) -> MatchingResult:
        trace_choices: list[MatchingChoice] = []
        log_prob = global_embedding.new_zeros(())
        entropy = global_embedding.new_zeros(())
        step = 0
        max_steps = min(planner.context.num_operations, planner.context.num_cells) + 1
        if max_assignments is not None:
            max_assignments = int(max_assignments)
            if max_assignments < 0:
                raise ValueError("max_assignments must be non-negative or None")
        selected_count = 0
        score_cache: dict[object, torch.Tensor] | None = (
            {} if replay is not None and bool(self.memoize_replay_candidate_scores) else None
        )
        index_maps = self._index_maps(graph) if self.tensorized_scoring_enabled else None
        while True:
            edges = planner.schedule_candidates()
            stop_allowed = planner.schedule_stop_allowed()
            if max_assignments is not None and selected_count >= max_assignments:
                edges = []
                stop_allowed = True
            if not edges and not stop_allowed:
                raise DecoderFeasibilityError(
                    "no schedule edge is feasible and STOP would physically deadlock"
                )
            logits = self._score_edges(
                edges=edges,
                node_embeddings=node_embeddings,
                cell_embeddings=cell_embeddings,
                global_embedding=global_embedding,
                graph=graph,
                planner=planner,
                score_cache=score_cache,
                index_maps=index_maps,
            )
            if not stop_allowed:
                logits = logits.clone()
                logits[-1] = -1.0e9
            choices = [MatchingChoice(e.operation_id, e.cell_id) for e in edges]
            all_choices = choices + [MatchingChoice.stop()]
            if replay is None:
                selected = self._select(logits, deterministic)
                choice = all_choices[selected]
            else:
                if step >= len(replay.choices):
                    raise DecoderFeasibilityError("schedule replay ended before STOP")
                choice = replay.choices[step]
                try:
                    selected = all_choices.index(choice)
                except ValueError as exc:
                    raise DecoderFeasibilityError(
                        f"schedule replay choice is no longer feasible: {choice}"
                    ) from exc
            if choice.is_stop and not stop_allowed:
                raise DecoderFeasibilityError("schedule STOP is masked because it would deadlock")
            lp, ent = self._step_stats(logits, selected)
            log_prob = log_prob + lp
            entropy = entropy + ent
            trace_choices.append(choice)
            if choice.is_stop:
                break
            edge = edges[selected]
            planner.apply_schedule(edge)
            selected_count += 1
            step += 1
            if step >= max_steps:
                raise DecoderFeasibilityError("schedule matcher exceeded finite source/cell bound")

        trace = MatchingTrace("operation_cell", tuple(trace_choices))
        trace.validate()
        if replay is not None and trace != replay:
            raise DecoderFeasibilityError("replayed schedule trace changed")
        return MatchingResult(trace=trace, log_prob=log_prob, entropy=entropy)

    def decode(self, **kwargs) -> MatchingResult:
        return self._decode_or_replay(replay=None, **kwargs)

    def replay(self, *, trace: MatchingTrace, **kwargs) -> MatchingResult:
        trace.validate()
        kwargs.pop("deterministic", None)
        return self._decode_or_replay(replay=trace, deterministic=True, **kwargs)


def trace_to_resource_assignments(trace: MatchingTrace) -> tuple[ResourceAssignment, ...]:
    trace.validate()
    return tuple(
        ResourceAssignment(int(choice.source_id), int(choice.target_id))
        for choice in trace.choices[:-1]
    )


def trace_to_schedule_assignments(trace: MatchingTrace) -> tuple[ScheduleAssignment, ...]:
    trace.validate()
    return tuple(
        ScheduleAssignment(int(choice.source_id), int(choice.target_id))
        for choice in trace.choices[:-1]
    )


__all__ = [
    "DecoderFeasibilityError",
    "MatchingChoice",
    "MatchingResult",
    "MatchingTrace",
    "OperationCellAutoregressiveMatcher",
    "PlannedConfiguration",
    "ResourceCellAutoregressiveMatcher",
    "STOP_TARGET",
    "UNCONFIGURED_TARGET",
    "trace_to_resource_assignments",
    "trace_to_schedule_assignments",
]
