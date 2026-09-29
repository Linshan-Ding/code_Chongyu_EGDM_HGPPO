"""Composite actor built in Phase G and replayed by the Phase H PPO layer.

The actor remains separate from PPO update logic.  It provides exact semantic
action traces and batched shared-encoder replay for variable-length set decoders.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from agent.base import EGDMRepresentationNetwork, RepresentationOutput
from agent.constraints import PolicyActionConstraints
from agent.critic import CriticOutput
from agent.gate import GateOutput
from agent.pooling import PoolingOutput
from agent.matching import (
    DecoderFeasibilityError,
    MatchingResult,
    MatchingTrace,
    OperationCellAutoregressiveMatcher,
    PlannedConfiguration,
    ResourceReplayPlan,
    ResourceCellAutoregressiveMatcher,
    ScheduleReplayPlan,
    trace_to_resource_assignments,
    trace_to_schedule_assignments,
)
from environment.action_context import ActionContext
from environment.graph_types import HeteroGraph, batch_heterographs
from environment.state import CompositeAction


@dataclass(frozen=True, slots=True)
class PolicyTrace:
    reconfigure: bool
    worker: MatchingTrace | None
    robot: MatchingTrace | None
    schedule: MatchingTrace


@dataclass(frozen=True, slots=True)
class CompositePolicyOutput:
    action: CompositeAction
    trace: PolicyTrace
    representation: RepresentationOutput
    gate_log_prob: torch.Tensor
    worker_log_prob: torch.Tensor
    robot_log_prob: torch.Tensor
    schedule_log_prob: torch.Tensor
    total_log_prob: torch.Tensor
    gate_entropy: torch.Tensor
    worker_entropy: torch.Tensor
    robot_entropy: torch.Tensor
    schedule_entropy: torch.Tensor


class EGDMCompositePolicy(nn.Module):
    """Paper actor through Phase G, without PPO update logic."""

    def __init__(self, cfg, reference_graph: HeteroGraph) -> None:
        super().__init__()
        self.cfg = cfg
        self.representation = EGDMRepresentationNetwork(cfg, reference_graph)
        self.worker_matcher = ResourceCellAutoregressiveMatcher(cfg, kind="worker")
        self.robot_matcher = ResourceCellAutoregressiveMatcher(cfg, kind="robot")
        self.schedule_matcher = OperationCellAutoregressiveMatcher(cfg)
        embed_dim = int(cfg.algo.embed_dim)
        # Paper Eq. after worker matching: h_tilde_m = h_m + W_H h_h(m).
        self.worker_cell_update = nn.Linear(embed_dim, embed_dim, bias=False)
        # Scheduling occurs after both resource matchers.  Adding the planned robot
        # representation is an explicit implementation extension so the scheduling
        # scorer sees the resulting H/R/HR configuration, not only the old cell state.
        self.robot_cell_update = nn.Linear(embed_dim, embed_dim, bias=False)

        # L1.7.2 diagnostic-only candidate.  During compact PPO replay the
        # semantic graph remains on CPU and already carries the exact hard
        # reconfiguration-feasibility bit.  The legacy forced-gate path checks
        # the corresponding *GPU* masked logit with ``float(...)``, which
        # introduces one CUDA synchronization per replayed transition.  Keep
        # the legacy behavior by default; the pilot may opt into an equivalent
        # CPU-side hard-feasibility check.
        self.trusted_replay_gate_feasibility = False
        self.batched_rollout_decoding_enabled = False
        self.replay_plan_cache_enabled = False
        self._replay_plan_cache: dict[tuple, dict[str, object]] = {}
        self.replay_plan_cache_hits = 0
        self.replay_plan_cache_misses = 0

    def set_tensorized_decoder_scoring(self, enabled: bool) -> None:
        """Enable tensorized candidate scoring and lockstep rollout decoding."""
        value = bool(enabled)
        self.worker_matcher.tensorized_scoring_enabled = value
        self.robot_matcher.tensorized_scoring_enabled = value
        self.schedule_matcher.tensorized_scoring_enabled = value
        self.batched_rollout_decoding_enabled = value

    def set_replay_plan_cache(self, enabled: bool) -> None:
        """Bound parameter-independent replay plans to one PPO update."""
        self.replay_plan_cache_enabled = bool(enabled)
        self._replay_plan_cache.clear()
        self.replay_plan_cache_hits = 0
        self.replay_plan_cache_misses = 0

    def _replay_plan_entry(
        self,
        context: ActionContext,
        trace: PolicyTrace,
        constraint: PolicyActionConstraints,
    ) -> dict[str, object]:
        if not self.replay_plan_cache_enabled:
            return {}
        key = (id(context), trace, constraint)
        entry = self._replay_plan_cache.get(key)
        if entry is None:
            entry = {}
            self._replay_plan_cache[key] = entry
            self.replay_plan_cache_misses += 1
        else:
            self.replay_plan_cache_hits += 1
        return entry

    @staticmethod
    def _cache_parameter_independent_plan(entry: dict[str, object], key: str, plan) -> None:
        """Cache only semantic replay metadata, never trainable tensors."""
        if not isinstance(plan, (ResourceReplayPlan, ScheduleReplayPlan)):
            raise TypeError(
                "replay-plan cache accepts only parameter-independent plan types"
            )
        entry[str(key)] = plan

    @staticmethod
    def _node_local_map(graph: HeteroGraph, node_type: str) -> dict[int, int]:
        return {
            int(node_id): local
            for local, node_id in enumerate(graph.nodes[node_type].ids.tolist())
        }

    def _worker_updated_cells(
        self,
        *,
        representation: RepresentationOutput,
        graph: HeteroGraph,
        planner: PlannedConfiguration,
    ) -> torch.Tensor:
        cells = representation.node_embeddings["cell"].clone()
        workers = representation.node_embeddings["worker"]
        worker_local = self._node_local_map(graph, "worker")
        cell_local = self._node_local_map(graph, "cell")
        planned = planner.eventual_worker_by_cell()
        pairs = [
            (cell_local[cell_id], worker_local[int(worker_id)])
            for cell_id, worker_id in enumerate(planned.tolist())
            if int(worker_id) >= 0
        ]
        if pairs:
            cell_indices = torch.tensor(
                [pair[0] for pair in pairs], dtype=torch.long, device=cells.device
            )
            worker_indices = torch.tensor(
                [pair[1] for pair in pairs], dtype=torch.long, device=workers.device
            )
            updates = self.worker_cell_update(
                workers.index_select(0, worker_indices)
            ).to(dtype=cells.dtype)
            cells.index_add_(0, cell_indices, updates)
        return cells

    def _final_cells(
        self,
        *,
        worker_updated: torch.Tensor,
        representation: RepresentationOutput,
        graph: HeteroGraph,
        planner: PlannedConfiguration,
    ) -> torch.Tensor:
        cells = worker_updated.clone()
        robots = representation.node_embeddings["robot"]
        robot_local = self._node_local_map(graph, "robot")
        cell_local = self._node_local_map(graph, "cell")
        planned = planner.eventual_robot_by_cell()
        pairs = [
            (cell_local[cell_id], robot_local[int(robot_id)])
            for cell_id, robot_id in enumerate(planned.tolist())
            if int(robot_id) >= 0
        ]
        if pairs:
            cell_indices = torch.tensor(
                [pair[0] for pair in pairs], dtype=torch.long, device=cells.device
            )
            robot_indices = torch.tensor(
                [pair[1] for pair in pairs], dtype=torch.long, device=robots.device
            )
            updates = self.robot_cell_update(
                robots.index_select(0, robot_indices)
            ).to(dtype=cells.dtype)
            cells.index_add_(0, cell_indices, updates)
        return cells

    def _resource_updated_cells_batch(
        self,
        *,
        kind: str,
        base_cells: list[torch.Tensor],
        representations: list[RepresentationOutput],
        graphs: list[HeteroGraph],
        planners: list[PlannedConfiguration],
    ) -> list[torch.Tensor]:
        """Apply one batched resource-to-cell projection across a rollout wave."""
        if not (
            len(base_cells) == len(representations) == len(graphs) == len(planners)
        ):
            raise ValueError("resource cell-update batch fields are not aligned")
        if kind not in {"worker", "robot"}:
            raise ValueError("resource cell-update kind must be worker or robot")
        cell_offsets = [0]
        resource_offsets = [0]
        for cells, representation in zip(base_cells, representations):
            cell_offsets.append(cell_offsets[-1] + int(cells.shape[0]))
            resource_offsets.append(
                resource_offsets[-1]
                + int(representation.node_embeddings[kind].shape[0])
            )
        cells_all = torch.cat(base_cells, dim=0).clone()
        resources_all = torch.cat(
            [representation.node_embeddings[kind] for representation in representations],
            dim=0,
        )
        cell_indices: list[int] = []
        resource_indices: list[int] = []
        for index, (graph, planner) in enumerate(zip(graphs, planners)):
            cell_local = self._node_local_map(graph, "cell")
            resource_local = self._node_local_map(graph, kind)
            planned = (
                planner.eventual_worker_by_cell()
                if kind == "worker"
                else planner.eventual_robot_by_cell()
            )
            for cell_id, resource_id in enumerate(planned.tolist()):
                resource_id = int(resource_id)
                if resource_id < 0:
                    continue
                cell_indices.append(cell_offsets[index] + cell_local[cell_id])
                resource_indices.append(
                    resource_offsets[index] + resource_local[resource_id]
                )
        if cell_indices:
            cell_index_tensor = torch.tensor(
                cell_indices, dtype=torch.long, device=cells_all.device
            )
            resource_index_tensor = torch.tensor(
                resource_indices, dtype=torch.long, device=resources_all.device
            )
            projection = (
                self.worker_cell_update
                if kind == "worker" else self.robot_cell_update
            )
            updates = projection(
                resources_all.index_select(0, resource_index_tensor)
            ).to(dtype=cells_all.dtype)
            cells_all.index_add_(0, cell_index_tensor, updates)
        return [
            cells_all[cell_offsets[index]:cell_offsets[index + 1]]
            for index in range(len(base_cells))
        ]

    def _worker_updated_cells_batch(
        self,
        *,
        representations: list[RepresentationOutput],
        graphs: list[HeteroGraph],
        planners: list[PlannedConfiguration],
    ) -> list[torch.Tensor]:
        return self._resource_updated_cells_batch(
            kind="worker",
            base_cells=[
                representation.node_embeddings["cell"]
                for representation in representations
            ],
            representations=representations,
            graphs=graphs,
            planners=planners,
        )

    def _final_cells_batch(
        self,
        *,
        worker_updated: list[torch.Tensor],
        representations: list[RepresentationOutput],
        graphs: list[HeteroGraph],
        planners: list[PlannedConfiguration],
    ) -> list[torch.Tensor]:
        return self._resource_updated_cells_batch(
            kind="robot",
            base_cells=worker_updated,
            representations=representations,
            graphs=graphs,
            planners=planners,
        )

    @staticmethod
    def _gate_step(
        logits: torch.Tensor,
        *,
        deterministic: bool,
        forced: bool | None,
        validate_forced_logit: bool = True,
    ) -> tuple[bool, torch.Tensor, torch.Tensor]:
        if logits.shape != (1, 2):
            raise ValueError("Phase G action decoding expects one graph at a time")
        dist = Categorical(logits=logits[0])
        if forced is None:
            index = int(torch.argmax(logits[0]).item()) if deterministic else int(dist.sample().item())
        else:
            index = 1 if forced else 0
            if validate_forced_logit and float(logits[0, index].detach()) <= -1.0e8:
                raise DecoderFeasibilityError("forced gate action is masked infeasible")
        idx = torch.tensor(index, dtype=torch.long, device=logits.device)
        return bool(index), dist.log_prob(idx), dist.entropy()

    def _decode_from_representation(
        self,
        graph: HeteroGraph,
        context: ActionContext,
        representation: RepresentationOutput,
        *,
        deterministic: bool,
        force_gate: bool | None,
        replay_trace: PolicyTrace | None,
        constraints: PolicyActionConstraints | None = None,
    ) -> CompositePolicyOutput:
        constraints = constraints or PolicyActionConstraints.unconstrained()
        constraints.validate()
        if force_gate is not None and constraints.force_gate is not None and force_gate != constraints.force_gate:
            raise ValueError("force_gate conflicts with PolicyActionConstraints.force_gate")
        effective_force_gate = constraints.force_gate if force_gate is None else force_gate
        if replay_trace is None:
            reconfigure, gate_lp, gate_entropy = self._gate_step(
                representation.gate.masked_logits,
                deterministic=deterministic,
                forced=effective_force_gate,
            )
        else:
            if effective_force_gate is not None and bool(replay_trace.reconfigure) != bool(effective_force_gate):
                raise DecoderFeasibilityError("stored gate trace violates curriculum action constraints")
            trusted_gate = bool(getattr(self, "trusted_replay_gate_feasibility", False))
            if trusted_gate:
                # The semantic replay graph is deliberately retained on CPU by
                # ``evaluate_traces_batch``.  Reconfiguration feasibility is a
                # state-derived hard mask, independent of current policy
                # parameters.  Checking it here preserves the exact legacy
                # rejection semantics without synchronizing a scalar CUDA logit.
                if graph.reconfigure_feasible.device.type != "cpu":
                    raise ValueError(
                        "trusted replay gate feasibility requires the CPU semantic graph"
                    )
                if bool(replay_trace.reconfigure) and not bool(graph.reconfigure_feasible[0]):
                    raise DecoderFeasibilityError("forced gate action is masked infeasible")
            reconfigure, gate_lp, gate_entropy = self._gate_step(
                representation.gate.masked_logits,
                deterministic=True,
                forced=replay_trace.reconfigure,
                validate_forced_logit=not trusted_gate,
            )

        zero = representation.pooling.global_embedding.new_zeros(())
        planner = PlannedConfiguration(context, validate_context=False)
        worker_result: MatchingResult | None = None
        robot_result: MatchingResult | None = None

        if reconfigure:
            if replay_trace is not None and (replay_trace.worker is None or replay_trace.robot is None):
                raise DecoderFeasibilityError("reconfigure trace requires worker and robot traces")
            kwargs = dict(
                node_embeddings=representation.node_embeddings,
                cell_embeddings=representation.node_embeddings["cell"],
                global_embedding=representation.pooling.global_embedding,
                graph=graph,
                planner=planner,
                deterministic=deterministic,
                max_changes=constraints.max_worker_moves,
            )
            if replay_trace is None:
                worker_result = self.worker_matcher.decode(**kwargs)
            else:
                worker_result = self.worker_matcher.replay(
                    trace=replay_trace.worker,
                    **kwargs,
                )

            worker_updated = self._worker_updated_cells(
                representation=representation, graph=graph, planner=planner
            )
            robot_kwargs = dict(
                node_embeddings=representation.node_embeddings,
                cell_embeddings=worker_updated,
                global_embedding=representation.pooling.global_embedding,
                graph=graph,
                planner=planner,
                deterministic=deterministic,
                max_changes=constraints.max_robot_moves,
            )
            if replay_trace is None:
                robot_result = self.robot_matcher.decode(**robot_kwargs)
            else:
                robot_result = self.robot_matcher.replay(
                    trace=replay_trace.robot,
                    **robot_kwargs,
                )
            if not planner.final_reconfiguration_valid():
                raise DecoderFeasibilityError("decoded resource set violates final hard constraints")
        else:
            if replay_trace is not None and (replay_trace.worker is not None or replay_trace.robot is not None):
                raise DecoderFeasibilityError("KEEP trace must not contain resource traces")
            worker_updated = self._worker_updated_cells(
                representation=representation, graph=graph, planner=planner
            )

        final_cells = self._final_cells(
            worker_updated=worker_updated,
            representation=representation,
            graph=graph,
            planner=planner,
        )
        schedule_kwargs = dict(
            node_embeddings=representation.node_embeddings,
            cell_embeddings=final_cells,
            global_embedding=representation.pooling.global_embedding,
            graph=graph,
            planner=planner,
            deterministic=deterministic,
            max_assignments=constraints.max_schedule_assignments,
        )
        if replay_trace is None:
            schedule_result = self.schedule_matcher.decode(**schedule_kwargs)
        else:
            schedule_result = self.schedule_matcher.replay(
                trace=replay_trace.schedule,
                **schedule_kwargs,
            )

        worker_trace = None if worker_result is None else worker_result.trace
        robot_trace = None if robot_result is None else robot_result.trace
        trace = PolicyTrace(
            reconfigure=reconfigure,
            worker=worker_trace,
            robot=robot_trace,
            schedule=schedule_result.trace,
        )
        if replay_trace is not None and trace != replay_trace:
            raise DecoderFeasibilityError("composite replay trace changed")

        action = CompositeAction(
            reconfigure=reconfigure,
            worker_assignments=() if worker_trace is None else trace_to_resource_assignments(worker_trace),
            robot_assignments=() if robot_trace is None else trace_to_resource_assignments(robot_trace),
            schedule_assignments=trace_to_schedule_assignments(schedule_result.trace),
        )
        worker_lp = zero if worker_result is None else worker_result.log_prob
        robot_lp = zero if robot_result is None else robot_result.log_prob
        worker_ent = zero if worker_result is None else worker_result.entropy
        robot_ent = zero if robot_result is None else robot_result.entropy
        total_lp = gate_lp + worker_lp + robot_lp + schedule_result.log_prob
        return CompositePolicyOutput(
            action=action,
            trace=trace,
            representation=representation,
            gate_log_prob=gate_lp,
            worker_log_prob=worker_lp,
            robot_log_prob=robot_lp,
            schedule_log_prob=schedule_result.log_prob,
            total_log_prob=total_lp,
            gate_entropy=gate_entropy,
            worker_entropy=worker_ent,
            robot_entropy=robot_ent,
            schedule_entropy=schedule_result.entropy,
        )

    def _run(
        self,
        graph: HeteroGraph,
        context: ActionContext,
        *,
        deterministic: bool,
        force_gate: bool | None,
        replay_trace: PolicyTrace | None,
        constraints: PolicyActionConstraints | None = None,
        validate_inputs: bool = True,
    ) -> CompositePolicyOutput:
        if validate_inputs:
            graph.validate()
            context.validate()
        if graph.batch_size != 1:
            raise ValueError("single-action decoding expects one graph at a time")
        representation = self.representation(graph)
        return self._decode_from_representation(
            graph, context, representation,
            deterministic=deterministic,
            force_gate=force_gate,
            replay_trace=replay_trace,
            constraints=constraints,
        )

    @staticmethod
    def _slice_representation(
        batched: RepresentationOutput,
        batched_graph: HeteroGraph,
        index: int,
    ) -> RepresentationOutput:
        node_embeddings = {}
        for node_type, embeddings in batched.node_embeddings.items():
            ptr = batched_graph.nodes[node_type].ptr
            start = int(ptr[index].item())
            end = int(ptr[index + 1].item())
            node_embeddings[node_type] = embeddings[start:end]

        typed = {
            node_type: tensor[index:index + 1]
            for node_type, tensor in batched.pooling.typed_embeddings.items()
        }
        pooling = PoolingOutput(
            global_embedding=batched.pooling.global_embedding[index:index + 1],
            typed_embeddings=typed,
            operation_embedding=batched.pooling.operation_embedding[index:index + 1],
            stage_embedding=batched.pooling.stage_embedding[index:index + 1],
            cell_embedding=batched.pooling.cell_embedding[index:index + 1],
            resource_embedding=batched.pooling.resource_embedding[index:index + 1],
        )
        gate = GateOutput(
            raw_reconfigure_logit=batched.gate.raw_reconfigure_logit[index:index + 1],
            masked_logits=batched.gate.masked_logits[index:index + 1],
            probabilities=batched.gate.probabilities[index:index + 1],
        )
        critics = CriticOutput(
            v_gate=batched.critics.v_gate[index:index + 1],
            v_rec=batched.critics.v_rec[index:index + 1],
            v_sch=batched.critics.v_sch[index:index + 1],
            event_embedding=batched.critics.event_embedding[index:index + 1],
        )
        return RepresentationOutput(
            node_embeddings=node_embeddings,
            pooling=pooling,
            gate=gate,
            critics=critics,
        )

    def _batched_representation_from_semantic_graphs(
        self, graphs: list[HeteroGraph], *, validate_inputs: bool = True
    ) -> tuple[HeteroGraph, RepresentationOutput]:
        """Encode CPU semantic graphs in one accelerator batch.

        The autoregressive feasibility planner is intentionally CPU-side: it performs
        many Python ``int/bool/float`` reads from exact action-context tensors. Moving
        those semantic tensors to CUDA forces a device synchronization for every such
        scalar read.  Only the learnable heterogeneous graph is transferred to the
        model device; the original per-event graph/context stay on CPU for decoding.
        """
        if not graphs:
            raise ValueError("at least one semantic graph is required")
        semantic_batch = batch_heterographs(graphs, validate=validate_inputs)
        model_device = next(self.parameters()).device
        encoder_batch = semantic_batch.to(model_device, validate=validate_inputs)
        # The formal EGDM representation exposes ``validate_graph`` so replay can
        # skip a duplicate structural scan after batch_heterographs() validated the
        # input.  K2 comparison representations predate that keyword and retain a
        # one-argument forward contract; keep both implementations batch-compatible.
        if isinstance(self.representation, EGDMRepresentationNetwork):
            representation = self.representation(
                encoder_batch, validate_graph=validate_inputs
            )
        else:
            representation = self.representation(encoder_batch)
        return semantic_batch, representation

    def _evaluate_traces_tensorized_batch(
        self,
        *,
        graphs: list[HeteroGraph],
        contexts: list[ActionContext],
        traces: list[PolicyTrace],
        constraints: list[PolicyActionConstraints],
        semantic_batch: HeteroGraph,
        batched_representation: RepresentationOutput,
    ) -> tuple[CompositePolicyOutput, ...]:
        """Replay a microbatch with lockstep neural decoder scoring.

        Each planner still advances independently on CPU, so hard feasibility and
        the stored semantic trace are unchanged.  Decoder scorer calls are grouped
        by autoregressive depth and issued as one GPU batch.
        """
        count = len(traces)
        representations = [
            self._slice_representation(batched_representation, semantic_batch, index)
            for index in range(count)
        ]
        planners = [
            PlannedConfiguration(context, validate_context=False)
            for context in contexts
        ]
        plan_entries = [
            self._replay_plan_entry(context, trace, constraint)
            for context, trace, constraint in zip(contexts, traces, constraints)
        ]
        reconfigure_flags = [bool(trace.reconfigure) for trace in traces]

        for trace, constraint in zip(traces, constraints):
            trace.schedule.validate()
            constraint.validate()
            if constraint.force_gate is not None and bool(constraint.force_gate) != bool(trace.reconfigure):
                raise DecoderFeasibilityError(
                    "stored gate trace violates policy action constraints"
                )
            if trace.reconfigure and (trace.worker is None or trace.robot is None):
                raise DecoderFeasibilityError(
                    "reconfigure trace requires worker and robot traces"
                )
            if not trace.reconfigure and (trace.worker is not None or trace.robot is not None):
                raise DecoderFeasibilityError(
                    "KEEP trace must not contain resource traces"
                )

        gate_logits = torch.cat(
            [representation.gate.masked_logits for representation in representations],
            dim=0,
        )
        gate_indices = torch.tensor(
            [1 if flag else 0 for flag in reconfigure_flags],
            dtype=torch.long,
            device=gate_logits.device,
        )
        gate_dist = Categorical(logits=gate_logits)
        trusted_gate = bool(getattr(self, "trusted_replay_gate_feasibility", False))
        if trusted_gate:
            # The gate's only hard mask is the state-derived semantic bit already
            # validated on CPU above. Avoid synchronizing one CUDA scalar per item.
            for index, flag in enumerate(reconfigure_flags):
                if flag and not bool(graphs[index].reconfigure_feasible[0]):
                    raise DecoderFeasibilityError("forced gate action is masked infeasible")
        else:
            # Preserve the legacy numerical sentinel check, but reduce it to one
            # synchronization for the whole replay microbatch.
            selected_gate_logits = gate_logits.gather(
                1, gate_indices[:, None]
            ).squeeze(1)
            if bool(torch.any(selected_gate_logits.detach() <= -1.0e8)):
                raise DecoderFeasibilityError("forced gate action is masked infeasible")
        gate_log_probs = gate_dist.log_prob(gate_indices)
        gate_entropies = gate_dist.entropy()

        reconfigure_indices = [
            index for index, flag in enumerate(reconfigure_flags) if flag
        ]
        worker_results: list[MatchingResult | None] = [None] * count
        robot_results: list[MatchingResult | None] = [None] * count
        if reconfigure_indices:
            worker_plans: list[ResourceReplayPlan] = []
            for index in reconfigure_indices:
                plan = plan_entries[index].get("worker")
                if plan is None:
                    plan = self.worker_matcher.prepare_replay_plan(
                        planner=planners[index],
                        trace=traces[index].worker,
                        max_changes=constraints[index].max_worker_moves,
                    )
                    if self.replay_plan_cache_enabled:
                        self._cache_parameter_independent_plan(
                            plan_entries[index], "worker", plan
                        )
                worker_plans.append(plan)
            worker_batch = self.worker_matcher.replay_prepared_batch(
                node_embeddings=[representations[i].node_embeddings for i in reconfigure_indices],
                cell_embeddings=[representations[i].node_embeddings["cell"] for i in reconfigure_indices],
                global_embeddings=[representations[i].pooling.global_embedding for i in reconfigure_indices],
                graphs=[graphs[i] for i in reconfigure_indices],
                planners=[planners[i] for i in reconfigure_indices],
                traces=[traces[i].worker for i in reconfigure_indices],
                plans=worker_plans,
            )
            for index, result in zip(reconfigure_indices, worker_batch):
                worker_results[index] = result

        worker_updated = self._worker_updated_cells_batch(
            representations=representations,
            graphs=graphs,
            planners=planners,
        )
        if reconfigure_indices:
            robot_plans: list[ResourceReplayPlan] = []
            for index in reconfigure_indices:
                plan = plan_entries[index].get("robot")
                if plan is None:
                    plan = self.robot_matcher.prepare_replay_plan(
                        planner=planners[index],
                        trace=traces[index].robot,
                        max_changes=constraints[index].max_robot_moves,
                    )
                    if self.replay_plan_cache_enabled:
                        self._cache_parameter_independent_plan(
                            plan_entries[index], "robot", plan
                        )
                robot_plans.append(plan)
            robot_batch = self.robot_matcher.replay_prepared_batch(
                node_embeddings=[representations[i].node_embeddings for i in reconfigure_indices],
                cell_embeddings=[worker_updated[i] for i in reconfigure_indices],
                global_embeddings=[representations[i].pooling.global_embedding for i in reconfigure_indices],
                graphs=[graphs[i] for i in reconfigure_indices],
                planners=[planners[i] for i in reconfigure_indices],
                traces=[traces[i].robot for i in reconfigure_indices],
                plans=robot_plans,
            )
            for index, result in zip(reconfigure_indices, robot_batch):
                robot_results[index] = result
            for index in reconfigure_indices:
                if not planners[index].final_reconfiguration_valid():
                    raise DecoderFeasibilityError(
                        "decoded resource set violates final hard constraints"
                    )

        final_cells = self._final_cells_batch(
            worker_updated=worker_updated,
            representations=representations,
            graphs=graphs,
            planners=planners,
        )
        schedule_plans: list[ScheduleReplayPlan] = []
        for index in range(count):
            plan = plan_entries[index].get("schedule")
            if plan is None:
                plan = self.schedule_matcher.prepare_replay_plan(
                    planner=planners[index],
                    trace=traces[index].schedule,
                    max_assignments=constraints[index].max_schedule_assignments,
                )
                if self.replay_plan_cache_enabled:
                    self._cache_parameter_independent_plan(
                        plan_entries[index], "schedule", plan
                    )
            schedule_plans.append(plan)
        schedule_batch = self.schedule_matcher.replay_prepared_batch(
            node_embeddings=[representation.node_embeddings for representation in representations],
            cell_embeddings=final_cells,
            global_embeddings=[representation.pooling.global_embedding for representation in representations],
            graphs=graphs,
            planners=planners,
            traces=[trace.schedule for trace in traces],
            plans=schedule_plans,
        )

        outputs: list[CompositePolicyOutput] = []
        for index, trace in enumerate(traces):
            worker_result = worker_results[index]
            robot_result = robot_results[index]
            schedule_result = schedule_batch[index]
            worker_trace = None if worker_result is None else worker_result.trace
            robot_trace = None if robot_result is None else robot_result.trace
            replayed_trace = PolicyTrace(
                reconfigure=reconfigure_flags[index],
                worker=worker_trace,
                robot=robot_trace,
                schedule=schedule_result.trace,
            )
            if replayed_trace != trace:
                raise DecoderFeasibilityError("composite replay trace changed")
            zero = representations[index].pooling.global_embedding.new_zeros(())
            worker_lp = zero if worker_result is None else worker_result.log_prob
            robot_lp = zero if robot_result is None else robot_result.log_prob
            worker_ent = zero if worker_result is None else worker_result.entropy
            robot_ent = zero if robot_result is None else robot_result.entropy
            total_lp = (
                gate_log_probs[index]
                + worker_lp
                + robot_lp
                + schedule_result.log_prob
            )
            outputs.append(CompositePolicyOutput(
                action=CompositeAction(
                    reconfigure=reconfigure_flags[index],
                    worker_assignments=(
                        () if worker_trace is None
                        else trace_to_resource_assignments(worker_trace)
                    ),
                    robot_assignments=(
                        () if robot_trace is None
                        else trace_to_resource_assignments(robot_trace)
                    ),
                    schedule_assignments=trace_to_schedule_assignments(
                        schedule_result.trace
                    ),
                ),
                trace=replayed_trace,
                representation=representations[index],
                gate_log_prob=gate_log_probs[index],
                worker_log_prob=worker_lp,
                robot_log_prob=robot_lp,
                schedule_log_prob=schedule_result.log_prob,
                total_log_prob=total_lp,
                gate_entropy=gate_entropies[index],
                worker_entropy=worker_ent,
                robot_entropy=robot_ent,
                schedule_entropy=schedule_result.entropy,
            ))
        return tuple(outputs)

    def _decode_tensorized_rollout_batch(
        self,
        *,
        graphs: list[HeteroGraph],
        contexts: list[ActionContext],
        constraints: list[PolicyActionConstraints],
        semantic_batch: HeteroGraph,
        batched_representation: RepresentationOutput,
        deterministic: bool,
        force_gate: bool | None,
    ) -> tuple[CompositePolicyOutput, ...]:
        """Decode one environment wave in lockstep while keeping exact CPU masks."""
        count = len(graphs)
        representations = [
            self._slice_representation(batched_representation, semantic_batch, index)
            for index in range(count)
        ]
        effective_force: list[bool | None] = []
        for constraint in constraints:
            if (
                force_gate is not None
                and constraint.force_gate is not None
                and bool(force_gate) != bool(constraint.force_gate)
            ):
                raise ValueError(
                    "force_gate conflicts with PolicyActionConstraints.force_gate"
                )
            effective_force.append(
                bool(force_gate)
                if force_gate is not None
                else constraint.force_gate
            )

        gate_logits = torch.cat(
            [representation.gate.masked_logits for representation in representations],
            dim=0,
        )
        gate_dist = Categorical(logits=gate_logits)
        gate_indices = torch.empty(
            count, dtype=torch.long, device=gate_logits.device
        )
        free_indices = [
            index for index, forced in enumerate(effective_force) if forced is None
        ]
        if free_indices:
            free_tensor = torch.tensor(
                free_indices, dtype=torch.long, device=gate_logits.device
            )
            free_logits = gate_logits.index_select(0, free_tensor)
            free_selected = (
                torch.argmax(free_logits, dim=1)
                if deterministic
                else Categorical(logits=free_logits).sample()
            )
            gate_indices.index_copy_(0, free_tensor, free_selected)
        forced_indices = [
            index for index, forced in enumerate(effective_force) if forced is not None
        ]
        if forced_indices:
            forced_tensor = torch.tensor(
                forced_indices, dtype=torch.long, device=gate_logits.device
            )
            forced_values = torch.tensor(
                [1 if effective_force[index] else 0 for index in forced_indices],
                dtype=torch.long,
                device=gate_logits.device,
            )
            gate_indices.index_copy_(0, forced_tensor, forced_values)
            forced_logits = gate_logits.index_select(0, forced_tensor).gather(
                1, forced_values[:, None]
            )
            if bool(torch.any(forced_logits.detach() <= -1.0e8)):
                raise DecoderFeasibilityError("forced gate action is masked infeasible")

        gate_log_probs = gate_dist.log_prob(gate_indices)
        gate_entropies = gate_dist.entropy()
        reconfigure_flags = [
            bool(value) for value in gate_indices.detach().cpu().tolist()
        ]
        planners = [
            PlannedConfiguration(context, validate_context=False)
            for context in contexts
        ]
        reconfigure_indices = [
            index for index, flag in enumerate(reconfigure_flags) if flag
        ]

        worker_results: list[MatchingResult | None] = [None] * count
        robot_results: list[MatchingResult | None] = [None] * count
        if reconfigure_indices:
            worker_batch = self.worker_matcher.decode_batch(
                node_embeddings=[
                    representations[index].node_embeddings
                    for index in reconfigure_indices
                ],
                cell_embeddings=[
                    representations[index].node_embeddings["cell"]
                    for index in reconfigure_indices
                ],
                global_embeddings=[
                    representations[index].pooling.global_embedding
                    for index in reconfigure_indices
                ],
                graphs=[graphs[index] for index in reconfigure_indices],
                planners=[planners[index] for index in reconfigure_indices],
                deterministic=deterministic,
                max_changes=[
                    constraints[index].max_worker_moves
                    for index in reconfigure_indices
                ],
            )
            for index, result in zip(reconfigure_indices, worker_batch):
                worker_results[index] = result

        worker_updated = self._worker_updated_cells_batch(
            representations=representations,
            graphs=graphs,
            planners=planners,
        )
        if reconfigure_indices:
            robot_batch = self.robot_matcher.decode_batch(
                node_embeddings=[
                    representations[index].node_embeddings
                    for index in reconfigure_indices
                ],
                cell_embeddings=[worker_updated[index] for index in reconfigure_indices],
                global_embeddings=[
                    representations[index].pooling.global_embedding
                    for index in reconfigure_indices
                ],
                graphs=[graphs[index] for index in reconfigure_indices],
                planners=[planners[index] for index in reconfigure_indices],
                deterministic=deterministic,
                max_changes=[
                    constraints[index].max_robot_moves
                    for index in reconfigure_indices
                ],
            )
            for index, result in zip(reconfigure_indices, robot_batch):
                robot_results[index] = result
            for index in reconfigure_indices:
                if not planners[index].final_reconfiguration_valid():
                    raise DecoderFeasibilityError(
                        "decoded resource set violates final hard constraints"
                    )

        final_cells = self._final_cells_batch(
            worker_updated=worker_updated,
            representations=representations,
            graphs=graphs,
            planners=planners,
        )
        schedule_results = self.schedule_matcher.decode_batch(
            node_embeddings=[
                representation.node_embeddings for representation in representations
            ],
            cell_embeddings=final_cells,
            global_embeddings=[
                representation.pooling.global_embedding
                for representation in representations
            ],
            graphs=graphs,
            planners=planners,
            deterministic=deterministic,
            max_assignments=[
                constraint.max_schedule_assignments for constraint in constraints
            ],
        )

        outputs = []
        for index in range(count):
            worker_result = worker_results[index]
            robot_result = robot_results[index]
            schedule_result = schedule_results[index]
            worker_trace = None if worker_result is None else worker_result.trace
            robot_trace = None if robot_result is None else robot_result.trace
            trace = PolicyTrace(
                reconfigure=reconfigure_flags[index],
                worker=worker_trace,
                robot=robot_trace,
                schedule=schedule_result.trace,
            )
            zero = representations[index].pooling.global_embedding.new_zeros(())
            worker_lp = zero if worker_result is None else worker_result.log_prob
            robot_lp = zero if robot_result is None else robot_result.log_prob
            worker_entropy = zero if worker_result is None else worker_result.entropy
            robot_entropy = zero if robot_result is None else robot_result.entropy
            outputs.append(CompositePolicyOutput(
                action=CompositeAction(
                    reconfigure=reconfigure_flags[index],
                    worker_assignments=(
                        () if worker_trace is None
                        else trace_to_resource_assignments(worker_trace)
                    ),
                    robot_assignments=(
                        () if robot_trace is None
                        else trace_to_resource_assignments(robot_trace)
                    ),
                    schedule_assignments=trace_to_schedule_assignments(
                        schedule_result.trace
                    ),
                ),
                trace=trace,
                representation=representations[index],
                gate_log_prob=gate_log_probs[index],
                worker_log_prob=worker_lp,
                robot_log_prob=robot_lp,
                schedule_log_prob=schedule_result.log_prob,
                total_log_prob=(
                    gate_log_probs[index]
                    + worker_lp
                    + robot_lp
                    + schedule_result.log_prob
                ),
                gate_entropy=gate_entropies[index],
                worker_entropy=worker_entropy,
                robot_entropy=robot_entropy,
                schedule_entropy=schedule_result.entropy,
            ))
        return tuple(outputs)

    def act_batch(
        self,
        graphs: list[HeteroGraph] | tuple[HeteroGraph, ...],
        contexts: list[ActionContext] | tuple[ActionContext, ...],
        *,
        deterministic: bool = False,
        force_gate: bool | None = None,
        constraints: list[PolicyActionConstraints] | tuple[PolicyActionConstraints, ...] | PolicyActionConstraints | None = None,
        validate_inputs: bool = True,
    ) -> tuple[CompositePolicyOutput, ...]:
        """Batch only the shared encoder; decode independent event actions in order.

        Environments are independent, so collecting one current event from each slot
        before stepping any slot is semantically equivalent to the old round-robin
        execution.  Decoding remains per event to preserve exact hard masks and the
        paper's autoregressive action semantics.
        """
        graphs, contexts = list(graphs), list(contexts)
        if not graphs or len(graphs) != len(contexts):
            raise ValueError("graphs/contexts must be non-empty and equal length")
        if constraints is None:
            constraints_list = [PolicyActionConstraints.unconstrained() for _ in graphs]
        elif isinstance(constraints, PolicyActionConstraints):
            constraints_list = [constraints for _ in graphs]
        else:
            constraints_list = list(constraints)
        if len(constraints_list) != len(graphs):
            raise ValueError("constraints length must equal graph batch length")
        for graph, context, constraint in zip(graphs, contexts, constraints_list):
            if validate_inputs:
                graph.validate()
                context.validate()
            constraint.validate()
            if graph.batch_size != 1:
                raise ValueError("act_batch expects single-graph semantic items")
            # L1.5 keeps exact feasibility facts on CPU to avoid thousands of
            # implicit CUDA synchronizations from Python scalar conversions.
            if context.operation_ids.device.type != "cpu":
                raise ValueError("act_batch expects CPU ActionContext tensors")
        semantic_batch, batched_representation = self._batched_representation_from_semantic_graphs(
            graphs, validate_inputs=validate_inputs
        )
        if self.batched_rollout_decoding_enabled and len(graphs) > 1:
            return self._decode_tensorized_rollout_batch(
                graphs=graphs,
                contexts=contexts,
                constraints=constraints_list,
                semantic_batch=semantic_batch,
                batched_representation=batched_representation,
                deterministic=deterministic,
                force_gate=force_gate,
            )
        outputs = []
        for index, (graph, context, constraint) in enumerate(zip(graphs, contexts, constraints_list)):
            representation = self._slice_representation(
                batched_representation, semantic_batch, index
            )
            outputs.append(self._decode_from_representation(
                graph, context, representation,
                deterministic=deterministic,
                force_gate=force_gate,
                replay_trace=None,
                constraints=constraint,
            ))
        return tuple(outputs)

    def evaluate_traces_batch(
        self,
        graphs: list[HeteroGraph] | tuple[HeteroGraph, ...],
        contexts: list[ActionContext] | tuple[ActionContext, ...],
        traces: list[PolicyTrace] | tuple[PolicyTrace, ...],
        constraints: list[PolicyActionConstraints] | tuple[PolicyActionConstraints, ...] | None = None,
        validate_inputs: bool = True,
    ) -> tuple[CompositePolicyOutput, ...]:
        """Batch the expensive shared graph encoder, then replay variable traces per event.

        Compact replay materialization stays on CPU.  This removes the previous
        CPU->GPU transfer of every ActionContext and, more importantly, avoids CUDA
        synchronization on each Python scalar feasibility read inside the decoders.
        """
        graphs, contexts, traces = list(graphs), list(contexts), list(traces)
        constraints = (
            [PolicyActionConstraints.unconstrained() for _ in graphs]
            if constraints is None else list(constraints)
        )
        if not graphs or not (len(graphs) == len(contexts) == len(traces) == len(constraints)):
            raise ValueError("graphs/contexts/traces/constraints must be non-empty and equal length")
        for graph, context in zip(graphs, contexts):
            if validate_inputs:
                graph.validate()
                context.validate()
            if graph.batch_size != 1:
                raise ValueError("evaluate_traces_batch expects single-graph items")
            if context.operation_ids.device.type != "cpu":
                raise ValueError("evaluate_traces_batch expects CPU ActionContext tensors")
        semantic_batch, batched_representation = self._batched_representation_from_semantic_graphs(
            graphs, validate_inputs=validate_inputs
        )
        if (
            self.worker_matcher.tensorized_scoring_enabled
            and self.robot_matcher.tensorized_scoring_enabled
            and self.schedule_matcher.tensorized_scoring_enabled
        ):
            return self._evaluate_traces_tensorized_batch(
                graphs=graphs,
                contexts=contexts,
                traces=traces,
                constraints=constraints,
                semantic_batch=semantic_batch,
                batched_representation=batched_representation,
            )
        outputs = []
        for index, (graph, context, trace, constraint) in enumerate(zip(graphs, contexts, traces, constraints)):
            representation = self._slice_representation(
                batched_representation, semantic_batch, index
            )
            outputs.append(self._decode_from_representation(
                graph, context, representation,
                deterministic=True,
                force_gate=None,
                replay_trace=trace,
                constraints=constraint,
            ))
        return tuple(outputs)

    def act(
        self,
        graph: HeteroGraph,
        context: ActionContext,
        *,
        deterministic: bool = False,
        force_gate: bool | None = None,
        constraints: PolicyActionConstraints | None = None,
        validate_inputs: bool = True,
    ) -> CompositePolicyOutput:
        return self._run(
            graph,
            context,
            deterministic=deterministic,
            force_gate=force_gate,
            replay_trace=None,
            constraints=constraints,
            validate_inputs=validate_inputs,
        )

    def evaluate_trace(
        self,
        graph: HeteroGraph,
        context: ActionContext,
        trace: PolicyTrace,
        *,
        constraints: PolicyActionConstraints | None = None,
        validate_inputs: bool = True,
    ) -> CompositePolicyOutput:
        """Recompute current-policy probabilities for an exact stored action trace."""
        return self._run(
            graph,
            context,
            deterministic=True,
            force_gate=None,
            replay_trace=trace,
            constraints=constraints,
            validate_inputs=validate_inputs,
        )


__all__ = [
    "CompositePolicyOutput",
    "EGDMCompositePolicy",
    "PolicyTrace",
]
