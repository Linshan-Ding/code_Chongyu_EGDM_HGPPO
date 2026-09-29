"""Compact exact-replay state for memory-safe formal PPO rollouts.

Phase L1.2 keeps the paper algorithm unchanged while avoiding storage of a full
heterogeneous graph and a full :class:`ActionContext` at every event.  The
collector stores only dynamic simulator facts plus a shared reference to the
immutable ``AssemblyInstance``.  During PPO replay, the exact policy-visible
simulator view is reconstructed and passed through the existing graph builder and
action-context builder.

The compact snapshot intentionally stores no future-event time/type.  It keeps
only the boolean fact ``pending_physical_event`` because that same fact was
already part of the Phase-I ActionContext contract used to prevent an immediate
STOP deadlock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from data.schema import AssemblyInstance
from environment.action_context import (
    ActionContext,
    ActionContextStaticFacts,
    build_action_context,
    build_action_context_static_facts,
)
from environment.entities import (
    CellActivity,
    CellRuntime,
    OperationRuntime,
    OperationStatus,
    OrderRuntime,
    ResourceActivity,
    RobotRuntime,
    WorkerRuntime,
)
from environment.events import EventType
from environment.graph_builder import DynamicHeteroGraphBuilder
from environment.graph_normalization import GraphContinuousNormalizer
from environment.graph_types import HeteroGraph


_OPERATION_STATUSES = (
    OperationStatus.NOT_RELEASED,
    OperationStatus.BLOCKED,
    OperationStatus.READY,
    OperationStatus.PROCESSING,
    OperationStatus.DONE,
)
_RESOURCE_ACTIVITIES = (
    ResourceActivity.IDLE,
    ResourceActivity.BUSY,
    ResourceActivity.RELOCATING,
)
_CELL_ACTIVITIES = (CellActivity.IDLE, CellActivity.BUSY)
_EVENT_TYPES = (
    EventType.ORDER_ARRIVAL,
    EventType.OPERATION_FINISH,
    EventType.RELOCATION_FINISH,
)

_STATUS_TO_CODE = {value: i for i, value in enumerate(_OPERATION_STATUSES)}
_RESOURCE_TO_CODE = {value: i for i, value in enumerate(_RESOURCE_ACTIVITIES)}
_CELL_TO_CODE = {value: i for i, value in enumerate(_CELL_ACTIVITIES)}
_EVENT_TO_BIT = {value: 1 << i for i, value in enumerate(_EVENT_TYPES)}


def _none_to_neg1(value: int | None) -> int:
    return -1 if value is None else int(value)


def _none_to_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _neg1_to_none(value: int) -> int | None:
    return None if int(value) < 0 else int(value)


def _nan_to_none(value: float) -> float | None:
    value = float(value)
    return None if value != value else value


@dataclass(frozen=True, slots=True)
class CompactDecisionSnapshot:
    """Small dynamic state sufficient to reconstruct graph + ActionContext exactly."""

    episode_static: CompactEpisodeStatic
    time: float
    last_reconfiguration_time: float
    decision_index: int
    event_mask: int
    pending_physical_event: bool

    # Orders: [J] arrived, [J] completion time (NaN = none)
    order_arrived: torch.Tensor
    order_completion: torch.Tensor

    # Operations: [O,2] => status, assigned_cell; [O,2] => start, completion.
    operation_codes: torch.Tensor
    operation_times: torch.Tensor

    # Cells: [M,5] => activity, configured_worker, configured_robot,
    # reserved_worker, reserved_robot; [M] busy_until.
    cell_codes: torch.Tensor
    cell_busy_until: torch.Tensor

    # Resources: [N,3] => activity, configured_cell, physical_cell; [N] busy_until.
    worker_codes: torch.Tensor
    worker_busy_until: torch.Tensor
    robot_codes: torch.Tensor
    robot_busy_until: torch.Tensor

    @property
    def estimated_bytes(self) -> int:
        tensors = (
            self.order_arrived,
            self.order_completion,
            self.operation_codes,
            self.operation_times,
            self.cell_codes,
            self.cell_busy_until,
            self.worker_codes,
            self.worker_busy_until,
            self.robot_codes,
            self.robot_busy_until,
        )
        return int(sum(x.numel() * x.element_size() for x in tensors))


class _EventPresence:
    """Only ``len(events)`` is policy-visible during replay."""

    __slots__ = ("present",)

    def __init__(self, present: bool) -> None:
        self.present = bool(present)

    def __len__(self) -> int:
        return int(self.present)


@dataclass(frozen=True, slots=True)
class CompactEpisodeStatic:
    """Immutable replay facts shared by every snapshot of one episode.

    The object is cached on the active simulator and referenced by snapshots, so
    large processing/compatibility tensors exist once per episode rather than
    once per event.  It is released naturally when the rollout buffer and active
    simulator episode no longer reference it.
    """

    instance: AssemblyInstance
    minimum_dwell_time: float
    operation_meta: tuple[tuple[int, int, int, int, int], ...]
    order_operation_ids: tuple[tuple[int, ...], ...]
    action_context: ActionContextStaticFacts


class ReplaySimulatorView:
    """Read-only simulator-shaped object consumed by existing builders/masks."""

    pass


def _episode_static(sim) -> CompactEpisodeStatic:
    cached = getattr(sim, "_compact_replay_episode_static", None)
    if cached is not None and cached.instance is sim.instance:
        return cached
    if sim.instance is None:
        raise RuntimeError("simulator must be reset before compact replay static construction")
    operation_meta = tuple(
        (
            int(op.operation_id),
            int(op.order_id),
            int(op.product_type),
            int(op.stage),
            int(op.route_position),
        )
        for op in sim.operations
    )
    order_operation_ids = tuple(tuple(int(x) for x in row) for row in sim.order_operation_ids)
    action_static = getattr(sim, "_action_context_static_facts", None)
    if action_static is None:
        action_static = build_action_context_static_facts(sim)
        sim._action_context_static_facts = action_static
    cached = CompactEpisodeStatic(
        instance=sim.instance,
        minimum_dwell_time=float(sim.minimum_dwell_time),
        operation_meta=operation_meta,
        order_operation_ids=order_operation_ids,
        action_context=action_static,
    )
    sim._compact_replay_episode_static = cached
    return cached


def capture_compact_snapshot(sim) -> CompactDecisionSnapshot:
    """Capture the current pre-action policy-visible simulator state.

    The immutable instance object is shared by reference across all snapshots of
    the same episode; only dynamic arrays are copied.
    """

    if sim.instance is None:
        raise RuntimeError("simulator must be reset before replay snapshot capture")
    episode_static = _episode_static(sim)

    order_arrived = torch.tensor([o.arrived for o in sim.orders], dtype=torch.bool)
    order_completion = torch.tensor(
        [_none_to_nan(o.completion_time) for o in sim.orders], dtype=torch.float64
    )

    operation_codes = torch.tensor(
        [
            [_STATUS_TO_CODE[op.status], _none_to_neg1(op.assigned_cell)]
            for op in sim.operations
        ],
        dtype=torch.int16,
    )
    operation_times = torch.tensor(
        [
            [_none_to_nan(op.start_time), _none_to_nan(op.completion_time)]
            for op in sim.operations
        ],
        dtype=torch.float64,
    )

    cell_codes = torch.tensor(
        [
            [
                _CELL_TO_CODE[cell.activity],
                _none_to_neg1(cell.configured_worker),
                _none_to_neg1(cell.configured_robot),
                _none_to_neg1(cell.reserved_worker),
                _none_to_neg1(cell.reserved_robot),
            ]
            for cell in sim.cells
        ],
        dtype=torch.int16,
    )
    cell_busy_until = torch.tensor(
        [_none_to_nan(cell.busy_until) for cell in sim.cells], dtype=torch.float64
    )

    worker_codes = torch.tensor(
        [
            [
                _RESOURCE_TO_CODE[w.activity],
                _none_to_neg1(w.configured_cell),
                _none_to_neg1(w.physical_cell),
            ]
            for w in sim.workers
        ],
        dtype=torch.int16,
    )
    worker_busy_until = torch.tensor(
        [_none_to_nan(w.busy_until) for w in sim.workers], dtype=torch.float64
    )
    robot_codes = torch.tensor(
        [
            [
                _RESOURCE_TO_CODE[r.activity],
                _none_to_neg1(r.configured_cell),
                _none_to_neg1(r.physical_cell),
            ]
            for r in sim.robots
        ],
        dtype=torch.int16,
    )
    robot_busy_until = torch.tensor(
        [_none_to_nan(r.busy_until) for r in sim.robots], dtype=torch.float64
    )

    event_mask = 0
    for event_type in sim.current_event_types:
        event_mask |= _EVENT_TO_BIT[event_type]

    return CompactDecisionSnapshot(
        episode_static=episode_static,
        time=float(sim.time),
        last_reconfiguration_time=float(sim.last_reconfiguration_time),
        decision_index=int(sim.decision_index),
        event_mask=int(event_mask),
        pending_physical_event=bool(len(sim.events) > 0),
        order_arrived=order_arrived,
        order_completion=order_completion,
        operation_codes=operation_codes,
        operation_times=operation_times,
        cell_codes=cell_codes,
        cell_busy_until=cell_busy_until,
        worker_codes=worker_codes,
        worker_busy_until=worker_busy_until,
        robot_codes=robot_codes,
        robot_busy_until=robot_busy_until,
    )


class CompactReplayMaterializer:
    """Reconstruct exact graph/context lazily from compact snapshots."""

    def __init__(
        self, cfg, normalizer: GraphContinuousNormalizer, *,
        static_processing_cache: bool = False,
        trusted_replay_normalization: bool = False,
    ) -> None:
        self.cfg = cfg
        self.normalizer = normalizer
        self.graph_builder = DynamicHeteroGraphBuilder(cfg)
        # L1.6.6 opt-in diagnostic fast path.  Formal execution remains unchanged
        # unless explicitly enabled by the isolated pilot.  The cache contains
        # only immutable per-operation processing minima derived from the same
        # AssemblyInstance formulas; it never stores dynamic simulator state.
        self.static_processing_cache = bool(static_processing_cache)
        # L1.6.7 opt-in diagnostic fast path.  The graph builder validates raw
        # replay graphs and PPO batching validates normalized graphs, so the normalizer's
        # own duplicate validations can be skipped without changing tensor values.
        self.trusted_replay_normalization = bool(trusted_replay_normalization)
        self._static_processing_by_episode: dict[int, tuple[tuple[float, ...], tuple[tuple[float, float, float, int, int, int], ...]]] = {}

    @staticmethod
    def _build_static_processing_cache(static: CompactEpisodeStatic):
        inst = static.instance
        by_key: dict[tuple[int, int], tuple[float, tuple[float, float, float, int, int, int]]] = {}
        minimum_by_operation: list[float] = []
        mode_by_operation: list[tuple[float, float, float, int, int, int]] = []
        for _op_id, _order_id, product_type, stage, _route_position in static.operation_meta:
            key = (int(product_type), int(stage))
            cached = by_key.get(key)
            if cached is None:
                h_vals = [
                    p for h in range(inst.num_workers)
                    if (p := inst.processing_time_h(key[0], key[1], h)) is not None
                ]
                r_vals = [
                    p for r in range(inst.num_robots)
                    if (p := inst.processing_time_r(key[0], key[1], r)) is not None
                ]
                hr_vals = [
                    p for h in range(inst.num_workers) for r in range(inst.num_robots)
                    if (p := inst.processing_time_hr(key[0], key[1], h, r)) is not None
                ]
                all_vals = [*h_vals, *r_vals, *hr_vals]
                if not all_vals:
                    raise ValueError(f"product={key[0]}, stage={key[1]} has no executable mode")
                mode = (
                    float(min(h_vals)) if h_vals else 0.0,
                    float(min(r_vals)) if r_vals else 0.0,
                    float(min(hr_vals)) if hr_vals else 0.0,
                    int(bool(h_vals)),
                    int(bool(r_vals)),
                    int(bool(hr_vals)),
                )
                cached = (float(min(all_vals)), mode)
                by_key[key] = cached
            minimum, mode = cached
            minimum_by_operation.append(minimum)
            mode_by_operation.append(mode)
        return tuple(minimum_by_operation), tuple(mode_by_operation)

    def _processing_cache_for(self, static: CompactEpisodeStatic):
        key = id(static)
        cached = self._static_processing_by_episode.get(key)
        if cached is None:
            cached = self._build_static_processing_cache(static)
            self._static_processing_by_episode[key] = cached
        return cached

    def simulator_view(self, snapshot: CompactDecisionSnapshot) -> Any:
        static = snapshot.episode_static
        inst = static.instance
        if snapshot.operation_codes.shape != (len(static.operation_meta), 2):
            raise ValueError("compact replay operation shape mismatch")

        sim = ReplaySimulatorView()
        sim.cfg = self.cfg
        sim.instance = inst
        sim.minimum_dwell_time = float(static.minimum_dwell_time)
        sim.time = float(snapshot.time)
        sim.last_reconfiguration_time = float(snapshot.last_reconfiguration_time)
        sim.decision_index = int(snapshot.decision_index)
        sim.current_event_types = tuple(
            event_type for i, event_type in enumerate(_EVENT_TYPES)
            if snapshot.event_mask & (1 << i)
        )
        sim.events = _EventPresence(snapshot.pending_physical_event)
        sim._action_context_static_facts = static.action_context
        sim._compact_replay_episode_static = static
        if self.static_processing_cache:
            minimum_by_operation, mode_by_operation = self._processing_cache_for(static)
            sim._replay_minimum_processing_by_operation = minimum_by_operation
            sim._replay_mode_min_by_operation = mode_by_operation

        sim.orders = []
        for j in range(inst.num_orders):
            order = OrderRuntime(order_id=j)
            order.arrived = bool(snapshot.order_arrived[j])
            order.completion_time = _nan_to_none(float(snapshot.order_completion[j]))
            sim.orders.append(order)

        sim.operations = []
        for i, (op_id, order_id, product_type, stage, route_position) in enumerate(static.operation_meta):
            code = snapshot.operation_codes[i]
            times = snapshot.operation_times[i]
            op = OperationRuntime(
                operation_id=op_id,
                order_id=order_id,
                product_type=product_type,
                stage=stage,
                route_position=route_position,
                status=_OPERATION_STATUSES[int(code[0])],
                start_time=_nan_to_none(float(times[0])),
                completion_time=_nan_to_none(float(times[1])),
                assigned_cell=_neg1_to_none(int(code[1])),
            )
            sim.operations.append(op)
        sim.order_operation_ids = [list(x) for x in static.order_operation_ids]

        sim.cells = []
        for m, stage in enumerate(inst.cell_stage):
            code = snapshot.cell_codes[m]
            sim.cells.append(
                CellRuntime(
                    cell_id=m,
                    stage=int(stage),
                    activity=_CELL_ACTIVITIES[int(code[0])],
                    configured_worker=_neg1_to_none(int(code[1])),
                    configured_robot=_neg1_to_none(int(code[2])),
                    reserved_worker=_neg1_to_none(int(code[3])),
                    reserved_robot=_neg1_to_none(int(code[4])),
                    busy_until=_nan_to_none(float(snapshot.cell_busy_until[m])),
                )
            )

        sim.workers = []
        for h in range(inst.num_workers):
            code = snapshot.worker_codes[h]
            sim.workers.append(
                WorkerRuntime(
                    worker_id=h,
                    activity=_RESOURCE_ACTIVITIES[int(code[0])],
                    configured_cell=_neg1_to_none(int(code[1])),
                    physical_cell=_neg1_to_none(int(code[2])),
                    busy_until=_nan_to_none(float(snapshot.worker_busy_until[h])),
                )
            )
        sim.robots = []
        for r in range(inst.num_robots):
            code = snapshot.robot_codes[r]
            sim.robots.append(
                RobotRuntime(
                    robot_id=r,
                    activity=_RESOURCE_ACTIVITIES[int(code[0])],
                    configured_cell=_neg1_to_none(int(code[1])),
                    physical_cell=_neg1_to_none(int(code[2])),
                    busy_until=_nan_to_none(float(snapshot.robot_busy_until[r])),
                )
            )
        return sim

    @torch.no_grad()
    def materialize(self, snapshot: CompactDecisionSnapshot) -> tuple[HeteroGraph, ActionContext]:
        sim = self.simulator_view(snapshot)
        raw_graph = self.graph_builder.build(sim)
        if self.trusted_replay_normalization:
            graph = self.normalizer.transform_replay_trusted(raw_graph)
        else:
            graph = self.normalizer.transform(raw_graph)
        context = build_action_context(sim)
        return graph, context


__all__ = [
    "CompactDecisionSnapshot",
    "CompactEpisodeStatic",
    "CompactReplayMaterializer",
    "ReplaySimulatorView",
    "capture_compact_snapshot",
]
