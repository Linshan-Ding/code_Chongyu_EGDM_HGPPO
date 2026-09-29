"""Stable action and observation-side data structures for Phase D."""

from __future__ import annotations

from dataclasses import dataclass, field

from environment.entities import (
    CellActivity,
    ExecutionMode,
    OperationStatus,
    ResourceActivity,
)
from environment.events import EventType


@dataclass(frozen=True, slots=True)
class ResourceAssignment:
    """One autoregressive resource-matching decision.

    target_cell >= 0 : configure/move to that cell.
    target_cell == -1: enter the temporary unconfigured state *in place*.

    An idle resource omitted from the list keeps its current configuration.
    """

    resource_id: int
    target_cell: int


@dataclass(frozen=True, slots=True)
class ScheduleAssignment:
    operation_id: int
    cell_id: int


@dataclass(frozen=True, slots=True)
class CompositeAction:
    reconfigure: bool = False
    worker_assignments: tuple[ResourceAssignment, ...] = ()
    robot_assignments: tuple[ResourceAssignment, ...] = ()
    schedule_assignments: tuple[ScheduleAssignment, ...] = ()

    @classmethod
    def keep(
        cls,
        schedule_assignments: tuple[ScheduleAssignment, ...] | list[ScheduleAssignment] = (),
    ) -> "CompositeAction":
        return cls(
            reconfigure=False,
            schedule_assignments=tuple(schedule_assignments),
        )

    @classmethod
    def reconfigure_then_schedule(
        cls,
        worker_assignments: tuple[ResourceAssignment, ...] | list[ResourceAssignment] = (),
        robot_assignments: tuple[ResourceAssignment, ...] | list[ResourceAssignment] = (),
        schedule_assignments: tuple[ScheduleAssignment, ...] | list[ScheduleAssignment] = (),
    ) -> "CompositeAction":
        return cls(
            reconfigure=True,
            worker_assignments=tuple(worker_assignments),
            robot_assignments=tuple(robot_assignments),
            schedule_assignments=tuple(schedule_assignments),
        )


@dataclass(frozen=True, slots=True)
class OrderView:
    order_id: int
    product_type: int
    release_time: float
    due_date: float
    weight: int
    completed: bool
    completion_time: float | None


@dataclass(frozen=True, slots=True)
class OperationView:
    operation_id: int
    order_id: int
    product_type: int
    stage: int
    route_position: int
    status: OperationStatus
    start_time: float | None
    completion_time: float | None
    assigned_cell: int | None
    mode: ExecutionMode | None


@dataclass(frozen=True, slots=True)
class CellView:
    cell_id: int
    stage: int
    activity: CellActivity
    configured_worker: int | None
    configured_robot: int | None
    reserved_worker: int | None
    reserved_robot: int | None
    processing_operation: int | None
    busy_until: float | None


@dataclass(frozen=True, slots=True)
class ResourceView:
    resource_id: int
    activity: ResourceActivity
    configured_cell: int | None
    physical_cell: int | None
    relocation_target: int | None
    busy_operation: int | None
    busy_until: float | None


@dataclass(frozen=True, slots=True)
class DecisionState:
    """Policy-visible Phase D state.

    Future/unreleased orders are intentionally excluded. Phase E will transform
    this state into the paper's five-node dynamic heterogeneous graph.
    """

    time: float
    event_types: tuple[EventType, ...]
    orders: tuple[OrderView, ...]
    operations: tuple[OperationView, ...]
    cells: tuple[CellView, ...]
    workers: tuple[ResourceView, ...]
    robots: tuple[ResourceView, ...]
    last_reconfiguration_time: float
    cumulative_base_reward: float
    decision_index: int

    @property
    def ready_operation_ids(self) -> tuple[int, ...]:
        return tuple(
            op.operation_id for op in self.operations if op.status == OperationStatus.READY
        )


@dataclass(frozen=True, slots=True)
class ResourceCandidateSet:
    resource_id: int
    target_cells: tuple[int, ...]
    can_unconfigure: bool
    can_stay: bool


@dataclass(frozen=True, slots=True)
class ScheduleCandidate:
    operation_id: int
    cell_id: int
    mode: ExecutionMode
    processing_time: float


@dataclass(frozen=True, slots=True)
class ActionCandidates:
    reconfigure_feasible: bool
    worker: tuple[ResourceCandidateSet, ...] = field(default_factory=tuple)
    robot: tuple[ResourceCandidateSet, ...] = field(default_factory=tuple)
    schedule: tuple[ScheduleCandidate, ...] = field(default_factory=tuple)
