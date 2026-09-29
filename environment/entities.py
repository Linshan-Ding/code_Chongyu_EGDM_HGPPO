"""Runtime entities for the EGDM-HGPPO discrete-event simulator.

The data layer stores immutable/static instance facts. This module stores only
runtime state that changes while an episode is simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OperationStatus(str, Enum):
    NOT_RELEASED = "NOT_RELEASED"
    BLOCKED = "BLOCKED"
    READY = "READY"
    PROCESSING = "PROCESSING"
    DONE = "DONE"


class ResourceActivity(str, Enum):
    IDLE = "IDLE"
    BUSY = "BUSY"
    RELOCATING = "RELOCATING"


class CellActivity(str, Enum):
    IDLE = "IDLE"
    BUSY = "BUSY"


class ExecutionMode(str, Enum):
    H = "H"
    R = "R"
    HR = "HR"
    IDLE = "IDLE"


@dataclass(slots=True)
class OrderRuntime:
    order_id: int
    arrived: bool = False
    completion_time: float | None = None

    @property
    def completed(self) -> bool:
        return self.completion_time is not None


@dataclass(slots=True)
class OperationRuntime:
    operation_id: int
    order_id: int
    product_type: int
    stage: int
    route_position: int
    status: OperationStatus = OperationStatus.NOT_RELEASED
    start_time: float | None = None
    completion_time: float | None = None
    assigned_cell: int | None = None
    worker_id: int | None = None
    robot_id: int | None = None
    mode: ExecutionMode | None = None


@dataclass(slots=True)
class CellRuntime:
    cell_id: int
    stage: int
    activity: CellActivity = CellActivity.IDLE
    configured_worker: int | None = None
    configured_robot: int | None = None
    reserved_worker: int | None = None
    reserved_robot: int | None = None
    processing_operation: int | None = None
    busy_until: float | None = None


@dataclass(slots=True)
class WorkerRuntime:
    worker_id: int
    activity: ResourceActivity = ResourceActivity.IDLE
    configured_cell: int | None = None
    physical_cell: int | None = None
    relocation_target: int | None = None
    busy_operation: int | None = None
    busy_until: float | None = None


@dataclass(slots=True)
class RobotRuntime:
    robot_id: int
    activity: ResourceActivity = ResourceActivity.IDLE
    configured_cell: int | None = None
    physical_cell: int | None = None
    relocation_target: int | None = None
    busy_operation: int | None = None
    busy_until: float | None = None
