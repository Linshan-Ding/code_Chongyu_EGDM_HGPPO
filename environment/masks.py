"""Hard feasibility masks and composite-action validation.

All structural feasibility is enforced here / in the simulator, not through soft
reward penalties. The policy in later phases will consume the same candidate
logic, so rule baselines and learned policies share the physical constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from environment.entities import CellActivity, ExecutionMode, OperationStatus, ResourceActivity
from environment.state import (
    ActionCandidates,
    CompositeAction,
    ResourceCandidateSet,
    ScheduleCandidate,
)


class InvalidActionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlannedResourceChange:
    kind: str  # "worker" | "robot"
    resource_id: int
    source_physical_cell: int
    old_configured_cell: int | None
    target_cell: int | None  # None means temporary unconfigured
    relocation_time: float

    @property
    def is_relocation(self) -> bool:
        return self.target_cell is not None and self.source_physical_cell != self.target_cell

    @property
    def is_immediate(self) -> bool:
        return not self.is_relocation or self.relocation_time <= 1e-12


@dataclass(frozen=True, slots=True)
class PlannedStart:
    operation_id: int
    cell_id: int
    mode: ExecutionMode
    worker_id: int | None
    robot_id: int | None
    processing_time: float


@dataclass(frozen=True, slots=True)
class ValidatedAction:
    reconfigure: bool
    worker_changes: tuple[PlannedResourceChange, ...]
    robot_changes: tuple[PlannedResourceChange, ...]
    starts: tuple[PlannedStart, ...]


def _configured_mode_and_time(sim, operation_id: int, cell_id: int, *, worker_cfg, robot_cfg):
    op = sim.operations[operation_id]
    cell = sim.cells[cell_id]
    product = op.product_type
    stage = op.stage

    worker = worker_cfg[cell_id]
    robot = robot_cfg[cell_id]
    if worker is None and robot is None:
        return None
    if worker is not None and robot is None:
        p = sim.instance.processing_time_h(product, stage, worker)
        return None if p is None else (ExecutionMode.H, worker, None, float(p))
    if worker is None and robot is not None:
        p = sim.instance.processing_time_r(product, stage, robot)
        return None if p is None else (ExecutionMode.R, None, robot, float(p))
    p = sim.instance.processing_time_hr(product, stage, worker, robot)
    return None if p is None else (ExecutionMode.HR, worker, robot, float(p))


def _current_resource_targets(sim, kind: str, resource_id: int) -> tuple[int, ...]:
    if kind == "worker":
        resource = sim.workers[resource_id]
        compat = sim.instance.worker_skill[resource_id]
        occupied = {
            c.cell_id
            for c in sim.cells
            if c.configured_worker is not None or c.reserved_worker is not None
        }
    else:
        resource = sim.robots[resource_id]
        compat = sim.instance.robot_capability[resource_id]
        occupied = {
            c.cell_id
            for c in sim.cells
            if c.configured_robot is not None or c.reserved_robot is not None
        }

    if resource.activity != ResourceActivity.IDLE:
        return ()

    out: list[int] = []
    for cell in sim.cells:
        if cell.activity != CellActivity.IDLE:
            continue
        if compat[cell.stage] != 1:
            continue
        # The resource's own current cell is a valid "stay" action and is handled
        # separately, so only actually free alternative targets are returned here.
        if cell.cell_id in occupied and resource.configured_cell != cell.cell_id:
            continue
        out.append(cell.cell_id)
    return tuple(out)


def gate_reconfigure_feasible(sim) -> bool:
    dwell = float(sim.minimum_dwell_time)
    if sim.time - sim.last_reconfiguration_time < dwell - 1e-12:
        return False

    # Conservative Phase-D coverage rule: while a previous relocation has left a
    # stage without a currently configured executable resource, do not start a
    # second reconfiguration. In-transit reservations remain unavailable supply.
    if bool(sim.cfg.env.action_constraints.enforce_basic_executable_coverage_after_reconfiguration):
        current_worker = [c.configured_worker for c in sim.cells]
        current_robot = [c.configured_robot for c in sim.cells]
        if not _coverage_ok(sim, current_worker, current_robot):
            return False

    # The paper masks reconfiguration if there is no idle movable resource or all
    # compatible target capacities are full. A pure "unconfigure" action is not
    # sufficient to make the gate useful/feasible here.
    for h, worker in enumerate(sim.workers):
        if worker.activity != ResourceActivity.IDLE:
            continue
        for target in _current_resource_targets(sim, "worker", h):
            if worker.configured_cell != target:
                return True
    for r, robot in enumerate(sim.robots):
        if robot.activity != ResourceActivity.IDLE:
            continue
        for target in _current_resource_targets(sim, "robot", r):
            if robot.configured_cell != target:
                return True
    return False


def build_action_candidates(sim) -> ActionCandidates:
    workers: list[ResourceCandidateSet] = []
    robots: list[ResourceCandidateSet] = []
    for h, worker in enumerate(sim.workers):
        if worker.activity == ResourceActivity.IDLE:
            workers.append(
                ResourceCandidateSet(
                    resource_id=h,
                    target_cells=_current_resource_targets(sim, "worker", h),
                    can_unconfigure=worker.configured_cell is not None,
                    can_stay=True,
                )
            )
    for r, robot in enumerate(sim.robots):
        if robot.activity == ResourceActivity.IDLE:
            robots.append(
                ResourceCandidateSet(
                    resource_id=r,
                    target_cells=_current_resource_targets(sim, "robot", r),
                    can_unconfigure=robot.configured_cell is not None,
                    can_stay=True,
                )
            )

    schedule: list[ScheduleCandidate] = []
    worker_cfg = [c.configured_worker for c in sim.cells]
    robot_cfg = [c.configured_robot for c in sim.cells]
    for op in sim.operations:
        if op.status != OperationStatus.READY:
            continue
        for cell in sim.cells:
            if cell.activity != CellActivity.IDLE or cell.stage != op.stage:
                continue
            if cell.reserved_worker is not None or cell.reserved_robot is not None:
                continue
            mode_info = _configured_mode_and_time(
                sim, op.operation_id, cell.cell_id, worker_cfg=worker_cfg, robot_cfg=robot_cfg
            )
            if mode_info is None:
                continue
            mode, _, _, p = mode_info
            schedule.append(
                ScheduleCandidate(
                    operation_id=op.operation_id,
                    cell_id=cell.cell_id,
                    mode=mode,
                    processing_time=p,
                )
            )
    return ActionCandidates(
        reconfigure_feasible=gate_reconfigure_feasible(sim),
        worker=tuple(workers),
        robot=tuple(robots),
        schedule=tuple(schedule),
    )


def _initial_physical_anchor(sim, kind: str, resource_id: int) -> int:
    """Deterministic physical anchor for an initially unconfigured resource.

    Phase C permits ``initial_*_cell == -1`` because the paper permits temporary
    unconfigured resources but does not specify a separate pool-location parameter.
    Phase D therefore uses an explicit implementation convention: such a resource
    waits physically beside the first compatible dedicated cell while not counting
    toward that cell's configured capacity. This choice is documented in env.yaml.
    """

    compat = (
        sim.instance.worker_skill[resource_id]
        if kind == "worker"
        else sim.instance.robot_capability[resource_id]
    )
    for cell in sim.cells:
        if compat[cell.stage] == 1:
            return cell.cell_id
    raise InvalidActionError(f"{kind} {resource_id} has no compatible physical anchor")


def _cell_configuration_legal(sim, cell_id: int, worker: int | None, robot: int | None) -> bool:
    stage = sim.cells[cell_id].stage
    if worker is None and robot is None:
        return False
    if worker is not None and sim.instance.worker_skill[worker][stage] != 1:
        return False
    if robot is not None and sim.instance.robot_capability[robot][stage] != 1:
        return False
    if worker is not None and robot is not None:
        return bool(sim.instance.hr_compatibility[worker][robot][stage])
    return True


def _coverage_ok(sim, worker_cfg: list[int | None], robot_cfg: list[int | None]) -> bool:
    covered = [False] * sim.instance.num_stages
    for cell in sim.cells:
        w = worker_cfg[cell.cell_id]
        r = robot_cfg[cell.cell_id]
        if _cell_configuration_legal(sim, cell.cell_id, w, r):
            covered[cell.stage] = True
    return all(covered)


def validate_composite_action(sim, action: CompositeAction) -> ValidatedAction:
    if not isinstance(action, CompositeAction):
        raise InvalidActionError("env.step expects a CompositeAction")

    if not action.reconfigure and (action.worker_assignments or action.robot_assignments):
        raise InvalidActionError("KEEP gate requires empty resource-reconfiguration sets")
    if action.reconfigure and not gate_reconfigure_feasible(sim):
        raise InvalidActionError("RECONFIGURE is masked as infeasible at this event")

    worker_cfg = [c.configured_worker for c in sim.cells]
    robot_cfg = [c.configured_robot for c in sim.cells]
    worker_res = [c.reserved_worker for c in sim.cells]
    robot_res = [c.reserved_robot for c in sim.cells]

    worker_changes: list[PlannedResourceChange] = []
    robot_changes: list[PlannedResourceChange] = []

    def plan_resource(kind: str, assignment, seen: set[int]) -> PlannedResourceChange:
        rid = int(assignment.resource_id)
        target_raw = int(assignment.target_cell)
        resources = sim.workers if kind == "worker" else sim.robots
        if not 0 <= rid < len(resources):
            raise InvalidActionError(f"invalid {kind} id {rid}")
        if rid in seen:
            raise InvalidActionError(f"{kind} {rid} selected twice in one set action")
        seen.add(rid)
        res = resources[rid]
        if res.activity != ResourceActivity.IDLE:
            raise InvalidActionError(f"{kind} {rid} is not idle")

        old_cell = res.configured_cell
        source = res.physical_cell
        if source is None:
            source = _initial_physical_anchor(sim, kind, rid)

        cfg = worker_cfg if kind == "worker" else robot_cfg
        reserved = worker_res if kind == "worker" else robot_res
        compat = (
            sim.instance.worker_skill[rid]
            if kind == "worker"
            else sim.instance.robot_capability[rid]
        )

        # First free the resource's current configured slot in the planned state.
        if old_cell is not None:
            if cfg[old_cell] != rid:
                raise InvalidActionError(f"internal {kind} configuration inconsistency")
            cfg[old_cell] = None

        if target_raw == -1:
            return PlannedResourceChange(kind, rid, source, old_cell, None, 0.0)
        if not 0 <= target_raw < sim.instance.num_cells:
            raise InvalidActionError(f"invalid target cell {target_raw}")
        target = target_raw
        cell = sim.cells[target]
        if cell.activity != CellActivity.IDLE:
            raise InvalidActionError("resource cannot be reconfigured into a busy cell")
        if compat[cell.stage] != 1:
            raise InvalidActionError(f"{kind} {rid} is incompatible with target stage")
        if cfg[target] is not None or reserved[target] is not None:
            raise InvalidActionError(f"target cell {target} {kind} capacity is unavailable")

        matrix = (
            sim.instance.worker_relocation_time
            if kind == "worker"
            else sim.instance.robot_relocation_time
        )
        relocation = float(matrix[rid][source][target])
        if source == target or relocation <= 1e-12:
            cfg[target] = rid
        else:
            reserved[target] = rid
        return PlannedResourceChange(kind, rid, source, old_cell, target, relocation)

    if action.reconfigure:
        seen_h: set[int] = set()
        for assignment in action.worker_assignments:
            worker_changes.append(plan_resource("worker", assignment, seen_h))
        seen_r: set[int] = set()
        for assignment in action.robot_assignments:
            robot_changes.append(plan_resource("robot", assignment, seen_r))

        # Both immediate and in-transit planned occupants must be HR-compatible.
        # Otherwise a later RELOCATION_FINISH event could silently create an
        # illegal H+R cell after the composite action had already been accepted.
        for cell in sim.cells:
            w = worker_cfg[cell.cell_id]
            if w is None:
                w = worker_res[cell.cell_id]
            r = robot_cfg[cell.cell_id]
            if r is None:
                r = robot_res[cell.cell_id]
            if w is not None and r is not None:
                stage = cell.stage
                if not bool(sim.instance.hr_compatibility[w][r][stage]):
                    raise InvalidActionError(
                        f"cell {cell.cell_id} planned worker/robot pair is not HR-compatible"
                    )

        if bool(sim.cfg.env.action_constraints.enforce_basic_executable_coverage_after_reconfiguration):
            if not _coverage_ok(sim, worker_cfg, robot_cfg):
                raise InvalidActionError(
                    "reconfiguration violates the configured basic stage-coverage rule"
                )

    # Scheduling is validated against the post-reconfiguration *currently arrived*
    # resources. In-transit resources reserve their destination but are not usable.
    starts: list[PlannedStart] = []
    used_ops: set[int] = set()
    used_cells: set[int] = set()
    used_workers: set[int] = set()
    used_robots: set[int] = set()

    for assignment in action.schedule_assignments:
        op_id = int(assignment.operation_id)
        cell_id = int(assignment.cell_id)
        if not 0 <= op_id < len(sim.operations):
            raise InvalidActionError(f"invalid operation id {op_id}")
        if not 0 <= cell_id < len(sim.cells):
            raise InvalidActionError(f"invalid cell id {cell_id}")
        if op_id in used_ops:
            raise InvalidActionError(f"operation {op_id} selected twice")
        if cell_id in used_cells:
            raise InvalidActionError(f"cell {cell_id} selected twice")
        op = sim.operations[op_id]
        cell = sim.cells[cell_id]
        if op.status != OperationStatus.READY:
            raise InvalidActionError(f"operation {op_id} is not READY")
        if cell.activity != CellActivity.IDLE:
            raise InvalidActionError(f"cell {cell_id} is busy")
        if cell.stage != op.stage:
            raise InvalidActionError("operation stage and dedicated cell stage do not match")
        if worker_res[cell_id] is not None or robot_res[cell_id] is not None:
            raise InvalidActionError("cell with an incoming resource reservation cannot start work")

        mode_info = _configured_mode_and_time(
            sim, op_id, cell_id, worker_cfg=worker_cfg, robot_cfg=robot_cfg
        )
        if mode_info is None:
            raise InvalidActionError("cell configuration does not form a legal execution mode")
        mode, worker, robot, p = mode_info
        if worker is not None:
            if sim.workers[worker].activity != ResourceActivity.IDLE or worker in used_workers:
                raise InvalidActionError("configured worker is unavailable or reused")
            used_workers.add(worker)
        if robot is not None:
            if sim.robots[robot].activity != ResourceActivity.IDLE or robot in used_robots:
                raise InvalidActionError("configured robot is unavailable or reused")
            used_robots.add(robot)
        used_ops.add(op_id)
        used_cells.add(cell_id)
        starts.append(PlannedStart(op_id, cell_id, mode, worker, robot, p))

    return ValidatedAction(
        reconfigure=action.reconfigure,
        worker_changes=tuple(worker_changes),
        robot_changes=tuple(robot_changes),
        starts=tuple(starts),
    )
