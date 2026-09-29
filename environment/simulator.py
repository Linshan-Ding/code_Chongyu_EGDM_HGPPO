"""Physical discrete-event simulator for EGDM-HGPPO.

This module contains no neural-network logic. All learned policies and all
baselines must share this exact simulator in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.schema import AssemblyInstance
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
from environment.events import Event, EventQueue, EventType
from environment.masks import ValidatedAction
from environment.reward import final_twt, interval_twt_reward
from environment.state import (
    CellView,
    DecisionState,
    OperationView,
    OrderView,
    ResourceView,
)


class SimulatorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AdvanceResult:
    delta_t: float
    base_reward: float
    event_types: tuple[EventType, ...]


class AssemblySimulator:
    def __init__(self, cfg, *, minimum_dwell_time: float | None = None) -> None:
        self.cfg = cfg
        self.minimum_dwell_time = (
            float(cfg.instance.minimum_dwell_time_minutes.training_default)
            if minimum_dwell_time is None
            else float(minimum_dwell_time)
        )
        if self.minimum_dwell_time < 0:
            raise ValueError("minimum_dwell_time must be non-negative")

        self.instance: AssemblyInstance | None = None
        self.time = 0.0
        self.orders: list[OrderRuntime] = []
        self.operations: list[OperationRuntime] = []
        self.order_operation_ids: list[list[int]] = []
        self.cells: list[CellRuntime] = []
        self.workers: list[WorkerRuntime] = []
        self.robots: list[RobotRuntime] = []
        self.events = EventQueue()
        self.current_event_types: tuple[EventType, ...] = ()
        self.last_reconfiguration_time = 0.0
        self.reconfiguration_count = 0
        self.cumulative_base_reward = 0.0
        self.decision_index = -1
        # Event-level ActionContext contains large per-instance processing tables.
        # Cache them only for the active episode and invalidate on every reset.
        self._action_context_static_facts = None
        self._compact_replay_episode_static = None
        self._replay_minimum_processing_by_operation = None
        self._replay_mode_min_by_operation = None

    def reset(self, instance: AssemblyInstance) -> DecisionState:
        instance.validate()
        self.instance = instance
        self.time = 0.0
        self.events.clear()
        self.current_event_types = ()
        self.last_reconfiguration_time = 0.0
        self.reconfiguration_count = 0
        self.cumulative_base_reward = 0.0
        self.decision_index = -1
        self._action_context_static_facts = None
        self._compact_replay_episode_static = None
        self._replay_minimum_processing_by_operation = None
        self._replay_mode_min_by_operation = None

        self.orders = [OrderRuntime(order_id=o.order_id) for o in instance.orders]
        self.cells = [
            CellRuntime(cell_id=m, stage=int(stage))
            for m, stage in enumerate(instance.cell_stage)
        ]
        self.workers = []
        for h, configured in enumerate(instance.initial_worker_cell):
            configured_cell = None if configured < 0 else int(configured)
            physical = configured_cell
            if physical is None:
                physical = self._first_compatible_cell("worker", h)
            self.workers.append(
                WorkerRuntime(
                    worker_id=h,
                    configured_cell=configured_cell,
                    physical_cell=physical,
                )
            )
            if configured_cell is not None:
                if self.cells[configured_cell].configured_worker is not None:
                    raise SimulatorError("initial worker cell capacity conflict")
                self.cells[configured_cell].configured_worker = h

        self.robots = []
        for r, configured in enumerate(instance.initial_robot_cell):
            configured_cell = None if configured < 0 else int(configured)
            physical = configured_cell
            if physical is None:
                physical = self._first_compatible_cell("robot", r)
            self.robots.append(
                RobotRuntime(
                    robot_id=r,
                    configured_cell=configured_cell,
                    physical_cell=physical,
                )
            )
            if configured_cell is not None:
                if self.cells[configured_cell].configured_robot is not None:
                    raise SimulatorError("initial robot cell capacity conflict")
                self.cells[configured_cell].configured_robot = r

        self.operations = []
        self.order_operation_ids = [[] for _ in instance.orders]
        for order in instance.orders:
            route = instance.product_routes[order.product_type]
            for pos, stage in enumerate(route):
                op_id = len(self.operations)
                self.operations.append(
                    OperationRuntime(
                        operation_id=op_id,
                        order_id=order.order_id,
                        product_type=order.product_type,
                        stage=int(stage),
                        route_position=pos,
                    )
                )
                self.order_operation_ids[order.order_id].append(op_id)

        # These graph features depend only on the immutable instance. Computing
        # them once avoids repeating worker x robot scans at every decision.
        mode_by_product_stage = {}
        minimum_by_operation = []
        mode_by_operation = []
        for operation in self.operations:
            key = (int(operation.product_type), int(operation.stage))
            cached = mode_by_product_stage.get(key)
            if cached is None:
                h_values = [
                    value for worker_id in range(instance.num_workers)
                    if (value := instance.processing_time_h(*key, worker_id)) is not None
                ]
                r_values = [
                    value for robot_id in range(instance.num_robots)
                    if (value := instance.processing_time_r(*key, robot_id)) is not None
                ]
                hr_values = [
                    value
                    for worker_id in range(instance.num_workers)
                    for robot_id in range(instance.num_robots)
                    if (
                        value := instance.processing_time_hr(
                            *key, worker_id, robot_id
                        )
                    ) is not None
                ]
                all_values = [*h_values, *r_values, *hr_values]
                if not all_values:
                    raise SimulatorError(
                        f"product={key[0]}, stage={key[1]} has no executable mode"
                    )
                cached = (
                    float(min(all_values)),
                    (
                        float(min(h_values)) if h_values else 0.0,
                        float(min(r_values)) if r_values else 0.0,
                        float(min(hr_values)) if hr_values else 0.0,
                        int(bool(h_values)),
                        int(bool(r_values)),
                        int(bool(hr_values)),
                    ),
                )
                mode_by_product_stage[key] = cached
            minimum, mode = cached
            minimum_by_operation.append(minimum)
            mode_by_operation.append(mode)
        self._replay_minimum_processing_by_operation = tuple(minimum_by_operation)
        self._replay_mode_min_by_operation = tuple(mode_by_operation)

        for order in instance.orders:
            self.events.push(
                float(order.release_time),
                EventType.ORDER_ARRIVAL,
                order_id=order.order_id,
            )

        if len(self.events) == 0:
            raise SimulatorError("instance contains no arrival events")
        self._advance_to_next_batch(initial=True)
        self.assert_consistent()
        return self.snapshot()

    def _first_compatible_cell(self, kind: str, resource_id: int) -> int:
        assert self.instance is not None
        compat = (
            self.instance.worker_skill[resource_id]
            if kind == "worker"
            else self.instance.robot_capability[resource_id]
        )
        for m, stage in enumerate(self.instance.cell_stage):
            if compat[stage] == 1:
                return m
        raise SimulatorError(f"{kind} {resource_id} has no compatible physical anchor")

    def apply_validated_action(self, plan: ValidatedAction) -> None:
        if plan.reconfigure:
            self.last_reconfiguration_time = float(self.time)
            self.reconfiguration_count += 1
            for change in plan.worker_changes:
                self._apply_resource_change(change)
            for change in plan.robot_changes:
                self._apply_resource_change(change)

        for start in plan.starts:
            self._start_operation(start)
        self.assert_consistent()

    def _apply_resource_change(self, change) -> None:
        if change.kind == "worker":
            resource = self.workers[change.resource_id]
            configured_attr = "configured_worker"
            reserved_attr = "reserved_worker"
            event_kind = "worker"
        else:
            resource = self.robots[change.resource_id]
            configured_attr = "configured_robot"
            reserved_attr = "reserved_robot"
            event_kind = "robot"

        if resource.activity != ResourceActivity.IDLE:
            raise SimulatorError("validated resource became non-idle before application")

        old_cell = resource.configured_cell
        if old_cell is not None:
            cell = self.cells[old_cell]
            if getattr(cell, configured_attr) != change.resource_id:
                raise SimulatorError("resource/cell configuration mismatch")
            setattr(cell, configured_attr, None)
        resource.configured_cell = None

        if change.target_cell is None:
            # Temporary unconfigured state. Physical location is retained.
            resource.physical_cell = change.source_physical_cell
            resource.relocation_target = None
            return

        target = int(change.target_cell)
        if change.source_physical_cell == target or change.relocation_time <= 1e-12:
            target_cell = self.cells[target]
            if getattr(target_cell, configured_attr) is not None:
                raise SimulatorError("target capacity unexpectedly occupied")
            setattr(target_cell, configured_attr, change.resource_id)
            resource.configured_cell = target
            resource.physical_cell = target
            resource.relocation_target = None
            resource.activity = ResourceActivity.IDLE
            return

        target_cell = self.cells[target]
        if getattr(target_cell, reserved_attr) not in {None, change.resource_id}:
            raise SimulatorError("target capacity unexpectedly reserved")
        setattr(target_cell, reserved_attr, change.resource_id)
        resource.activity = ResourceActivity.RELOCATING
        resource.physical_cell = change.source_physical_cell
        resource.relocation_target = target
        resource.busy_until = self.time + float(change.relocation_time)
        self.events.push(
            resource.busy_until,
            EventType.RELOCATION_FINISH,
            resource_kind=event_kind,
            resource_id=change.resource_id,
            target_cell=target,
        )

    def _start_operation(self, start) -> None:
        assert self.instance is not None
        op = self.operations[start.operation_id]
        cell = self.cells[start.cell_id]
        if op.status != OperationStatus.READY or cell.activity != CellActivity.IDLE:
            raise SimulatorError("validated scheduling action became invalid")

        finish = self.time + float(start.processing_time)
        op.status = OperationStatus.PROCESSING
        op.start_time = float(self.time)
        op.assigned_cell = start.cell_id
        op.worker_id = start.worker_id
        op.robot_id = start.robot_id
        op.mode = start.mode

        cell.activity = CellActivity.BUSY
        cell.processing_operation = start.operation_id
        cell.busy_until = finish

        if start.worker_id is not None:
            worker = self.workers[start.worker_id]
            worker.activity = ResourceActivity.BUSY
            worker.busy_operation = start.operation_id
            worker.busy_until = finish
        if start.robot_id is not None:
            robot = self.robots[start.robot_id]
            robot.activity = ResourceActivity.BUSY
            robot.busy_operation = start.operation_id
            robot.busy_until = finish

        self.events.push(
            finish,
            EventType.OPERATION_FINISH,
            operation_id=start.operation_id,
        )

    def advance(self) -> AdvanceResult:
        if self.is_done:
            return AdvanceResult(0.0, 0.0, ())
        if len(self.events) == 0:
            raise SimulatorError(
                "deadlock: unfinished operations remain but the future-event queue is empty"
            )
        return self._advance_to_next_batch(initial=False)

    def _advance_to_next_batch(self, *, initial: bool) -> AdvanceResult:
        assert self.instance is not None
        t_next = self.events.peek_time()
        if t_next < self.time - 1e-12:
            raise SimulatorError("event queue attempted to move time backwards")
        t0 = float(self.time)
        base_reward = interval_twt_reward(self.instance, self.orders, t0, t_next)
        self.cumulative_base_reward += base_reward
        self.time = float(t_next)
        batch = self.events.pop_time_batch()
        for event in batch:
            self._process_event(event)

        # Unique event types preserve simultaneous-event information without leaking
        # multiplicity into the future event-type critic interface.
        seen: list[EventType] = []
        for event in batch:
            if event.event_type not in seen:
                seen.append(event.event_type)
        self.current_event_types = tuple(seen)
        self.decision_index = 0 if initial else self.decision_index + 1
        self.assert_consistent()
        return AdvanceResult(
            delta_t=float(t_next - t0),
            base_reward=float(base_reward),
            event_types=self.current_event_types,
        )

    def _process_event(self, event: Event) -> None:
        if event.event_type == EventType.ORDER_ARRIVAL:
            self._process_arrival(int(event.payload["order_id"]))
        elif event.event_type == EventType.OPERATION_FINISH:
            self._process_operation_finish(int(event.payload["operation_id"]))
        elif event.event_type == EventType.RELOCATION_FINISH:
            self._process_relocation_finish(
                str(event.payload["resource_kind"]),
                int(event.payload["resource_id"]),
                int(event.payload["target_cell"]),
            )
        else:
            raise SimulatorError(f"unsupported event type {event.event_type}")

    def _process_arrival(self, order_id: int) -> None:
        order = self.orders[order_id]
        if order.arrived:
            raise SimulatorError("duplicate order-arrival event")
        order.arrived = True
        op_ids = self.order_operation_ids[order_id]
        if not op_ids:
            raise SimulatorError("order has no required operation")
        self.operations[op_ids[0]].status = OperationStatus.READY
        for op_id in op_ids[1:]:
            self.operations[op_id].status = OperationStatus.BLOCKED

    def _process_operation_finish(self, operation_id: int) -> None:
        op = self.operations[operation_id]
        if op.status != OperationStatus.PROCESSING:
            raise SimulatorError("finish event for non-processing operation")
        cell = self.cells[op.assigned_cell]
        if cell.processing_operation != operation_id:
            raise SimulatorError("cell does not own finishing operation")

        op.status = OperationStatus.DONE
        op.completion_time = float(self.time)
        cell.activity = CellActivity.IDLE
        cell.processing_operation = None
        cell.busy_until = None

        if op.worker_id is not None:
            worker = self.workers[op.worker_id]
            worker.activity = ResourceActivity.IDLE
            worker.busy_operation = None
            worker.busy_until = None
        if op.robot_id is not None:
            robot = self.robots[op.robot_id]
            robot.activity = ResourceActivity.IDLE
            robot.busy_operation = None
            robot.busy_until = None

        op_ids = self.order_operation_ids[op.order_id]
        next_pos = op.route_position + 1
        if next_pos < len(op_ids):
            next_op = self.operations[op_ids[next_pos]]
            if next_op.status != OperationStatus.BLOCKED:
                raise SimulatorError("next operation is not BLOCKED at predecessor completion")
            next_op.status = OperationStatus.READY
        else:
            runtime = self.orders[op.order_id]
            runtime.completion_time = float(self.time)

    def _process_relocation_finish(self, kind: str, resource_id: int, target: int) -> None:
        if kind == "worker":
            resource = self.workers[resource_id]
            cell = self.cells[target]
            if cell.reserved_worker != resource_id:
                raise SimulatorError("worker relocation target reservation missing")
            if cell.configured_worker is not None:
                raise SimulatorError("worker target capacity occupied on arrival")
            cell.reserved_worker = None
            cell.configured_worker = resource_id
        elif kind == "robot":
            resource = self.robots[resource_id]
            cell = self.cells[target]
            if cell.reserved_robot != resource_id:
                raise SimulatorError("robot relocation target reservation missing")
            if cell.configured_robot is not None:
                raise SimulatorError("robot target capacity occupied on arrival")
            cell.reserved_robot = None
            cell.configured_robot = resource_id
        else:
            raise SimulatorError(f"unknown resource kind {kind}")

        if resource.activity != ResourceActivity.RELOCATING:
            raise SimulatorError("relocation-finish event for non-relocating resource")
        if resource.relocation_target != target:
            raise SimulatorError("relocation target mismatch")
        resource.activity = ResourceActivity.IDLE
        resource.configured_cell = target
        resource.physical_cell = target
        resource.relocation_target = None
        resource.busy_until = None

    @property
    def is_done(self) -> bool:
        return bool(self.orders) and all(order.completed for order in self.orders)

    def twt(self) -> float:
        if not self.is_done:
            raise SimulatorError("TWT is final only after all orders complete")
        assert self.instance is not None
        return final_twt(self.instance, self.orders)

    def snapshot(self) -> DecisionState:
        assert self.instance is not None
        visible_order_ids = {o.order_id for o in self.orders if o.arrived}
        order_views = tuple(
            OrderView(
                order_id=data.order_id,
                product_type=data.product_type,
                release_time=float(data.release_time),
                due_date=float(data.due_date),
                weight=int(data.weight),
                completed=runtime.completed,
                completion_time=runtime.completion_time,
            )
            for runtime, data in zip(self.orders, self.instance.orders, strict=True)
            if runtime.arrived
        )
        operation_views = tuple(
            OperationView(
                operation_id=op.operation_id,
                order_id=op.order_id,
                product_type=op.product_type,
                stage=op.stage,
                route_position=op.route_position,
                status=op.status,
                start_time=op.start_time,
                completion_time=op.completion_time,
                assigned_cell=op.assigned_cell,
                mode=op.mode,
            )
            for op in self.operations
            if op.order_id in visible_order_ids
        )
        cells = tuple(
            CellView(
                cell_id=c.cell_id,
                stage=c.stage,
                activity=c.activity,
                configured_worker=c.configured_worker,
                configured_robot=c.configured_robot,
                reserved_worker=c.reserved_worker,
                reserved_robot=c.reserved_robot,
                processing_operation=c.processing_operation,
                busy_until=c.busy_until,
            )
            for c in self.cells
        )
        workers = tuple(
            ResourceView(
                resource_id=w.worker_id,
                activity=w.activity,
                configured_cell=w.configured_cell,
                physical_cell=w.physical_cell,
                relocation_target=w.relocation_target,
                busy_operation=w.busy_operation,
                busy_until=w.busy_until,
            )
            for w in self.workers
        )
        robots = tuple(
            ResourceView(
                resource_id=r.robot_id,
                activity=r.activity,
                configured_cell=r.configured_cell,
                physical_cell=r.physical_cell,
                relocation_target=r.relocation_target,
                busy_operation=r.busy_operation,
                busy_until=r.busy_until,
            )
            for r in self.robots
        )
        return DecisionState(
            time=float(self.time),
            event_types=self.current_event_types,
            orders=order_views,
            operations=operation_views,
            cells=cells,
            workers=workers,
            robots=robots,
            last_reconfiguration_time=float(self.last_reconfiguration_time),
            cumulative_base_reward=float(self.cumulative_base_reward),
            decision_index=int(self.decision_index),
        )

    def assert_consistent(self) -> None:
        # Cell -> resource ownership.
        seen_h: set[int] = set()
        seen_r: set[int] = set()
        for cell in self.cells:
            if cell.configured_worker is not None:
                h = cell.configured_worker
                if h in seen_h:
                    raise SimulatorError("worker configured in more than one cell")
                seen_h.add(h)
                if self.workers[h].configured_cell != cell.cell_id:
                    raise SimulatorError("worker configuration is not bidirectionally consistent")
            if cell.configured_robot is not None:
                r = cell.configured_robot
                if r in seen_r:
                    raise SimulatorError("robot configured in more than one cell")
                seen_r.add(r)
                if self.robots[r].configured_cell != cell.cell_id:
                    raise SimulatorError("robot configuration is not bidirectionally consistent")
            if cell.activity == CellActivity.BUSY and cell.processing_operation is None:
                raise SimulatorError("busy cell has no processing operation")
            if cell.activity == CellActivity.IDLE and cell.processing_operation is not None:
                raise SimulatorError("idle cell still owns a processing operation")

        for w in self.workers:
            if w.activity == ResourceActivity.RELOCATING and w.configured_cell is not None:
                raise SimulatorError("relocating worker cannot be configured")
            if w.activity == ResourceActivity.BUSY and w.busy_operation is None:
                raise SimulatorError("busy worker has no operation")
        for r in self.robots:
            if r.activity == ResourceActivity.RELOCATING and r.configured_cell is not None:
                raise SimulatorError("relocating robot cannot be configured")
            if r.activity == ResourceActivity.BUSY and r.busy_operation is None:
                raise SimulatorError("busy robot has no operation")
