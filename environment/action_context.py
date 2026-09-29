"""Policy-side action context for Phase G autoregressive matching.

The heterogeneous graph carries learnable state features.  The matching decoders
also need exact, non-learned feasibility facts to update masks after each selected
edge without reaching into simulator internals.  ``ActionContext`` is that stable
read-only contract.

Only already-arrived operations are included, so the Phase D no-future-leakage
contract remains intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from environment.entities import CellActivity, OperationStatus, ResourceActivity


@dataclass(frozen=True, slots=True)
class ActionContextStaticFacts:
    """Per-instance tensors shared by every event of one episode."""

    cell_stage: torch.Tensor
    worker_skill: torch.Tensor
    worker_relocation_time: torch.Tensor
    robot_capability: torch.Tensor
    robot_relocation_time: torch.Tensor
    hr_compatibility: torch.Tensor
    processing_h_all: torch.Tensor
    processing_r_all: torch.Tensor
    processing_hr_all: torch.Tensor


@dataclass(frozen=True, slots=True)
class ActionContext:
    operation_ids: torch.Tensor                 # [O] global operation ids
    operation_stage: torch.Tensor               # [O]
    operation_ready: torch.Tensor               # [O] bool
    operation_slack_minutes: torch.Tensor       # [O]

    cell_stage: torch.Tensor                    # [M]
    cell_idle: torch.Tensor                     # [M] bool
    cell_worker: torch.Tensor                   # [M], -1 if none
    cell_robot: torch.Tensor                    # [M], -1 if none
    cell_reserved_worker: torch.Tensor          # [M], -1 if none
    cell_reserved_robot: torch.Tensor           # [M], -1 if none

    worker_idle: torch.Tensor                   # [H] bool
    worker_configured_cell: torch.Tensor        # [H], -1 if none
    worker_physical_cell: torch.Tensor          # [H]
    worker_skill: torch.Tensor                  # [H,S] bool
    worker_relocation_time: torch.Tensor        # [H,M,M]

    robot_idle: torch.Tensor                    # [R] bool
    robot_configured_cell: torch.Tensor         # [R], -1 if none
    robot_physical_cell: torch.Tensor           # [R]
    robot_capability: torch.Tensor              # [R,S] bool
    robot_relocation_time: torch.Tensor         # [R,M,M]

    hr_compatibility: torch.Tensor              # [H,R,S] bool
    processing_h: torch.Tensor                  # [O,H], inf if illegal
    processing_r: torch.Tensor                  # [O,R], inf if illegal
    processing_hr: torch.Tensor                 # [O,H,R], inf if illegal

    # Phase I feasibility fact: whether the simulator already has a future
    # physical event queued before the current composite action is applied.
    # This exposes no event time/type and is used only to mask a schedule STOP
    # that would otherwise create an immediate physical deadlock.
    pending_physical_event: torch.Tensor         # [1] bool

    @property
    def num_operations(self) -> int:
        return int(self.operation_ids.numel())

    @property
    def num_cells(self) -> int:
        return int(self.cell_stage.numel())

    @property
    def num_workers(self) -> int:
        return int(self.worker_idle.numel())

    @property
    def num_robots(self) -> int:
        return int(self.robot_idle.numel())

    @property
    def num_stages(self) -> int:
        if self.cell_stage.numel() == 0:
            return 0
        return int(self.cell_stage.max().item()) + 1

    def to(self, device: torch.device | str) -> "ActionContext":
        return ActionContext(**{
            field: getattr(self, field).to(device)
            for field in self.__dataclass_fields__
        })

    def validate(self) -> None:
        o, m, h, r = (
            self.num_operations,
            self.num_cells,
            self.num_workers,
            self.num_robots,
        )
        if self.operation_stage.shape != (o,):
            raise ValueError("operation_stage shape mismatch")
        if self.operation_ready.shape != (o,) or self.operation_ready.dtype != torch.bool:
            raise ValueError("operation_ready must be [O] bool")
        if self.operation_slack_minutes.shape != (o,):
            raise ValueError("operation_slack_minutes shape mismatch")
        if self.cell_stage.shape != (m,):
            raise ValueError("cell_stage shape mismatch")
        if self.cell_idle.shape != (m,) or self.cell_idle.dtype != torch.bool:
            raise ValueError("cell_idle must be [M] bool")
        for name in (
            "cell_worker", "cell_robot", "cell_reserved_worker", "cell_reserved_robot"
        ):
            if getattr(self, name).shape != (m,):
                raise ValueError(f"{name} shape mismatch")
        if self.worker_idle.shape != (h,) or self.worker_idle.dtype != torch.bool:
            raise ValueError("worker_idle must be [H] bool")
        if self.worker_configured_cell.shape != (h,) or self.worker_physical_cell.shape != (h,):
            raise ValueError("worker location shape mismatch")
        if self.robot_idle.shape != (r,) or self.robot_idle.dtype != torch.bool:
            raise ValueError("robot_idle must be [R] bool")
        if self.robot_configured_cell.shape != (r,) or self.robot_physical_cell.shape != (r,):
            raise ValueError("robot location shape mismatch")
        s = self.num_stages
        if self.worker_skill.shape != (h, s) or self.worker_skill.dtype != torch.bool:
            raise ValueError("worker_skill must be [H,S] bool")
        if self.robot_capability.shape != (r, s) or self.robot_capability.dtype != torch.bool:
            raise ValueError("robot_capability must be [R,S] bool")
        if self.worker_relocation_time.shape != (h, m, m):
            raise ValueError("worker_relocation_time must be [H,M,M]")
        if self.robot_relocation_time.shape != (r, m, m):
            raise ValueError("robot_relocation_time must be [R,M,M]")
        if self.hr_compatibility.shape != (h, r, s) or self.hr_compatibility.dtype != torch.bool:
            raise ValueError("hr_compatibility must be [H,R,S] bool")
        if self.processing_h.shape != (o, h):
            raise ValueError("processing_h must be [O,H]")
        if self.processing_r.shape != (o, r):
            raise ValueError("processing_r must be [O,R]")
        if self.processing_hr.shape != (o, h, r):
            raise ValueError("processing_hr must be [O,H,R]")
        if self.pending_physical_event.shape != (1,) or self.pending_physical_event.dtype != torch.bool:
            raise ValueError("pending_physical_event must be [1] bool")
        if o and int(self.operation_ids.min()) < 0:
            raise ValueError("operation ids must be non-negative")
        if m and (int(self.cell_stage.min()) < 0 or int(self.cell_stage.max()) >= s):
            raise ValueError("cell stage ids out of range")


def build_action_context_static_facts(sim) -> ActionContextStaticFacts:
    """Build immutable feasibility/processing tensors once per instance."""
    if sim.instance is None:
        raise RuntimeError("simulator must be reset before static action-context construction")
    inst = sim.instance
    inf = float("inf")
    p_h = torch.full((len(sim.operations), inst.num_workers), inf, dtype=torch.float32)
    p_r = torch.full((len(sim.operations), inst.num_robots), inf, dtype=torch.float32)
    p_hr = torch.full(
        (len(sim.operations), inst.num_workers, inst.num_robots),
        inf,
        dtype=torch.float32,
    )
    for op in sim.operations:
        oid = int(op.operation_id)
        for h in range(inst.num_workers):
            value = inst.processing_time_h(op.product_type, op.stage, h)
            if value is not None:
                p_h[oid, h] = float(value)
        for r in range(inst.num_robots):
            value = inst.processing_time_r(op.product_type, op.stage, r)
            if value is not None:
                p_r[oid, r] = float(value)
        for h in range(inst.num_workers):
            for r in range(inst.num_robots):
                value = inst.processing_time_hr(op.product_type, op.stage, h, r)
                if value is not None:
                    p_hr[oid, h, r] = float(value)
    return ActionContextStaticFacts(
        cell_stage=torch.tensor(inst.cell_stage, dtype=torch.long),
        worker_skill=torch.tensor(inst.worker_skill, dtype=torch.bool),
        worker_relocation_time=torch.tensor(inst.worker_relocation_time, dtype=torch.float32),
        robot_capability=torch.tensor(inst.robot_capability, dtype=torch.bool),
        robot_relocation_time=torch.tensor(inst.robot_relocation_time, dtype=torch.float32),
        hr_compatibility=torch.tensor(inst.hr_compatibility, dtype=torch.bool),
        processing_h_all=p_h,
        processing_r_all=p_r,
        processing_hr_all=p_hr,
    )


def _remaining_min_work(sim, order_id: int) -> float:
    total = 0.0
    for op_id in sim.order_operation_ids[order_id]:
        op = sim.operations[op_id]
        if op.status == OperationStatus.DONE:
            continue
        if op.status == OperationStatus.PROCESSING:
            if op.completion_time is not None:
                total += max(0.0, float(op.completion_time - sim.time))
            continue
        replay_cache = getattr(sim, "_replay_minimum_processing_by_operation", None)
        if replay_cache is not None:
            total += float(replay_cache[int(op.operation_id)])
        else:
            total += float(sim.instance.minimum_processing_time(op.product_type, op.stage))
    return total


def build_action_context(sim) -> ActionContext:
    if sim.instance is None:
        raise RuntimeError("simulator must be reset before action-context construction")
    inst = sim.instance
    visible_ops = [op for op in sim.operations if sim.orders[op.order_id].arrived]

    operation_ids = [op.operation_id for op in visible_ops]
    operation_stage = [op.stage for op in visible_ops]
    operation_ready = [op.status == OperationStatus.READY for op in visible_ops]
    operation_slack = []
    for op in visible_ops:
        order = inst.orders[op.order_id]
        slack = float(order.due_date - sim.time - _remaining_min_work(sim, op.order_id))
        operation_slack.append(slack)

    static = getattr(sim, "_action_context_static_facts", None)
    if static is None:
        static = build_action_context_static_facts(sim)
        sim._action_context_static_facts = static
    visible_ids = torch.tensor(operation_ids, dtype=torch.long)
    p_h = static.processing_h_all.index_select(0, visible_ids)
    p_r = static.processing_r_all.index_select(0, visible_ids)
    p_hr = static.processing_hr_all.index_select(0, visible_ids)

    context = ActionContext(
        operation_ids=torch.tensor(operation_ids, dtype=torch.long),
        operation_stage=torch.tensor(operation_stage, dtype=torch.long),
        operation_ready=torch.tensor(operation_ready, dtype=torch.bool),
        operation_slack_minutes=torch.tensor(operation_slack, dtype=torch.float32),
        cell_stage=static.cell_stage,
        cell_idle=torch.tensor(
            [cell.activity == CellActivity.IDLE for cell in sim.cells], dtype=torch.bool
        ),
        cell_worker=torch.tensor(
            [-1 if cell.configured_worker is None else cell.configured_worker for cell in sim.cells],
            dtype=torch.long,
        ),
        cell_robot=torch.tensor(
            [-1 if cell.configured_robot is None else cell.configured_robot for cell in sim.cells],
            dtype=torch.long,
        ),
        cell_reserved_worker=torch.tensor(
            [-1 if cell.reserved_worker is None else cell.reserved_worker for cell in sim.cells],
            dtype=torch.long,
        ),
        cell_reserved_robot=torch.tensor(
            [-1 if cell.reserved_robot is None else cell.reserved_robot for cell in sim.cells],
            dtype=torch.long,
        ),
        worker_idle=torch.tensor(
            [w.activity == ResourceActivity.IDLE for w in sim.workers], dtype=torch.bool
        ),
        worker_configured_cell=torch.tensor(
            [-1 if w.configured_cell is None else w.configured_cell for w in sim.workers],
            dtype=torch.long,
        ),
        worker_physical_cell=torch.tensor([w.physical_cell for w in sim.workers], dtype=torch.long),
        worker_skill=static.worker_skill,
        worker_relocation_time=static.worker_relocation_time,
        robot_idle=torch.tensor(
            [r.activity == ResourceActivity.IDLE for r in sim.robots], dtype=torch.bool
        ),
        robot_configured_cell=torch.tensor(
            [-1 if r.configured_cell is None else r.configured_cell for r in sim.robots],
            dtype=torch.long,
        ),
        robot_physical_cell=torch.tensor([r.physical_cell for r in sim.robots], dtype=torch.long),
        robot_capability=static.robot_capability,
        robot_relocation_time=static.robot_relocation_time,
        hr_compatibility=static.hr_compatibility,
        processing_h=p_h,
        processing_r=p_r,
        processing_hr=p_hr,
        pending_physical_event=torch.tensor([len(sim.events) > 0], dtype=torch.bool),
    )
    context.validate()
    return context


__all__ = ["ActionContext", "ActionContextStaticFacts", "build_action_context", "build_action_context_static_facts"]
