"""Build the paper's five-node dynamic heterogeneous graph from simulator state.

The builder is deterministic and contains no learnable parameters.  It is part of
the environment/state representation, not the agent.  Only arrived orders are
materialized as operation nodes, which preserves the Phase D no-future-leakage
contract.
"""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Iterable

import torch

from environment.entities import CellActivity, ExecutionMode, OperationStatus, ResourceActivity
from environment.events import EventType
from environment.graph_types import EdgeStore, EdgeType, HeteroGraph, NodeStore
from environment.masks import gate_reconfigure_feasible


NODE_TYPES = ("operation", "stage", "cell", "worker", "robot")
EVENT_TYPES = (
    EventType.ORDER_ARRIVAL,
    EventType.OPERATION_FINISH,
    EventType.RELOCATION_FINISH,
)
STATUS_INDEX = {
    OperationStatus.NOT_RELEASED: 0,
    OperationStatus.BLOCKED: 1,
    OperationStatus.READY: 2,
    OperationStatus.PROCESSING: 3,
    OperationStatus.DONE: 4,
}
MODE_INDEX = {
    ExecutionMode.IDLE: 0,
    ExecutionMode.H: 1,
    ExecutionMode.R: 2,
    ExecutionMode.HR: 3,
}


class GraphBuildError(RuntimeError):
    pass


def _matrix(rows: list[list[float]], width: int) -> torch.Tensor:
    if rows:
        return torch.tensor(rows, dtype=torch.float32)
    return torch.empty((0, width), dtype=torch.float32)


def _binary_matrix(rows: list[list[float]], width: int) -> torch.Tensor:
    if rows:
        return torch.tensor(rows, dtype=torch.float32)
    return torch.empty((0, width), dtype=torch.float32)


def _node_store(
    ids: list[int],
    continuous: list[list[float]],
    binary: list[list[float]],
    categorical: dict[str, list[int]],
    continuous_names: tuple[str, ...],
    binary_names: tuple[str, ...],
) -> NodeStore:
    n = len(ids)
    return NodeStore(
        ids=torch.tensor(ids, dtype=torch.long),
        continuous=_matrix(continuous, len(continuous_names)),
        binary=_binary_matrix(binary, len(binary_names)),
        categorical={k: torch.tensor(v, dtype=torch.long) for k, v in categorical.items()},
        continuous_names=continuous_names,
        binary_names=binary_names,
        batch=torch.zeros((n,), dtype=torch.long),
        ptr=torch.tensor([0, n], dtype=torch.long),
    )


def _edge_store(
    pairs: list[tuple[int, int]],
    continuous: list[list[float]],
    binary: list[list[float]],
    continuous_names: tuple[str, ...] = (),
    binary_names: tuple[str, ...] = (),
) -> EdgeStore:
    e = len(pairs)
    if pairs:
        edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return EdgeStore(
        edge_index=edge_index,
        continuous=_matrix(continuous, len(continuous_names)),
        binary=_binary_matrix(binary, len(binary_names)),
        continuous_names=continuous_names,
        binary_names=binary_names,
        batch=torch.zeros((e,), dtype=torch.long),
    )


def _coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if abs(mean) <= 1e-12:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return sqrt(max(variance, 0.0)) / abs(mean)


class DynamicHeteroGraphBuilder:
    """Convert one current simulator state into the dynamic heterogeneous graph."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        graph_cfg = cfg.env.graph
        self.max_stages = int(graph_cfg.max_stages)
        if self.max_stages <= 0:
            raise ValueError("env.graph.max_stages must be positive")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, sim) -> HeteroGraph:
        if sim.instance is None:
            raise GraphBuildError("simulator must be reset before graph construction")
        inst = sim.instance
        if inst.num_stages > self.max_stages:
            raise GraphBuildError(
                f"instance has {inst.num_stages} stages but graph max_stages={self.max_stages}"
            )

        visible_ops = [op for op in sim.operations if sim.orders[op.order_id].arrived]
        visible_order_ids = {op.order_id for op in visible_ops}
        op_local = {op.operation_id: i for i, op in enumerate(visible_ops)}

        order_remaining = {
            order_id: self._order_remaining_min_work(sim, order_id)
            for order_id in visible_order_ids
        }
        order_slack = {
            order_id: self._order_slack(sim, order_id, order_remaining[order_id])
            for order_id in visible_order_ids
        }
        order_pressure = {
            order_id: self._order_due_pressure(sim, order_id, order_slack[order_id])
            for order_id in visible_order_ids
        }

        stage_stats = self._stage_statistics(
            sim,
            visible_ops=visible_ops,
            order_slack=order_slack,
            order_pressure=order_pressure,
        )

        nodes = {
            "operation": self._build_operation_nodes(
                sim, visible_ops, order_remaining, order_slack
            ),
            "stage": self._build_stage_nodes(sim, stage_stats),
            "cell": self._build_cell_nodes(sim, stage_stats),
            "worker": self._build_resource_nodes(sim, "worker"),
            "robot": self._build_resource_nodes(sim, "robot"),
        }
        edges = self._build_edges(
            sim,
            visible_ops=visible_ops,
            op_local=op_local,
            order_slack=order_slack,
            order_pressure=order_pressure,
        )

        workloads = [float(s["remaining_workload_minutes"]) for s in stage_stats]
        ratios = [float(s["load_capacity_ratio"]) for s in stage_stats]
        idle_resources = sum(w.activity == ResourceActivity.IDLE for w in sim.workers) + sum(
            r.activity == ResourceActivity.IDLE for r in sim.robots
        )
        dwell = max(0.0, float(sim.time - sim.last_reconfiguration_time))
        global_due_pressure = sum(
            order_pressure[j]
            for j in visible_order_ids
            if not sim.orders[j].completed
        )
        gate_names = (
            "stage_workload_cv",
            "max_load_capacity_ratio",
            "idle_resource_count",
            "dwell_since_last_reconfiguration_minutes",
            "weighted_due_pressure",
        )
        gate = torch.tensor(
            [[
                _coefficient_of_variation(workloads),
                max(ratios, default=0.0),
                float(idle_resources),
                dwell,
                float(global_due_pressure),
            ]],
            dtype=torch.float32,
        )

        event_multi = torch.zeros((1, len(EVENT_TYPES)), dtype=torch.float32)
        active_event_types = set(sim.current_event_types)
        for i, event_type in enumerate(EVENT_TYPES):
            if event_type in active_event_types:
                event_multi[0, i] = 1.0

        graph = HeteroGraph(
            nodes=nodes,
            edges=edges,
            gate_continuous=gate,
            gate_continuous_names=gate_names,
            event_type_multihot=event_multi,
            reconfigure_feasible=torch.tensor(
                [gate_reconfigure_feasible(sim)], dtype=torch.bool
            ),
            decision_time=torch.tensor([float(sim.time)], dtype=torch.float32),
            decision_index=torch.tensor([int(sim.decision_index)], dtype=torch.long),
            instance_ids=(inst.instance_id,),
            batch_size=1,
        )
        graph.validate()
        return graph

    # ------------------------------------------------------------------
    # Feature semantics
    # ------------------------------------------------------------------
    def _order_remaining_min_work(self, sim, order_id: int) -> float:
        inst = sim.instance
        assert inst is not None
        total = 0.0
        for op_id in sim.order_operation_ids[order_id]:
            op = sim.operations[op_id]
            if op.status == OperationStatus.DONE:
                continue
            if op.status == OperationStatus.PROCESSING and op.assigned_cell is not None:
                busy_until = sim.cells[op.assigned_cell].busy_until
                if busy_until is None:
                    raise GraphBuildError("processing operation has no cell busy_until")
                total += max(0.0, float(busy_until - sim.time))
            else:
                total += self._minimum_processing_time(sim, op)
        return total

    def _order_slack(self, sim, order_id: int, remaining_min_work: float) -> float:
        order = sim.instance.orders[order_id]
        return float(order.due_date - sim.time - remaining_min_work)

    def _order_due_pressure(self, sim, order_id: int, slack: float) -> float:
        """Scale-free implementation of the paper's weighted due-pressure feature.

        The paper specifies the feature concept but not a closed formula. The
        implementation uses the following scale-free due-pressure definition:

            pressure_j = w_j * max(0, 1 - slack_j / (d_j - r_j))

        It is zero for very loose jobs, rises smoothly before tardiness, equals
        ``w_j`` at zero slack, and continues increasing once slack is negative.
        """

        order = sim.instance.orders[order_id]
        due_window = max(float(order.due_date - order.release_time), 1e-12)
        return float(order.weight) * max(0.0, 1.0 - float(slack) / due_window)

    def _operation_ready_time(self, sim, op) -> float | None:
        order = sim.instance.orders[op.order_id]
        if op.route_position == 0:
            return float(order.release_time)
        predecessor_id = sim.order_operation_ids[op.order_id][op.route_position - 1]
        predecessor = sim.operations[predecessor_id]
        return predecessor.completion_time

    def _operation_waiting_time(self, sim, op) -> float:
        ready_time = self._operation_ready_time(sim, op)
        if ready_time is None:
            return 0.0
        if op.start_time is not None:
            return max(0.0, float(op.start_time - ready_time))
        if op.status == OperationStatus.READY:
            return max(0.0, float(sim.time - ready_time))
        return 0.0

    def _mode_min_times(self, sim, op) -> tuple[float, float, float, int, int, int]:
        replay_cache = getattr(sim, "_replay_mode_min_by_operation", None)
        if replay_cache is not None:
            return replay_cache[int(op.operation_id)]
        inst = sim.instance
        h_vals = [
            p for h in range(inst.num_workers)
            if (p := inst.processing_time_h(op.product_type, op.stage, h)) is not None
        ]
        r_vals = [
            p for r in range(inst.num_robots)
            if (p := inst.processing_time_r(op.product_type, op.stage, r)) is not None
        ]
        hr_vals = [
            p for h in range(inst.num_workers) for r in range(inst.num_robots)
            if (p := inst.processing_time_hr(op.product_type, op.stage, h, r)) is not None
        ]
        return (
            float(min(h_vals)) if h_vals else 0.0,
            float(min(r_vals)) if r_vals else 0.0,
            float(min(hr_vals)) if hr_vals else 0.0,
            int(bool(h_vals)),
            int(bool(r_vals)),
            int(bool(hr_vals)),
        )

    def _minimum_processing_time(self, sim, op) -> float:
        replay_cache = getattr(sim, "_replay_minimum_processing_by_operation", None)
        if replay_cache is not None:
            return float(replay_cache[int(op.operation_id)])
        return float(sim.instance.minimum_processing_time(op.product_type, op.stage))

    def _remaining_predecessors(self, sim, op) -> int:
        ids = sim.order_operation_ids[op.order_id][: op.route_position]
        return sum(sim.operations[i].status != OperationStatus.DONE for i in ids)

    def _cell_mode(self, sim, cell) -> tuple[ExecutionMode, bool]:
        inst = sim.instance
        worker = cell.configured_worker
        robot = cell.configured_robot
        if worker is None and robot is None:
            return ExecutionMode.IDLE, False
        if worker is not None and robot is None:
            legal = bool(inst.worker_skill[worker][cell.stage])
            return (ExecutionMode.H if legal else ExecutionMode.IDLE), legal
        if worker is None and robot is not None:
            legal = bool(inst.robot_capability[robot][cell.stage])
            return (ExecutionMode.R if legal else ExecutionMode.IDLE), legal
        legal = bool(inst.hr_compatibility[worker][robot][cell.stage])
        return (ExecutionMode.HR if legal else ExecutionMode.IDLE), legal

    def _configured_processing_time(self, sim, op, cell) -> tuple[float, ExecutionMode, bool]:
        inst = sim.instance
        mode, legal = self._cell_mode(sim, cell)
        if not legal:
            return 0.0, ExecutionMode.IDLE, False
        if mode == ExecutionMode.H:
            p = inst.processing_time_h(op.product_type, op.stage, cell.configured_worker)
        elif mode == ExecutionMode.R:
            p = inst.processing_time_r(op.product_type, op.stage, cell.configured_robot)
        elif mode == ExecutionMode.HR:
            p = inst.processing_time_hr(
                op.product_type,
                op.stage,
                cell.configured_worker,
                cell.configured_robot,
            )
        else:
            p = None
        return (0.0, ExecutionMode.IDLE, False) if p is None else (float(p), mode, True)

    def _stage_statistics(self, sim, *, visible_ops, order_slack, order_pressure):
        inst = sim.instance
        stats: list[dict[str, float]] = []
        for stage in range(inst.num_stages):
            stage_ops = [
                op for op in visible_ops
                if op.stage == stage and op.status != OperationStatus.DONE
            ]
            ready = [op for op in stage_ops if op.status == OperationStatus.READY]
            remaining_work = sum(
                self._minimum_processing_time(sim, op)
                if op.status != OperationStatus.PROCESSING
                else self._order_processing_residual(sim, op)
                for op in stage_ops
            )
            due_pressure = sum(order_pressure[op.order_id] for op in stage_ops)
            cells = [cell for cell in sim.cells if cell.stage == stage]
            busy_cells = sum(cell.activity == CellActivity.BUSY for cell in cells)
            service_cells = 0
            available_cells = 0
            unconfigured_cells = 0
            for cell in cells:
                _, legal = self._cell_mode(sim, cell)
                in_transition = cell.reserved_worker is not None or cell.reserved_robot is not None
                if legal and not in_transition:
                    service_cells += 1
                    if cell.activity == CellActivity.IDLE:
                        available_cells += 1
                else:
                    unconfigured_cells += 1
            human_supply = sum(
                cell.configured_worker is not None for cell in cells
            )
            robot_supply = sum(
                cell.configured_robot is not None for cell in cells
            )
            load_capacity_ratio = float(remaining_work) / max(1.0, float(service_cells))
            stats.append({
                "queue_length": float(len(ready)),
                "remaining_workload_minutes": float(remaining_work),
                "weighted_due_pressure": float(due_pressure),
                "parallel_cell_count": float(len(cells)),
                "available_cell_count": float(available_cells),
                "busy_cell_count": float(busy_cells),
                "unconfigured_cell_count": float(unconfigured_cells),
                "human_supply": float(human_supply),
                "robot_supply": float(robot_supply),
                "load_capacity_ratio": float(load_capacity_ratio),
                "zero_service_capacity": float(service_cells == 0),
                "ready_workload_minutes": float(sum(
                    self._minimum_processing_time(sim, op) for op in ready
                )),
            })
        return stats

    def _order_processing_residual(self, sim, op) -> float:
        if op.assigned_cell is None:
            return 0.0
        busy_until = sim.cells[op.assigned_cell].busy_until
        return 0.0 if busy_until is None else max(0.0, float(busy_until - sim.time))

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def _build_operation_nodes(self, sim, visible_ops, order_remaining, order_slack):
        cont_names = (
            "remaining_predecessor_count",
            "order_weight",
            "slack_minutes",
            "waiting_time_minutes",
            "min_process_H_minutes",
            "min_process_R_minutes",
            "min_process_HR_minutes",
            "order_remaining_min_work_minutes",
        )
        bin_names = (
            "is_ready",
            "is_tardy_now",
            "H_mode_available",
            "R_mode_available",
            "HR_mode_available",
        )
        ids: list[int] = []
        continuous: list[list[float]] = []
        binary: list[list[float]] = []
        categorical = {"product_type": [], "stage": [], "status": []}
        for op in visible_ops:
            order = sim.instance.orders[op.order_id]
            p_h, p_r, p_hr, a_h, a_r, a_hr = self._mode_min_times(sim, op)
            ids.append(op.operation_id)
            continuous.append([
                float(self._remaining_predecessors(sim, op)),
                float(order.weight),
                float(order_slack[op.order_id]),
                float(self._operation_waiting_time(sim, op)),
                p_h,
                p_r,
                p_hr,
                float(order_remaining[op.order_id]),
            ])
            binary.append([
                float(op.status == OperationStatus.READY),
                float(sim.time >= order.due_date and not sim.orders[op.order_id].completed),
                float(a_h),
                float(a_r),
                float(a_hr),
            ])
            categorical["product_type"].append(int(op.product_type))
            categorical["stage"].append(int(op.stage))
            categorical["status"].append(STATUS_INDEX[op.status])
        return _node_store(ids, continuous, binary, categorical, cont_names, bin_names)

    def _build_stage_nodes(self, sim, stage_stats):
        cont_names = (
            "queue_length",
            "remaining_workload_minutes",
            "weighted_due_pressure",
            "parallel_cell_count",
            "available_cell_count",
            "busy_cell_count",
            "unconfigured_cell_count",
            "human_supply",
            "robot_supply",
            "load_capacity_ratio",
        )
        bin_names = ("zero_service_capacity",)
        ids = list(range(sim.instance.num_stages))
        continuous = [[float(s[name]) for name in cont_names] for s in stage_stats]
        binary = [[float(s["zero_service_capacity"])] for s in stage_stats]
        categorical = {"stage": ids.copy()}
        return _node_store(ids, continuous, binary, categorical, cont_names, bin_names)

    def _build_cell_nodes(self, sim, stage_stats):
        cont_names = (
            "remaining_processing_time_minutes",
            "local_ready_workload_minutes",
            "next_available_in_minutes",
        )
        bin_names = (
            "is_busy",
            "has_worker",
            "has_robot",
            "has_worker_reservation",
            "has_robot_reservation",
            "configuration_executable",
        )
        ids: list[int] = []
        continuous: list[list[float]] = []
        binary: list[list[float]] = []
        categorical = {"stage": [], "worker_slot": [], "robot_slot": [], "mode": []}
        for cell in sim.cells:
            mode, legal = self._cell_mode(sim, cell)
            remaining = 0.0 if cell.busy_until is None else max(0.0, float(cell.busy_until - sim.time))
            next_available = remaining
            if cell.reserved_worker is not None:
                arrival = sim.workers[cell.reserved_worker].busy_until
                if arrival is not None:
                    next_available = max(next_available, float(arrival - sim.time))
            if cell.reserved_robot is not None:
                arrival = sim.robots[cell.reserved_robot].busy_until
                if arrival is not None:
                    next_available = max(next_available, float(arrival - sim.time))
            ids.append(cell.cell_id)
            continuous.append([
                remaining,
                float(stage_stats[cell.stage]["ready_workload_minutes"]),
                max(0.0, next_available),
            ])
            binary.append([
                float(cell.activity == CellActivity.BUSY),
                float(cell.configured_worker is not None),
                float(cell.configured_robot is not None),
                float(cell.reserved_worker is not None),
                float(cell.reserved_robot is not None),
                float(legal and cell.reserved_worker is None and cell.reserved_robot is None),
            ])
            categorical["stage"].append(int(cell.stage))
            categorical["worker_slot"].append(0 if cell.configured_worker is None else cell.configured_worker + 1)
            categorical["robot_slot"].append(0 if cell.configured_robot is None else cell.configured_robot + 1)
            categorical["mode"].append(MODE_INDEX[mode])
        return _node_store(ids, continuous, binary, categorical, cont_names, bin_names)

    def _resource_relocation_to_stages(self, sim, kind: str, resource_id: int, physical_cell: int) -> list[float]:
        inst = sim.instance
        matrix = (
            inst.worker_relocation_time[resource_id]
            if kind == "worker"
            else inst.robot_relocation_time[resource_id]
        )
        out: list[float] = []
        for stage in range(inst.num_stages):
            targets = [m for m, s in enumerate(inst.cell_stage) if s == stage]
            out.append(min(float(matrix[physical_cell][m]) for m in targets))
        out.extend([0.0] * (self.max_stages - inst.num_stages))
        return out

    def _build_resource_nodes(self, sim, kind: str):
        inst = sim.instance
        resources = sim.workers if kind == "worker" else sim.robots
        compat = inst.worker_skill if kind == "worker" else inst.robot_capability
        prefix = "skill" if kind == "worker" else "capability"
        cont_names = (
            "remaining_busy_or_relocation_time_minutes",
            *tuple(f"relocation_to_stage_{s}_minutes" for s in range(self.max_stages)),
            "eligible_stage_ratio",
        )
        bin_names = (
            *tuple(f"{prefix}_stage_{s}" for s in range(self.max_stages)),
            "is_idle",
            "is_busy",
            "is_relocating",
        )
        ids: list[int] = []
        continuous: list[list[float]] = []
        binary: list[list[float]] = []
        categorical = {
            "configured_cell": [],
            "physical_cell": [],
            "configured_stage": [],
            "physical_stage": [],
        }
        for resource in resources:
            rid = resource.worker_id if kind == "worker" else resource.robot_id
            physical = resource.physical_cell
            if physical is None:
                raise GraphBuildError(f"{kind} {rid} has no physical cell")
            remaining = 0.0 if resource.busy_until is None else max(0.0, float(resource.busy_until - sim.time))
            reloc = self._resource_relocation_to_stages(sim, kind, rid, physical)
            row = [int(x) for x in compat[rid]] + [0] * (self.max_stages - inst.num_stages)
            ids.append(rid)
            continuous.append([
                remaining,
                *reloc,
                float(sum(compat[rid])) / float(inst.num_stages),
            ])
            binary.append([
                *[float(x) for x in row],
                float(resource.activity == ResourceActivity.IDLE),
                float(resource.activity == ResourceActivity.BUSY),
                float(resource.activity == ResourceActivity.RELOCATING),
            ])
            configured = resource.configured_cell
            configured_stage = None if configured is None else inst.cell_stage[configured]
            physical_stage = inst.cell_stage[physical]
            categorical["configured_cell"].append(0 if configured is None else configured + 1)
            categorical["physical_cell"].append(physical + 1)
            categorical["configured_stage"].append(0 if configured_stage is None else configured_stage + 1)
            categorical["physical_stage"].append(physical_stage + 1)
        return _node_store(ids, continuous, binary, categorical, cont_names, bin_names)

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    def _build_edges(self, sim, *, visible_ops, op_local, order_slack, order_pressure):
        inst = sim.instance
        edges: dict[EdgeType, EdgeStore] = {}

        # Direct technological precedence, plus reverse relation.
        precedence: list[tuple[int, int]] = []
        for order_id in {op.order_id for op in visible_ops}:
            ids = [op_id for op_id in sim.order_operation_ids[order_id] if op_id in op_local]
            precedence.extend((op_local[a], op_local[b]) for a, b in zip(ids, ids[1:]))
        edges[("operation", "precedes", "operation")] = _edge_store(precedence, [[] for _ in precedence], [[] for _ in precedence])
        reverse_precedence = [(b, a) for a, b in precedence]
        edges[("operation", "succeeds", "operation")] = _edge_store(reverse_precedence, [[] for _ in reverse_precedence], [[] for _ in reverse_precedence])

        # Operation-stage membership.
        op_stage = [(op_local[op.operation_id], op.stage) for op in visible_ops]
        edges[("operation", "belongs_to", "stage")] = _edge_store(op_stage, [[] for _ in op_stage], [[] for _ in op_stage])
        edges[("stage", "has_operation", "operation")] = _edge_store([(b, a) for a, b in op_stage], [[] for _ in op_stage], [[] for _ in op_stage])

        # Stage-cell fixed ownership.
        stage_cell = [(cell.stage, cell.cell_id) for cell in sim.cells]
        edges[("stage", "has_cell", "cell")] = _edge_store(stage_cell, [[] for _ in stage_cell], [[] for _ in stage_cell])
        edges[("cell", "of_stage", "stage")] = _edge_store([(b, a) for a, b in stage_cell], [[] for _ in stage_cell], [[] for _ in stage_cell])

        # Operation-cell stage compatibility and current execution context.
        oc_pairs: list[tuple[int, int]] = []
        oc_cont: list[list[float]] = []
        oc_bin: list[list[float]] = []
        for op in visible_ops:
            wait = self._operation_waiting_time(sim, op)
            for cell in sim.cells:
                if cell.stage != op.stage:
                    continue
                p, _, legal_cfg = self._configured_processing_time(sim, op, cell)
                dispatchable = (
                    legal_cfg
                    and cell.activity == CellActivity.IDLE
                    and cell.reserved_worker is None
                    and cell.reserved_robot is None
                    and op.status == OperationStatus.READY
                )
                oc_pairs.append((op_local[op.operation_id], cell.cell_id))
                oc_cont.append([p, wait, float(order_pressure[op.order_id])])
                oc_bin.append([
                    1.0,
                    float(legal_cfg),
                    float(dispatchable),
                    float(op.status == OperationStatus.PROCESSING and op.assigned_cell == cell.cell_id),
                ])
        oc_cont_names = (
            "configured_processing_time_minutes",
            "operation_waiting_time_minutes",
            "order_due_pressure",
        )
        oc_bin_names = ("stage_compatible", "configured_mode_legal", "currently_dispatchable", "is_processing_here")
        edges[("operation", "compatible_with", "cell")] = _edge_store(
            oc_pairs, oc_cont, oc_bin, oc_cont_names, oc_bin_names
        )
        edges[("cell", "can_process", "operation")] = _edge_store(
            [(b, a) for a, b in oc_pairs], oc_cont, oc_bin, oc_cont_names, oc_bin_names
        )

        # Resource-stage skills/capabilities.
        ws = [
            (h, s)
            for h in range(inst.num_workers)
            for s in range(inst.num_stages)
            if inst.worker_skill[h][s] == 1
        ]
        rs = [
            (r, s)
            for r in range(inst.num_robots)
            for s in range(inst.num_stages)
            if inst.robot_capability[r][s] == 1
        ]
        one_ws = [[1.0] for _ in ws]
        one_rs = [[1.0] for _ in rs]
        edges[("worker", "skilled_for", "stage")] = _edge_store(ws, [[] for _ in ws], one_ws, (), ("compatible",))
        edges[("stage", "has_skilled_worker", "worker")] = _edge_store([(b, a) for a, b in ws], [[] for _ in ws], one_ws, (), ("compatible",))
        edges[("robot", "capable_of", "stage")] = _edge_store(rs, [[] for _ in rs], one_rs, (), ("compatible",))
        edges[("stage", "has_capable_robot", "robot")] = _edge_store([(b, a) for a, b in rs], [[] for _ in rs], one_rs, (), ("compatible",))

        # Current configuration edges.
        wc = [(w.worker_id, w.configured_cell) for w in sim.workers if w.configured_cell is not None]
        rc = [(r.robot_id, r.configured_cell) for r in sim.robots if r.configured_cell is not None]
        edges[("worker", "configured_at", "cell")] = _edge_store(wc, [[] for _ in wc], [[1.0] for _ in wc], (), ("current_configuration",))
        edges[("cell", "has_worker", "worker")] = _edge_store([(b, a) for a, b in wc], [[] for _ in wc], [[1.0] for _ in wc], (), ("current_configuration",))
        edges[("robot", "configured_at", "cell")] = _edge_store(rc, [[] for _ in rc], [[1.0] for _ in rc], (), ("current_configuration",))
        edges[("cell", "has_robot", "robot")] = _edge_store([(b, a) for a, b in rc], [[] for _ in rc], [[1.0] for _ in rc], (), ("current_configuration",))

        # Candidate resource-cell edges make the paper's matcher edge scores and
        # relocation-time edge feature explicit. They do not bypass masks: busy
        # resources and occupied slots remain encoded and Phase F will mask them.
        for kind in ("worker", "robot"):
            resources = sim.workers if kind == "worker" else sim.robots
            compat = inst.worker_skill if kind == "worker" else inst.robot_capability
            reloc = inst.worker_relocation_time if kind == "worker" else inst.robot_relocation_time
            pairs: list[tuple[int, int]] = []
            cont: list[list[float]] = []
            bins: list[list[float]] = []
            for resource in resources:
                rid = resource.worker_id if kind == "worker" else resource.robot_id
                source = resource.physical_cell
                if source is None:
                    raise GraphBuildError(f"{kind} {rid} has no physical anchor")
                for cell in sim.cells:
                    if compat[rid][cell.stage] != 1:
                        continue
                    if kind == "worker":
                        slot_free = cell.configured_worker in {None, rid} and cell.reserved_worker in {None, rid}
                    else:
                        slot_free = cell.configured_robot in {None, rid} and cell.reserved_robot in {None, rid}
                    pairs.append((rid, cell.cell_id))
                    cont.append([float(reloc[rid][source][cell.cell_id])])
                    bins.append([
                        float(source == cell.cell_id),
                        float(slot_free and cell.activity == CellActivity.IDLE),
                        float(resource.activity == ResourceActivity.IDLE),
                    ])
            cont_names = ("relocation_time_minutes",)
            bin_names = ("same_physical_cell", "target_slot_free_now", "resource_idle")
            src_type = kind
            forward_rel = "can_relocate_to"
            reverse_rel = "eligible_for_worker" if kind == "worker" else "eligible_for_robot"
            edges[(src_type, forward_rel, "cell")] = _edge_store(pairs, cont, bins, cont_names, bin_names)
            edges[("cell", reverse_rel, src_type)] = _edge_store([(b, a) for a, b in pairs], cont, bins, cont_names, bin_names)

        # Same-stage operation competition among unfinished visible operations.
        # Phase L1.2 vectorizes the exact all-pairs relation.  The old nested
        # Python loops materialized hundreds of thousands of tiny tuples/lists on
        # L-scale states; repeat/repeat_interleave preserves the identical
        # left-major/right-minor edge order while avoiding that transient memory.
        by_stage: dict[int, list] = defaultdict(list)
        for op in visible_ops:
            if op.status != OperationStatus.DONE:
                by_stage[op.stage].append(op)
        comp_indices: list[torch.Tensor] = []
        comp_cont_tensors: list[torch.Tensor] = []
        comp_bin_tensors: list[torch.Tensor] = []
        for ops in by_stage.values():
            n_ops = len(ops)
            if n_ops <= 1:
                continue
            local = torch.tensor(
                [op_local[op.operation_id] for op in ops], dtype=torch.long
            )
            pressure = torch.tensor(
                [order_pressure[op.order_id] for op in ops], dtype=torch.float32
            )
            slack = torch.tensor(
                [order_slack[op.order_id] for op in ops], dtype=torch.float32
            )
            ready = torch.tensor(
                [op.status == OperationStatus.READY for op in ops], dtype=torch.bool
            )
            src = local.repeat_interleave(n_ops)
            dst = local.repeat(n_ops)
            keep = src != dst
            p_src = pressure.repeat_interleave(n_ops)
            p_dst = pressure.repeat(n_ops)
            s_src = slack.repeat_interleave(n_ops)
            s_dst = slack.repeat(n_ops)
            r_src = ready.repeat_interleave(n_ops)
            r_dst = ready.repeat(n_ops)
            comp_indices.append(torch.stack([src[keep], dst[keep]], dim=0))
            comp_cont_tensors.append(torch.stack([
                (p_src - p_dst)[keep],
                (s_src - s_dst)[keep],
            ], dim=1))
            comp_bin_tensors.append((r_src & r_dst)[keep].to(torch.float32).unsqueeze(1))

        if comp_indices:
            comp_edge_index = torch.cat(comp_indices, dim=1)
            comp_continuous = torch.cat(comp_cont_tensors, dim=0)
            comp_binary = torch.cat(comp_bin_tensors, dim=0)
        else:
            comp_edge_index = torch.empty((2, 0), dtype=torch.long)
            comp_continuous = torch.empty((0, 2), dtype=torch.float32)
            comp_binary = torch.empty((0, 1), dtype=torch.float32)
        edges[("operation", "competes_with", "operation")] = EdgeStore(
            edge_index=comp_edge_index,
            continuous=comp_continuous,
            binary=comp_binary,
            continuous_names=("relative_due_pressure", "relative_slack_minutes"),
            binary_names=("both_ready",),
            batch=torch.zeros((comp_edge_index.shape[1],), dtype=torch.long),
        )

        return edges


__all__ = ["DynamicHeteroGraphBuilder", "GraphBuildError", "NODE_TYPES"]
