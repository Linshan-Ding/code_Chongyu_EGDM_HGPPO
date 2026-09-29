"""Optimization baselines for Phase K2.

This module implements two distinct mathematical-programming roles from the paper:

* ``OfflineMILPReferenceSolver``: clairvoyant small-instance reference MILP.  It
  models release dates, operation precedence, dedicated parallel cells, H/R/HR
  execution modes, worker/robot non-overlap, and sequence-dependent resource
  relocation times.  The global minimum reconfiguration dwell is not represented
  in this disjunctive formulation; therefore the result is an exact reference only
  when dwell=0 and otherwise a valid *relaxation/lower-bound model*.  The status is
  carried explicitly so downstream experiments cannot silently call a relaxation
  an optimum of the dwell-constrained simulator.

* ``RollingHorizonMILPPolicy``: online event policy.  It never reads unreleased
  orders.  At each event it solves small binary assignment MILPs over the current
  idle-resource relocation candidates and currently READY operation/cell edges,
  then immediately re-optimizes at the next physical event.

SciPy/HiGHS is used instead of a proprietary solver so the open-source repository
can run without CPLEX/Gurobi.  The paper label "CP-SAT/MILP" permits either exact
backend; formal experiments should report the backend and proof status.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from agent.baselines.base import DecisionPolicy, RuleBaselineConfig
from agent.baselines.rules import (
    _candidate_resource_moves,
    _dispatch,
    _schedule_options,
    _stage_values,
    _trial_plan,
    _visible_order_map,
)
from data.schema import AssemblyInstance
from environment.state import CompositeAction, ResourceAssignment, ScheduleAssignment


@dataclass(frozen=True, slots=True)
class OptimizationBaselineConfig:
    offline_time_limit_seconds: float = 3600.0
    offline_mip_rel_gap: float = 0.0
    offline_max_operations: int = 140
    rolling_time_limit_seconds: float = 0.20
    rolling_max_reconfiguration_moves: int = 3
    rolling_relocation_penalty: float = 0.05
    rolling_tardiness_pressure: float = 2.0
    rolling_processing_penalty: float = 0.02

    def validate(self) -> None:
        if self.offline_time_limit_seconds <= 0 or self.rolling_time_limit_seconds <= 0:
            raise ValueError("MILP time limits must be positive")
        if self.offline_mip_rel_gap < 0:
            raise ValueError("offline_mip_rel_gap must be non-negative")
        if self.offline_max_operations <= 0:
            raise ValueError("offline_max_operations must be positive")
        if self.rolling_max_reconfiguration_moves < 0:
            raise ValueError("rolling_max_reconfiguration_moves must be non-negative")
        if self.rolling_relocation_penalty < 0 or self.rolling_processing_penalty < 0:
            raise ValueError("rolling penalties must be non-negative")
        if self.rolling_tardiness_pressure < 0:
            raise ValueError("rolling_tardiness_pressure must be non-negative")


@dataclass(frozen=True, slots=True)
class OfflineMILPResult:
    instance_id: str
    objective_twt: float | None
    dual_bound: float | None
    mip_gap: float | None
    optimal: bool
    feasible: bool
    status: int
    message: str
    wall_time_seconds: float
    num_operations: int
    num_variables: int
    num_constraints: int
    dwell_exact: bool


@dataclass(frozen=True, slots=True)
class _Op:
    op_id: int
    order_id: int
    product: int
    stage: int
    predecessor: int | None
    is_last: bool


class _LinearModel:
    """Minimal sparse MILP builder around scipy.optimize.milp."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integrality: list[int] = []
        self.cost: list[float] = []
        self.rows: list[dict[int, float]] = []
        self.row_lb: list[float] = []
        self.row_ub: list[float] = []

    def var(self, name: str, *, lb=0.0, ub=np.inf, integer=False, cost=0.0) -> int:
        idx = len(self.names)
        self.names.append(name)
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        self.integrality.append(1 if integer else 0)
        self.cost.append(float(cost))
        return idx

    def add(self, coeff: dict[int, float], *, lb=-np.inf, ub=np.inf) -> None:
        clean = {int(k): float(v) for k, v in coeff.items() if abs(float(v)) > 1e-14}
        self.rows.append(clean)
        self.row_lb.append(float(lb))
        self.row_ub.append(float(ub))

    def solve(self, *, time_limit: float, mip_rel_gap: float):
        rr: list[int] = []
        cc: list[int] = []
        vv: list[float] = []
        for r, row in enumerate(self.rows):
            for c, value in row.items():
                rr.append(r); cc.append(c); vv.append(value)
        A = coo_matrix((vv, (rr, cc)), shape=(len(self.rows), len(self.names))).tocsr()
        options = {"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap)}
        return milp(
            np.asarray(self.cost, dtype=float),
            integrality=np.asarray(self.integrality, dtype=np.int8),
            bounds=Bounds(np.asarray(self.lb), np.asarray(self.ub)),
            constraints=LinearConstraint(A, np.asarray(self.row_lb), np.asarray(self.row_ub)),
            options=options,
        )


def _ops(instance: AssemblyInstance) -> list[_Op]:
    out: list[_Op] = []
    for order in instance.orders:
        route = instance.product_routes[order.product_type]
        pred = None
        for pos, stage in enumerate(route):
            oid = len(out)
            out.append(_Op(
                op_id=oid,
                order_id=int(order.order_id),
                product=int(order.product_type),
                stage=int(stage),
                predecessor=pred,
                is_last=pos == len(route) - 1,
            ))
            pred = oid
    return out


def _first_compatible_cell(instance: AssemblyInstance, kind: str, rid: int) -> int:
    compat = instance.worker_skill[rid] if kind == "worker" else instance.robot_capability[rid]
    for m, stage in enumerate(instance.cell_stage):
        if compat[stage]:
            return m
    raise ValueError(f"{kind} {rid} has no compatible physical anchor")


def _mode_options(instance: AssemblyInstance, op: _Op):
    options: list[tuple[str, int | None, int | None, float]] = []
    for h in range(instance.num_workers):
        p = instance.processing_time_h(op.product, op.stage, h)
        if p is not None:
            options.append(("H", h, None, float(p)))
    for r in range(instance.num_robots):
        p = instance.processing_time_r(op.product, op.stage, r)
        if p is not None:
            options.append(("R", None, r, float(p)))
    for h in range(instance.num_workers):
        for r in range(instance.num_robots):
            p = instance.processing_time_hr(op.product, op.stage, h, r)
            if p is not None:
                options.append(("HR", h, r, float(p)))
    if not options:
        raise ValueError(f"operation {op.op_id} has no execution mode")
    return options


class OfflineMILPReferenceSolver:
    method_name = "CP-SAT/MILP"

    def __init__(self, cfg: OptimizationBaselineConfig, *, minimum_dwell_time: float = 0.0) -> None:
        cfg.validate()
        self.cfg = cfg
        self.minimum_dwell_time = float(minimum_dwell_time)

    def solve(self, instance: AssemblyInstance) -> OfflineMILPResult:
        instance.validate()
        ops = _ops(instance)
        if len(ops) > self.cfg.offline_max_operations:
            raise ValueError(
                f"offline MILP has {len(ops)} operations; configured cap is "
                f"{self.cfg.offline_max_operations}. Increase explicitly for formal S runs."
            )
        start_wall = perf_counter()
        model = _LinearModel()

        # A conservative horizon / Big-M.  It deliberately exceeds any serial
        # schedule in which every operation uses its slowest feasible mode and a
        # resource crosses the plant before every operation.
        max_release = max(float(o.release_time) for o in instance.orders)
        slow_sum = 0.0
        for op in ops:
            slow_sum += max(x[3] for x in _mode_options(instance, op))
        max_reloc = 0.0
        for matrix in (instance.worker_relocation_time, instance.robot_relocation_time):
            for resource in matrix:
                for row in resource:
                    max_reloc = max(max_reloc, max(map(float, row), default=0.0))
        horizon = max_release + slow_sum + (len(ops) + 2) * max_reloc + 10.0
        M = max(100.0, 2.0 * horizon)

        start_var: dict[int, int] = {}
        tardy_var: dict[int, int] = {}
        cell_var: dict[tuple[int, int], int] = {}
        mode_var: dict[tuple[int, int], int] = {}
        mode_rows: dict[int, list[tuple[int, str, int | None, int | None, float]]] = {}
        worker_use: dict[tuple[int, int], list[int]] = {}
        robot_use: dict[tuple[int, int], list[int]] = {}

        cells_by_stage = {
            s: [m for m, st in enumerate(instance.cell_stage) if st == s]
            for s in range(instance.num_stages)
        }

        for op in ops:
            order = instance.orders[op.order_id]
            start_var[op.op_id] = model.var(
                f"S[{op.op_id}]", lb=float(order.release_time), ub=horizon
            )
            modes = _mode_options(instance, op)
            mode_rows[op.op_id] = []
            for q, (mode, h, r, p) in enumerate(modes):
                v = model.var(f"mode[{op.op_id},{q},{mode}]", lb=0, ub=1, integer=True)
                mode_var[(op.op_id, q)] = v
                mode_rows[op.op_id].append((v, mode, h, r, p))
                if h is not None:
                    worker_use.setdefault((op.op_id, h), []).append(v)
                if r is not None:
                    robot_use.setdefault((op.op_id, r), []).append(v)
            model.add({v: 1.0 for v, *_ in mode_rows[op.op_id]}, lb=1.0, ub=1.0)

            for m in cells_by_stage[op.stage]:
                cell_var[(op.op_id, m)] = model.var(
                    f"cell[{op.op_id},{m}]", lb=0, ub=1, integer=True
                )
            model.add(
                {cell_var[(op.op_id, m)]: 1.0 for m in cells_by_stage[op.stage]},
                lb=1.0, ub=1.0,
            )

        for order in instance.orders:
            tardy_var[order.order_id] = model.var(
                f"T[{order.order_id}]", lb=0.0, ub=horizon,
                cost=float(order.weight),
            )

        def duration_coeff(op_id: int, scale: float = -1.0) -> dict[int, float]:
            return {v: scale * p for v, _mode, _h, _r, p in mode_rows[op_id]}

        # Technological precedence.
        for op in ops:
            if op.predecessor is None:
                continue
            coeff = {start_var[op.op_id]: 1.0, start_var[op.predecessor]: -1.0}
            for v, value in duration_coeff(op.predecessor, -1.0).items():
                coeff[v] = coeff.get(v, 0.0) + value
            model.add(coeff, lb=0.0)

        # Weighted tardiness on each order's last operation.
        last_by_order = {op.order_id: op for op in ops if op.is_last}
        for order in instance.orders:
            op = last_by_order[order.order_id]
            coeff = {
                tardy_var[order.order_id]: 1.0,
                start_var[op.op_id]: -1.0,
            }
            for v, value in duration_coeff(op.op_id, -1.0).items():
                coeff[v] = coeff.get(v, 0.0) + value
            model.add(coeff, lb=-float(order.due_date))

        # Cell disjunctive capacity. Only operations of the same stage can ever
        # compete for the same dedicated cell.
        for i, a in enumerate(ops):
            for b in ops[i + 1:]:
                if a.stage != b.stage:
                    continue
                for m in cells_by_stage[a.stage]:
                    y = model.var(f"cellord[{a.op_id},{b.op_id},{m}]", lb=0, ub=1, integer=True)
                    za = cell_var[(a.op_id, m)]
                    zb = cell_var[(b.op_id, m)]
                    # a before b when y=1
                    coeff = {start_var[b.op_id]: 1.0, start_var[a.op_id]: -1.0,
                             y: -M, za: -M, zb: -M}
                    for v, value in duration_coeff(a.op_id, -1.0).items():
                        coeff[v] = coeff.get(v, 0.0) + value
                    model.add(coeff, lb=-3.0 * M)
                    # b before a when y=0
                    coeff = {start_var[a.op_id]: 1.0, start_var[b.op_id]: -1.0,
                             y: M, za: -M, zb: -M}
                    for v, value in duration_coeff(b.op_id, -1.0).items():
                        coeff[v] = coeff.get(v, 0.0) + value
                    model.add(coeff, lb=-2.0 * M)

        # Shared human resources with sequence-dependent travel between selected cells.
        for h in range(instance.num_workers):
            usable = [op for op in ops if worker_use.get((op.op_id, h))]
            initial = instance.initial_worker_cell[h]
            initial = _first_compatible_cell(instance, "worker", h) if initial < 0 else initial
            for op in usable:
                u = worker_use[(op.op_id, h)]
                for m in cells_by_stage[op.stage]:
                    coeff = {start_var[op.op_id]: 1.0, cell_var[(op.op_id, m)]: -M}
                    for v in u:
                        coeff[v] = coeff.get(v, 0.0) - M
                    travel = float(instance.worker_relocation_time[h][initial][m])
                    model.add(coeff, lb=travel - 2.0 * M)
            for ii, a in enumerate(usable):
                for b in usable[ii + 1:]:
                    y = model.var(f"hord[{h},{a.op_id},{b.op_id}]", lb=0, ub=1, integer=True)
                    ua = worker_use[(a.op_id, h)]
                    ub = worker_use[(b.op_id, h)]
                    for ma in cells_by_stage[a.stage]:
                        for mb in cells_by_stage[b.stage]:
                            za = cell_var[(a.op_id, ma)]; zb = cell_var[(b.op_id, mb)]
                            travel = float(instance.worker_relocation_time[h][ma][mb])
                            coeff = {start_var[b.op_id]: 1.0, start_var[a.op_id]: -1.0,
                                     y: -M, za: -M, zb: -M}
                            for v in ua: coeff[v] = coeff.get(v, 0.0) - M
                            for v in ub: coeff[v] = coeff.get(v, 0.0) - M
                            for v, value in duration_coeff(a.op_id, -1.0).items():
                                coeff[v] = coeff.get(v, 0.0) + value
                            model.add(coeff, lb=travel - 5.0 * M)
                            travel_rev = float(instance.worker_relocation_time[h][mb][ma])
                            coeff = {start_var[a.op_id]: 1.0, start_var[b.op_id]: -1.0,
                                     y: M, za: -M, zb: -M}
                            for v in ua: coeff[v] = coeff.get(v, 0.0) - M
                            for v in ub: coeff[v] = coeff.get(v, 0.0) - M
                            for v, value in duration_coeff(b.op_id, -1.0).items():
                                coeff[v] = coeff.get(v, 0.0) + value
                            model.add(coeff, lb=travel_rev - 4.0 * M)

        # Shared robot resources with the same travel logic.
        for r in range(instance.num_robots):
            usable = [op for op in ops if robot_use.get((op.op_id, r))]
            initial = instance.initial_robot_cell[r]
            initial = _first_compatible_cell(instance, "robot", r) if initial < 0 else initial
            for op in usable:
                u = robot_use[(op.op_id, r)]
                for m in cells_by_stage[op.stage]:
                    coeff = {start_var[op.op_id]: 1.0, cell_var[(op.op_id, m)]: -M}
                    for v in u: coeff[v] = coeff.get(v, 0.0) - M
                    travel = float(instance.robot_relocation_time[r][initial][m])
                    model.add(coeff, lb=travel - 2.0 * M)
            for ii, a in enumerate(usable):
                for b in usable[ii + 1:]:
                    y = model.var(f"rord[{r},{a.op_id},{b.op_id}]", lb=0, ub=1, integer=True)
                    ua = robot_use[(a.op_id, r)]; ub = robot_use[(b.op_id, r)]
                    for ma in cells_by_stage[a.stage]:
                        for mb in cells_by_stage[b.stage]:
                            za = cell_var[(a.op_id, ma)]; zb = cell_var[(b.op_id, mb)]
                            travel = float(instance.robot_relocation_time[r][ma][mb])
                            coeff = {start_var[b.op_id]: 1.0, start_var[a.op_id]: -1.0,
                                     y: -M, za: -M, zb: -M}
                            for v in ua: coeff[v] = coeff.get(v, 0.0) - M
                            for v in ub: coeff[v] = coeff.get(v, 0.0) - M
                            for v, value in duration_coeff(a.op_id, -1.0).items():
                                coeff[v] = coeff.get(v, 0.0) + value
                            model.add(coeff, lb=travel - 5.0 * M)
                            travel_rev = float(instance.robot_relocation_time[r][mb][ma])
                            coeff = {start_var[a.op_id]: 1.0, start_var[b.op_id]: -1.0,
                                     y: M, za: -M, zb: -M}
                            for v in ua: coeff[v] = coeff.get(v, 0.0) - M
                            for v in ub: coeff[v] = coeff.get(v, 0.0) - M
                            for v, value in duration_coeff(b.op_id, -1.0).items():
                                coeff[v] = coeff.get(v, 0.0) + value
                            model.add(coeff, lb=travel_rev - 4.0 * M)

        res = model.solve(
            time_limit=self.cfg.offline_time_limit_seconds,
            mip_rel_gap=self.cfg.offline_mip_rel_gap,
        )
        objective = None if res.fun is None or not isfinite(float(res.fun)) else float(res.fun)
        dual = getattr(res, "mip_dual_bound", None)
        dual = None if dual is None or not isfinite(float(dual)) else float(dual)
        gap = getattr(res, "mip_gap", None)
        gap = None if gap is None or not isfinite(float(gap)) else float(gap)
        feasible = res.x is not None and objective is not None
        optimal = bool(res.status == 0 and feasible and (gap is None or gap <= 1e-9))
        return OfflineMILPResult(
            instance_id=instance.instance_id,
            objective_twt=objective,
            dual_bound=dual,
            mip_gap=gap,
            optimal=optimal,
            feasible=feasible,
            status=int(res.status),
            message=str(res.message),
            wall_time_seconds=perf_counter() - start_wall,
            num_operations=len(ops),
            num_variables=len(model.names),
            num_constraints=len(model.rows),
            dwell_exact=abs(self.minimum_dwell_time) <= 1e-12,
        )


def _binary_select(values: list[float], groups: list[list[int]], max_selected: int, time_limit: float) -> list[int]:
    """Solve max-value 0/1 selection with at-most-one constraints per group."""
    n = len(values)
    if n == 0 or max_selected <= 0:
        return []
    rows = []
    lb = []
    ub = []
    for group in groups:
        if not group:
            continue
        row = np.zeros(n, dtype=float); row[group] = 1.0
        rows.append(row); lb.append(-np.inf); ub.append(1.0)
    row = np.ones(n, dtype=float)
    rows.append(row); lb.append(-np.inf); ub.append(float(max_selected))
    A = np.stack(rows, axis=0)
    res = milp(
        -np.asarray(values, dtype=float),
        integrality=np.ones(n, dtype=np.int8),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(A, np.asarray(lb), np.asarray(ub)),
        options={"time_limit": float(time_limit)},
    )
    if res.x is None:
        return []
    return [i for i, x in enumerate(res.x) if x >= 0.5]




def _action_makes_progress(env, action: CompositeAction) -> bool:
    plan = _trial_plan(
        env, reconfigure=action.reconfigure,
        workers=tuple(action.worker_assignments), robots=tuple(action.robot_assignments),
        schedule=tuple(action.schedule_assignments),
    )
    if plan is None:
        return False
    if plan.starts:
        return True
    return any(change.is_relocation for change in (*plan.worker_changes, *plan.robot_changes))


def ensure_progress_action(env, rule_cfg: RuleBaselineConfig, action: CompositeAction) -> CompositeAction:
    """Repair a no-op action only when the physical event queue is empty.

    This mirrors the Phase-I STOP mask: an online baseline may wait when a future
    physical event already exists, but it may not choose an action that leaves
    unfinished work with no way for time to advance.
    """
    if len(env.sim.events) > 0 or env.sim.is_done or _action_makes_progress(env, action):
        return action
    keep_sched = _dispatch(
        env, rule="EDD", atc_k=rule_cfg.atc_k,
        reconfigure=False, workers=(), robots=(),
    )
    keep = CompositeAction.keep(keep_sched)
    if _action_makes_progress(env, keep):
        return keep
    if env.candidates().reconfigure_feasible:
        candidates = env.candidates()
        trials = []
        for cset in candidates.worker:
            rid = int(cset.resource_id)
            current = env.sim.workers[rid].configured_cell
            source = int(env.sim.workers[rid].physical_cell)
            for target in cset.target_cells:
                target = int(target)
                if current == target:
                    continue
                travel = float(env.sim.instance.worker_relocation_time[rid][source][target])
                trials.append((travel, "w", ResourceAssignment(rid, target)))
        for cset in candidates.robot:
            rid = int(cset.resource_id)
            current = env.sim.robots[rid].configured_cell
            source = int(env.sim.robots[rid].physical_cell)
            for target in cset.target_cells:
                target = int(target)
                if current == target:
                    continue
                travel = float(env.sim.instance.robot_relocation_time[rid][source][target])
                trials.append((travel, "r", ResourceAssignment(rid, target)))
        for _travel, kind, assignment in sorted(trials, key=lambda x: (x[0], x[1], x[2].resource_id, x[2].target_cell)):
            workers = (assignment,) if kind == "w" else ()
            robots = (assignment,) if kind == "r" else ()
            sched = _dispatch(
                env, rule="EDD", atc_k=rule_cfg.atc_k,
                reconfigure=True, workers=workers, robots=robots,
            )
            repaired = CompositeAction(
                reconfigure=True, worker_assignments=workers, robot_assignments=robots,
                schedule_assignments=sched,
            )
            if _action_makes_progress(env, repaired):
                return repaired
    raise RuntimeError("no physically progressive online baseline action exists")


class RollingHorizonMILPPolicy(DecisionPolicy):
    """Event-by-event MILP baseline using only current visible information."""

    method_name = "RH-MILP"

    def __init__(self, cfg: OptimizationBaselineConfig, rule_cfg: RuleBaselineConfig) -> None:
        cfg.validate(); rule_cfg.validate()
        self.cfg = cfg
        self.rule_cfg = rule_cfg

    def _resource_plan(self, env):
        if not env.candidates().reconfigure_feasible:
            return (), ()
        w_pool, r_pool, _ = _candidate_resource_moves(env, RuleBaselineConfig(
            periodic_interval_minutes=self.rule_cfg.periodic_interval_minutes,
            atc_k=self.rule_cfg.atc_k,
            threshold_load_capacity_ratio=self.rule_cfg.threshold_load_capacity_ratio,
            bottleneck_min_gap=self.rule_cfg.bottleneck_min_gap,
            relocation_time_penalty=self.cfg.rolling_relocation_penalty,
            max_reconfiguration_moves=self.cfg.rolling_max_reconfiguration_moves,
        ))
        tagged = [("w", *x) for x in w_pool if x[0] > 0] + [("r", *x) for x in r_pool if x[0] > 0]
        if not tagged:
            return (), ()
        values = [float(x[1]) for x in tagged]
        groups: list[list[int]] = []
        # same resource at most once; same type target-cell at most once
        keys = {}
        for i, (kind, _benefit, _travel, assignment) in enumerate(tagged):
            keys.setdefault((kind, "resource", assignment.resource_id), []).append(i)
            keys.setdefault((kind, "cell", assignment.target_cell), []).append(i)
        groups.extend(keys.values())
        chosen = _binary_select(
            values, groups, self.cfg.rolling_max_reconfiguration_moves,
            self.cfg.rolling_time_limit_seconds,
        )
        # MILP handles assignment cardinality; simulator validation handles HR
        # compatibility/basic-coverage interactions. Repair by dropping the least
        # valuable selected edge until the complete set is physically valid.
        chosen = sorted(chosen, key=lambda i: values[i], reverse=True)
        while chosen:
            workers = tuple(tagged[i][3] for i in chosen if tagged[i][0] == "w")
            robots = tuple(tagged[i][3] for i in chosen if tagged[i][0] == "r")
            if _trial_plan(env, reconfigure=True, workers=workers, robots=robots, schedule=()) is not None:
                return workers, robots
            chosen.pop()
        return (), ()

    def _schedule(self, env, workers, robots):
        reconfigure = bool(workers or robots)
        options = _schedule_options(
            env, reconfigure=reconfigure, workers=workers, robots=robots, prefix=()
        )
        if not options:
            return ()
        orders = _visible_order_map(env)
        now = float(env.sim.time)
        values = []
        assignments = []
        op_groups: dict[int, list[int]] = {}
        cell_groups: dict[int, list[int]] = {}
        for i, (assignment, p) in enumerate(options):
            op = env.sim.operations[assignment.operation_id]
            order = orders[int(op.order_id)]
            slack_after = float(order.due_date) - now - float(p)
            pressure = max(0.0, -slack_after) + 1.0 / max(1.0, float(order.due_date) - now + 1.0)
            score = (
                float(order.weight) * (1.0 + self.cfg.rolling_tardiness_pressure * pressure)
                - self.cfg.rolling_processing_penalty * float(p)
            )
            assignments.append(assignment); values.append(score)
            op_groups.setdefault(int(assignment.operation_id), []).append(i)
            cell_groups.setdefault(int(assignment.cell_id), []).append(i)
        groups = list(op_groups.values()) + list(cell_groups.values())
        chosen = _binary_select(
            values, groups, max_selected=min(len(op_groups), len(cell_groups)),
            time_limit=self.cfg.rolling_time_limit_seconds,
        )
        result = tuple(assignments[i] for i in chosen)
        if _trial_plan(env, reconfigure=reconfigure, workers=workers, robots=robots, schedule=result) is not None:
            return result
        # Robust fallback uses the same current-information EDD repair as K1.
        return _dispatch(
            env, rule="EDD", atc_k=self.rule_cfg.atc_k,
            reconfigure=reconfigure, workers=workers, robots=robots,
        )

    def act(self, env) -> CompositeAction:
        workers, robots = self._resource_plan(env)
        schedule = self._schedule(env, workers, robots)
        action = CompositeAction(
            reconfigure=bool(workers or robots),
            worker_assignments=workers,
            robot_assignments=robots,
            schedule_assignments=schedule,
        )
        return ensure_progress_action(env, self.rule_cfg, action)


__all__ = [
    "OfflineMILPReferenceSolver", "OfflineMILPResult",
    "OptimizationBaselineConfig", "RollingHorizonMILPPolicy", "ensure_progress_action",
]
