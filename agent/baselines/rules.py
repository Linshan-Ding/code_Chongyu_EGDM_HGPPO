"""Paper Table-8 rule baselines implemented on the shared discrete-event simulator.

Implemented in Phase K1:
- Fixed-EDD
- Periodic-ATC
- Threshold-EDD
- Bottleneck-Rule

The exact threshold/period/ATC-k values are not prescribed by the paper.  They
therefore live in ``configs/eval.yaml`` as explicit implementation choices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from agent.baselines.base import DecisionPolicy, RuleBaselineConfig
from environment.masks import InvalidActionError, validate_composite_action
from environment.state import CompositeAction, ResourceAssignment, ScheduleAssignment


def _stage_values(env, name: str) -> list[float]:
    graph = env.graph()
    store = graph.nodes["stage"]
    try:
        col = store.continuous_names.index(name)
    except ValueError as exc:
        raise RuntimeError(f"stage graph feature missing: {name}") from exc
    return [float(x) for x in store.continuous[:, col].tolist()]


def _visible_order_map(env) -> dict[int, object]:
    state = env._last_state
    if state is None:
        raise RuntimeError("environment has not been reset")
    return {int(order.order_id): order for order in state.orders}


def _trial_plan(
    env,
    *,
    reconfigure: bool,
    workers: tuple[ResourceAssignment, ...],
    robots: tuple[ResourceAssignment, ...],
    schedule: tuple[ScheduleAssignment, ...],
):
    action = CompositeAction(
        reconfigure=bool(reconfigure),
        worker_assignments=workers,
        robot_assignments=robots,
        schedule_assignments=schedule,
    )
    try:
        return validate_composite_action(env.sim, action)
    except InvalidActionError:
        return None


def _schedule_options(
    env,
    *,
    reconfigure: bool,
    workers: tuple[ResourceAssignment, ...],
    robots: tuple[ResourceAssignment, ...],
    prefix: tuple[ScheduleAssignment, ...],
):
    state = env._last_state
    if state is None:
        raise RuntimeError("environment has not been reset")
    used_ops = {x.operation_id for x in prefix}
    used_cells = {x.cell_id for x in prefix}
    ready = [op for op in state.operations if op.operation_id not in used_ops and op.status.value == "READY"]
    idle_cells = [c for c in state.cells if c.cell_id not in used_cells and c.activity.value == "IDLE"]
    options = []
    for op in ready:
        for cell in idle_cells:
            if int(cell.stage) != int(op.stage):
                continue
            candidate = ScheduleAssignment(int(op.operation_id), int(cell.cell_id))
            plan = _trial_plan(
                env,
                reconfigure=reconfigure,
                workers=workers,
                robots=robots,
                schedule=prefix + (candidate,),
            )
            if plan is None or not plan.starts:
                continue
            start = plan.starts[-1]
            if start.operation_id != candidate.operation_id or start.cell_id != candidate.cell_id:
                continue
            options.append((candidate, float(start.processing_time)))
    return options


def _dispatch(
    env,
    *,
    rule: str,
    atc_k: float,
    reconfigure: bool,
    workers: tuple[ResourceAssignment, ...],
    robots: tuple[ResourceAssignment, ...],
) -> tuple[ScheduleAssignment, ...]:
    orders = _visible_order_map(env)
    prefix: tuple[ScheduleAssignment, ...] = ()
    while True:
        options = _schedule_options(
            env,
            reconfigure=reconfigure,
            workers=workers,
            robots=robots,
            prefix=prefix,
        )
        if not options:
            break
        if rule == "EDD":
            def key(item):
                assignment, p = item
                op = env.sim.operations[assignment.operation_id]
                order = orders[int(op.order_id)]
                return (
                    float(order.due_date),
                    -int(order.weight),
                    float(p),
                    int(assignment.operation_id),
                    int(assignment.cell_id),
                )
            chosen, _ = min(options, key=key)
        elif rule == "ATC":
            mean_p = sum(p for _, p in options) / max(1, len(options))
            now = float(env.sim.time)
            def score(item):
                assignment, p = item
                op = env.sim.operations[assignment.operation_id]
                order = orders[int(op.order_id)]
                slack = max(0.0, float(order.due_date) - now - float(p))
                priority = (float(order.weight) / max(float(p), 1e-9)) * math.exp(
                    -slack / max(float(atc_k) * mean_p, 1e-9)
                )
                return (priority, -float(order.due_date), -int(assignment.operation_id))
            chosen, _ = max(options, key=score)
        else:
            raise ValueError(f"unknown dispatch rule: {rule}")
        prefix = prefix + (chosen,)
    return prefix


def _candidate_resource_moves(env, cfg: RuleBaselineConfig):
    candidates = env.candidates()
    ratios = _stage_values(env, "load_capacity_ratio")
    worker_moves = []
    robot_moves = []

    for cset in candidates.worker:
        rid = int(cset.resource_id)
        source_cell = int(env.sim.workers[rid].physical_cell)
        source_stage = int(env.sim.cells[source_cell].stage)
        for target in cset.target_cells:
            target = int(target)
            target_stage = int(env.sim.cells[target].stage)
            if target_stage == source_stage:
                continue
            travel = float(env.sim.instance.worker_relocation_time[rid][source_cell][target])
            benefit = ratios[target_stage] - ratios[source_stage] - cfg.relocation_time_penalty * travel
            worker_moves.append((benefit, travel, ResourceAssignment(rid, target)))

    for cset in candidates.robot:
        rid = int(cset.resource_id)
        source_cell = int(env.sim.robots[rid].physical_cell)
        source_stage = int(env.sim.cells[source_cell].stage)
        for target in cset.target_cells:
            target = int(target)
            target_stage = int(env.sim.cells[target].stage)
            if target_stage == source_stage:
                continue
            travel = float(env.sim.instance.robot_relocation_time[rid][source_cell][target])
            benefit = ratios[target_stage] - ratios[source_stage] - cfg.relocation_time_penalty * travel
            robot_moves.append((benefit, travel, ResourceAssignment(rid, target)))

    worker_moves.sort(key=lambda x: (-x[0], x[1], x[2].resource_id, x[2].target_cell))
    robot_moves.sort(key=lambda x: (-x[0], x[1], x[2].resource_id, x[2].target_cell))
    return worker_moves, robot_moves, ratios


def _greedy_reconfiguration(env, cfg: RuleBaselineConfig):
    if cfg.max_reconfiguration_moves <= 0 or not env.candidates().reconfigure_feasible:
        return (), ()
    worker_pool, robot_pool, _ = _candidate_resource_moves(env, cfg)
    merged = [("worker", *x) for x in worker_pool] + [("robot", *x) for x in robot_pool]
    merged.sort(key=lambda x: (-x[1], x[2], x[3].resource_id, x[3].target_cell))

    workers: tuple[ResourceAssignment, ...] = ()
    robots: tuple[ResourceAssignment, ...] = ()
    accepted = 0
    for kind, benefit, _travel, assignment in merged:
        if accepted >= cfg.max_reconfiguration_moves:
            break
        if float(benefit) <= 0.0:
            break
        new_workers = workers + ((assignment,) if kind == "worker" else ())
        new_robots = robots + ((assignment,) if kind == "robot" else ())
        plan = _trial_plan(
            env,
            reconfigure=True,
            workers=new_workers,
            robots=new_robots,
            schedule=(),
        )
        if plan is None:
            continue
        workers, robots = new_workers, new_robots
        accepted += 1
    return workers, robots


@dataclass
class _RulePolicy(DecisionPolicy):
    cfg: RuleBaselineConfig
    method_name: str
    dispatch_rule: str = "EDD"

    def __post_init__(self):
        self.cfg.validate()

    def _should_reconfigure(self, env) -> bool:
        return False

    def act(self, env) -> CompositeAction:
        do_reconfigure = bool(self._should_reconfigure(env))
        workers: tuple[ResourceAssignment, ...] = ()
        robots: tuple[ResourceAssignment, ...] = ()
        if do_reconfigure:
            workers, robots = _greedy_reconfiguration(env, self.cfg)
            do_reconfigure = bool(workers or robots)
        schedule = _dispatch(
            env,
            rule=self.dispatch_rule,
            atc_k=self.cfg.atc_k,
            reconfigure=do_reconfigure,
            workers=workers,
            robots=robots,
        )
        return CompositeAction(
            reconfigure=do_reconfigure,
            worker_assignments=workers,
            robot_assignments=robots,
            schedule_assignments=schedule,
        )


class FixedEDDPolicy(_RulePolicy):
    def __init__(self, cfg: RuleBaselineConfig):
        super().__init__(cfg=cfg, method_name="Fixed-EDD", dispatch_rule="EDD")


class PeriodicATCPolicy(_RulePolicy):
    def __init__(self, cfg: RuleBaselineConfig):
        super().__init__(cfg=cfg, method_name="Periodic-ATC", dispatch_rule="ATC")

    def _should_reconfigure(self, env) -> bool:
        if not env.candidates().reconfigure_feasible:
            return False
        elapsed = float(env.sim.time - env.sim.last_reconfiguration_time)
        return elapsed >= float(self.cfg.periodic_interval_minutes) - 1e-12


class ThresholdEDDPolicy(_RulePolicy):
    def __init__(self, cfg: RuleBaselineConfig):
        super().__init__(cfg=cfg, method_name="Threshold-EDD", dispatch_rule="EDD")

    def _should_reconfigure(self, env) -> bool:
        if not env.candidates().reconfigure_feasible:
            return False
        ratios = _stage_values(env, "load_capacity_ratio")
        return max(ratios, default=0.0) >= float(self.cfg.threshold_load_capacity_ratio)


class BottleneckRulePolicy(_RulePolicy):
    def __init__(self, cfg: RuleBaselineConfig):
        super().__init__(cfg=cfg, method_name="Bottleneck-Rule", dispatch_rule="EDD")

    def _should_reconfigure(self, env) -> bool:
        if not env.candidates().reconfigure_feasible:
            return False
        ratios = _stage_values(env, "load_capacity_ratio")
        if not ratios:
            return False
        return max(ratios) - min(ratios) >= float(self.cfg.bottleneck_min_gap)


def build_rule_policy(name: str, cfg: RuleBaselineConfig) -> DecisionPolicy:
    table = {
        "Fixed-EDD": FixedEDDPolicy,
        "Periodic-ATC": PeriodicATCPolicy,
        "Threshold-EDD": ThresholdEDDPolicy,
        "Bottleneck-Rule": BottleneckRulePolicy,
    }
    try:
        return table[str(name)](cfg)
    except KeyError as exc:
        raise KeyError(f"unknown Phase K1 rule baseline: {name}") from exc


RULE_METHODS = ("Fixed-EDD", "Periodic-ATC", "Threshold-EDD", "Bottleneck-Rule")

__all__ = [
    "BottleneckRulePolicy", "FixedEDDPolicy", "PeriodicATCPolicy",
    "RULE_METHODS", "ThresholdEDDPolicy", "build_rule_policy",
]
