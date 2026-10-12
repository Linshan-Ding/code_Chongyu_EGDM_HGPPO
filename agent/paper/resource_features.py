"""Current-state resource proposals and fixed ATC dispatch features."""
import math
from dataclasses import replace
import numpy as np
from agent.baselines.base import RuleBaselineConfig
from agent.baselines.rules import _greedy_reconfiguration, _schedule_options
from environment.masks import InvalidActionError, validate_composite_action
from environment.reward import _processing_remaining_time
from environment.state import CompositeAction, ResourceAssignment
STAGE_DIM = 12
GLOBAL_DIM = 6
ACTION_DIM = 12


class EventView:
    """Cache read-only graph/masks for one physical state, never across events."""

    def __init__(self, env):
        self.env = env
        self._graph = self._candidates = None

    def __getattr__(self, key):
        return getattr(self.env, key)

    def graph(self):
        if self._graph is None:
            self._graph = self.env.graph()
        return self._graph

    def candidates(self):
        if self._candidates is None:
            self._candidates = self.env.candidates()
        return self._candidates

def rule_config(options):
    return RuleBaselineConfig(
        periodic_interval_minutes=options["period_minutes"],
        atc_k=options["atc_k"],
        relocation_time_penalty=options["travel_penalty"],
        max_reconfiguration_moves=options["max_moves"],
    )

def signed_log(values):
    x = np.asarray(values, dtype=np.float32)
    return np.sign(x) * np.log1p(np.abs(x))

class ResourceFeatures:
    def __init__(self, env, options):
        self.env = EventView(env)
        self.sim = env.sim
        self.instance = env.sim.instance
        self.options = options
        self.cfg = rule_config(options)
        self.visible = env._last_state
        self.orders = {x.order_id: x for x in self.visible.orders}
        self.operations = {x.operation_id: x for x in self.visible.operations}
        n = self.instance.num_stages
        self.work = np.zeros(n)
        self.count = np.zeros((n, self.instance.num_products))
        self.lower = {}
        self.after = {}
        for op in self.visible.operations:
            if op.status.value == "DONE":
                continue
            key = (op.product_type, op.stage)
            if key not in self.lower:
                self.lower[key] = self.instance.minimum_processing_time(*key)
            remaining = self.lower[key]
            if op.status.value == "PROCESSING":
                remaining = _processing_remaining_time(
                    self.instance, self.sim.operations[op.operation_id], self.sim.time
                )
            self.work[op.stage] += remaining
            self.count[op.stage, op.product_type] += 1
        for op in self.visible.operations:
            route = self.instance.product_routes[op.product_type]
            self.after[op.operation_id] = sum(
                self.instance.minimum_processing_time(op.product_type, s)
                for s in route[op.route_position + 1 :]
            )
        self.mix = self.count / np.maximum(1, self.count.sum(axis=1, keepdims=True))
        self.base_capacity, self.base_pressure = self.pressures((), ())

    def pressures(self, workers, robots):
        wc = [c.configured_worker for c in self.visible.cells]
        rc = [c.configured_robot for c in self.visible.cells]
        for cell in self.visible.cells:
            if cell.reserved_worker is not None or cell.reserved_robot is not None:
                wc[cell.cell_id] = rc[cell.cell_id] = None
        for assignments, cfg in ((workers, wc), (robots, rc)):
            for a in assignments:
                for c, rid in enumerate(cfg):
                    if rid == a.resource_id:
                        cfg[c] = None
                cfg[a.target_cell] = a.resource_id
        capacity = np.zeros(self.instance.num_stages)
        for c in self.visible.cells:
            w, r = wc[c.cell_id], rc[c.cell_id]
            for product, q in enumerate(self.mix[c.stage]):
                if q == 0 or (w is None and r is None):
                    continue
                if r is None:
                    p = self.instance.processing_time_h(product, c.stage, w)
                elif w is None:
                    p = self.instance.processing_time_r(product, c.stage, r)
                else:
                    p = self.instance.processing_time_hr(product, c.stage, w, r)
                if p is not None:
                    capacity[c.stage] += q * self.lower[(product, c.stage)] / p
        return capacity, self.work / np.maximum(1, capacity)

    def opportunity(self):
        workers, robots = (), ()
        for _ in range(self.options["max_moves"]):
            _, before = self.pressures(workers, robots)
            choices = []
            masks = self.env.candidates()
            for kind, sets, resources, times, chosen in (
                (
                    "worker",
                    masks.worker,
                    self.sim.workers,
                    self.instance.worker_relocation_time,
                    workers,
                ),
                (
                    "robot",
                    masks.robot,
                    self.sim.robots,
                    self.instance.robot_relocation_time,
                    robots,
                ),
            ):
                used = {x.resource_id for x in chosen}
                for group in sets:
                    rid = group.resource_id
                    if rid in used:
                        continue
                    source = resources[rid].physical_cell
                    a = self.sim.cells[source].stage
                    for target in group.target_cells:
                        b = self.sim.cells[target].stage
                        if a == b:
                            continue
                        assignment = ResourceAssignment(rid, target)
                        ws = workers + ((assignment,) if kind == "worker" else ())
                        rs = robots + ((assignment,) if kind == "robot" else ())
                        try:
                            validate_composite_action(
                                self.sim, CompositeAction(True, ws, rs)
                            )
                        except InvalidActionError:
                            continue
                        _, after = self.pressures(ws, rs)
                        travel = times[rid][source][target]
                        score = (
                            before[b]
                            - after[b]
                            - (after[a] - before[a])
                            - self.options["travel_penalty"] * travel
                        )
                        if score > 0:
                            choices.append((-score, travel, kind, rid, target, ws, rs))
            if not choices:
                break
            best = min(choices)
            workers, robots = best[-2:]
        return workers, robots

    def resources(self, template):
        if template == 1:
            return (), ()
        interval = (
            self.options["short_period_minutes"]
            if template == 4
            else self.cfg.periodic_interval_minutes
        )
        if (
            not self.env.candidates().reconfigure_feasible
            or self.sim.time - self.sim.last_reconfiguration_time < interval - 1e-12
        ):
            return (), ()
        if template < 3:
            return _greedy_reconfiguration(
                self.env,
                replace(
                    self.cfg,
                    max_reconfiguration_moves=1
                    if template == 2
                    else self.cfg.max_reconfiguration_moves,
                ),
            )
        return self.opportunity()

    def dispatch(self, workers, robots, beta):
        prefix = ()
        while True:
            options = _schedule_options(
                self.env,
                reconfigure=bool(workers or robots),
                workers=workers,
                robots=robots,
                prefix=prefix,
            )
            if not options:
                return prefix
            mean_p = sum(p for _, p in options) / len(options)

            def score(item, mean_p=mean_p):
                a, p = item
                order = self.orders[self.operations[a.operation_id].order_id]
                slack = max(
                    0.0,
                    order.due_date
                    - self.sim.time
                    - p
                    - beta * self.after[a.operation_id],
                )
                priority = (
                    order.weight
                    / max(p, 1e-9)
                    * math.exp(-slack / max(self.cfg.atc_k * mean_p, 1e-9))
                )
                return priority, -order.due_date, -a.operation_id

            prefix += (max(options, key=score)[0],)

    def state_features(self):
        rows = []
        for s in range(self.instance.num_stages):
            ops = [
                o
                for o in self.visible.operations
                if o.stage == s and o.status.value != "DONE"
            ]
            cells = [c for c in self.visible.cells if c.stage == s]
            slack = [
                self.orders[o.order_id].due_date
                - self.sim.time
                - self.after[o.operation_id]
                for o in ops
            ]
            rows.append(
                [
                    self.work[s],
                    self.base_capacity[s],
                    self.base_pressure[s],
                    len(ops),
                    sum(o.status.value == "READY" for o in ops),
                    sum(o.status.value == "PROCESSING" for o in ops),
                    sum(self.orders[o.order_id].weight for o in ops),
                    min(slack, default=0),
                    sum(slack) / max(1, len(slack)),
                    sum(c.configured_worker is not None for c in cells),
                    sum(c.configured_robot is not None for c in cells),
                    sum(c.activity.value == "BUSY" for c in cells),
                ]
            )
        elapsed = self.sim.time - self.sim.last_reconfiguration_time
        if not math.isfinite(elapsed):
            elapsed = self.options["period_minutes"]
        global_values = [
            self.sim.time,
            elapsed,
            len(self.visible.orders),
            sum(not o.completed for o in self.visible.orders),
            self.work.sum(),
            self.sim.reconfiguration_count,
        ]
        return signed_log(rows), signed_log(global_values)

    def action_features(self, action):
        plan = validate_composite_action(self.sim, action)
        changes = plan.worker_changes + plan.robot_changes
        _, pressure = self.pressures(
            action.worker_assignments, action.robot_assignments
        )
        delta = pressure - self.base_pressure
        starts = plan.starts
        orders = [self.orders[self.operations[x.operation_id].order_id] for x in starts]
        return signed_log(
            [
                len(changes),
                sum(x.relocation_time for x in changes),
                np.maximum(delta, 0).sum(),
                np.maximum(-delta, 0).sum(),
                len(starts),
                sum(o.weight for o in orders),
                sum(
                    max(
                        0.0,
                        self.sim.time
                        + x.processing_time
                        + self.after[x.operation_id]
                        - o.due_date,
                    )
                    for x, o in zip(starts, orders)
                ),
                sum(x.processing_time for x in starts),
                max((x.processing_time for x in starts), default=0),
                sum(self.after[x.operation_id] for x in starts),
                len(action.worker_assignments),
                len(action.robot_assignments),
            ]
        )
