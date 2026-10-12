"""Finite resource plans; fixed beta=1 dispatch, without advancing the simulator."""
from dataclasses import dataclass
import numpy as np

from .resource_features import ResourceFeatures as BaseBuilder, signed_log
from environment.masks import InvalidActionError, validate_composite_action
from environment.state import CompositeAction, ResourceAssignment


@dataclass
class Observation:
    graph: object
    stages: np.ndarray
    state: np.ndarray
    features: np.ndarray
    prior: np.ndarray
    identifiers: list
    actions: list
    groups: np.ndarray
    moves: list
    anchor: int = 0

    @property
    def mask(self):
        # Invalid plans are filtered before construction; every retained action
        # is legal. This explicit mask is saved with event probability evidence.
        return np.ones(len(self.actions), dtype=bool)


class Builder(BaseBuilder):
    def ranked_moves(self, workers=(), robots=()):
        if not self.env.candidates().reconfigure_feasible:
            return []
        _, before = self.pressures(workers, robots)
        choices = []
        for kind, sets, resources, times, chosen in (
            ("worker", self.env.candidates().worker, self.sim.workers, self.instance.worker_relocation_time, workers),
            ("robot", self.env.candidates().robot, self.sim.robots, self.instance.robot_relocation_time, robots),
        ):
            used = {x.resource_id for x in chosen}
            for group in sets:
                rid = group.resource_id
                if rid in used:
                    continue
                source = resources[rid].physical_cell
                for target in group.target_cells:
                    if target == source:
                        continue
                    move = ResourceAssignment(rid, target)
                    ws = workers + ((move,) if kind == "worker" else ())
                    rs = robots + ((move,) if kind == "robot" else ())
                    try:
                        plan = validate_composite_action(self.sim, CompositeAction(True, ws, rs))
                    except InvalidActionError:
                        continue
                    if len(plan.worker_changes) + len(plan.robot_changes) != len(ws) + len(rs):
                        continue
                    _, after = self.pressures(ws, rs)
                    a, b = self.sim.cells[source].stage, self.sim.cells[target].stage
                    travel = times[rid][source][target]
                    # Same-stage moves can change heterogeneous capacity too. Total pressure
                    # reduction avoids counting the same stage twice in that case.
                    gain = before[b] - after[b]
                    if a != b:
                        gain -= after[a] - before[a]
                    score = gain - self.options["travel_penalty"] * travel
                    choices.append((-score, travel, kind, rid, target, ws, rs))
        return sorted(choices)

    def greedy(self):
        ws = rs = ()
        for _ in range(self.options["max_moves"]):
            choices = self.ranked_moves(ws, rs)
            if not choices or choices[0][0] >= 0:
                break
            ws, rs = choices[0][-2:]
        return ws, rs

    def build(self, method="full"):
        anchor_w, anchor_r = self.resources(0)
        plans = [("periodic_anchor", anchor_w, anchor_r), ("keep", (), ())]
        singles = self.ranked_moves()[:self.options["single_candidates"]]
        plans += [(f"single:{c[2]}:{c[3]}:{c[4]}", c[-2], c[-1]) for c in singles]
        for c in singles[:self.options["pair_candidates"]]:
            follow = self.ranked_moves(c[-2], c[-1])
            if follow:
                plans.append((f"pair:{c[2]}:{c[3]}:{c[4]}", *follow[0][-2:]))
        plans.append(("opportunity", *self.greedy()))
        actions, names, moves, groups = [], [], [], []
        keys = {}
        for name, ws, rs in plans:
            # Migration order is immaterial; canonical ordering also deduplicates pairs.
            action = CompositeAction(bool(ws or rs), ws, rs, self.dispatch(ws, rs, 1.0))
            plan = validate_composite_action(self.sim, action)
            key = (tuple(sorted(ws, key=lambda a: (a.resource_id, a.target_cell))),
                   tuple(sorted(rs, key=lambda a: (a.resource_id, a.target_cell))), action.schedule_assignments)
            if key in keys:
                names[keys[key]] += "|" + name
                continue
            keys[key] = len(actions)
            actions.append(action)
            names.append(name)
            changes = plan.worker_changes + plan.robot_changes
            groups.append(int(bool(changes)))
            links = []
            for kind, assignments, resources in (("worker", ws, self.sim.workers), ("robot", rs, self.sim.robots)):
                for a in assignments:
                    links.append((kind, a.resource_id, resources[a.resource_id].physical_cell, a.target_cell))
            moves.append(links)
        prior = np.full(len(actions), (1 - self.options["anchor_prior"]) / max(1, len(actions) - 1), np.float32)
        prior[0] = self.options["anchor_prior"] if len(actions) > 1 else 1.0
        if method == "no_rule_prior":
            prior[:] = 1 / len(prior)
        if method == "periodic_gate":
            eligible = (self.sim.time - self.sim.last_reconfiguration_time >= self.options["period_minutes"] - 1e-12)
            forced_group = int(eligible and any(groups))
            keep = [i for i, g in enumerate(groups) if g == forced_group]
            actions, names, moves, groups, prior = ([actions[i] for i in keep], [names[i] for i in keep],
                                                  [moves[i] for i in keep], [groups[i] for i in keep], prior[keep])
            prior /= prior.sum()
        from .graph import with_reservations
        graph = with_reservations(self.env.graph(), self.visible)
        stages, state = self.state_features()
        state = np.concatenate((state, signed_log(graph.gate_continuous[0].numpy()), graph.event_type_multihot[0].numpy()))
        return Observation(graph, stages, state, np.stack([self.action_features(a) for a in actions]),
                           prior, names, actions, np.asarray(groups), moves,
                           next((i for i, n in enumerate(names) if "periodic_anchor" in n), -1))
