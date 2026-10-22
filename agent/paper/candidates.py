"""Finite resource plans from several criteria; beta=1 ATC dispatch unless a learned dispatcher is attached."""
from dataclasses import dataclass, field
import numpy as np

from .resource_features import ResourceFeatures as BaseBuilder, signed_log, CRITERIA
from agent.baselines.rules import _candidate_resource_moves, _greedy_reconfiguration
from environment.events import EventType
from environment.masks import InvalidActionError, validate_composite_action
from environment.state import CompositeAction, ResourceAssignment

PROTECTED = ("periodic_anchor", "keep")


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
    restricted: bool = False
    dispatch_steps: list = field(default_factory=list)

    @property
    def mask(self):
        # Invalid plans are filtered before construction; every retained action
        # is legal. This explicit mask is saved with event probability evidence.
        return np.ones(len(self.actions), dtype=bool)


class Builder(BaseBuilder):
    # ------------------------------------------------------------------ moves
    def _valid(self, ws, rs):
        try:
            plan = validate_composite_action(self.sim, CompositeAction(True, ws, rs))
        except InvalidActionError:
            return False
        return len(plan.worker_changes) + len(plan.robot_changes) == len(ws) + len(rs)

    def _legal_moves(self, workers=(), robots=()):
        """Legal single moves of idle resources given an already chosen prefix."""
        if not self.env.candidates().reconfigure_feasible:
            return []
        out = []
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
                    if target == source or target < 0:
                        continue
                    move = ResourceAssignment(rid, target)
                    ws = workers + ((move,) if kind == "worker" else ())
                    rs = robots + ((move,) if kind == "robot" else ())
                    if not self._valid(ws, rs):
                        continue
                    out.append((kind, rid, source, target, float(times[rid][source][target]), ws, rs))
        return out

    def ranked_moves(self, workers=(), robots=()):
        """v1 ordering: pressure reduction minus travel penalty, unsigned."""
        if not self.env.candidates().reconfigure_feasible:
            return []
        _, before = self.pressures(workers, robots)
        choices = []
        for kind, rid, source, target, travel, ws, rs in self._legal_moves(workers, robots):
            _, after = self.pressures(ws, rs)
            a, b = self.sim.cells[source].stage, self.sim.cells[target].stage
            # Same-stage moves can change heterogeneous capacity too. Total pressure
            # reduction avoids counting the same stage twice in that case.
            gain = before[b] - after[b]
            if a != b:
                gain -= after[a] - before[a]
            score = gain - self.options["travel_penalty"] * travel
            choices.append((-score, travel, kind, rid, target, ws, rs))
        return sorted(choices)

    def score_moves(self, criterion, workers=(), robots=()):
        """v2 ordering under one criterion; only moves with a positive score survive."""
        if not self.env.candidates().reconfigure_feasible:
            return []
        out = []
        if criterion == "lcr":
            # The same load/capacity scorer Periodic-ATC uses, without its period.
            worker_moves, robot_moves, _ = _candidate_resource_moves(self.env, self.cfg)
            used_w = {x.resource_id for x in workers}
            used_r = {x.resource_id for x in robots}
            for kind, moves, used in (("worker", worker_moves, used_w), ("robot", robot_moves, used_r)):
                for benefit, travel, a in moves:
                    if benefit <= 0 or a.resource_id in used:
                        continue
                    ws = workers + ((a,) if kind == "worker" else ())
                    rs = robots + ((a,) if kind == "robot" else ())
                    if not self._valid(ws, rs):
                        continue
                    out.append((float(benefit), float(travel), kind, a.resource_id, a.target_cell, ws, rs))
        else:
            weighted = criterion == "due"
            _, before = self.pressures(workers, robots, weighted=weighted)
            for kind, rid, source, target, travel, ws, rs in self._legal_moves(workers, robots):
                _, after = self.pressures(ws, rs, weighted=weighted)
                a, b = self.sim.cells[source].stage, self.sim.cells[target].stage
                gain = before[b] - after[b]
                if a != b:
                    gain -= after[a] - before[a]
                score = gain - self.options["travel_penalty"] * travel
                if score > 0:
                    out.append((float(score), travel, kind, rid, target, ws, rs))
        return sorted(out, key=lambda x: (-x[0], x[1], x[2], x[3], x[4]))

    def greedy(self):
        ws = rs = ()
        for _ in range(self.options["max_moves"]):
            choices = self.ranked_moves(ws, rs)
            if not choices or choices[0][0] >= 0:
                break
            ws, rs = choices[0][-2:]
        return ws, rs

    def unconfigure_plans(self):
        """Idle resources leaving over-served stages (target -1 keeps them in place, unconfigured)."""
        plans = []
        if not self.env.candidates().reconfigure_feasible:
            return plans
        for kind, sets, resources in (("worker", self.env.candidates().worker, self.sim.workers),
                                      ("robot", self.env.candidates().robot, self.sim.robots)):
            for group in sets:
                rid = group.resource_id
                source = resources[rid].physical_cell
                if source is None or -1 not in group.target_cells:
                    continue
                stage = self.sim.cells[source].stage
                if self.base_pressure[stage] >= 0.5:
                    continue
                move = ResourceAssignment(rid, -1)
                ws, rs = ((move,), ()) if kind == "worker" else ((), (move,))
                if self._valid(ws, rs):
                    plans.append((f"unconfigure:{kind}:{rid}", ws, rs, ()))
                if len(plans) >= 2:
                    return plans
        return plans

    # ------------------------------------------------------------------ plans
    def plans_v1(self, anchor):
        plans = [("periodic_anchor", *anchor, ()), ("keep", (), (), ())]
        singles = self.ranked_moves()[:self.options["single_candidates"]]
        plans += [(f"single:{c[2]}:{c[3]}:{c[4]}", c[-2], c[-1], ("pressure",)) for c in singles]
        for c in singles[:self.options["pair_candidates"]]:
            follow = self.ranked_moves(c[-2], c[-1])
            if follow:
                plans.append((f"pair:{c[2]}:{c[3]}:{c[4]}", *follow[0][-2:], ("pressure",)))
        plans.append(("opportunity", *self.greedy(), ("pressure",)))
        return plans

    def plans_v2(self, anchor):
        plans = [("periodic_anchor", *anchor, ()), ("keep", (), (), ())]
        if not self.env.candidates().reconfigure_feasible:
            return plans
        now_w, now_r = _greedy_reconfiguration(self.env, self.cfg)
        if now_w or now_r:
            plans.append(("lcr_now", now_w, now_r, ("lcr",)))
        singles = {}
        for criterion in CRITERIA:
            for rank, (score, travel, kind, rid, target, ws, rs) in enumerate(self.score_moves(criterion)[:self.options["single_candidates"]]):
                key = (kind, rid, target)
                entry = singles.setdefault(key, dict(rank=rank, criteria=set(), ws=ws, rs=rs, travel=travel))
                entry["rank"] = min(entry["rank"], rank)
                entry["criteria"].add(criterion)
        ordered = sorted(singles.items(), key=lambda kv: (kv[1]["rank"], -len(kv[1]["criteria"]), kv[1]["travel"], kv[0]))
        chosen = ordered[:self.options["single_candidates"]]
        for key, e in chosen:
            plans.append((f"single:{key[0]}:{key[1]}:{key[2]}", e["ws"], e["rs"], tuple(c for c in CRITERIA if c in e["criteria"])))
        for position, (key, e) in enumerate(chosen[:self.options["pair_candidates"]]):
            criterion = next(c for c in CRITERIA if c in e["criteria"])
            follow = self.score_moves(criterion, e["ws"], e["rs"])
            if not follow:
                continue
            ws, rs = follow[0][-2:]
            plans.append((f"pair:{key[0]}:{key[1]}:{key[2]}", ws, rs, (criterion,)))
            if position == 0 and self.options["candidate_max_moves"] >= 3:
                third = self.score_moves(criterion, ws, rs)
                if third:
                    plans.append((f"triple:{key[0]}:{key[1]}:{key[2]}", *third[0][-2:], (criterion,)))
        plans.append(("opportunity", *self.greedy(), ("pressure",)))
        if self.options.get("allow_unconfigure", False):
            plans += self.unconfigure_plans()
        return plans

    def restricted_event(self):
        """B3: learned deviations only at trigger events or within the reconfiguration budget."""
        if self.options.get("gate_trigger_mode", "every_event") == "trigger_events":
            types = set(self.sim.current_event_types)
            trigger = EventType.ORDER_ARRIVAL in types or EventType.RELOCATION_FINISH in types
            if not trigger and EventType.OPERATION_FINISH in types:
                ready = np.zeros(self.instance.num_stages)
                configured = np.zeros(self.instance.num_stages)
                for op in self.visible.operations:
                    if op.status.value == "READY":
                        ready[op.stage] += 1
                for cell in self.visible.cells:
                    if cell.configured_worker is not None or cell.configured_robot is not None:
                        configured[cell.stage] += 1
                starved = bool(np.any((configured > 0) & (ready == 0)))
                overflow = bool(np.max(self.base_pressure, initial=0.0) >= 1.5)
                trigger = starved or overflow
            if not trigger:
                return True
        budget = self.options.get("reconfiguration_budget")
        if budget:
            limit = budget.get(self.instance.scale)
            if limit is not None and self.sim.reconfiguration_count >= int(limit):
                return True
        return False

    def build(self, method="full"):
        anchor = self.resources(0)
        version = self.options.get("candidate_version", "v1")
        plans = self.plans_v2(anchor) if version == "v2" else self.plans_v1(anchor)
        restricted = self.restricted_event()
        if restricted:
            plans = [p for p in plans if p[0] in PROTECTED]
        actions, names, moves, groups, criteria = [], [], [], [], []
        keys = {}
        for name, ws, rs, flags in plans:
            # Migration order is immaterial; canonical ordering also deduplicates pairs.
            action = CompositeAction(bool(ws or rs), ws, rs, self.dispatch(ws, rs, 1.0))
            plan = validate_composite_action(self.sim, action)
            key = (tuple(sorted(ws, key=lambda a: (a.resource_id, a.target_cell))),
                   tuple(sorted(rs, key=lambda a: (a.resource_id, a.target_cell))), action.schedule_assignments)
            if key in keys:
                names[keys[key]] += "|" + name
                criteria[keys[key]] |= set(flags)
                continue
            if len(actions) >= self.options["max_candidates"] and name not in PROTECTED:
                continue
            keys[key] = len(actions)
            actions.append(action)
            names.append(name)
            criteria.append(set(flags))
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
            actions, names, moves, groups, prior, criteria = ([actions[i] for i in keep], [names[i] for i in keep],
                [moves[i] for i in keep], [groups[i] for i in keep], prior[keep], [criteria[i] for i in keep])
            prior /= prior.sum()
        from .graph import with_reservations
        graph = with_reservations(self.env.graph(), self.visible)
        stages, state = self.state_features()
        state = np.concatenate((state, signed_log(graph.gate_continuous[0].numpy()), graph.event_type_multihot[0].numpy()))
        return Observation(graph, stages, state, np.stack([self.action_features(a, tuple(c)) for a, c in zip(actions, criteria)]),
                           prior, names, actions, np.asarray(groups), moves,
                           next((i for i, n in enumerate(names) if "periodic_anchor" in n), -1), restricted)
