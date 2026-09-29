"""Rolling-horizon ALNS baseline using only currently visible jobs/resources.

The ALNS optimizes the current event's resource-rematching set.  No unreleased
arrival, due date or route is queried.  After each destroy/repair move the complete
composite action is checked by the same simulator hard constraints used by RL.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random

from agent.baselines.base import DecisionPolicy, RuleBaselineConfig
from agent.baselines.optimization import ensure_progress_action
from agent.baselines.rules import (
    _candidate_resource_moves, _dispatch, _schedule_options, _trial_plan, _visible_order_map
)
from environment.state import CompositeAction


@dataclass(frozen=True, slots=True)
class ALNSBaselineConfig:
    iterations_per_event: int = 40
    max_reconfiguration_moves: int = 4
    reaction_factor: float = 0.20
    random_accept_temperature: float = 0.10
    relocation_penalty: float = 0.05
    imbalance_weight: float = 1.0
    schedule_weight: float = 1.0

    def validate(self) -> None:
        if self.iterations_per_event <= 0:
            raise ValueError("ALNS iterations_per_event must be positive")
        if self.max_reconfiguration_moves < 0:
            raise ValueError("ALNS max_reconfiguration_moves must be non-negative")
        if not 0 < self.reaction_factor <= 1:
            raise ValueError("ALNS reaction_factor must be in (0,1]")
        if self.random_accept_temperature < 0 or self.relocation_penalty < 0:
            raise ValueError("ALNS temperatures/penalties must be non-negative")


class RollingHorizonALNSPolicy(DecisionPolicy):
    method_name = "RH-ALNS"

    def __init__(self, cfg: ALNSBaselineConfig, rule_cfg: RuleBaselineConfig) -> None:
        cfg.validate(); rule_cfg.validate()
        self.cfg = cfg
        self.rule_cfg = rule_cfg
        self.rng = random.Random(0)
        self.weights = [1.0, 1.0, 1.0]

    def reset(self, env) -> None:
        # Stable per-instance randomness: deterministic evaluation while retaining
        # stochastic neighborhood search internally.
        iid = str(env.sim.instance.instance_id).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(iid).digest()[:8], "little")
        self.rng = random.Random(seed)
        self.weights = [1.0, 1.0, 1.0]

    def _pool(self, env):
        rcfg = RuleBaselineConfig(
            periodic_interval_minutes=self.rule_cfg.periodic_interval_minutes,
            atc_k=self.rule_cfg.atc_k,
            threshold_load_capacity_ratio=self.rule_cfg.threshold_load_capacity_ratio,
            bottleneck_min_gap=self.rule_cfg.bottleneck_min_gap,
            relocation_time_penalty=self.cfg.relocation_penalty,
            max_reconfiguration_moves=self.cfg.max_reconfiguration_moves,
        )
        w, r, _ = _candidate_resource_moves(env, rcfg)
        tagged = [("w", float(b), float(t), a) for b, t, a in w]
        tagged += [("r", float(b), float(t), a) for b, t, a in r]
        return [x for x in tagged if x[1] > 0]

    @staticmethod
    def _to_sets(tagged, indices):
        workers = tuple(tagged[i][3] for i in indices if tagged[i][0] == "w")
        robots = tuple(tagged[i][3] for i in indices if tagged[i][0] == "r")
        return workers, robots

    def _valid(self, env, tagged, indices):
        workers, robots = self._to_sets(tagged, indices)
        return _trial_plan(env, reconfigure=bool(indices), workers=workers, robots=robots, schedule=()) is not None

    def _score(self, env, tagged, indices):
        workers, robots = self._to_sets(tagged, indices)
        reconfigure = bool(indices)
        schedule = _dispatch(
            env, rule="ATC", atc_k=self.rule_cfg.atc_k,
            reconfigure=reconfigure, workers=workers, robots=robots,
        )
        move_gain = sum(tagged[i][1] for i in indices)
        orders = _visible_order_map(env)
        now = float(env.sim.time)
        sched_gain = 0.0
        plan = _trial_plan(
            env, reconfigure=reconfigure, workers=workers, robots=robots, schedule=schedule
        )
        if plan is not None:
            for start in plan.starts:
                op = env.sim.operations[start.operation_id]
                order = orders[op.order_id]
                tardy_pressure = max(0.0, now + start.processing_time - float(order.due_date))
                sched_gain += float(order.weight) * (1.0 + tardy_pressure) / max(start.processing_time, 1e-9)
        return self.cfg.imbalance_weight * move_gain + self.cfg.schedule_weight * sched_gain, schedule

    def _operator(self) -> int:
        total = sum(self.weights)
        x = self.rng.random() * total
        acc = 0.0
        for i, w in enumerate(self.weights):
            acc += w
            if x <= acc:
                return i
        return len(self.weights) - 1

    def _mutate(self, current: tuple[int, ...], pool_size: int, op: int) -> tuple[int, ...]:
        cur = list(current)
        if op == 0:  # destroy one
            if cur:
                cur.pop(self.rng.randrange(len(cur)))
        elif op == 1:  # repair/add one
            choices = [i for i in range(pool_size) if i not in cur]
            if choices and len(cur) < self.cfg.max_reconfiguration_moves:
                cur.append(self.rng.choice(choices))
        else:  # swap
            if cur:
                cur.pop(self.rng.randrange(len(cur)))
            choices = [i for i in range(pool_size) if i not in cur]
            if choices and len(cur) < self.cfg.max_reconfiguration_moves:
                cur.append(self.rng.choice(choices))
        return tuple(sorted(set(cur)))

    def act(self, env) -> CompositeAction:
        if not env.candidates().reconfigure_feasible:
            schedule = _dispatch(
                env, rule="ATC", atc_k=self.rule_cfg.atc_k,
                reconfigure=False, workers=(), robots=(),
            )
            return ensure_progress_action(env, self.rule_cfg, CompositeAction.keep(schedule))
        pool = self._pool(env)
        if not pool:
            schedule = _dispatch(
                env, rule="ATC", atc_k=self.rule_cfg.atc_k,
                reconfigure=False, workers=(), robots=(),
            )
            return ensure_progress_action(env, self.rule_cfg, CompositeAction.keep(schedule))

        # Greedy seed from best positive moves, repaired through exact hard constraints.
        current: tuple[int, ...] = ()
        for i in sorted(range(len(pool)), key=lambda k: pool[k][1], reverse=True):
            if len(current) >= self.cfg.max_reconfiguration_moves:
                break
            trial = tuple(sorted(current + (i,)))
            if self._valid(env, pool, trial):
                current = trial
        current_score, current_sched = self._score(env, pool, current)
        best, best_score, best_sched = current, current_score, current_sched

        for _ in range(self.cfg.iterations_per_event):
            op = self._operator()
            trial = self._mutate(current, len(pool), op)
            if len(trial) > self.cfg.max_reconfiguration_moves or not self._valid(env, pool, trial):
                self.weights[op] = (1-self.cfg.reaction_factor)*self.weights[op] + self.cfg.reaction_factor*0.1
                continue
            score, sched = self._score(env, pool, trial)
            delta = score - current_score
            accept = delta >= 0
            if not accept and self.cfg.random_accept_temperature > 0:
                accept = self.rng.random() < math.exp(delta / self.cfg.random_accept_temperature)
            reward = 0.5
            if accept:
                current, current_score, current_sched = trial, score, sched
                reward = 1.0
            if score > best_score + 1e-12:
                best, best_score, best_sched = trial, score, sched
                reward = 3.0
            self.weights[op] = (
                (1-self.cfg.reaction_factor)*self.weights[op]
                + self.cfg.reaction_factor*reward
            )

        workers, robots = self._to_sets(pool, best)
        action = CompositeAction(
            reconfigure=bool(best),
            worker_assignments=workers,
            robot_assignments=robots,
            schedule_assignments=best_sched,
        )
        if _trial_plan(
            env, reconfigure=action.reconfigure, workers=workers, robots=robots,
            schedule=best_sched,
        ) is None:
            # Defensive fallback: never send an ALNS-internal infeasible solution to env.step.
            schedule = _dispatch(
                env, rule="EDD", atc_k=self.rule_cfg.atc_k,
                reconfigure=False, workers=(), robots=(),
            )
            return CompositeAction.keep(schedule)
        return ensure_progress_action(env, self.rule_cfg, action)


__all__ = ["ALNSBaselineConfig", "RollingHorizonALNSPolicy"]
