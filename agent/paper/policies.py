"""Baseline adapters preserve declared search budgets and physical feasibility."""
import hashlib
import random
from .resource_features import EventView, rule_config
from agent.baselines.rules import FixedEDDPolicy, PeriodicATCPolicy, _RulePolicy
from agent.baselines.optimization import RollingHorizonMILPPolicy, OptimizationBaselineConfig
from agent.baselines.alns import RollingHorizonALNSPolicy, ALNSBaselineConfig
from environment.state import CompositeAction
from .candidates import Builder


class BetaView(EventView):
    dispatch_beta = 1.0


class Policy:
    def __init__(self, name, options, seed=0):
        self.name, self.options, self.seed = name, options, seed
        rule = rule_config(options)
        if name == "RH-MILP":
            self.policy = RollingHorizonMILPPolicy(OptimizationBaselineConfig(
                rolling_time_limit_seconds=options["milp_seconds"], rolling_max_reconfiguration_moves=options["milp_moves"]), rule)
        elif name == "RH-ALNS":
            self.policy = RollingHorizonALNSPolicy(ALNSBaselineConfig(iterations_per_event=options["alns_iterations"],
                max_reconfiguration_moves=options["alns_moves"]), rule)
        elif name != "Opportunity-ATC":
            self.policy = (_RulePolicy(rule, "Fixed-ATC", "ATC") if name == "Fixed-ATC" else
                           {"Fixed-EDD": FixedEDDPolicy, "Periodic-ATC": PeriodicATCPolicy}[name](rule))

    def reset(self, env):
        if hasattr(self, "policy"):
            self.policy.reset(env)
        if self.name == "RH-ALNS":
            data = f"{env.sim.instance.instance_id}:{self.seed}".encode()
            self.policy.rng = random.Random(int.from_bytes(hashlib.sha256(data).digest()[:8], "little"))

    def act(self, env):
        if self.name == "Opportunity-ATC":
            b = Builder(env, self.options)
            # Historical R3 resource logic retained: periodic eligibility and positive moves.
            ws, rs = b.resources(3)
            return CompositeAction(bool(ws or rs), ws, rs, b.dispatch(ws, rs, 1.))
        return self.policy.act(BetaView(env))


def learned_action(network, env, options, generator=None, deterministic=False):
    if network.method == "dan":
        return network.act(env, options, generator, deterministic)
    builder = Builder(env, options)
    observation = builder.build(network.method)
    index, logp, value, details = network.select(observation, generator, deterministic)
    action = observation.actions[index]
    if getattr(network, "learned_dispatch", False):
        steps, schedule, dispatch_logp, atc_steps = network.dispatch(observation, index, builder, generator, deterministic)
        observation.dispatch_steps = steps
        action = CompositeAction(action.reconfigure, action.worker_assignments, action.robot_assignments, schedule)
        logp += dispatch_logp
        details.update(dispatch_logp=dispatch_logp, dispatch_steps=len(steps), dispatch_atc_steps=atc_steps,
                       joint_logp=logp, choices=max([details["choices"]] + [len(s.options) for s in steps]),
                       dispatch=[(s.identifiers[s.choice], s.choice == s.atc) for s in steps])
        index = (index, tuple(s.choice for s in steps))
    return observation, index, action, logp, value, details
