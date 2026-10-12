import hashlib
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from environment.env import AssemblyEnv
from environment.masks import validate_composite_action
from .worker_pool import phi
from .config import project_config
from .policies import Policy, learned_action
from .storage import write_json, preserve_file


def evaluate(instance, options, output, *, method="full", seed=0, network=None, baseline=None, timing_scope="in_process"):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preserve_file(output)
    preserve_file(output.with_suffix(".trace.jsonl"))
    env = AssemblyEnv(project_config(options))
    env.reset(instance)
    policy = None if network is not None else Policy(method, options, seed)
    if policy:
        policy.reset(env)
    if network is not None:
        network.eval()
    initial_phi, base, shaping = phi(env), 0., 0.
    durations, trace, selected = [], hashlib.sha256(), {}
    row = dict(instance_id=instance.instance_id, method=method, seed=seed, feasible=False, failure="", twt=None,
               gap=None, reward_error=None, shaping_error=None, trace_sha256=None, decisions=0, moves=0,
               travel=0., reconfigurations=0, gate_fraction=None, anchor_fraction=None,
               decision_p95_ms=None, wall_seconds=None, timing_scope=timing_scope,
               reference_kind="empirical_policy", baseline_twt=baseline, fallback_status="not_applied",
               beta=1.0 if method != "dan" else None)
    begin, anchors, gates = time.perf_counter(), 0, 0
    try:
        with output.with_suffix(".trace.jsonl").open("w", encoding="utf-8") as stream, torch.no_grad():
            for event in range(options["event_limit"]):
                start = time.perf_counter()
                if network is not None:
                    _, _, action, _, _, details = learned_action(network, env, options, deterministic=True)
                else:
                    action = policy.act(env)
                    details = dict(candidate=method, gate=int(action.reconfigure), anchor=method == "Periodic-ATC")
                plan = validate_composite_action(env.sim, action)
                durations.append((time.perf_counter() - start) * 1000)
                changes = plan.worker_changes + plan.robot_changes
                row["moves"] += len(changes)
                row["travel"] += sum(c.relocation_time for c in changes)
                anchors += int(details["anchor"])
                gates += int(details["gate"])
                selected[details["candidate"]] = selected.get(details["candidate"], 0) + 1
                canonical = dict(time=env.sim.time, action=asdict(action))
                event_types = [t.value for t in env._last_state.event_types]
                trace.update(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode())
                _, _, done, info = env.step(action)
                base += info["reward_twt"]
                shaping += info["reward_shaping"]
                stream.write(json.dumps(dict(canonical, event=event, event_types=event_types, details=details,
                                             reward_twt=info["reward_twt"], reward_shaping=info["reward_shaping"])) + "\n")
                row["decisions"] += 1
                if done:
                    row.update(twt=info["twt"], reward_error=abs(base + info["twt"]), shaping_error=abs(shaping + initial_phi))
                    row["feasible"] = row["reward_error"] <= options["reward_tolerance"] and row["shaping_error"] <= options["reward_tolerance"]
                    if not row["feasible"]:
                        row["failure"] = "reward_or_potential_identity"
                    break
            else:
                row["failure"] = "event_limit"
    except Exception:
        row["failure"] = traceback.format_exc()
    row.update(wall_seconds=time.perf_counter() - begin, decision_p95_ms=float(np.percentile(durations, 95)) if durations else None,
               reconfigurations=env.sim.reconfiguration_count, trace_sha256=trace.hexdigest(),
               gate_fraction=gates / max(1, row["decisions"]), anchor_fraction=anchors / max(1, row["decisions"]))
    if row["feasible"] and baseline is not None:
        row["gap"] = 100 * (row["twt"] - baseline) / max(baseline, 1.)
    write_json(output, dict(row, counts=selected))
    return row


def selection_score(results, baselines):
    failures = sum(not r["feasible"] for r in results)
    score = sum(r["twt"] / max(baselines[r["instance_id"]], 1.) for r in results if r["feasible"]) / max(1, len(results) - failures)
    return failures, score
