"""Persistent workers; one same-instance terminal episode per slot and epoch."""
import hashlib
import json
import multiprocessing as mp
import pickle
import time
import traceback
from dataclasses import asdict, dataclass
import numpy as np
import torch
from environment.env import AssemblyEnv
from environment.masks import validate_composite_action
from .worker_pool import WorkerPool as BasePool, WorkerFailure, phi
from .config import FORMAT, project_config
from .network import create_network
from .policies import learned_action
from .storage import json_digest, write_json, save_checkpoint
from pathlib import Path


@dataclass
class Transition:
    observation: object
    action: object
    logp: float
    value: float
    reward: float
    done: bool
    env_id: int
    episode: int
    event: int
    version: int
    choice: bool
    details: dict
    instance_id: str
    advantage: float = 0.
    target: float = 0.


def stream_seed(seed, epoch, env_id):
    raw = hashlib.sha256(f"paper-batch-v1:{seed}:{epoch}:{env_id}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % (2**63 - 1)


def batch_summary(episodes, expected):
    """Never report the mean of only the successful subset."""
    if (len(episodes) != expected or {r["env_id"] for r in episodes} != set(range(expected))
        or not all(r["feasible"] and r["twt"] is not None and np.isfinite(r["twt"]) for r in episodes)):
        raise ValueError("Incomplete/failed batch has no valid TWT mean")
    for field in ("epoch", "instance_id", "instance_sha256", "initial_state_sha256", "policy_version"):
        if len({r[field] for r in episodes}) != 1:
            raise ValueError(f"Mixed batch {field}")
    if len({r["action_seed"] for r in episodes}) != expected:
        raise ValueError("Action random streams are not independent")
    values = np.asarray([r["twt"] for r in episodes], dtype=np.float64)
    return dict(mean_twt=float(values.mean()), std_twt=float(values.std(ddof=0)),
                min_twt=float(values.min()), max_twt=float(values.max()),
                completed=expected, failed=0, batch_events=sum(r["episode_events"] for r in episodes))


class Collector:
    def __init__(self, options, instances, method, seed, ids):
        self.options, self.instances, self.method, self.seed = options, instances, method, seed
        self.ids = ids
        self.envs = {i: AssemblyEnv(project_config(options)) for i in ids}
        reference = AssemblyEnv(project_config(options))
        reference.reset(instances[0])
        self.network = create_network(options, reference.graph(), method)
        self.executed, self.episodes = 0, []

    @torch.no_grad()
    def collect(self, weights, version, epoch, instance_index, baselines, deterministic=False, evidence_dir=None):
        if epoch < 1:
            raise ValueError("Sampling epochs start at 1; epoch 0 is evaluation only")
        self.network.load_state_dict(weights, strict=True)
        self.network.eval()
        self.executed, self.episodes = 0, []
        records, bootstrap, terminal_states = [], {}, []
        instance = self.instances[instance_index]
        iid, instance_hash = instance.instance_id, json_digest(asdict(instance))
        for env_id in self.ids:
            started = time.perf_counter()
            env = self.envs[env_id]
            env.reset(instance)
            action_seed = stream_seed(self.seed, epoch, env_id)
            generator = torch.Generator().manual_seed(action_seed)
            initial_hash = json_digest(asdict(env.sim.snapshot()))
            initial_phi, base, shaping, moves, travel, anchors, event_count = phi(env), 0., 0., 0, 0., 0, 0
            trace = hashlib.sha256()
            row = dict(epoch=epoch, instance_id=iid, instance_sha256=instance_hash,
                       initial_state_sha256=initial_hash, env_id=env_id, episode=epoch,
                       policy_version=version, action_seed=action_seed, twt=None, feasible=False, failure="")
            try:
                for event in range(self.options["event_limit"]):
                    obs, action_id, action, logp, value, details = learned_action(
                        self.network, env, self.options, generator, deterministic=deterministic)
                    details = dict(details, physical_action=asdict(action), physical_time=env.sim.time)
                    plan = validate_composite_action(env.sim, action)
                    changes = plan.worker_changes + plan.robot_changes
                    moves += len(changes)
                    travel += sum(c.relocation_time for c in changes)
                    anchors += int(details["anchor"])
                    trace.update(json.dumps(dict(time=env.sim.time, action=asdict(action)),
                                            sort_keys=True, separators=(",", ":")).encode())
                    _, reward, done, info = env.step(action)
                    records.append(Transition(obs, action_id, logp, value, reward / max(baselines[iid], 1.), done,
                        env_id, epoch, event, version, details["choices"] > 1, details, iid))
                    self.executed += 1
                    event_count += 1
                    base += info["reward_twt"]
                    shaping += info["reward_shaping"]
                    if done:
                        row.update(twt=info["twt"], feasible=(abs(base + info["twt"]) <= self.options["reward_tolerance"]
                                   and abs(shaping + initial_phi) <= self.options["reward_tolerance"]))
                        if not row["feasible"]:
                            row["failure"] = "reward_or_potential_identity"
                        break
                else:
                    row["failure"] = "event_limit: incomplete episode"
            except Exception:
                row["failure"] = traceback.format_exc()
            row.update(episode_events=event_count, moves=moves, travel=travel, anchor_fraction=anchors / max(1, event_count),
                       wall_seconds=time.perf_counter() - started, trace_sha256=trace.hexdigest())
            self.episodes.append(row)
            if evidence_dir is not None:
                write_json(Path(evidence_dir) / "environment-results" / f"{env_id}.json", row)
            # Audit terminal state/RNG. Next epoch resets exactly once using its
            # explicit case and derived stream; finished slots never auto-reset.
            terminal_states.append(dict(env_id=env_id, epoch=epoch, instance_id=iid, state=env.sim.snapshot(),
                                        action_rng=generator.get_state(), terminal=row["feasible"]))
            bootstrap[env_id] = 0.
        return dict(format=FORMAT, version=version, epoch=epoch, records=records, bootstrap=bootstrap,
                    episodes=self.episodes, executed=self.executed, terminal_states=terminal_states)


def worker_main(pipe, options, instances, method, seed, ids):
    collector = None
    try:
        torch.set_num_threads(1)
        collector = Collector(options, instances, method, seed, ids)
        pipe.send_bytes(pickle.dumps(("ready", None)))
        while True:
            command, payload = pickle.loads(pipe.recv_bytes())
            if command == "close":
                break
            if command != "collect":
                raise ValueError(f"Unknown worker command: {command}")
            result = collector.collect(**payload)
            pipe.send_bytes(pickle.dumps(("ok", result), protocol=pickle.HIGHEST_PROTOCOL))
            del result
    except BaseException:
        try:
            pipe.send_bytes(pickle.dumps(("error", dict(traceback=traceback.format_exc(),
                executed=collector.executed if collector else 0, episodes=collector.episodes if collector else []))))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        pipe.close()


class Pool(BasePool):
    def __init__(self, options, instances, method, seed):
        self.options, self.workers, self.last_evidence = options, [], []
        self.ncases, self.nslots = len(instances), options["environments"]
        self.terminal_states = []
        context = mp.get_context("spawn")
        try:
            for worker in range(options["workers"]):
                ids = list(range(worker, self.nslots, options["workers"]))
                parent, child = context.Pipe()
                p = context.Process(target=worker_main, args=(child, options, instances, method, seed, ids), daemon=True)
                p.start()
                child.close()
                self.workers.append((p, parent))
            self.receive("ready")
        except BaseException:
            self.close()
            raise

    def collect(self, network, version, epoch, instance_index, baselines, deterministic=False, evidence_dir=None):
        payload = dict(weights=network.state_dict(), version=version, epoch=epoch,
                       instance_index=instance_index, baselines=baselines, deterministic=deterministic, evidence_dir=evidence_dir)
        message = pickle.dumps(("collect", payload), protocol=pickle.HIGHEST_PROTOCOL)
        for _, pipe in self.workers:
            pipe.send_bytes(message)
        records, bootstrap, episodes, errors = [], {}, [], []
        self.terminal_states = []
        # Drain all workers after failures, preserving successful/partial evidence.
        for i, (process, pipe) in enumerate(self.workers):
            try:
                status, part = pickle.loads(pipe.recv_bytes())
            except (EOFError, OSError) as exc:
                errors.append(dict(worker=i, exitcode=process.exitcode, failure=repr(exc)))
                continue
            if status != "ok":
                errors.append(dict(worker=i, **part))
                episodes.extend(part.get("episodes", []))
                continue
            if (part["format"], part["version"], part["epoch"]) != (FORMAT, version, epoch):
                errors.append(dict(worker=i, failure="Mixed rollout format/epoch/policy version"))
            records.extend(part["records"])
            episodes.extend(part["episodes"])
            bootstrap.update(part["bootstrap"])
            self.terminal_states.extend(part["terminal_states"])
        records.sort(key=lambda r: (r.env_id, r.event))
        episodes.sort(key=lambda r: r["env_id"])
        self.terminal_states.sort(key=lambda r: r["env_id"])
        try:
            batch_summary(episodes, self.nslots)
        except ValueError as exc:
            errors.append(dict(failure=str(exc)))
        if errors:
            if evidence_dir is not None:
                save_checkpoint(Path(evidence_dir) / "failed-rollout.pt", dict(records=records,
                    bootstrap=bootstrap, epoch=epoch, policy_version=version, errors=errors))
            raise WorkerFailure(dict(epoch=epoch, policy_version=version, errors=errors, episodes=episodes,
                                     executed_events=len(records), missing_env_ids=sorted(set(range(self.nslots)) - {r["env_id"] for r in episodes})))
        return records, bootstrap, episodes
