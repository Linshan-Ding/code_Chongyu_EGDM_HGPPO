"""Persistent workers; each update samples a plan of (case, replicate) episode slots."""
import hashlib
import json
import multiprocessing as mp
import pickle
import time
import traceback
from collections import defaultdict
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


def cpu_weights(network):
    """Workers always sample on CPU; the learner may live on another device."""
    return {k: v.detach().cpu() for k, v in network.state_dict().items()}


def stream_seed(seed, epoch, env_id):
    raw = hashlib.sha256(f"paper-batch-v1:{seed}:{epoch}:{env_id}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % (2**63 - 1)


def case_order(seed, cycle, n_instances):
    """Seeded permutation of the registered training cases for one visiting cycle."""
    raw = hashlib.sha256(f"paper-cycle-v2:{seed}:{cycle}".encode()).digest()
    return np.random.default_rng(int.from_bytes(raw[:8], "big")).permutation(n_instances)


def perturbation_seed(seed, epoch, instance_index):
    raw = hashlib.sha256(f"paper-arrivals-v2:{seed}:{epoch}:{instance_index}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % (2**31 - 1)


def update_plan(options, seed, epoch, n_instances):
    """Episode slots of one update: instances_per_update cases x episodes_per_instance replicates.

    Cases follow seeded permutations of the training set, so every case is visited
    equally often over a cycle and an interrupted run replans identically.
    """
    if epoch < 1:
        raise ValueError("Sampling updates start at 1; update 0 is evaluation only")
    k, n = int(options["instances_per_update"]), int(options["episodes_per_instance"])
    perturb = options.get("train_instance_perturbation", "none") == "arrivals"
    slots = []
    for j in range(k):
        cycle, offset = divmod((epoch - 1) * k + j, n_instances)
        index = int(case_order(seed, cycle, n_instances)[offset])
        pseed = perturbation_seed(seed, epoch, index) if perturb else None
        for r in range(n):
            slots.append(dict(env_id=j * n + r, instance_index=index, replicate=r, perturbation_seed=pseed))
    return slots


def batch_summary(episodes, expected):
    """Never report the mean of only the successful subset."""
    if (len(episodes) != expected or {r["env_id"] for r in episodes} != set(range(expected))
        or not all(r["feasible"] and r["twt"] is not None and np.isfinite(r["twt"]) for r in episodes)):
        raise ValueError("Incomplete/failed batch has no valid TWT mean")
    for field in ("epoch", "policy_version"):
        if len({r[field] for r in episodes}) != 1:
            raise ValueError(f"Mixed batch {field}")
    groups = defaultdict(list)
    for r in episodes:
        groups[r["group"]].append(r)
    for rows in groups.values():
        for field in ("instance_id", "instance_sha256", "initial_state_sha256", "baseline_twt"):
            if len({r[field] for r in rows}) != 1:
                raise ValueError(f"Mixed case group {field}")
    if len({r["action_seed"] for r in episodes}) != expected:
        raise ValueError("Action random streams are not independent")
    values = np.asarray([r["twt"] for r in episodes], dtype=np.float64)
    relative = np.asarray([r["twt"] / max(float(r["baseline_twt"]), 1.) for r in episodes], dtype=np.float64)
    return dict(mean_twt=float(values.mean()), std_twt=float(values.std(ddof=0)),
                min_twt=float(values.min()), max_twt=float(values.max()),
                rel_twt_mean=float(relative.mean()), rel_twt_std=float(relative.std(ddof=0)),
                rel_twt_geomean=float(np.exp(np.log(np.maximum(relative, 1e-9)).mean())),
                instances="|".join(sorted({r["instance_id"] for r in episodes})), groups=len(groups),
                gate_rate=float(np.mean([r["gate_fraction"] for r in episodes])),
                anchor_rate=float(np.mean([r["anchor_fraction"] for r in episodes])),
                moves_per_episode=float(np.mean([r["moves"] for r in episodes])),
                travel_per_episode=float(np.mean([r["travel"] for r in episodes])),
                decision_seconds_per_episode=float(np.mean([r.get("decision_seconds", 0.) for r in episodes])),
                step_seconds_per_episode=float(np.mean([r.get("step_seconds", 0.) for r in episodes])),
                completed=expected, failed=0, batch_events=sum(r["episode_events"] for r in episodes))


class Collector:
    def __init__(self, options, instances, method, seed, ids, validation=()):
        self.options, self.instances, self.method, self.seed = options, instances, method, seed
        self.validation = list(validation)
        self.ids = ids
        self.envs = {i: AssemblyEnv(project_config(options)) for i in ids}
        reference = AssemblyEnv(project_config(options))
        reference.reset(instances[0])
        self.network = create_network(options, reference.graph(), method)
        self.executed, self.episodes = 0, []
        self.perturbed = {}

    def case(self, slot, baselines):
        """The registered case, or its seeded arrival perturbation with a freshly measured anchor TWT."""
        instance = self.instances[slot["instance_index"]]
        pseed = slot.get("perturbation_seed")
        if pseed is None:
            return instance, float(baselines[instance.instance_id])
        key = (instance.instance_id, int(pseed))
        if key not in self.perturbed:
            from data.generator import perturb_arrivals
            from .evaluate import rule_twt
            perturbed = perturb_arrivals(instance, int(pseed))
            self.perturbed = {key: (perturbed, rule_twt(perturbed, self.options))}
        return self.perturbed[key]

    @torch.no_grad()
    def collect(self, weights, version, epoch, plan, baselines, deterministic=False, evidence_dir=None):
        if epoch < 1:
            raise ValueError("Sampling epochs start at 1; epoch 0 is evaluation only")
        self.network.load_state_dict(weights, strict=True)
        self.network.eval()
        self.executed, self.episodes = 0, []
        records, bootstrap, terminal_states = [], {}, []
        for slot in [s for s in plan if s["env_id"] in self.ids]:
            env_id = slot["env_id"]
            started = time.perf_counter()
            env = self.envs[env_id]
            instance, baseline = self.case(slot, baselines)
            iid, instance_hash = instance.instance_id, json_digest(asdict(instance))
            group = iid if slot.get("perturbation_seed") is None else f"{iid}#{slot['perturbation_seed']}"
            env.reset(instance)
            action_seed = stream_seed(self.seed, epoch, env_id)
            generator = torch.Generator().manual_seed(action_seed)
            initial_hash = json_digest(asdict(env.sim.snapshot()))
            initial_phi, base, shaping, moves, travel, anchors, gates, event_count = phi(env), 0., 0., 0, 0., 0, 0, 0
            decision_seconds, step_seconds = 0., 0.
            trace = hashlib.sha256()
            row = dict(epoch=epoch, instance_id=iid, instance_index=int(slot["instance_index"]), replicate=int(slot["replicate"]),
                       group=group, perturbation_seed=slot.get("perturbation_seed"), instance_sha256=instance_hash,
                       initial_state_sha256=initial_hash, env_id=env_id, episode=epoch, policy_version=version,
                       action_seed=action_seed, baseline_twt=baseline, twt=None, rel_twt=None, feasible=False, failure="")
            try:
                for event in range(self.options["event_limit"]):
                    decision_start = time.perf_counter()
                    obs, action_id, action, logp, value, details = learned_action(
                        self.network, env, self.options, generator, deterministic=deterministic)
                    decision_seconds += time.perf_counter() - decision_start
                    details = dict(details, physical_action=asdict(action), physical_time=env.sim.time)
                    plan_check = validate_composite_action(env.sim, action)
                    changes = plan_check.worker_changes + plan_check.robot_changes
                    moves += len(changes)
                    travel += sum(c.relocation_time for c in changes)
                    anchors += int(details["anchor"])
                    gates += int(details["gate"])
                    trace.update(json.dumps(dict(time=env.sim.time, action=asdict(action)),
                                            sort_keys=True, separators=(",", ":")).encode())
                    step_start = time.perf_counter()
                    _, reward, done, info = env.step(action)
                    step_seconds += time.perf_counter() - step_start
                    records.append(Transition(obs, action_id, logp, value, reward / max(baseline, 1.), done,
                        env_id, epoch, event, version, details["choices"] > 1, details, iid))
                    self.executed += 1
                    event_count += 1
                    base += info["reward_twt"]
                    shaping += info["reward_shaping"]
                    if done:
                        row.update(twt=info["twt"], rel_twt=info["twt"] / max(baseline, 1.),
                                   feasible=(abs(base + info["twt"]) <= self.options["reward_tolerance"]
                                             and abs(shaping + initial_phi) <= self.options["reward_tolerance"]))
                        if not row["feasible"]:
                            row["failure"] = "reward_or_potential_identity"
                        break
                else:
                    row["failure"] = "event_limit: incomplete episode"
            except Exception:
                row["failure"] = traceback.format_exc()
            row.update(episode_events=event_count, moves=moves, travel=travel, anchor_fraction=anchors / max(1, event_count),
                       gate_fraction=gates / max(1, event_count), wall_seconds=time.perf_counter() - started,
                       decision_seconds=decision_seconds, step_seconds=step_seconds, trace_sha256=trace.hexdigest())
            self.episodes.append(row)
            if evidence_dir is not None:
                write_json(Path(evidence_dir) / "environment-results" / f"{env_id}.json", row)
            # Audit terminal state/RNG. Next update resets exactly once using its
            # explicit case and derived stream; finished slots never auto-reset.
            terminal_states.append(dict(env_id=env_id, epoch=epoch, instance_id=iid, state=env.sim.snapshot(),
                                        action_rng=generator.get_state(), terminal=row["feasible"]))
            bootstrap[env_id] = 0.
        return dict(format=FORMAT, version=version, epoch=epoch, records=records, bootstrap=bootstrap,
                    episodes=self.episodes, executed=self.executed, terminal_states=terminal_states)

    @torch.no_grad()
    def validate(self, weights, epoch, indices, baselines, output_dir):
        """Deterministic greedy evaluation of registered development cases."""
        from .evaluate import evaluate
        self.network.load_state_dict(weights, strict=True)
        self.network.eval()
        rows = []
        for index in indices:
            instance = self.validation[index]
            output = Path(output_dir) / f"{instance.instance_id}.json"
            result = evaluate(instance, self.options, output, method=self.method, seed=self.seed, network=self.network,
                              baseline=float(baselines[instance.instance_id]), timing_scope="training_validation", trace=False)
            rows.append(dict(epoch=epoch, index=index, instance_id=instance.instance_id, twt=result["twt"],
                             baseline_twt=float(baselines[instance.instance_id]),
                             rel_twt=(result["twt"] / max(float(baselines[instance.instance_id]), 1.)) if result["feasible"] else None,
                             feasible=result["feasible"], failure=result["failure"], gate_fraction=result["gate_fraction"],
                             anchor_fraction=result["anchor_fraction"], moves=result["moves"], travel=result["travel"],
                             decision_p95_ms=result["decision_p95_ms"], wall_seconds=result["wall_seconds"]))
        return dict(format=FORMAT, epoch=epoch, rows=rows)


def worker_main(pipe, options, instances, method, seed, ids, validation):
    collector = None
    try:
        torch.set_num_threads(1)
        collector = Collector(options, instances, method, seed, ids, validation)
        pipe.send_bytes(pickle.dumps(("ready", None)))
        while True:
            command, payload = pickle.loads(pipe.recv_bytes())
            if command == "close":
                break
            if command == "collect":
                result = collector.collect(**payload)
            elif command == "validate":
                result = collector.validate(**payload)
            else:
                raise ValueError(f"Unknown worker command: {command}")
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
    def __init__(self, options, instances, method, seed, validation=()):
        self.options, self.workers, self.last_evidence = options, [], []
        self.ncases, self.nslots = len(instances), options["environments"]
        self.validation = list(validation)
        self.terminal_states = []
        context = mp.get_context("spawn")
        try:
            for worker in range(options["workers"]):
                ids = list(range(worker, self.nslots, options["workers"]))
                parent, child = context.Pipe()
                p = context.Process(target=worker_main, args=(child, options, instances, method, seed, ids, self.validation), daemon=True)
                p.start()
                child.close()
                self.workers.append((p, parent))
            self.receive("ready")
        except BaseException:
            self.close()
            raise

    def collect(self, network, version, epoch, plan, baselines, deterministic=False, evidence_dir=None):
        if {s["env_id"] for s in plan} != set(range(self.nslots)):
            raise ValueError("Update plan must cover every episode slot exactly once")
        payload = dict(weights=cpu_weights(network), version=version, epoch=epoch, plan=plan,
                       baselines=baselines, deterministic=deterministic, evidence_dir=evidence_dir)
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
        planned = {s["env_id"]: s for s in plan}
        for r in episodes:
            if r["env_id"] in planned and r["instance_index"] != planned[r["env_id"]]["instance_index"]:
                errors.append(dict(failure=f"Slot {r['env_id']} sampled an unplanned case"))
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

    def validate(self, network, epoch, baselines, output_dir):
        """Greedy development evaluation spread over the persistent workers."""
        if not self.validation:
            return []
        assignments = [list(range(w, len(self.validation), len(self.workers))) for w in range(len(self.workers))]
        for (_, pipe), indices in zip(self.workers, assignments):
            payload = dict(weights=cpu_weights(network), epoch=epoch, indices=indices, baselines=baselines, output_dir=str(output_dir))
            pipe.send_bytes(pickle.dumps(("validate", payload), protocol=pickle.HIGHEST_PROTOCOL))
        rows = []
        for part in self.receive("ok"):
            if (part["format"], part["epoch"]) != (FORMAT, epoch):
                raise WorkerFailure(dict(failure="Mixed validation format/epoch", epoch=epoch))
            rows.extend(part["rows"])
        rows.sort(key=lambda r: r["index"])
        if [r["index"] for r in rows] != list(range(len(self.validation))):
            raise WorkerFailure(dict(failure="Incomplete validation", epoch=epoch, returned=[r["index"] for r in rows]))
        return rows
