"""L1.6 process-local rollout collection for CPU-parallel environment sampling.

This module implements the user's first acceleration strategy without changing
EGDM-HGPPO itself.  Each worker process owns several independent environments
*and a read-only CPU copy of the frozen rollout policy*.  It therefore performs
many environment interactions locally and sends only the compact on-policy
rollout payload back to the parent once per PPO iteration.  The parent keeps the
single authoritative optimizer/GPU model and performs the unchanged PPO update.

Why this design: sending a full heterogeneous graph + ActionContext through a
process pipe at every event is slower than the original serial executor.  Local
worker inference avoids that per-event IPC while still parallelising the costly
simulator/graph/mask work across CPU cores.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import random
import time
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch

from agent.buffer import HeadValues, RolloutBuffer
from agent.policy import EGDMCompositePolicy
from environment.replay_state import CompactReplayMaterializer
from agent.training.curriculum import configure_curriculum_stage
from agent.training.rollout import RolloutCollector, RolloutStats
from agent.training.vector_env import EpisodeSummary, EventVectorEnv


@dataclass(frozen=True, slots=True)
class MultiprocessRolloutConfig:
    worker_processes: int = 4
    worker_torch_threads: int = 2
    start_method: str = "spawn"

    def validate(self, *, parallel_envs: int, rollout_events: int) -> None:
        if self.worker_processes <= 1:
            raise ValueError("multiprocess rollout requires at least two workers")
        if self.worker_processes > int(parallel_envs):
            raise ValueError("worker_processes cannot exceed parallel_envs")
        if int(parallel_envs) % self.worker_processes != 0:
            raise ValueError("parallel_envs must divide evenly across workers")
        if int(rollout_events) % self.worker_processes != 0:
            raise ValueError("rollout_events must divide evenly across workers")
        envs_per_worker = int(parallel_envs) // self.worker_processes
        events_per_worker = int(rollout_events) // self.worker_processes
        if events_per_worker % envs_per_worker != 0:
            raise ValueError(
                "events_per_worker must be divisible by envs_per_worker so every "
                "logical environment contributes equally"
            )
        if self.worker_torch_threads <= 0:
            raise ValueError("worker_torch_threads must be positive")
        if self.start_method not in {"spawn", "forkserver", "fork"}:
            raise ValueError("unsupported multiprocessing start method")


@dataclass(frozen=True, slots=True)
class _WorkerRequest:
    worker_id: int
    cfg: object
    slots: tuple
    global_env_ids: tuple[int, ...]
    max_episode_decisions: int
    normalizer: object
    policy_state: dict
    policy_method: str | None
    ablation_variant: str | None
    ablation_periodic_gate_period: int
    stage_index: int
    constraints: object
    target_events: int
    storage_device: str
    storage_mode: str
    worker_seed: int
    torch_threads: int
    tensorized_decoder_scoring: bool
    replay_candidate_score_memoization: bool
    trusted_policy_inputs: bool


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    worker_id: int
    global_env_ids: tuple[int, ...]
    transitions: tuple
    bootstrap_values: dict
    episode_summaries: tuple
    stats: RolloutStats
    final_slots: tuple
    next_env: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class MultiprocessRolloutResult:
    buffer: RolloutBuffer
    bootstrap_values: dict[tuple[int, int], HeadValues]
    episode_summaries: tuple[EpisodeSummary, ...]
    stats: RolloutStats
    worker_wall_seconds: tuple[float, ...]
    wall_seconds: float
    worker_processes: int
    envs_per_worker: int
    events_per_worker: int
    next_env: int
    vector_env_state_committed: bool


def _seed_worker(seed: int, torch_threads: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    # Avoid 4 workers each creating a full-machine OpenMP pool.
    torch.set_num_threads(int(torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class _CPUInferenceAdapter:
    """Minimal RolloutCollector interface around a CPU policy copy."""

    def __init__(self, policy: EGDMCompositePolicy) -> None:
        self.policy = policy
        self.device = torch.device("cpu")

    @torch.no_grad()
    def act_batch(self, graphs, contexts, **kwargs):
        self.policy.eval()
        return self.policy.act_batch(graphs, contexts, **kwargs)

    @torch.no_grad()
    def value(self, graph) -> HeadValues:
        self.policy.eval()
        representation = self.policy.representation(graph.to("cpu"))
        critics = representation.critics
        return HeadValues(
            float(critics.v_gate.squeeze().cpu()),
            float(critics.v_rec.squeeze().cpu()),
            float(critics.v_sch.squeeze().cpu()),
        )


def _vector_from_slots(cfg, slots, max_episode_decisions: int) -> EventVectorEnv:
    vec = EventVectorEnv.__new__(EventVectorEnv)
    vec.cfg = cfg
    vec.max_episode_decisions = int(max_episode_decisions)
    vec.slots = list(slots)
    return vec


def _worker_collect(request: _WorkerRequest) -> _WorkerResult:
    _seed_worker(request.worker_seed, request.torch_threads)
    started = time.perf_counter()
    vector_env = _vector_from_slots(
        request.cfg, request.slots, request.max_episode_decisions
    )
    reference_graph = request.normalizer.transform(vector_env.graph(0))
    if request.ablation_variant is not None:
        from agent.ablation_variants import build_ablation_policy
        policy = build_ablation_policy(
            request.ablation_variant,
            request.cfg,
            reference_graph,
            periodic_gate_period=request.ablation_periodic_gate_period,
        ).to("cpu")
    elif request.policy_method is None:
        policy = EGDMCompositePolicy(request.cfg, reference_graph).to("cpu")
    else:
        # Lazy import avoids making the production EGDM executor depend on the
        # comparison-policy module unless a learned baseline is actually run.
        from agent.baselines.learned_variants import build_learned_baseline_policy
        policy = build_learned_baseline_policy(
            request.policy_method, request.cfg, reference_graph
        ).to("cpu")
    configure_curriculum_stage(policy, int(request.stage_index))
    policy.load_state_dict(request.policy_state)
    if hasattr(policy, "set_tensorized_decoder_scoring"):
        policy.set_tensorized_decoder_scoring(
            bool(request.tensorized_decoder_scoring)
        )
    for matcher_name in ("worker_matcher", "robot_matcher", "schedule_matcher"):
        matcher = getattr(policy, matcher_name, None)
        if matcher is not None and hasattr(matcher, "memoize_replay_candidate_scores"):
            matcher.memoize_replay_candidate_scores = bool(
                request.replay_candidate_score_memoization
            )
    adapter = _CPUInferenceAdapter(policy)
    collector = RolloutCollector(
        vector_env=vector_env,
        agent=adapter,
        normalizer=request.normalizer,
        storage_device=request.storage_device,
        storage_mode=request.storage_mode,
        progress_every_events=0,
        action_batch_size=len(vector_env),
        trusted_normalization=bool(request.trusted_policy_inputs),
        trusted_policy_inputs=bool(request.trusted_policy_inputs),
        constraints=request.constraints,
    )
    rollout = collector.collect(int(request.target_events))
    return _WorkerResult(
        worker_id=int(request.worker_id),
        global_env_ids=tuple(int(x) for x in request.global_env_ids),
        transitions=tuple(rollout.buffer.transitions),
        bootstrap_values=dict(rollout.bootstrap_values),
        episode_summaries=tuple(rollout.episode_summaries),
        stats=rollout.stats,
        final_slots=tuple(vector_env.slots),
        next_env=int(collector.next_env),
        wall_seconds=float(time.perf_counter() - started),
    )


def _worker_entry(conn, request: _WorkerRequest) -> None:
    """Collect locally and return one ordinary pickle blob.

    ``Connection.send`` uses multiprocessing's Torch tensor reducers, which keep
    file-descriptor/resource-sharer state alive across processes.  A normal
    ``pickle.dumps`` + ``send_bytes`` copies this compact once-per-iteration
    payload inline and lets the worker exit cleanly on Windows/Linux.
    """
    try:
        payload = ("ok", _worker_collect(request))
    except Exception:
        import traceback
        payload = ("error", traceback.format_exc())
    try:
        conn.send_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    finally:
        conn.close()


def _child_seed(base_seed: int, *parts: int) -> int:
    seq = np.random.SeedSequence([int(base_seed), *map(int, parts)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _partition_global_envs(
    *, parallel_envs: int, worker_processes: int, next_env: int
) -> tuple[tuple[int, ...], ...]:
    """Split one legacy round-robin wave into contiguous worker chunks."""
    order = tuple((int(next_env) + i) % int(parallel_envs) for i in range(int(parallel_envs)))
    per = int(parallel_envs) // int(worker_processes)
    return tuple(order[i * per : (i + 1) * per] for i in range(int(worker_processes)))


def _merge_worker_results(
    *,
    cfg,
    normalizer,
    storage_device: str,
    storage_mode: str,
    worker_results: Sequence[_WorkerResult],
    static_processing_cache: bool = False,
    trusted_replay_normalization: bool = False,
) -> tuple[RolloutBuffer, dict, tuple[EpisodeSummary, ...], RolloutStats]:
    results = sorted(worker_results, key=lambda x: x.worker_id)
    if not results:
        raise ValueError("no worker rollout results")
    envs_per_worker = len(results[0].global_env_ids)
    if any(len(x.global_env_ids) != envs_per_worker for x in results):
        raise ValueError("worker environment partitions are unbalanced")
    per_worker_events = len(results[0].transitions)
    if any(len(x.transitions) != per_worker_events for x in results):
        raise ValueError("worker rollout event counts differ")
    if per_worker_events % envs_per_worker != 0:
        raise ValueError("worker transition stream is not aligned to local env waves")

    materializer = (
        CompactReplayMaterializer(
            cfg,
            normalizer,
            static_processing_cache=bool(static_processing_cache),
            trusted_replay_normalization=bool(trusted_replay_normalization),
        )
        if storage_mode == "compact_replay_state" else None
    )
    merged = RolloutBuffer(
        storage_device=storage_device,
        storage_mode=storage_mode,
        replay_materializer=materializer,
    )

    # Interleave local waves so the parent buffer keeps the same global logical
    # env order as the original 32-slot round-robin collector.
    for start in range(0, per_worker_events, envs_per_worker):
        for result in results:
            chunk = result.transitions[start : start + envs_per_worker]
            for transition in chunk:
                local_env = int(transition.env_id)
                global_env = int(result.global_env_ids[local_env])
                merged.transitions.append(replace(transition, env_id=global_env))

    bootstrap: dict[tuple[int, int], HeadValues] = {}
    summaries: list[EpisodeSummary] = []
    for result in results:
        for (local_env, episode_id), value in result.bootstrap_values.items():
            global_env = int(result.global_env_ids[int(local_env)])
            bootstrap[(global_env, int(episode_id))] = value
        for summary in result.episode_summaries:
            global_env = int(result.global_env_ids[int(summary.env_id)])
            summaries.append(replace(summary, env_id=global_env))

    total_events = sum(int(x.stats.events) for x in results)
    if total_events <= 0:
        raise ValueError("worker rollout stats contain zero events")
    def weighted(field: str) -> float:
        return sum(float(getattr(x.stats, field)) * int(x.stats.events) for x in results) / total_events

    stats = RolloutStats(
        events=int(total_events),
        completed_episodes=sum(int(x.stats.completed_episodes) for x in results),
        reward_sum=sum(float(x.stats.reward_sum) for x in results),
        reward_mean=sum(float(x.stats.reward_sum) for x in results) / total_events,
        mean_delta_t=weighted("mean_delta_t"),
        mean_schedule_edges=weighted("mean_schedule_edges"),
        reconfiguration_fraction=weighted("reconfiguration_fraction"),
        mean_raw_reconfigure_probability=weighted("mean_raw_reconfigure_probability"),
        max_worker_moves=max(int(x.stats.max_worker_moves) for x in results),
        max_robot_moves=max(int(x.stats.max_robot_moves) for x in results),
        estimated_replay_megabytes=merged.estimated_replay_bytes / (1024.0 * 1024.0),
    )
    return merged, bootstrap, tuple(summaries), stats


def _commit_worker_slots(vector_env: EventVectorEnv, worker_results: Sequence[_WorkerResult]) -> None:
    """Copy each worker's post-rollout slot back to the authoritative parent vector env.

    L1.6.0--L1.6.7 pilots intentionally discarded worker-local simulator/sampler
    progress because they benchmarked only one isolated iteration.  Formal training
    must keep that state: otherwise every PPO iteration would restart from the same
    checkpoint slots.  The worker result already crosses the process boundary via
    ordinary pickle, so assigning those returned slots is an exact state transfer,
    not a reconstruction.
    """
    seen: set[int] = set()
    for result in sorted(worker_results, key=lambda x: x.worker_id):
        if len(result.final_slots) != len(result.global_env_ids):
            raise RuntimeError("worker returned a mismatched final-slot partition")
        for local_env, global_env in enumerate(result.global_env_ids):
            global_env = int(global_env)
            if global_env in seen:
                raise RuntimeError(f"duplicate worker ownership for env {global_env}")
            if not 0 <= global_env < len(vector_env):
                raise RuntimeError(f"worker returned invalid global env id {global_env}")
            vector_env.slots[global_env] = result.final_slots[int(local_env)]
            seen.add(global_env)
    if seen != set(range(len(vector_env))):
        missing = sorted(set(range(len(vector_env))) - seen)
        raise RuntimeError(f"worker state merge did not cover parent envs: {missing}")


def collect_rollout_multiprocess(
    *,
    cfg,
    vector_env: EventVectorEnv,
    policy,
    normalizer,
    stage_spec,
    target_events: int,
    training_seed: int,
    global_iteration: int,
    collector_next_env: int,
    storage_device: str,
    storage_mode: str,
    mp_config: MultiprocessRolloutConfig,
    commit_vector_env_state: bool = False,
    static_processing_cache: bool = False,
    trusted_replay_normalization: bool = False,
) -> MultiprocessRolloutResult:
    """Collect one on-policy rollout with process-local CPU policy inference.

    ``commit_vector_env_state`` is the L1.6.8 stateful-continuation switch.  It is
    false by default so all earlier one-iteration diagnostics retain their exact
    behavior.  Formal-style continuation sets it true and receives the worker's
    post-rollout simulator *and sampler RNG* state in the parent vector environment.
    """

    parallel_envs = len(vector_env)
    target_events = int(target_events)
    mp_config.validate(parallel_envs=parallel_envs, rollout_events=target_events)
    if not isinstance(vector_env, EventVectorEnv):
        raise TypeError("L1.6 pilot currently expects a serial EventVectorEnv checkpoint")

    partitions = _partition_global_envs(
        parallel_envs=parallel_envs,
        worker_processes=mp_config.worker_processes,
        next_env=int(collector_next_env),
    )
    events_per_worker = target_events // mp_config.worker_processes
    state_cpu = {k: v.detach().cpu() for k, v in policy.state_dict().items()}

    ctx = mp.get_context(mp_config.start_method)
    processes = []
    conns = []
    started = time.perf_counter()
    for worker_id, global_ids in enumerate(partitions):
        slots = tuple(vector_env.slots[int(env_id)] for env_id in global_ids)
        request = _WorkerRequest(
            worker_id=worker_id,
            cfg=cfg,
            slots=slots,
            global_env_ids=tuple(global_ids),
            max_episode_decisions=vector_env.max_episode_decisions,
            normalizer=normalizer,
            policy_state=state_cpu,
            policy_method=(
                None if getattr(policy, "baseline_name", None) is None
                else str(getattr(policy, "baseline_name"))
            ),
            ablation_variant=(
                None if getattr(policy, "ablation_variant", None) is None
                else str(getattr(policy, "ablation_variant"))
            ),
            ablation_periodic_gate_period=int(
                getattr(policy, "ablation_periodic_gate_period", 4)
            ),
            stage_index=int(stage_spec.stage_index),
            constraints=stage_spec.constraints,
            target_events=events_per_worker,
            storage_device=str(storage_device),
            storage_mode=str(storage_mode),
            worker_seed=_child_seed(
                int(training_seed), 1600, int(global_iteration), int(worker_id)
            ),
            torch_threads=int(mp_config.worker_torch_threads),
            tensorized_decoder_scoring=bool(
                getattr(policy.worker_matcher, "tensorized_scoring_enabled", False)
                and getattr(policy.robot_matcher, "tensorized_scoring_enabled", False)
                and getattr(policy.schedule_matcher, "tensorized_scoring_enabled", False)
            ),
            replay_candidate_score_memoization=bool(
                getattr(policy.worker_matcher, "memoize_replay_candidate_scores", False)
                and getattr(policy.robot_matcher, "memoize_replay_candidate_scores", False)
                and getattr(policy.schedule_matcher, "memoize_replay_candidate_scores", False)
            ),
            trusted_policy_inputs=bool(trusted_replay_normalization),
        )
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_worker_entry,
            args=(child, request),
            name=f"egdm-rollout-worker-{worker_id}",
        )
        proc.start()
        child.close()
        processes.append(proc)
        conns.append(parent)

    results = []
    try:
        for worker_id, conn in enumerate(conns):
            status, payload = pickle.loads(conn.recv_bytes())
            if status != "ok":
                raise RuntimeError(f"rollout worker {worker_id} failed:\n{payload}")
            results.append(payload)
    finally:
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass
        for proc in processes:
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)

    wall = time.perf_counter() - started
    merged, bootstrap, summaries, stats = _merge_worker_results(
        cfg=cfg,
        normalizer=normalizer,
        storage_device=storage_device,
        storage_mode=storage_mode,
        worker_results=results,
        static_processing_cache=bool(static_processing_cache),
        trusted_replay_normalization=bool(trusted_replay_normalization),
    )
    if len(merged) != target_events:
        raise RuntimeError(
            f"multiprocess rollout produced {len(merged)} events, expected {target_events}"
        )
    if int(stats.events) != target_events:
        raise RuntimeError(
            f"merged rollout stats report {stats.events} events, expected {target_events}"
        )
    next_env = (int(collector_next_env) + target_events) % parallel_envs
    if commit_vector_env_state:
        _commit_worker_slots(vector_env, results)
    return MultiprocessRolloutResult(
        buffer=merged,
        bootstrap_values=bootstrap,
        episode_summaries=summaries,
        stats=stats,
        worker_wall_seconds=tuple(float(x.wall_seconds) for x in sorted(results, key=lambda x: x.worker_id)),
        wall_seconds=float(wall),
        worker_processes=int(mp_config.worker_processes),
        envs_per_worker=int(parallel_envs // mp_config.worker_processes),
        events_per_worker=int(events_per_worker),
        next_env=int(next_env),
        vector_env_state_committed=bool(commit_vector_env_state),
    )


__all__ = [
    "MultiprocessRolloutConfig",
    "MultiprocessRolloutResult",
    "collect_rollout_multiprocess",
]
