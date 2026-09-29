"""Real on-policy event rollout collection for Phase I/L1 formal training."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import torch

from agent.buffer import HeadValues, PolicyStorageScalars, RolloutBuffer
from agent.constraints import PolicyActionConstraints
from environment.replay_state import CompactReplayMaterializer


@dataclass(frozen=True, slots=True)
class RolloutStats:
    events: int
    completed_episodes: int
    reward_sum: float
    reward_mean: float
    mean_delta_t: float
    mean_schedule_edges: float
    reconfiguration_fraction: float
    mean_raw_reconfigure_probability: float
    max_worker_moves: int
    max_robot_moves: int
    estimated_replay_megabytes: float


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    buffer: RolloutBuffer
    bootstrap_values: dict[tuple[int, int], HeadValues]
    episode_summaries: tuple
    stats: RolloutStats


class RolloutCollector:
    def __init__(
        self,
        *,
        vector_env,
        agent,
        normalizer,
        storage_device="cpu",
        force_gate: bool | None = None,
        constraints: PolicyActionConstraints | None = None,
        storage_mode: str = "full_graph_context",
        progress_every_events: int = 0,
        action_batch_size: int = 1,
        trusted_normalization: bool = False,
        trusted_policy_inputs: bool = False,
        static_processing_cache: bool = False,
    ) -> None:
        self.vector_env = vector_env
        self.agent = agent
        self.normalizer = normalizer
        self.storage_device = storage_device
        self.force_gate = force_gate
        self.constraints = constraints or PolicyActionConstraints(
            force_gate=force_gate
        )
        self.constraints.validate()
        self.storage_mode = str(storage_mode)
        self.progress_every_events = max(0, int(progress_every_events))
        self.action_batch_size = max(1, int(action_batch_size))
        self.trusted_normalization = bool(trusted_normalization)
        self.trusted_policy_inputs = bool(trusted_policy_inputs)
        if self.action_batch_size > len(self.vector_env):
            raise ValueError("action_batch_size cannot exceed vector environment count")
        self.replay_materializer = (
            CompactReplayMaterializer(
                self.vector_env.cfg,
                self.normalizer,
                static_processing_cache=bool(static_processing_cache),
                trusted_replay_normalization=self.trusted_normalization,
            )
            if self.storage_mode == "compact_replay_state" else None
        )
        self.next_env = 0

    def _normalized_graph(self, env_id: int):
        raw_graph = self.vector_env.graph(env_id)
        if self.trusted_normalization:
            return self.normalizer.transform_replay_trusted(raw_graph)
        return self.normalizer.transform(raw_graph)

    @staticmethod
    def _pack_policy_scalars(outputs) -> tuple[list[PolicyStorageScalars], list[float]]:
        """Move rollout scalars from CUDA to CPU once per action batch.

        The legacy path called ``.cpu().item()`` seven/eight times per event.  On
        CUDA that creates a device synchronization for every scalar.  Packing all
        old-policy statistics into one tensor preserves the stored numbers while
        reducing synchronization frequency by roughly the action batch size.
        """
        rows = []
        for out in outputs:
            critics = out.representation.critics
            rows.append(torch.stack([
                out.total_log_prob.reshape(()),
                out.gate_log_prob.reshape(()),
                (out.worker_log_prob + out.robot_log_prob).reshape(()),
                out.schedule_log_prob.reshape(()),
                critics.v_gate.reshape(()),
                critics.v_rec.reshape(()),
                critics.v_sch.reshape(()),
                out.representation.gate.probabilities[0, 1].reshape(()),
            ]))
        packed = torch.stack(rows, dim=0).detach().cpu()
        scalars: list[PolicyStorageScalars] = []
        gate_probs: list[float] = []
        for row in packed.tolist():
            scalars.append(PolicyStorageScalars(
                total_log_prob=float(row[0]),
                gate_log_prob=float(row[1]),
                resource_log_prob=float(row[2]),
                schedule_log_prob=float(row[3]),
                values=HeadValues(float(row[4]), float(row[5]), float(row[6])),
            ))
            gate_probs.append(float(row[7]))
        return scalars, gate_probs

    def collect(self, target_events: int) -> RolloutBatch:
        target_events = int(target_events)
        if target_events <= 0:
            raise ValueError("target_events must be positive")
        buffer = RolloutBuffer(
            storage_device=self.storage_device,
            storage_mode=self.storage_mode,
            replay_materializer=self.replay_materializer,
        )
        summaries = []
        rewards = []
        deltas = []
        schedule_counts = []
        reconfig_flags = []
        raw_gate_probs = []
        worker_move_counts = []
        robot_move_counts = []

        completed_events = 0
        while completed_events < target_events:
            wave_size = min(self.action_batch_size, target_events - completed_events)
            env_ids = []
            graphs = []
            contexts = []
            replay_states = []

            # Each wave touches distinct environment slots at most once.  The old
            # executor stepped these same independent slots in exactly this order;
            # batching only postpones the independent steps until after one shared
            # encoder call and therefore does not couple simulator trajectories.
            for _ in range(wave_size):
                env_id = self.next_env
                self.next_env = (self.next_env + 1) % len(self.vector_env)
                env_ids.append(env_id)
                graphs.append(self._normalized_graph(env_id))
                contexts.append(self.vector_env.action_context(env_id))
                replay_states.append(
                    self.vector_env.replay_snapshot(env_id)
                    if buffer.compact_storage else None
                )

            outputs = self.agent.act_batch(
                graphs,
                contexts,
                deterministic=False,
                force_gate=self.force_gate,
                constraints=self.constraints,
                validate_inputs=not self.trusted_policy_inputs,
            )
            if len(outputs) != wave_size:
                raise RuntimeError("batched policy returned wrong number of actions")
            policy_scalars, wave_gate_probs = self._pack_policy_scalars(outputs)

            for env_id, graph, context, replay_state, output, stored_scalars, gate_prob in zip(
                env_ids, graphs, contexts, replay_states, outputs, policy_scalars, wave_gate_probs
            ):
                _, reward, done, info, summary = self.vector_env.step(env_id, output.action)
                key = self.vector_env.current_key(env_id)
                buffer.add(
                    graph=None if buffer.compact_storage else graph,
                    context=None if buffer.compact_storage else context,
                    policy_output=output,
                    replay_state=replay_state,
                    constraints=self.constraints,
                    policy_scalars=stored_scalars,
                    reward=reward,
                    done=done,
                    delta_t=info["delta_t"],
                    env_id=key[0],
                    episode_id=key[1],
                )
                rewards.append(float(reward))
                deltas.append(float(info["delta_t"]))
                schedule_counts.append(len(output.action.schedule_assignments))
                reconfig_flags.append(bool(output.action.reconfigure))
                worker_move_counts.append(sum(
                    int(a.target_cell != int(context.worker_configured_cell[a.resource_id]))
                    for a in output.action.worker_assignments
                ))
                robot_move_counts.append(sum(
                    int(a.target_cell != int(context.robot_configured_cell[a.resource_id]))
                    for a in output.action.robot_assignments
                ))
                raw_gate_probs.append(gate_prob)
                if summary is not None:
                    summaries.append(summary)
                    self.vector_env.reset_slot(env_id)

            completed_events += wave_size
            if self.progress_every_events and (
                completed_events % self.progress_every_events == 0
                or completed_events == target_events
            ):
                replay_mb = buffer.estimated_replay_bytes / (1024.0 * 1024.0)
                print(
                    f"[rollout] {completed_events}/{target_events} events; "
                    f"action_batch={self.action_batch_size}; compact_replay={buffer.compact_storage}; "
                    f"replay_payload≈{replay_mb:.1f} MiB",
                    flush=True,
                )

        # Only the final fragment for each env can be truncated at the rollout boundary.
        last_by_key = {}
        for transition in buffer.transitions:
            last_by_key[(transition.env_id, transition.episode_id)] = transition
        bootstrap: dict[tuple[int, int], HeadValues] = {}
        for env_id in range(len(self.vector_env)):
            key = self.vector_env.current_key(env_id)
            last = last_by_key.get(key)
            if last is None or last.done:
                continue
            bootstrap[key] = self.agent.value(self._normalized_graph(env_id))

        stats = RolloutStats(
            events=len(buffer),
            completed_episodes=len(summaries),
            reward_sum=sum(rewards),
            reward_mean=mean(rewards),
            mean_delta_t=mean(deltas),
            mean_schedule_edges=mean(schedule_counts),
            reconfiguration_fraction=mean(float(x) for x in reconfig_flags),
            mean_raw_reconfigure_probability=mean(raw_gate_probs),
            max_worker_moves=max(worker_move_counts, default=0),
            max_robot_moves=max(robot_move_counts, default=0),
            estimated_replay_megabytes=buffer.estimated_replay_bytes / (1024.0 * 1024.0),
        )
        return RolloutBatch(
            buffer=buffer,
            bootstrap_values=bootstrap,
            episode_summaries=tuple(summaries),
            stats=stats,
        )


__all__ = ["RolloutBatch", "RolloutCollector", "RolloutStats"]
