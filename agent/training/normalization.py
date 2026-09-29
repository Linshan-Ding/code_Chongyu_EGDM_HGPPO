"""Training-only graph-statistics collection for Phase I."""

from __future__ import annotations

from dataclasses import dataclass

from environment.env import AssemblyEnv
from environment.graph_normalization import GraphContinuousNormalizer
from environment.state import CompositeAction, ScheduleAssignment


@dataclass(frozen=True, slots=True)
class NormalizationCollectionSummary:
    graphs: int
    episodes_started: int
    episodes_completed: int


def _greedy_fixed_configuration_action(env: AssemblyEnv) -> CompositeAction:
    candidates = sorted(
        env.candidates().schedule,
        key=lambda c: (c.operation_id, c.cell_id),
    )
    selected = []
    used_ops: set[int] = set()
    used_cells: set[int] = set()
    for candidate in candidates:
        if candidate.operation_id in used_ops or candidate.cell_id in used_cells:
            continue
        selected.append(ScheduleAssignment(candidate.operation_id, candidate.cell_id))
        used_ops.add(candidate.operation_id)
        used_cells.add(candidate.cell_id)
    return CompositeAction.keep(selected)


def fit_training_normalizer(
    cfg,
    sampler,
    *,
    episodes: int,
    max_graphs: int,
    max_episode_decisions: int,
    progress_every_graphs: int = 0,
):
    """Fit z-score statistics on online training states only, streaming one graph at a time.

    The greedy dispatcher is used solely to visit physically varied training
    states (idle/busy/late etc.). Its actions are not stored as demonstrations
    and never enter PPO supervision.  Phase L1.2 does not retain the graph list,
    so peak memory is bounded by one raw graph plus the small moment accumulators.
    """
    episodes = int(episodes)
    max_graphs = int(max_graphs)
    progress_every_graphs = max(0, int(progress_every_graphs))
    if episodes <= 0 or max_graphs <= 0:
        raise ValueError("normalization episodes/max_graphs must be positive")

    counters = {"graphs": 0, "completed": 0, "started": 0}

    def graph_stream():
        for _ in range(episodes):
            if counters["graphs"] >= max_graphs:
                break
            env = AssemblyEnv(cfg)
            env.reset(sampler.sample())
            counters["started"] += 1
            for _decision in range(int(max_episode_decisions)):
                yield env.graph()
                counters["graphs"] += 1
                if progress_every_graphs and (
                    counters["graphs"] % progress_every_graphs == 0
                    or counters["graphs"] >= max_graphs
                ):
                    print(
                        f"[normalizer] {counters['graphs']}/{max_graphs} graphs streamed",
                        flush=True,
                    )
                if counters["graphs"] >= max_graphs:
                    break
                _, _, done, _ = env.step(_greedy_fixed_configuration_action(env))
                if done:
                    counters["completed"] += 1
                    break
            else:
                raise RuntimeError("normalization rollout exceeded max_episode_decisions")
            if counters["graphs"] >= max_graphs:
                break

    normalizer = GraphContinuousNormalizer(
        min_std=float(cfg.env.graph.normalizer_min_std)
    )
    normalizer.fit(graph_stream())
    return normalizer, NormalizationCollectionSummary(
        graphs=int(counters["graphs"]),
        episodes_started=int(counters["started"]),
        episodes_completed=int(counters["completed"]),
    )


__all__ = ["NormalizationCollectionSummary", "fit_training_normalizer"]
