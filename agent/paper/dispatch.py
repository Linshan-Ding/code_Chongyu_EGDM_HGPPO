"""Learned operation-to-cell dispatch as a residual on the ATC(beta=1) ranking.

Each dispatch step offers the legal (ready operation, idle configured cell) pairs
of the current prefix, exactly the options the fixed ATC rule iterates over. The
prior puts ``dispatch_prior`` mass on the ATC choice and spreads the rest, and a
zero-initialised residual head starts at that prior, so the untrained policy
dispatches exactly like the rule baselines.
"""
import math
from dataclasses import dataclass, field
import numpy as np
from agent.baselines.rules import _schedule_options
from .resource_features import signed_log

DISPATCH_DIM = 11


@dataclass
class DispatchStep:
    options: list            # [(ScheduleAssignment, processing_time)]
    features: np.ndarray     # [n, DISPATCH_DIM]
    operations: list         # local operation-node index per option
    cells: list              # cell-node index per option
    prior: np.ndarray        # [n]
    atc: int                 # ATC(beta=1) choice
    choice: int | None = None
    identifiers: list = field(default_factory=list)


def atc_scores(builder, options, beta=1.0):
    mean_p = sum(p for _, p in options) / len(options)
    scores = []
    for a, p in options:
        order = builder.orders[builder.operations[a.operation_id].order_id]
        slack = max(0.0, order.due_date - builder.sim.time - p - beta * builder.after[a.operation_id])
        priority = order.weight / max(p, 1e-9) * math.exp(-slack / max(builder.cfg.atc_k * mean_p, 1e-9))
        scores.append((priority, -order.due_date, -a.operation_id))
    return scores


def dispatch_step(builder, workers, robots, prefix, options_cfg, graph):
    """The next dispatch decision for a chosen resource plan, or None when nothing is dispatchable."""
    legal = _schedule_options(builder.env, reconfigure=bool(workers or robots), workers=workers, robots=robots, prefix=prefix)
    if not legal:
        return None
    scores = atc_scores(builder, legal, 1.0)
    atc = max(range(len(legal)), key=lambda i: scores[i])
    opids = {int(oid): i for i, oid in enumerate(graph.nodes["operation"].ids.tolist())}
    ready = {}
    for op in builder.visible.operations:
        if op.status.value == "READY":
            ready[op.stage] = ready.get(op.stage, 0) + 1
    rows, operations, cells, identifiers = [], [], [], []
    for (a, p), score in zip(legal, scores):
        op = builder.operations[a.operation_id]
        order = builder.orders[op.order_id]
        cell = builder.visible.cells[a.cell_id]
        worker = cell.configured_worker is not None
        robot = cell.configured_robot is not None
        rows.append([p, order.due_date - builder.sim.time - p - builder.after[a.operation_id], order.weight,
                     builder.after[a.operation_id], order.due_date - builder.sim.time,
                     float(worker and not robot), float(robot and not worker), float(worker and robot),
                     ready.get(op.stage, 0), math.log(max(score[0], 1e-12)), len(prefix)])
        operations.append(opids.get(a.operation_id, -1))
        cells.append(int(a.cell_id))
        identifiers.append((int(a.operation_id), int(a.cell_id)))
    prior = np.full(len(legal), (1.0 - options_cfg["dispatch_prior"]) / max(1, len(legal) - 1), np.float32)
    prior[atc] = options_cfg["dispatch_prior"] if len(legal) > 1 else 1.0
    return DispatchStep(legal, signed_log(np.asarray(rows, dtype=np.float32)), operations, cells, prior, atc, None, identifiers)
