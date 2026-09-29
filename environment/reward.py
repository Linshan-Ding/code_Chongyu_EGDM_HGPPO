"""Objective-consistent reward utilities.

Important implementation detail
-------------------------------
The paper states that due dates are not decision events, while also requiring the
continuous tardiness integral to sum exactly to -TWT. If a due date lies strictly
inside [t_e, t_{e+1}), using only the set of orders already late at t_e would miss
part of the integral. We therefore integrate the same tardiness rate exactly across
internal due-date crossings *without* creating extra policy decision events.
"""

from __future__ import annotations

from math import isclose

from data.schema import AssemblyInstance
from environment.entities import OrderRuntime


def interval_twt_reward(
    instance: AssemblyInstance,
    orders: list[OrderRuntime],
    t0: float,
    t1: float,
) -> float:
    """Exact negative TWT integral over [t0, t1).

    During the open interval before the next event, any order that has arrived and
    is not yet completed accrues weight after its due date. Completion events occur
    at t1, so an order completing at t1 is unfinished throughout [t0, t1).
    """

    if t1 < t0:
        raise ValueError("t1 must be >= t0")
    if isclose(t1, t0, abs_tol=1e-15):
        return 0.0

    penalty = 0.0
    for runtime, data in zip(orders, instance.orders, strict=True):
        if not runtime.arrived or runtime.completed:
            continue
        late_start = max(float(t0), float(data.due_date))
        if t1 > late_start:
            penalty += float(data.weight) * (float(t1) - late_start)
    return -penalty


def final_twt(instance: AssemblyInstance, orders: list[OrderRuntime]) -> float:
    total = 0.0
    for runtime, data in zip(orders, instance.orders, strict=True):
        if runtime.completion_time is None:
            raise ValueError("final_twt requires all orders to be completed")
        total += float(data.weight) * max(
            0.0, float(runtime.completion_time) - float(data.due_date)
        )
    return total


def duration_discount(gamma: float, delta_t: float, tau_0: float) -> float:
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if delta_t < 0:
        raise ValueError("delta_t must be non-negative")
    if tau_0 <= 0:
        raise ValueError("tau_0 must be positive")
    return float(gamma ** (delta_t / tau_0))


def _processing_remaining_time(instance: AssemblyInstance, op, current_time: float) -> float:
    """Actual remaining duration of an operation that is already PROCESSING.

    Paper Eq. (16) uses the remaining required work *from the current state*.
    Re-counting the full processing duration after an operation has already been
    partly executed would overstate ``R_min_j(e)``.  The running operation's
    selected H/R/HR mode is already known at the decision state, so its remaining
    duration is the selected-mode duration minus elapsed processing time.
    """

    from environment.entities import ExecutionMode

    if op.start_time is None or op.mode is None:
        raise ValueError("PROCESSING operation must have start_time and mode")

    if op.mode == ExecutionMode.H:
        if op.worker_id is None:
            raise ValueError("H-mode PROCESSING operation has no worker_id")
        duration = instance.processing_time_h(op.product_type, op.stage, op.worker_id)
    elif op.mode == ExecutionMode.R:
        if op.robot_id is None:
            raise ValueError("R-mode PROCESSING operation has no robot_id")
        duration = instance.processing_time_r(op.product_type, op.stage, op.robot_id)
    elif op.mode == ExecutionMode.HR:
        if op.worker_id is None or op.robot_id is None:
            raise ValueError("HR-mode PROCESSING operation lacks worker/robot id")
        duration = instance.processing_time_hr(
            op.product_type, op.stage, op.worker_id, op.robot_id
        )
    else:
        raise ValueError(f"invalid PROCESSING execution mode: {op.mode}")

    if duration is None:
        raise ValueError("PROCESSING operation's recorded mode is incompatible with instance")
    elapsed = max(0.0, float(current_time) - float(op.start_time))
    return max(0.0, float(duration) - elapsed)


def tardiness_potential(
    instance: AssemblyInstance,
    orders: list[OrderRuntime],
    operations,
    current_time: float,
    eta: float,
) -> float:
    """Paper Eq. (16) potential using only already-arrived orders.

    ``R_min_j(e)`` is the remaining lower-bound work content at the *current*
    decision state.  For operations not yet started, we use the instance-level
    minimum feasible processing time.  For an operation already PROCESSING, its
    selected mode is fixed and non-preemptive, so only its actual remaining
    selected-mode duration is counted.  DONE operations contribute zero.

    This remains state-only and never exposes unreleased orders.  The paper leaves
    numerical ``eta`` unspecified; Phase L1.3b therefore keeps eta under
    validation-only selection before any formal five-seed run.
    """

    eta = float(eta)
    if eta < 0.0:
        raise ValueError("potential eta must be non-negative")
    if eta == 0.0:
        return 0.0

    from environment.entities import OperationStatus

    remaining_by_order = {
        runtime.order_id: 0.0
        for runtime in orders
        if runtime.arrived and not runtime.completed
    }
    for op in operations:
        if op.order_id not in remaining_by_order or op.status == OperationStatus.DONE:
            continue
        if op.status == OperationStatus.PROCESSING:
            remaining = _processing_remaining_time(instance, op, current_time)
        else:
            remaining = instance.minimum_processing_time(op.product_type, op.stage)
        remaining_by_order[op.order_id] += float(remaining)

    potential = 0.0
    for runtime, data in zip(orders, instance.orders, strict=True):
        if not runtime.arrived or runtime.completed:
            continue
        predicted_lateness = max(
            0.0,
            float(current_time)
            + float(remaining_by_order.get(runtime.order_id, 0.0))
            - float(data.due_date),
        )
        potential -= eta * float(data.weight) * predicted_lateness
    return float(potential)
