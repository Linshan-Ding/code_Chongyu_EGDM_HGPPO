"""Discrete events and priority queue used by the simulator."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Any


class EventType(str, Enum):
    ORDER_ARRIVAL = "ORDER_ARRIVAL"
    OPERATION_FINISH = "OPERATION_FINISH"
    RELOCATION_FINISH = "RELOCATION_FINISH"


# Same-time events are processed as one batch before a decision. The priority is
# only a deterministic tie-breaker inside that batch; it does not create extra
# policy decision times.
_EVENT_PRIORITY = {
    EventType.OPERATION_FINISH: 0,
    EventType.RELOCATION_FINISH: 1,
    EventType.ORDER_ARRIVAL: 2,
}


@dataclass(order=True, slots=True)
class Event:
    time: float
    priority: int
    sequence: int
    event_type: EventType = field(compare=False)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter = count()

    def __len__(self) -> int:
        return len(self._heap)

    def clear(self) -> None:
        self._heap.clear()
        self._counter = count()

    def push(self, time: float, event_type: EventType, **payload: Any) -> Event:
        if time < 0:
            raise ValueError("event time must be non-negative")
        event = Event(
            time=float(time),
            priority=_EVENT_PRIORITY[event_type],
            sequence=next(self._counter),
            event_type=event_type,
            payload=dict(payload),
        )
        heapq.heappush(self._heap, event)
        return event

    def peek_time(self) -> float:
        if not self._heap:
            raise IndexError("event queue is empty")
        return float(self._heap[0].time)

    def pop_time_batch(self, tol: float = 1e-10) -> list[Event]:
        if not self._heap:
            return []
        t = self._heap[0].time
        out: list[Event] = []
        while self._heap and abs(self._heap[0].time - t) <= tol:
            out.append(heapq.heappop(self._heap))
        out.sort(key=lambda e: (e.priority, e.sequence))
        return out
