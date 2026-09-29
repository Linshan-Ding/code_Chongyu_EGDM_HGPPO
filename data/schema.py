"""Data schema for fixed and online-generated EGDM-HGPPO assembly instances.

The data layer is intentionally independent from the simulator and the agent.
It stores only static instance facts. Runtime states (READY/BUSY/RELOCATING,
queues, completion times, etc.) belong to ``environment`` and are not stored here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil, isfinite
from typing import Any


@dataclass(slots=True)
class OrderData:
    order_id: int
    product_type: int
    release_time: float
    due_date: float
    weight: int

    def validate(self, num_product_types: int) -> None:
        if self.order_id < 0:
            raise ValueError("order_id must be non-negative")
        if not 0 <= self.product_type < num_product_types:
            raise ValueError(f"invalid product_type={self.product_type}")
        if self.release_time < 0:
            raise ValueError("release_time must be non-negative")
        if self.due_date <= self.release_time:
            raise ValueError("due_date must be strictly larger than release_time")
        if self.weight not in {1, 2, 4}:
            raise ValueError(f"unsupported paper weight: {self.weight}")


@dataclass(slots=True)
class AssemblyInstance:
    """Complete static instance consumed by the future discrete-event simulator.

    Indices are zero-based in code. Paper symbols remain one-based conceptually.
    ``cell_stage[m]`` is the fixed stage of cell ``m``.
    Product routes are strictly increasing stage index lists.
    """

    instance_id: str
    scale: str
    scenario: str
    generation_seed: int | None

    target_load_ratio: float
    due_tightness: str
    worker_skill_density: float
    robot_capability_density: float
    relocation_multiplier: float

    num_stages: int
    cell_stage: list[int]
    product_routes: list[list[int]]
    base_process_time: list[list[float]]  # [product][stage]

    worker_skill: list[list[int]]  # [worker][stage]
    worker_efficiency: list[list[float]]  # [worker][stage]
    robot_capability: list[list[int]]  # [robot][stage]
    robot_efficiency: list[list[float]]  # [robot][stage]

    hr_compatibility: list[list[list[int]]]  # [worker][robot][stage]
    hr_synergy: list[list[list[float]]]  # [worker][robot][stage]

    worker_relocation_time: list[list[list[float]]]  # [worker][src_cell][dst_cell]
    robot_relocation_time: list[list[list[float]]]  # [robot][src_cell][dst_cell]

    initial_worker_cell: list[int]  # -1 means temporarily unconfigured
    initial_robot_cell: list[int]

    orders: list[OrderData]
    generation_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_cells(self) -> int:
        return len(self.cell_stage)

    @property
    def num_products(self) -> int:
        return len(self.product_routes)

    @property
    def num_workers(self) -> int:
        return len(self.worker_skill)

    @property
    def num_robots(self) -> int:
        return len(self.robot_capability)

    @property
    def num_orders(self) -> int:
        return len(self.orders)

    def processing_time_h(self, product: int, stage: int, worker: int) -> float | None:
        if self.worker_skill[worker][stage] != 1:
            return None
        return self.base_process_time[product][stage] / self.worker_efficiency[worker][stage]

    def processing_time_r(self, product: int, stage: int, robot: int) -> float | None:
        if self.robot_capability[robot][stage] != 1:
            return None
        return self.base_process_time[product][stage] / self.robot_efficiency[robot][stage]

    def processing_time_hr(
        self,
        product: int,
        stage: int,
        worker: int,
        robot: int,
    ) -> float | None:
        if self.hr_compatibility[worker][robot][stage] != 1:
            return None
        p_h = self.processing_time_h(product, stage, worker)
        p_r = self.processing_time_r(product, stage, robot)
        if p_h is None or p_r is None:
            raise ValueError("HR compatibility cannot exist without H and R compatibility")
        return self.hr_synergy[worker][robot][stage] * min(p_h, p_r)

    def minimum_processing_time(self, product: int, stage: int) -> float:
        candidates: list[float] = []
        for h in range(self.num_workers):
            p = self.processing_time_h(product, stage, h)
            if p is not None:
                candidates.append(p)
        for r in range(self.num_robots):
            p = self.processing_time_r(product, stage, r)
            if p is not None:
                candidates.append(p)
        for h in range(self.num_workers):
            for r in range(self.num_robots):
                p = self.processing_time_hr(product, stage, h, r)
                if p is not None:
                    candidates.append(p)
        if not candidates:
            raise ValueError(f"product={product}, stage={stage} has no executable mode")
        return min(candidates)

    def minimum_route_work_content(self, product: int) -> float:
        return sum(self.minimum_processing_time(product, s) for s in self.product_routes[product])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if self.scale not in {"S", "M", "L", "XL"}:
            raise ValueError(f"invalid scale={self.scale!r}")
        if self.scenario not in {"D1", "D2", "D3", "D4", "D5"}:
            raise ValueError(f"invalid scenario={self.scenario!r}")
        if self.num_stages <= 0:
            raise ValueError("num_stages must be positive")
        if not 0 < self.target_load_ratio:
            raise ValueError("target_load_ratio must be positive")
        if self.due_tightness not in {"tight", "medium", "loose"}:
            raise ValueError("due_tightness must be tight/medium/loose")
        if self.relocation_multiplier < 0:
            raise ValueError("relocation_multiplier must be non-negative")

        # Fixed cell-stage structure: 1-3 cells per stage.
        if not self.cell_stage:
            raise ValueError("at least one cell is required")
        counts = [0] * self.num_stages
        for s in self.cell_stage:
            if not 0 <= s < self.num_stages:
                raise ValueError(f"cell has invalid stage index {s}")
            counts[s] += 1
        if any(c < 1 or c > 3 for c in counts):
            raise ValueError(f"each stage must have 1-3 cells, got counts={counts}")
        required_multi = ceil(0.30 * self.num_stages - 1e-12)
        if sum(c >= 2 for c in counts) < required_multi:
            raise ValueError("fewer than 30% of stages have at least two parallel cells")

        # Product routes and base processing times.
        if not self.product_routes:
            raise ValueError("at least one product type is required")
        if len(self.base_process_time) != self.num_products:
            raise ValueError("base_process_time product dimension mismatch")
        min_route_len = ceil(0.70 * self.num_stages - 1e-12)
        route_stage_union: set[int] = set()
        for v, route in enumerate(self.product_routes):
            if route != sorted(set(route)):
                raise ValueError(f"product {v} route must be sorted and unique: {route}")
            if not min_route_len <= len(route) <= self.num_stages:
                raise ValueError(f"product {v} route coverage outside 70-100%: {route}")
            if any(s < 0 or s >= self.num_stages for s in route):
                raise ValueError(f"product {v} contains invalid stage")
            route_stage_union.update(route)
            row = self.base_process_time[v]
            if len(row) != self.num_stages or any((not isfinite(x) or x <= 0) for x in row):
                raise ValueError(f"invalid base_process_time row for product {v}")
        if route_stage_union != set(range(self.num_stages)):
            raise ValueError("generated product family must cover all stages at least once")

        # Human and robot static matrices.
        if not self.worker_skill:
            raise ValueError("at least one worker is required")
        if not self.robot_capability:
            raise ValueError("at least one robot is required")
        if len(self.worker_efficiency) != self.num_workers:
            raise ValueError("worker_efficiency worker dimension mismatch")
        if len(self.robot_efficiency) != self.num_robots:
            raise ValueError("robot_efficiency robot dimension mismatch")
        for h in range(self.num_workers):
            if len(self.worker_skill[h]) != self.num_stages:
                raise ValueError("worker_skill stage dimension mismatch")
            if len(self.worker_efficiency[h]) != self.num_stages:
                raise ValueError("worker_efficiency stage dimension mismatch")
            if any(x not in {0, 1} for x in self.worker_skill[h]):
                raise ValueError("worker_skill must be binary")
            if any((not isfinite(x) or x <= 0) for x in self.worker_efficiency[h]):
                raise ValueError("worker_efficiency must be positive finite")
        for r in range(self.num_robots):
            if len(self.robot_capability[r]) != self.num_stages:
                raise ValueError("robot_capability stage dimension mismatch")
            if len(self.robot_efficiency[r]) != self.num_stages:
                raise ValueError("robot_efficiency stage dimension mismatch")
            if any(x not in {0, 1} for x in self.robot_capability[r]):
                raise ValueError("robot_capability must be binary")
            if any((not isfinite(x) or x <= 0) for x in self.robot_efficiency[r]):
                raise ValueError("robot_efficiency must be positive finite")

        # The paper requires worker-skill generation to keep every stage covered.
        for s in range(self.num_stages):
            if not any(self.worker_skill[h][s] for h in range(self.num_workers)):
                raise ValueError(f"stage {s} has no compatible worker")

        # HR compatibility can only exist on jointly feasible H/R pairs.
        if len(self.hr_compatibility) != self.num_workers or len(self.hr_synergy) != self.num_workers:
            raise ValueError("HR worker dimension mismatch")
        for h in range(self.num_workers):
            if len(self.hr_compatibility[h]) != self.num_robots:
                raise ValueError("HR robot dimension mismatch")
            if len(self.hr_synergy[h]) != self.num_robots:
                raise ValueError("HR synergy robot dimension mismatch")
            for r in range(self.num_robots):
                if len(self.hr_compatibility[h][r]) != self.num_stages:
                    raise ValueError("HR stage dimension mismatch")
                if len(self.hr_synergy[h][r]) != self.num_stages:
                    raise ValueError("HR synergy stage dimension mismatch")
                for s in range(self.num_stages):
                    compat = self.hr_compatibility[h][r][s]
                    if compat not in {0, 1}:
                        raise ValueError("HR compatibility must be binary")
                    if compat and not (self.worker_skill[h][s] and self.robot_capability[r][s]):
                        raise ValueError("HR compatibility violates H/R compatibility")
                    synergy = self.hr_synergy[h][r][s]
                    if not isfinite(synergy) or synergy <= 0:
                        raise ValueError("HR synergy must be positive finite")

        # Every routed product-stage must be executable by at least one mode.
        for v, route in enumerate(self.product_routes):
            for s in route:
                _ = self.minimum_processing_time(v, s)

        # Relocation matrices: diagonal 0, off-diagonal positive when multiplier > 0.
        self._validate_relocation(self.worker_relocation_time, self.num_workers, "worker")
        self._validate_relocation(self.robot_relocation_time, self.num_robots, "robot")

        # Initial configuration: same-type cell capacity <= 1 and compatibility respected.
        self._validate_initial_configuration(
            self.initial_worker_cell,
            self.num_workers,
            self.worker_skill,
            "worker",
        )
        self._validate_initial_configuration(
            self.initial_robot_cell,
            self.num_robots,
            self.robot_capability,
            "robot",
        )
        # If H and R share an initially configured cell, the resulting HR mode
        # must itself be legal.  Earlier phases validated each resource type
        # independently; Phase K2 exposed that cross-type co-location can create
        # an initially unusable cell and later deadlock online baselines.
        worker_by_cell = {m: h for h, m in enumerate(self.initial_worker_cell) if m >= 0}
        robot_by_cell = {m: r for r, m in enumerate(self.initial_robot_cell) if m >= 0}
        for m in set(worker_by_cell).intersection(robot_by_cell):
            h = worker_by_cell[m]; r = robot_by_cell[m]; s = self.cell_stage[m]
            if self.hr_compatibility[h][r][s] != 1:
                raise ValueError(
                    f"initial worker {h} / robot {r} co-location at cell {m} "
                    "is HR-incompatible"
                )

        # Orders.
        if not self.orders:
            raise ValueError("at least one order is required")
        seen_ids: set[int] = set()
        last_release = -1.0
        for order in self.orders:
            order.validate(self.num_products)
            if order.order_id in seen_ids:
                raise ValueError("duplicate order_id")
            seen_ids.add(order.order_id)
            if order.release_time + 1e-12 < last_release:
                raise ValueError("orders must be sorted by release_time")
            last_release = order.release_time

    def _validate_relocation(self, matrix: list[list[list[float]]], n_resources: int, label: str) -> None:
        if len(matrix) != n_resources:
            raise ValueError(f"{label} relocation resource dimension mismatch")
        for resource_matrix in matrix:
            if len(resource_matrix) != self.num_cells:
                raise ValueError(f"{label} relocation source-cell dimension mismatch")
            for src, row in enumerate(resource_matrix):
                if len(row) != self.num_cells:
                    raise ValueError(f"{label} relocation target-cell dimension mismatch")
                for dst, value in enumerate(row):
                    if not isfinite(value) or value < 0:
                        raise ValueError(f"invalid {label} relocation time")
                    if src == dst and abs(value) > 1e-12:
                        raise ValueError(f"{label} relocation diagonal must be zero")
                    if src != dst and self.relocation_multiplier > 0 and value <= 0:
                        raise ValueError(f"{label} off-diagonal relocation must be positive")

    def _validate_initial_configuration(
        self,
        assignments: list[int],
        n_resources: int,
        compatibility: list[list[int]],
        label: str,
    ) -> None:
        if len(assignments) != n_resources:
            raise ValueError(f"initial_{label}_cell length mismatch")
        occupied: set[int] = set()
        for resource_id, cell in enumerate(assignments):
            if cell == -1:
                continue
            if not 0 <= cell < self.num_cells:
                raise ValueError(f"initial {label} cell index out of range")
            if cell in occupied:
                raise ValueError(f"two {label}s assigned to the same cell")
            occupied.add(cell)
            stage = self.cell_stage[cell]
            if compatibility[resource_id][stage] != 1:
                raise ValueError(f"initial {label} is incompatible with assigned cell")
