"""Instance generator for EGDM-HGPPO.

Training uses this module online. Fixed validation/test datasets call the same
generator and then materialize each instance through ``data.io``.

Paper-specified distributions come from ``configs/instance.yaml``. Quantities
that the paper leaves qualitative (e.g. the exact D3 switch location) are kept
under ``instance.implementation_choices`` instead of being hidden in code.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from configs.config import ProjectConfig, load_config
from data.schema import AssemblyInstance, OrderData


def _inclusive_int(rng: np.random.Generator, bounds: list[int]) -> int:
    lo, hi = int(bounds[0]), int(bounds[1])
    return int(rng.integers(lo, hi + 1))


def _uniform(rng: np.random.Generator, bounds: list[float], size=None):
    return rng.uniform(float(bounds[0]), float(bounds[1]), size=size)


def _choice_float(rng: np.random.Generator, values: list[float]) -> float:
    return float(values[int(rng.integers(0, len(values)))])


def load_parameter_table(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate a Scheme-2 structural scale parameter table.

    The canonical paper symbols are ``S, M, J, H, R, V``: stages, cells,
    orders, workers, robots and product types.  Historical tables may use
    ``A`` for stages; it is accepted only as a read-time compatibility alias
    and is normalized to ``S`` in the returned mapping.
    """
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rows = payload.get("instances", payload)
    if not isinstance(rows, dict) or not rows:
        raise ValueError("instance parameter table must contain a non-empty 'instances' mapping")
    required = {"S", "M", "R", "J", "H", "V", "scale"}
    out: dict[str, dict[str, Any]] = {}
    for case_id, raw in rows.items():
        if not isinstance(raw, dict):
            raise ValueError(f"parameter-table case {case_id!r} must be a mapping")
        item = dict(raw)
        # ``A`` was used by an earlier implementation as an informal alias
        # for stage count.  Never expose it as the canonical schema.
        if "S" not in item and "A" in item:
            item["S"] = item["A"]
        if "S" in item and "A" in item and int(item["S"]) != int(item["A"]):
            raise ValueError(f"parameter-table case {case_id!r} has conflicting S/A stage counts")
        missing = required.difference(item)
        if missing:
            raise ValueError(f"parameter-table case {case_id!r} missing {sorted(missing)}")
        item["parameter_case_id"] = str(case_id)
        item["M"] = int(item["M"])
        item["S"] = int(item["S"])
        item["R"] = int(item["R"])
        item["J"] = int(item["J"])
        item["H"] = int(item["H"])
        item["V"] = int(item["V"])
        item["scale"] = str(item["scale"])
        item.pop("A", None)
        if item["M"] <= 0 or item["S"] <= 0 or item["R"] <= 0 or item["J"] <= 0 or item["H"] <= 0 or item["V"] <= 0:
            raise ValueError(f"parameter-table case {case_id!r} dimensions must be positive")
        if item["scale"] not in {"S", "M", "L", "XL"}:
            raise ValueError(f"parameter-table case {case_id!r} has invalid scale")
        out[str(case_id)] = item
    return out


def _dominant_probabilities(n: int, dominant: int, share: float) -> np.ndarray:
    if n == 1:
        return np.array([1.0], dtype=float)
    share = float(share)
    if not 0.0 < share < 1.0:
        raise ValueError("dominant product share must be in (0,1)")
    probs = np.full(n, (1.0 - share) / (n - 1), dtype=float)
    probs[dominant % n] = share
    return probs


class InstanceGenerator:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg

    def sample(
        self,
        *,
        scale: str = "S",
        scenario: str = "D1",
        seed: int | None = None,
        load_ratio: float | None = None,
        due_tightness: str | None = None,
        worker_skill_density: float | None = None,
        robot_capability_density: float | None = None,
        relocation_multiplier: float = 1.0,
        instance_id: str | None = None,
        instance_parameters: dict[str, Any] | None = None,
    ) -> AssemblyInstance:
        if scale not in {"S", "M", "L", "XL"}:
            raise ValueError(f"unknown scale {scale!r}")
        if scenario not in {"D1", "D2", "D3", "D4", "D5"}:
            raise ValueError(f"unknown scenario {scenario!r}")

        rng = np.random.default_rng(seed)
        inst_cfg = self.cfg.instance
        profile = inst_cfg.scales[scale]
        choices = inst_cfg.implementation_choices

        # Scheme-2 experiment protocol:
        # If an explicit parameter combination is provided from the instance table,
        # use it as the main scale-defining factor. Otherwise keep the original
        # scale-based random generation for backward compatibility.
        if instance_parameters is not None:
            # Scheme-2 table semantics are explicit.  Legacy aliases remain
            # accepted for old diagnostic files, but are no longer the primary
            # representation of scale.
            if "S" in instance_parameters and "A" in instance_parameters:
                if int(instance_parameters["S"]) != int(instance_parameters["A"]):
                    raise ValueError("instance_parameters has conflicting S/A stage counts")
            num_stages = int(
                instance_parameters.get(
                    "S", instance_parameters.get("A", instance_parameters.get("stages", profile.num_stages))
                )
            )
            num_workers = int(instance_parameters.get("H", instance_parameters.get("workers", 0)))
            num_robots = int(instance_parameters.get("R", instance_parameters.get("robots", 0)))
            num_products = int(instance_parameters.get("V", instance_parameters.get("product_types", 0)))
            num_orders = int(instance_parameters.get("J", instance_parameters.get("orders", 0)))

            if num_workers <= 0:
                num_workers = _inclusive_int(rng, profile.workers)
            if num_robots <= 0:
                num_robots = _inclusive_int(rng, profile.robots)
            if num_products <= 0:
                num_products = _inclusive_int(rng, profile.product_types)
            if num_orders <= 0:
                num_orders = _inclusive_int(rng, profile.orders)
        else:
            num_stages = int(profile.num_stages)
            num_workers = _inclusive_int(rng, profile.workers)
            num_robots = _inclusive_int(rng, profile.robots)
            num_products = _inclusive_int(rng, profile.product_types)
            num_orders = _inclusive_int(rng, profile.orders)

        if instance_parameters is not None and "M" in instance_parameters:
            total_cells = int(instance_parameters["M"])
            total_cell_bounds = (total_cells, total_cells)
        elif instance_parameters is not None and "cells" in instance_parameters:
            cells_cfg = instance_parameters["cells"]

            if isinstance(cells_cfg, (list, tuple)):
                total_cell_bounds = (
                    int(cells_cfg[0]),
                    int(cells_cfg[1]),
                )
            else:
                total_cells = int(cells_cfg)
                total_cell_bounds = (
                    total_cells,
                    total_cells,
                )
        else:
            total_cell_bounds = profile.total_cells

        cell_stage = self._sample_cells(
            rng,
            total_cell_bounds,
            num_stages,
        )
        num_cells = len(cell_stage)

        product_routes = self._sample_product_routes(rng, num_products, num_stages)
        base_process = _uniform(
            rng,
            inst_cfg.base_process_time_minutes,
            size=(num_products, num_stages),
        )

        worker_skill_density = (
            float(worker_skill_density)
            if worker_skill_density is not None
            else _choice_float(rng, inst_cfg.worker_skill_density_choices)
        )
        robot_capability_density = (
            float(robot_capability_density)
            if robot_capability_density is not None
            else _choice_float(rng, inst_cfg.robot_capability_density_choices)
        )

        worker_skill = (rng.random((num_workers, num_stages)) < worker_skill_density).astype(int)
        robot_cap = (rng.random((num_robots, num_stages)) < robot_capability_density).astype(int)

        # Paper Table 6 says worker skill generation should keep every stage covered.
        for s in range(num_stages):
            if worker_skill[:, s].sum() == 0:
                worker_skill[int(rng.integers(0, num_workers)), s] = 1
        # Avoid a completely unusable resource.
        for h in range(num_workers):
            if worker_skill[h].sum() == 0:
                worker_skill[h, int(rng.integers(0, num_stages))] = 1
        for r in range(num_robots):
            if robot_cap[r].sum() == 0:
                robot_cap[r, int(rng.integers(0, num_stages))] = 1

        # Engineering feasibility repair: reserve one distinct compatible resource per
        # stage for the initial load-balancing configuration. This does not expose future
        # orders and is recorded in generation_meta.
        anchor_resources = self._ensure_unique_stage_anchor_resources(
            rng, worker_skill, robot_cap, num_stages
        )

        worker_eff = _uniform(
            rng,
            inst_cfg.worker_efficiency,
            size=(num_workers, num_stages),
        )
        robot_eff = _uniform(
            rng,
            inst_cfg.robot_efficiency,
            size=(num_robots, num_stages),
        )

        hr_rate = float(_uniform(rng, inst_cfg.hr_compatibility_rate))
        hr_compat = np.zeros((num_workers, num_robots, num_stages), dtype=int)
        candidate = (
            worker_skill[:, None, :].astype(bool)
            & robot_cap[None, :, :].astype(bool)
        )
        draws = rng.random(candidate.shape) < hr_rate
        hr_compat[candidate & draws] = 1
        hr_synergy = _uniform(
            rng,
            inst_cfg.hr_synergy_factor,
            size=(num_workers, num_robots, num_stages),
        )

        relocation_multiplier = float(relocation_multiplier)
        if relocation_multiplier < 0:
            raise ValueError("relocation_multiplier must be non-negative")
        worker_reloc = self._sample_relocation_matrix(
            rng,
            num_resources=num_workers,
            cell_stage=cell_stage,
            multiplier=relocation_multiplier,
        )
        robot_reloc = self._sample_relocation_matrix(
            rng,
            num_resources=num_robots,
            cell_stage=cell_stage,
            multiplier=relocation_multiplier,
        )

        initial_worker_cell, initial_robot_cell = self._initial_configuration(
            rng=rng,
            cell_stage=cell_stage,
            product_routes=product_routes,
            base_process=base_process,
            worker_skill=worker_skill,
            robot_cap=robot_cap,
            hr_compat=hr_compat,
            anchor_resources=anchor_resources,
        )

        load_ratio = (
            float(load_ratio)
            if load_ratio is not None
            else _choice_float(rng, inst_cfg.load_ratio_choices)
        )
        if due_tightness is None:
            due_keys = ["tight", "medium", "loose"]
            due_tightness = due_keys[int(rng.integers(0, len(due_keys)))]
        if due_tightness not in {"tight", "medium", "loose"}:
            raise ValueError("due_tightness must be tight/medium/loose")

        min_work = self._minimum_product_workloads(
            base_process=base_process,
            product_routes=product_routes,
            worker_skill=worker_skill,
            worker_eff=worker_eff,
            robot_cap=robot_cap,
            robot_eff=robot_eff,
            hr_compat=hr_compat,
            hr_synergy=hr_synergy,
        )
        effective_servers = min(num_cells, num_workers + num_robots)
        mean_work = float(np.mean(min_work))
        base_arrival_rate = load_ratio * effective_servers / mean_work
        if base_arrival_rate <= 0:
            raise RuntimeError("arrival-rate calibration produced a non-positive rate")

        release_times, product_types, urgent_mask, scenario_meta = self._sample_dynamic_orders(
            rng=rng,
            num_orders=num_orders,
            num_products=num_products,
            scenario=scenario,
            base_arrival_rate=base_arrival_rate,
        )

        weight_values = np.asarray(inst_cfg.order_weight.values, dtype=int)
        weight_probs = np.asarray(inst_cfg.order_weight.probabilities, dtype=float)
        weights = rng.choice(weight_values, size=num_orders, p=weight_probs).astype(int)
        if scenario == "D5":
            weights[urgent_mask] = 4

        due_range = getattr(inst_cfg.due_date_factor, due_tightness)
        kappas = _uniform(rng, due_range, size=num_orders)
        due_dates = np.array(
            [
                release_times[j] + kappas[j] * min_work[int(product_types[j])]
                for j in range(num_orders)
            ],
            dtype=float,
        )

        # Stable sort is important if D5 inserts several orders at nearly the same time.
        order_perm = np.argsort(release_times, kind="stable")
        orders: list[OrderData] = []
        for new_id, old_idx in enumerate(order_perm.tolist()):
            orders.append(
                OrderData(
                    order_id=new_id,
                    product_type=int(product_types[old_idx]),
                    release_time=float(release_times[old_idx]),
                    due_date=float(due_dates[old_idx]),
                    weight=int(weights[old_idx]),
                )
            )

        if instance_id is None:
            seed_label = "online" if seed is None else str(seed)
            instance_id = f"{scale}_{scenario}_{seed_label}"

        generation_meta: dict[str, Any] = {
            "paper_vs_implementation": {
                "paper_specified": [
                    "scale ranges",
                    "70-100% route coverage",
                    "processing/efficiency/synergy distributions",
                    "skill/capability density choices",
                    "D1-D5 qualitative scenario definitions",
                    "load-ratio choices",
                    "weight distribution",
                    "due-date-factor ranges",
                    "relocation-time ranges",
                    "load-balance initial configuration with random perturbation",
                ],
                "implementation_choices": {
                    "arrival_rate_calibration": str(choices.arrival_rate_calibration),
                    "same_stage_cells_are_adjacent": bool(choices.same_stage_cells_are_adjacent),
                    "initial_configuration_perturbation_rate": float(
                        choices.initial_configuration_perturbation_rate
                    ),
                    "scenario_timing_basis": str(choices.scenario_timing_basis),
                    "base_product_mix": str(choices.base_product_mix),
                },
            },
            "base_arrival_rate_orders_per_minute": float(base_arrival_rate),
            "effective_parallel_servers_for_load_calibration": int(effective_servers),
            "mean_minimum_route_work_minutes": mean_work,
            "actual_hr_compatibility_rate_parameter": hr_rate,
            "actual_worker_skill_density": float(worker_skill.mean()),
            "actual_robot_capability_density": float(robot_cap.mean()),
            "instance_parameter_table": (
                self._canonical_parameter_metadata(instance_parameters)
                if instance_parameters is not None else None
            ),
            "parameter_case_id": (
                None if instance_parameters is None else instance_parameters.get("parameter_case_id")
            ),
            "dimensions": {
                "S": int(num_stages),
                "M": int(num_cells),
                "J": int(num_orders),
                "H": int(num_workers),
                "R": int(num_robots),
                "V": int(num_products),
            },
            "scenario_parameters": scenario_meta,
            "route_stage_coverage": list(inst_cfg.route_stage_coverage),
            "min_multi_cell_stage_ratio": float(inst_cfg.min_multi_cell_stage_ratio),
        }

        instance = AssemblyInstance(
            instance_id=instance_id,
            scale=scale,
            scenario=scenario,
            generation_seed=seed,
            target_load_ratio=load_ratio,
            due_tightness=due_tightness,
            worker_skill_density=worker_skill_density,
            robot_capability_density=robot_capability_density,
            relocation_multiplier=relocation_multiplier,
            num_stages=num_stages,
            cell_stage=[int(x) for x in cell_stage],
            product_routes=[[int(x) for x in route] for route in product_routes],
            base_process_time=base_process.tolist(),
            worker_skill=worker_skill.tolist(),
            worker_efficiency=worker_eff.tolist(),
            robot_capability=robot_cap.tolist(),
            robot_efficiency=robot_eff.tolist(),
            hr_compatibility=hr_compat.tolist(),
            hr_synergy=hr_synergy.tolist(),
            worker_relocation_time=worker_reloc.tolist(),
            robot_relocation_time=robot_reloc.tolist(),
            initial_worker_cell=[int(x) for x in initial_worker_cell],
            initial_robot_cell=[int(x) for x in initial_robot_cell],
            orders=orders,
            generation_meta=generation_meta,
        )
        instance.validate()
        return instance

    @staticmethod
    def _canonical_parameter_metadata(parameters: dict[str, Any]) -> dict[str, Any]:
        """Return table metadata with the paper's canonical symbols only."""
        out = dict(parameters)
        if "S" not in out and "A" in out:
            out["S"] = out["A"]
        out.pop("A", None)
        return out

    def _sample_cells(
        self,
        rng: np.random.Generator,
        total_cell_bounds: list[int],
        num_stages: int,
    ) -> list[int]:
        max_per_stage = int(self.cfg.instance.parallel_cells_per_stage[1])
        min_multi_ratio = float(self.cfg.instance.min_multi_cell_stage_ratio)
        required_multi = ceil(min_multi_ratio * num_stages - 1e-12)

        paper_low, paper_high = int(total_cell_bounds[0]), int(total_cell_bounds[1])
        # The intersection of Table 5 and the >=30% parallel-stage requirement.
        feasible_low = max(paper_low, num_stages + required_multi)
        feasible_high = min(paper_high, num_stages * max_per_stage)
        if feasible_low > feasible_high:
            raise ValueError(
                "No cell count satisfies both the paper total-cell range and the "
                ">=30% multi-cell-stage requirement."
            )
        total_cells = int(rng.integers(feasible_low, feasible_high + 1))

        counts = np.ones(num_stages, dtype=int)
        chosen = rng.choice(num_stages, size=required_multi, replace=False)
        counts[chosen] += 1
        remaining = total_cells - int(counts.sum())
        while remaining > 0:
            candidates = np.flatnonzero(counts < max_per_stage)
            s = int(rng.choice(candidates))
            counts[s] += 1
            remaining -= 1

        cell_stage: list[int] = []
        for s, count in enumerate(counts.tolist()):
            cell_stage.extend([s] * int(count))
        return cell_stage

    def _sample_product_routes(
        self,
        rng: np.random.Generator,
        num_products: int,
        num_stages: int,
    ) -> list[list[int]]:
        coverage = self.cfg.instance.route_stage_coverage
        min_len = ceil(float(coverage[0]) * num_stages - 1e-12)
        max_len = min(num_stages, int(np.floor(float(coverage[1]) * num_stages + 1e-12)))
        routes: list[list[int]] = []
        for _ in range(num_products):
            length = int(rng.integers(min_len, max_len + 1))
            route = sorted(rng.choice(num_stages, size=length, replace=False).astype(int).tolist())
            routes.append(route)

        # Make the product family cover all stages; adding a missing stage never violates
        # the upper coverage bound because max_len == num_stages in the paper config.
        covered = {s for route in routes for s in route}
        for missing in sorted(set(range(num_stages)) - covered):
            candidates = [v for v, route in enumerate(routes) if len(route) < max_len]
            if not candidates:
                raise RuntimeError("cannot repair product-route stage coverage")
            v = int(rng.choice(candidates))
            routes[v] = sorted(routes[v] + [missing])
        return routes

    def _ensure_unique_stage_anchor_resources(
        self,
        rng: np.random.Generator,
        worker_skill: np.ndarray,
        robot_cap: np.ndarray,
        num_stages: int,
    ) -> list[tuple[str, int]]:
        num_workers = worker_skill.shape[0]
        num_robots = robot_cap.shape[0]
        tokens: list[tuple[str, int]] = [
            *(('H', h) for h in range(num_workers)),
            *(('R', r) for r in range(num_robots)),
        ]
        if len(tokens) < num_stages:
            raise ValueError("total H+R resources must be at least the number of stages")
        rng.shuffle(tokens)
        anchors = tokens[:num_stages]
        for stage, (kind, idx) in enumerate(anchors):
            if kind == 'H':
                worker_skill[idx, stage] = 1
            else:
                robot_cap[idx, stage] = 1
        return anchors

    def _sample_relocation_matrix(
        self,
        rng: np.random.Generator,
        *,
        num_resources: int,
        cell_stage: list[int],
        multiplier: float,
    ) -> np.ndarray:
        m = len(cell_stage)
        out = np.zeros((num_resources, m, m), dtype=float)
        adjacent = self.cfg.instance.relocation_time_minutes.adjacent_cells
        cross = self.cfg.instance.relocation_time_minutes.cross_stage
        same_stage_is_adjacent = bool(
            self.cfg.instance.implementation_choices.same_stage_cells_are_adjacent
        )
        for q in range(num_resources):
            for src in range(m):
                for dst in range(m):
                    if src == dst:
                        continue
                    same_stage = cell_stage[src] == cell_stage[dst]
                    bounds = adjacent if (same_stage and same_stage_is_adjacent) else cross
                    out[q, src, dst] = float(_uniform(rng, bounds)) * multiplier
        return out

    def _minimum_product_workloads(
        self,
        *,
        base_process: np.ndarray,
        product_routes: list[list[int]],
        worker_skill: np.ndarray,
        worker_eff: np.ndarray,
        robot_cap: np.ndarray,
        robot_eff: np.ndarray,
        hr_compat: np.ndarray,
        hr_synergy: np.ndarray,
    ) -> np.ndarray:
        num_products = base_process.shape[0]
        workloads = np.zeros(num_products, dtype=float)
        for v in range(num_products):
            total = 0.0
            for s in product_routes[v]:
                candidates: list[float] = []
                for h in range(worker_skill.shape[0]):
                    if worker_skill[h, s]:
                        candidates.append(float(base_process[v, s] / worker_eff[h, s]))
                for r in range(robot_cap.shape[0]):
                    if robot_cap[r, s]:
                        candidates.append(float(base_process[v, s] / robot_eff[r, s]))
                for h in range(worker_skill.shape[0]):
                    for r in range(robot_cap.shape[0]):
                        if hr_compat[h, r, s]:
                            p_h = float(base_process[v, s] / worker_eff[h, s])
                            p_r = float(base_process[v, s] / robot_eff[r, s])
                            candidates.append(float(hr_synergy[h, r, s] * min(p_h, p_r)))
                if not candidates:
                    raise RuntimeError(f"product {v} stage {s} is not executable")
                total += min(candidates)
            workloads[v] = total
        return workloads

    def _sample_dynamic_orders(
        self,
        *,
        rng: np.random.Generator,
        num_orders: int,
        num_products: int,
        scenario: str,
        base_arrival_rate: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        c = self.cfg.instance.implementation_choices
        dominant_share = float(c.dominant_product_share)
        base_mix = str(c.base_product_mix)
        if base_mix != "uniform":
            raise ValueError("Phase C currently supports base_product_mix=uniform only")
        urgent_fraction = float(c.d5_urgent_batch_fraction)
        urgent_center = float(c.d5_urgent_batch_center_ratio)
        urgent_rate_multiplier = float(c.d5_urgent_arrival_rate_multiplier)

        release = np.zeros(num_orders, dtype=float)
        products = np.zeros(num_orders, dtype=int)
        urgent = np.zeros(num_orders, dtype=bool)

        d2_start = float(c.d2_burst_start_ratio)
        d2_end = float(c.d2_burst_end_ratio)
        d3_switch = float(c.d3_mix_switch_ratio)
        d3_sequence = [int(x) for x in c.d3_dominant_product_sequence]
        d4_changes = [float(x) for x in c.d4_change_ratios]
        d4_rates = [float(x) for x in c.d4_arrival_rate_multipliers]
        d4_sequence = [int(x) for x in c.d4_dominant_product_sequence]
        if len(d3_sequence) != 2:
            raise ValueError("D3 implementation choices require two dominant products")
        if len(d4_changes) != 2 or len(d4_rates) != 3 or len(d4_sequence) != 3:
            raise ValueError("D4 implementation choices require 2 change ratios and 3 rate/mix segments")
        d2_multiplier = float(
            rng.uniform(float(c.d2_burst_multiplier[0]), float(c.d2_burst_multiplier[1]))
        )

        urgent_count = max(1, int(ceil(urgent_fraction * num_orders)))
        urgent_center_idx = int(round(urgent_center * max(0, num_orders - 1)))
        urgent_start = max(0, urgent_center_idx - urgent_count // 2)
        urgent_end = min(num_orders, urgent_start + urgent_count)
        urgent_start = max(0, urgent_end - urgent_count)
        if scenario == "D5":
            urgent[urgent_start:urgent_end] = True

        scenario_meta: dict[str, Any] = {
            "timing_basis": str(c.scenario_timing_basis),
            "base_product_mix": base_mix,
            "d2_sampled_burst_multiplier": d2_multiplier if scenario == "D2" else None,
            "d2_burst_window": [d2_start, d2_end],
            "d3_mix_switch_ratio": d3_switch,
            "d3_dominant_product_sequence": d3_sequence,
            "d4_change_ratios": d4_changes,
            "d4_arrival_rate_multipliers": d4_rates,
            "d4_dominant_product_sequence": d4_sequence,
            "d5_urgent_index_window": [urgent_start, urgent_end],
            "d5_urgent_arrival_rate_multiplier": urgent_rate_multiplier,
            "d5_urgent_dominant_product": int(c.d5_urgent_dominant_product),
            "dominant_product_share": dominant_share,
        }

        for j in range(num_orders):
            frac = 0.0 if num_orders <= 1 else j / (num_orders - 1)
            rate_multiplier = 1.0
            if scenario == "D2" and d2_start <= frac <= d2_end:
                rate_multiplier = d2_multiplier
            elif scenario == "D4":
                if frac < d4_changes[0]:
                    rate_multiplier = d4_rates[0]
                elif frac < d4_changes[1]:
                    rate_multiplier = d4_rates[1]
                else:
                    rate_multiplier = d4_rates[2]
            elif scenario == "D5" and urgent[j]:
                rate_multiplier = urgent_rate_multiplier

            if j > 0:
                release[j] = release[j - 1] + float(
                    rng.exponential(1.0 / (base_arrival_rate * rate_multiplier))
                )

            if scenario in {"D1", "D2"}:
                probs = np.full(num_products, 1.0 / num_products)
            elif scenario == "D3":
                dominant = d3_sequence[0] if frac < d3_switch else d3_sequence[1]
                dominant = min(dominant, num_products - 1)
                probs = _dominant_probabilities(num_products, dominant, dominant_share)
            elif scenario == "D4":
                if frac < d4_changes[0]:
                    dominant = min(d4_sequence[0], num_products - 1)
                elif frac < d4_changes[1]:
                    dominant = min(d4_sequence[1], num_products - 1)
                else:
                    dominant = min(d4_sequence[2], num_products - 1)
                probs = _dominant_probabilities(num_products, dominant, dominant_share)
            else:  # D5
                if urgent[j]:
                    dominant = min(int(c.d5_urgent_dominant_product), num_products - 1)
                    probs = _dominant_probabilities(num_products, dominant, dominant_share)
                else:
                    probs = np.full(num_products, 1.0 / num_products)
            products[j] = int(rng.choice(num_products, p=probs))

        return release, products, urgent, scenario_meta

    def _initial_configuration(
        self,
        *,
        rng: np.random.Generator,
        cell_stage: list[int],
        product_routes: list[list[int]],
        base_process: np.ndarray,
        worker_skill: np.ndarray,
        robot_cap: np.ndarray,
        hr_compat: np.ndarray,
        anchor_resources: list[tuple[str, int]],
    ) -> tuple[list[int], list[int]]:
        num_workers = worker_skill.shape[0]
        num_robots = robot_cap.shape[0]
        num_stages = worker_skill.shape[1]
        worker_cell = [-1] * num_workers
        robot_cell = [-1] * num_robots
        occupied_h: set[int] = set()
        occupied_r: set[int] = set()

        cells_by_stage = {
            s: [m for m, sm in enumerate(cell_stage) if sm == s]
            for s in range(num_stages)
        }

        # Expected-load score uses product-family information, not realized future orders.
        stage_scores = np.zeros(num_stages, dtype=float)
        for s in range(num_stages):
            for v, route in enumerate(product_routes):
                if s in route:
                    stage_scores[s] += float(base_process[v, s])
        stage_scores /= max(1, len(product_routes))

        # One distinct anchor resource per stage.
        anchor_token_set = set(anchor_resources)
        for stage, (kind, idx) in enumerate(anchor_resources):
            if kind == "H":
                cell = cells_by_stage[stage][0]
                worker_cell[idx] = cell
                occupied_h.add(cell)
            else:
                cell = cells_by_stage[stage][0]
                robot_cell[idx] = cell
                occupied_r.add(cell)

        def cross_legal(kind: str, idx: int, cell: int) -> bool:
            stage = cell_stage[cell]
            if kind == "H":
                robot = next((r for r, m in enumerate(robot_cell) if m == cell), None)
                return robot is None or bool(hr_compat[idx, robot, stage])
            worker = next((h for h, m in enumerate(worker_cell) if m == cell), None)
            return worker is None or bool(hr_compat[worker, idx, stage])

        def best_cell_for_resource(kind: str, idx: int) -> int:
            compat = worker_skill[idx] if kind == "H" else robot_cap[idx]
            occupied = occupied_h if kind == "H" else occupied_r
            stages = [s for s in range(num_stages) if compat[s]]
            stages.sort(key=lambda s: (-stage_scores[s], s))
            for s in stages:
                free = [
                    m for m in cells_by_stage[s]
                    if m not in occupied and cross_legal(kind, idx, m)
                ]
                if free:
                    return int(rng.choice(free))
            return -1

        for h in range(num_workers):
            if ("H", h) in anchor_token_set:
                continue
            cell = best_cell_for_resource("H", h)
            worker_cell[h] = cell
            if cell >= 0:
                occupied_h.add(cell)
        for r in range(num_robots):
            if ("R", r) in anchor_token_set:
                continue
            cell = best_cell_for_resource("R", r)
            robot_cell[r] = cell
            if cell >= 0:
                occupied_r.add(cell)

        # Paper says "load-balancing heuristic + random perturbation" but not its
        # magnitude. Only non-anchor resources are perturbed so stage anchor coverage
        # remains intact. The rate is config-visible.
        perturb_rate = float(
            self.cfg.instance.implementation_choices.initial_configuration_perturbation_rate
        )
        for kind, assignments, compat_matrix, occupied in [
            ("H", worker_cell, worker_skill, occupied_h),
            ("R", robot_cell, robot_cap, occupied_r),
        ]:
            for idx, current in enumerate(list(assignments)):
                if (kind, idx) in anchor_token_set or current < 0 or rng.random() >= perturb_rate:
                    continue
                candidates = [
                    m
                    for m, s in enumerate(cell_stage)
                    if m not in occupied and compat_matrix[idx, s] and cross_legal(kind, idx, m)
                ]
                if not candidates:
                    continue
                new_cell = int(rng.choice(candidates))
                occupied.remove(current)
                occupied.add(new_cell)
                assignments[idx] = new_cell

        return worker_cell, robot_cell


def _summary(instance: AssemblyInstance) -> dict[str, Any]:
    cells_per_stage = [instance.cell_stage.count(s) for s in range(instance.num_stages)]
    return {
        "instance_id": instance.instance_id,
        "scale": instance.scale,
        "scenario": instance.scenario,
        "stages": instance.num_stages,
        "cells": instance.num_cells,
        "cells_per_stage": cells_per_stage,
        "orders": instance.num_orders,
        "workers": instance.num_workers,
        "robots": instance.num_robots,
        "product_types": instance.num_products,
        "target_load_ratio": instance.target_load_ratio,
        "due_tightness": instance.due_tightness,
        "first_three_orders": [order.__dict__ if hasattr(order, '__dict__') else {
            "order_id": order.order_id,
            "product_type": order.product_type,
            "release_time": order.release_time,
            "due_date": order.due_date,
            "weight": order.weight,
        } for order in instance.orders[:3]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one EGDM-HGPPO assembly instance.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=[
            "configs/instance.yaml",
            "configs/env.yaml",
            "configs/algo.yaml",
            "configs/curriculum.yaml",
        ],
    )
    parser.add_argument("--scale", choices=["S", "M", "L", "XL"], default="S")
    parser.add_argument("--scenario", choices=["D1", "D2", "D3", "D4", "D5"], default="D1")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--load-ratio", type=float, default=None)
    parser.add_argument("--due-tightness", choices=["tight", "medium", "loose"], default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    instance = InstanceGenerator(cfg).sample(
        scale=args.scale,
        scenario=args.scenario,
        seed=args.seed,
        load_ratio=args.load_ratio,
        due_tightness=args.due_tightness,
    )
    print(json.dumps(_summary(instance), ensure_ascii=False, indent=2))
    print("Instance validation: PASSED")


if __name__ == "__main__":
    main()
