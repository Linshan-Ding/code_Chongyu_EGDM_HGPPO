"""CSV materialization for one complete EGDM-HGPPO instance per file.

The file format is a normalized long table. It deliberately avoids pickle/PT
binaries so fixed validation/test instances can be inspected and published.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from data.schema import AssemblyInstance, OrderData


FIELDNAMES = [
    "record_type",
    "key",
    "i",
    "j",
    "k",
    "l",
    "value",
    "value2",
    "value3",
    "text",
]


def _row(
    record_type: str,
    *,
    key: str = "",
    i: int | str = "",
    j: int | str = "",
    k: int | str = "",
    l: int | str = "",
    value: float | int | str = "",
    value2: float | int | str = "",
    value3: float | int | str = "",
    text: str = "",
) -> dict[str, Any]:
    return {
        "record_type": record_type,
        "key": key,
        "i": i,
        "j": j,
        "k": k,
        "l": l,
        "value": value,
        "value2": value2,
        "value3": value3,
        "text": text,
    }


def save_instance_csv(instance: AssemblyInstance, path: str | Path) -> Path:
    instance.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    meta = {
        "instance_id": instance.instance_id,
        "scale": instance.scale,
        "scenario": instance.scenario,
        "generation_seed": instance.generation_seed,
        "target_load_ratio": instance.target_load_ratio,
        "due_tightness": instance.due_tightness,
        "worker_skill_density": instance.worker_skill_density,
        "robot_capability_density": instance.robot_capability_density,
        "relocation_multiplier": instance.relocation_multiplier,
        "num_stages": instance.num_stages,
        "generation_meta": instance.generation_meta,
    }
    for key, val in meta.items():
        rows.append(_row("META", key=key, text=json.dumps(val, ensure_ascii=False, separators=(",", ":"))))

    for cell_id, stage_id in enumerate(instance.cell_stage):
        rows.append(_row("CELL", i=cell_id, j=stage_id))

    for product_id, route in enumerate(instance.product_routes):
        for pos, stage_id in enumerate(route):
            rows.append(_row("ROUTE", i=product_id, j=pos, k=stage_id))

    for v, row in enumerate(instance.base_process_time):
        for s, value in enumerate(row):
            rows.append(_row("BASE_PROCESS", i=v, j=s, value=repr(float(value))))

    for h, row in enumerate(instance.worker_skill):
        for s, value in enumerate(row):
            rows.append(_row("WORKER_SKILL", i=h, j=s, value=int(value)))
    for h, row in enumerate(instance.worker_efficiency):
        for s, value in enumerate(row):
            rows.append(_row("WORKER_EFF", i=h, j=s, value=repr(float(value))))

    for r, row in enumerate(instance.robot_capability):
        for s, value in enumerate(row):
            rows.append(_row("ROBOT_CAP", i=r, j=s, value=int(value)))
    for r, row in enumerate(instance.robot_efficiency):
        for s, value in enumerate(row):
            rows.append(_row("ROBOT_EFF", i=r, j=s, value=repr(float(value))))

    for h in range(instance.num_workers):
        for r in range(instance.num_robots):
            for s in range(instance.num_stages):
                rows.append(
                    _row(
                        "HR_COMPAT",
                        i=h,
                        j=r,
                        k=s,
                        value=int(instance.hr_compatibility[h][r][s]),
                    )
                )
                rows.append(
                    _row(
                        "HR_SYNERGY",
                        i=h,
                        j=r,
                        k=s,
                        value=repr(float(instance.hr_synergy[h][r][s])),
                    )
                )

    for h in range(instance.num_workers):
        for src in range(instance.num_cells):
            for dst in range(instance.num_cells):
                rows.append(
                    _row(
                        "WORKER_RELOC",
                        i=h,
                        j=src,
                        k=dst,
                        value=repr(float(instance.worker_relocation_time[h][src][dst])),
                    )
                )
    for r in range(instance.num_robots):
        for src in range(instance.num_cells):
            for dst in range(instance.num_cells):
                rows.append(
                    _row(
                        "ROBOT_RELOC",
                        i=r,
                        j=src,
                        k=dst,
                        value=repr(float(instance.robot_relocation_time[r][src][dst])),
                    )
                )

    for h, cell in enumerate(instance.initial_worker_cell):
        rows.append(_row("INITIAL_WORKER", i=h, j=cell))
    for r, cell in enumerate(instance.initial_robot_cell):
        rows.append(_row("INITIAL_ROBOT", i=r, j=cell))

    for order in instance.orders:
        rows.append(
            _row(
                "ORDER",
                i=order.order_id,
                j=order.product_type,
                k=order.weight,
                value=repr(float(order.release_time)),
                value2=repr(float(order.due_date)),
            )
        )

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"unexpected instance CSV schema: {reader.fieldnames}")
        return list(reader)


def _int(cell: str) -> int:
    return int(cell)


def _float(cell: str) -> float:
    return float(cell)


def load_instance_csv(path: str | Path) -> AssemblyInstance:
    path = Path(path)
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"empty instance file: {path}")

    meta: dict[str, Any] = {}
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        record_type = row["record_type"]
        if record_type == "META":
            meta[row["key"]] = json.loads(row["text"])
        else:
            groups.setdefault(record_type, []).append(row)

    required_meta = {
        "instance_id",
        "scale",
        "scenario",
        "generation_seed",
        "target_load_ratio",
        "due_tightness",
        "worker_skill_density",
        "robot_capability_density",
        "relocation_multiplier",
        "num_stages",
        "generation_meta",
    }
    missing = required_meta.difference(meta)
    if missing:
        raise ValueError(f"instance CSV missing META fields: {sorted(missing)}")

    num_stages = int(meta["num_stages"])

    cell_rows = groups.get("CELL", [])
    num_cells = 1 + max(_int(r["i"]) for r in cell_rows)
    cell_stage = [0] * num_cells
    for r in cell_rows:
        cell_stage[_int(r["i"])] = _int(r["j"])

    route_rows = groups.get("ROUTE", [])
    num_products = 1 + max(_int(r["i"]) for r in route_rows)
    product_routes: list[list[tuple[int, int]]] = [[] for _ in range(num_products)]
    for r in route_rows:
        product_routes[_int(r["i"])].append((_int(r["j"]), _int(r["k"])))
    routes = [[stage for _, stage in sorted(items)] for items in product_routes]

    base_process = [[0.0] * num_stages for _ in range(num_products)]
    for r in groups.get("BASE_PROCESS", []):
        base_process[_int(r["i"])][_int(r["j"])] = _float(r["value"])

    worker_skill_rows = groups.get("WORKER_SKILL", [])
    num_workers = 1 + max(_int(r["i"]) for r in worker_skill_rows)
    worker_skill = [[0] * num_stages for _ in range(num_workers)]
    worker_eff = [[0.0] * num_stages for _ in range(num_workers)]
    for r in worker_skill_rows:
        worker_skill[_int(r["i"])][_int(r["j"])] = _int(r["value"])
    for r in groups.get("WORKER_EFF", []):
        worker_eff[_int(r["i"])][_int(r["j"])] = _float(r["value"])

    robot_cap_rows = groups.get("ROBOT_CAP", [])
    num_robots = 1 + max(_int(r["i"]) for r in robot_cap_rows)
    robot_cap = [[0] * num_stages for _ in range(num_robots)]
    robot_eff = [[0.0] * num_stages for _ in range(num_robots)]
    for r in robot_cap_rows:
        robot_cap[_int(r["i"])][_int(r["j"])] = _int(r["value"])
    for r in groups.get("ROBOT_EFF", []):
        robot_eff[_int(r["i"])][_int(r["j"])] = _float(r["value"])

    hr_compat = [
        [[0] * num_stages for _ in range(num_robots)]
        for _ in range(num_workers)
    ]
    hr_synergy = [
        [[0.0] * num_stages for _ in range(num_robots)]
        for _ in range(num_workers)
    ]
    for r in groups.get("HR_COMPAT", []):
        hr_compat[_int(r["i"])][_int(r["j"])][_int(r["k"])] = _int(r["value"])
    for r in groups.get("HR_SYNERGY", []):
        hr_synergy[_int(r["i"])][_int(r["j"])][_int(r["k"])] = _float(r["value"])

    worker_reloc = [
        [[0.0] * num_cells for _ in range(num_cells)]
        for _ in range(num_workers)
    ]
    robot_reloc = [
        [[0.0] * num_cells for _ in range(num_cells)]
        for _ in range(num_robots)
    ]
    for r in groups.get("WORKER_RELOC", []):
        worker_reloc[_int(r["i"])][_int(r["j"])][_int(r["k"])] = _float(r["value"])
    for r in groups.get("ROBOT_RELOC", []):
        robot_reloc[_int(r["i"])][_int(r["j"])][_int(r["k"])] = _float(r["value"])

    initial_worker = [-1] * num_workers
    initial_robot = [-1] * num_robots
    for r in groups.get("INITIAL_WORKER", []):
        initial_worker[_int(r["i"])] = _int(r["j"])
    for r in groups.get("INITIAL_ROBOT", []):
        initial_robot[_int(r["i"])] = _int(r["j"])

    orders = [
        OrderData(
            order_id=_int(r["i"]),
            product_type=_int(r["j"]),
            release_time=_float(r["value"]),
            due_date=_float(r["value2"]),
            weight=_int(r["k"]),
        )
        for r in groups.get("ORDER", [])
    ]
    orders.sort(key=lambda x: x.order_id)

    instance = AssemblyInstance(
        instance_id=str(meta["instance_id"]),
        scale=str(meta["scale"]),
        scenario=str(meta["scenario"]),
        generation_seed=None if meta["generation_seed"] is None else int(meta["generation_seed"]),
        target_load_ratio=float(meta["target_load_ratio"]),
        due_tightness=str(meta["due_tightness"]),
        worker_skill_density=float(meta["worker_skill_density"]),
        robot_capability_density=float(meta["robot_capability_density"]),
        relocation_multiplier=float(meta["relocation_multiplier"]),
        num_stages=num_stages,
        cell_stage=cell_stage,
        product_routes=routes,
        base_process_time=base_process,
        worker_skill=worker_skill,
        worker_efficiency=worker_eff,
        robot_capability=robot_cap,
        robot_efficiency=robot_eff,
        hr_compatibility=hr_compat,
        hr_synergy=hr_synergy,
        worker_relocation_time=worker_reloc,
        robot_relocation_time=robot_reloc,
        initial_worker_cell=initial_worker,
        initial_robot_cell=initial_robot,
        orders=orders,
        generation_meta=meta["generation_meta"],
    )
    instance.validate()
    return instance
