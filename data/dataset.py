"""Materialize fixed validation/test instances and maintain data/instances/index.csv.

Training instances are intentionally NOT generated here. The future training loop
will call ``InstanceGenerator.sample`` online every rollout/iteration.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from configs.config import load_config
from data.generator import InstanceGenerator
from data.io import load_instance_csv, save_instance_csv


INDEX_FIELDS = [
    "instance_id",
    "relative_path",
    "tier",
    "scale",
    "scenario",
    "generation_seed",
    "target_load_ratio",
    "due_tightness",
    "num_stages",
    "num_cells",
    "num_orders",
    "num_workers",
    "num_robots",
    "num_products",
]

TIER_DEFAULT_SCALE = {
    "small": "S",
    "main": "M",
    "large": "L",
    "xl": "XL",
    "val": "S",
}


def _load_existing_index(index_path: Path) -> dict[str, dict[str, str]]:
    if not index_path.exists():
        return {}
    with index_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {row["instance_id"]: row for row in reader}


def _write_index(index_path: Path, rows: dict[str, dict[str, Any]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (r["tier"], r["scale"], r["scenario"], r["instance_id"]))
    with index_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)


def make_fixed_instances(
    *,
    cfg_paths: list[str | Path],
    root: str | Path = "data/instances",
    tier: str,
    scale: str | None = None,
    scenario: str = "D1",
    count: int,
    base_seed: int = 10_000,
    load_ratio: float | None = None,
    due_tightness: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    if tier not in TIER_DEFAULT_SCALE:
        raise ValueError(f"unknown tier={tier!r}")
    if count <= 0:
        raise ValueError("count must be positive")
    scale = scale or TIER_DEFAULT_SCALE[tier]

    cfg = load_config(cfg_paths)
    generator = InstanceGenerator(cfg)

    root = Path(root)
    tier_dir = root / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.csv"
    index_rows = _load_existing_index(index_path)

    created_or_reused: list[Path] = []
    for i in range(count):
        seed = int(base_seed + i)
        rho_label = "rhoMix" if load_ratio is None else f"rho{int(round(load_ratio * 100)):03d}"
        due_label = "dueMix" if due_tightness is None else due_tightness
        instance_id = f"{scale}_{scenario}_{rho_label}_{due_label}_{i:04d}"
        path = tier_dir / f"{instance_id}.csv"

        if path.exists() and not overwrite:
            instance = load_instance_csv(path)
        else:
            instance = generator.sample(
                scale=scale,
                scenario=scenario,
                seed=seed,
                load_ratio=load_ratio,
                due_tightness=due_tightness,
                instance_id=instance_id,
            )
            save_instance_csv(instance, path)

        index_rows[instance.instance_id] = {
            "instance_id": instance.instance_id,
            "relative_path": path.relative_to(root).as_posix(),
            "tier": tier,
            "scale": instance.scale,
            "scenario": instance.scenario,
            "generation_seed": "" if instance.generation_seed is None else instance.generation_seed,
            "target_load_ratio": instance.target_load_ratio,
            "due_tightness": instance.due_tightness,
            "num_stages": instance.num_stages,
            "num_cells": instance.num_cells,
            "num_orders": instance.num_orders,
            "num_workers": instance.num_workers,
            "num_robots": instance.num_robots,
            "num_products": instance.num_products,
        }
        created_or_reused.append(path)

    _write_index(index_path, index_rows)
    return created_or_reused


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed EGDM-HGPPO validation/test instances."
    )
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
    parser.add_argument("--root", default="data/instances")
    parser.add_argument("--tier", choices=list(TIER_DEFAULT_SCALE), required=True)
    parser.add_argument("--scale", choices=["S", "M", "L", "XL"], default=None)
    parser.add_argument("--scenario", choices=["D1", "D2", "D3", "D4", "D5"], default="D1")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=10_000)
    parser.add_argument("--load-ratio", type=float, default=None)
    parser.add_argument("--due-tightness", choices=["tight", "medium", "loose"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = make_fixed_instances(
        cfg_paths=args.config,
        root=args.root,
        tier=args.tier,
        scale=args.scale,
        scenario=args.scenario,
        count=args.count,
        base_seed=args.base_seed,
        load_ratio=args.load_ratio,
        due_tightness=args.due_tightness,
        overwrite=args.overwrite,
    )
    print(f"Fixed dataset ready: {len(paths)} instances")
    for path in paths[:5]:
        print(path)
    if len(paths) > 5:
        print("...")
    print(f"Index: {Path(args.root) / 'index.csv'}")


if __name__ == "__main__":
    main()
