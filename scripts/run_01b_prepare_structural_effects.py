"""Materialize the controlled S/M/R/J factor-analysis test suite."""

from pathlib import Path

import yaml

from _bootstrap import ROOT
from configs.config import load_config
from data.io import load_instance_csv
from agent.evaluation.test_suite import ensure_fixed_test_suite


CFG_PATHS = [
    "configs/instance.yaml",
    "configs/env.yaml",
    "configs/algo.yaml",
    "configs/curriculum.yaml",
]
DESIGN_PATH = ROOT / "configs" / "structural_effect_design.yaml"
OUTPUT_ROOT = ROOT / "data" / "instances" / "test_structural_effects_v2"


def main() -> None:
    payload = yaml.safe_load(DESIGN_PATH.read_text(encoding="utf-8")) or {}
    conditions = dict(payload.get("fixed_conditions") or {})
    cases = dict(payload.get("instances") or {})
    if len(cases) != 9:
        raise RuntimeError(f"expected 9 controlled design cells, found {len(cases)}")

    cfg = load_config([ROOT / path for path in CFG_PATHS])
    records = ensure_fixed_test_suite(
        cfg,
        root=OUTPUT_ROOT,
        base_seed=410000,
        scales=(str(conditions["scale"]),),
        scenarios=(str(conditions["scenario"]),),
        load_ratios=(float(conditions["load_ratio"]),),
        due_tightness=(str(conditions["due_tightness"]),),
        instances_per_combination=1,
        prefix="struct",
        instance_parameter_table_path=DESIGN_PATH,
        one_per_parameter_case=True,
        relocation_multiplier=float(conditions["relocation_multiplier"]),
        worker_skill_density=float(conditions["worker_skill_density"]),
        robot_capability_density=float(conditions["robot_capability_density"]),
    )
    if len(records) != len(cases):
        raise RuntimeError(f"expected {len(cases)} records, generated {len(records)}")

    # Ensure every requested design cell has explicit metadata in its instance
    # and no existing file was silently reused with a different structural tuple.
    expected = {
        str(case_id): tuple(int(case[key]) for key in ("S", "M", "R", "J", "H", "V"))
        for case_id, case in cases.items()
    }
    observed = {}
    for record in records:
        instance = load_instance_csv(record.path)
        observed[str(record.parameter_case_id)] = (
            int(record.num_stages), int(record.num_cells), int(record.num_robots),
            int(record.num_orders),
            int(instance.num_workers), int(instance.num_products),
        )
    if observed != expected:
        raise RuntimeError(f"structural metadata mismatch: expected={expected}, observed={observed}")
    print(f"Controlled structural-effects suite ready: {len(records)} instances")
    print(f"Index: {OUTPUT_ROOT / 'index.csv'}")

if __name__ == "__main__":
    main()
