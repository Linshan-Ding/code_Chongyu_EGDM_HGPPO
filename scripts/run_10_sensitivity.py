"""Run the fixed relocation/resource-capability sensitivity experiment."""

from __future__ import annotations

import csv
from pathlib import Path

from _bootstrap import ROOT, formal_iterations, run
from data.generator import load_parameter_table
from agent.experiments.scheme2 import load_scheme2_config


RELOCATION_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0)
WORKER_SKILL_DENSITIES = (0.60, 0.75, 0.90)
ROBOT_CAPABILITY_DENSITIES = (0.50, 0.65, 0.80)


def _checkpoint(run_dir: Path) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None
    candidate = run_dir / "checkpoints" / "best_model.pt"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"validation-selected checkpoint is missing: {candidate}. "
            "Complete formal validation before sensitivity evaluation."
        )
    return candidate


def _methods() -> tuple[list[str], list[str]]:
    methods = ["Fixed-EDD", "Periodic-ATC", "Threshold-EDD", "Bottleneck-Rule", "RH-MILP", "RH-ALNS"]
    checkpoint_args: list[str] = []
    formal_dir = ROOT / "result" / "scheme2_runs" / f"egdm_hgppo_scheme2_seed0_n{formal_iterations()}"
    egdm = _checkpoint(formal_dir)
    if egdm is not None:
        methods.append("EGDM-HGPPO")
        checkpoint_args.extend(["--checkpoint", str(egdm)])
    baseline_root = ROOT / "result" / "baseline_scheme2_runs"
    for method in ("MLP-PPO", "GAT-PPO", "HGT-PPO-Flat", "HGT-MAPPO"):
        safe = method.lower().replace("-", "_")
        checkpoint = _checkpoint(baseline_root / f"{safe}_scheme2_seed0_n{formal_iterations()}")
        if checkpoint is not None:
            methods.append(method)
            checkpoint_args.extend(["--baseline-checkpoint", f"{method}={checkpoint}"])
    if "EGDM-HGPPO" not in methods:
        raise FileNotFoundError("formal EGDM-HGPPO checkpoint is required before sensitivity evaluation")
    return methods, checkpoint_args


def _merge(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    for item in inputs:
        with item.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            incoming = list(reader.fieldnames or [])
            if fieldnames is None:
                fieldnames = incoming
            else:
                # Older completed cells may predate diagnostic columns such as
                # ``failure_reason``.  Preserve them and fill missing values;
                # reject only a genuinely empty/incompatible file.
                if not incoming:
                    raise ValueError(f"empty sensitivity CSV schema: {item}")
                for name in incoming:
                    if name not in fieldnames:
                        fieldnames.append(name)
            rows.extend(dict(row) for row in reader)
    if not fieldnames:
        raise RuntimeError("sensitivity evaluation produced no rows")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cell_complete(path: Path, expected_methods: list[str], expected_instances: int) -> bool:
    """Return true only for a complete cell produced with the current method set."""
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    observed_methods = {row.get("method", "") for row in rows}
    observed_instances = {row.get("instance_id", "") for row in rows}
    return (
        observed_methods == set(expected_methods)
        and len(observed_instances) == int(expected_instances)
        and len(rows) == len(expected_methods) * int(expected_instances)
    )


methods, checkpoint_args = _methods()
s2 = load_scheme2_config()
parameter_table = load_parameter_table(s2.instance_parameter_table_path)
expected_instances = sum(
    1 for params in parameter_table.values() if str(params["scale"]) in set(s2.scales)
)
raw_root = ROOT / "result" / "sensitivity" / "raw"
outputs: list[Path] = []
for relocation in RELOCATION_MULTIPLIERS:
    for worker_density in WORKER_SKILL_DENSITIES:
        for robot_density in ROBOT_CAPABILITY_DENSITIES:
            label = (
                f"rm{int(round(relocation * 10)):02d}_"
                f"wsd{int(round(worker_density * 100)):03d}_"
                f"rcd{int(round(robot_density * 100)):03d}"
            )
            suite_root = ROOT / "data" / "instances" / "sensitivity" / label
            output = raw_root / f"{label}.csv"
            if _cell_complete(output, methods, expected_instances=expected_instances):
                print(f"SKIP completed sensitivity cell: {label}")
                outputs.append(output)
                continue
            command = [
                "eval.py", "--prepare", "--methods", *methods,
                "--test-root", str(suite_root), "--prefix", "sensitivity",
                "--scales", "S", "M", "L", "--scenarios", "D1", "D2", "D3", "D4", "D5",
                "--load-ratios", "0.65", "0.80", "0.95", "--due-tightness", "tight", "medium", "loose",
                "--count", "1", "--base-seed", "410000",
                "--relocation-multiplier", str(relocation),
                "--worker-skill-density", str(worker_density),
                "--robot-capability-density", str(robot_density),
                "--output", str(output), "--run-id", f"scheme2_seed0_sensitivity_{label}", "--tier", "sensitivity",
                *checkpoint_args,
            ]
            print(f"Sensitivity cell: {label}; methods={', '.join(methods)}")
            run(*command)
            outputs.append(output)

merged = ROOT / "result" / "sensitivity.csv"
_merge(outputs, merged)
print(f"Sensitivity experiment completed: {len(outputs)} cells, merged CSV: {merged}")
