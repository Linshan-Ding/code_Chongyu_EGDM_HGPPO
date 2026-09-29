"""Offline MILP reference result I/O for optimality-gap evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

REFERENCE_FIELDS = (
    "instance_id", "method", "objective_twt", "dual_bound", "mip_gap",
    "optimal", "feasible", "status", "message", "wall_time_seconds",
    "num_operations", "num_variables", "num_constraints", "dwell_exact",
)


def write_reference_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REFERENCE_FIELDS)
        w.writeheader()
        for row in rows:
            data = {k: getattr(row, k) for k in REFERENCE_FIELDS if k != "method"}
            data["method"] = "CP-SAT/MILP"
            w.writerow(data)
    return path


def load_reference_objectives(path, *, prefer="objective_twt") -> dict[str, float]:
    out = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            value = row.get(prefer, "")
            if value in {"", "None", None}:
                continue
            out[row["instance_id"]] = float(value)
    return out


__all__ = ["REFERENCE_FIELDS", "load_reference_objectives", "write_reference_csv"]
