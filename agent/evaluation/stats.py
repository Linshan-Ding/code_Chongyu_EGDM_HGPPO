"""Deterministic aggregation for fixed-instance evaluation CSVs.

The formal Scheme-2 run uses one seed and one instance per design cell, so this
module deliberately reports descriptive summaries rather than inventing
confidence intervals or significance tests. It accepts evaluation files from
multiple methods/runs and keeps the grouping fields needed to trace every
summary back to the fixed design metadata.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Mapping, Sequence


SUMMARY_FIELDS = (
    "source_file",
    "tier",
    "method",
    "run_id",
    "scale",
    "parameter_case_id",
    "scenario",
    "rho",
    "due_tightness",
    "n",
    "feasible_rate",
    "mean_twt",
    "std_twt",
    "mean_optimality_gap",
    "std_optimality_gap",
    "mean_tardy_ratio",
    "mean_flow_time",
    "mean_wall_time_seconds",
    "mean_decision_time_ms",
)

_GROUP_FIELDS = (
    "tier",
    "method",
    "scale",
    "parameter_case_id",
    "scenario",
    "rho",
    "due_tightness",
)
_REQUIRED_EVAL_FIELDS = {"instance_id", "method", "tier", "twt", "feasible"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = _REQUIRED_EVAL_FIELDS.difference(fields)
        if missing:
            raise ValueError(
                f"{path} is not an evaluation CSV; missing columns: {sorted(missing)}"
            )
        return [dict(row) for row in reader]


def _float(row: Mapping[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or str(value).strip() in {"", "None", "nan", "NaN"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _key_value(row: Mapping[str, str], key: str) -> str:
    value = row.get(key, "")
    if key == "rho":
        parsed = _float(row, key)
        return "" if parsed is None else f"{parsed:.8g}"
    return "" if value is None else str(value)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else float(mean(values))


def _std(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else float(pstdev(values))


def aggregate_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    source_file: str = "",
) -> list[dict[str, object]]:
    """Aggregate evaluation rows by method/run and fixed design factors.

    Failed rows are retained in ``feasible_rate`` while objective statistics use
    only finite values. This prevents a partially completed evaluation from
    silently looking better than it is.
    """
    groups: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for row in rows:
        missing = _REQUIRED_EVAL_FIELDS.difference(row)
        if missing:
            raise ValueError(f"evaluation row is missing columns: {sorted(missing)}")
        key = tuple(_key_value(row, field) for field in _GROUP_FIELDS)
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        grouped = dict(zip(_GROUP_FIELDS, key, strict=True))
        run_ids = sorted({str(row.get("run_id", "")) for row in group if row.get("run_id")})
        source_files = sorted({str(row.get("__source_file", source_file)) for row in group})
        twt = [value for value in (_float(row, "twt") for row in group) if value is not None]
        gaps = [
            value for value in (_float(row, "optimality_gap") for row in group)
            if value is not None
        ]
        feasible = [
            str(row.get("feasible", "")).strip().lower() in {"1", "true", "yes"}
            for row in group
        ]
        item: dict[str, object] = {
            "source_file": ";".join(source_files),
            **grouped,
            "run_id": run_ids[0] if len(run_ids) == 1 else ("multiple" if run_ids else ""),
            "n": len(group),
            "feasible_rate": float(sum(feasible) / len(feasible)) if feasible else None,
            "mean_twt": _mean(twt),
            "std_twt": _std(twt),
            "mean_optimality_gap": _mean(gaps),
            "std_optimality_gap": _std(gaps),
        }
        for field in (
            "tardy_ratio",
            "mean_flow_time",
            "wall_time_seconds",
            "mean_decision_time_ms",
        ):
            values = [value for value in (_float(row, field) for row in group) if value is not None]
            item[
                {
                    "tardy_ratio": "mean_tardy_ratio",
                    "mean_flow_time": "mean_flow_time",
                    "wall_time_seconds": "mean_wall_time_seconds",
                    "mean_decision_time_ms": "mean_decision_time_ms",
                }[field]
            ] = _mean(values)
        summaries.append({field: item.get(field) for field in SUMMARY_FIELDS})
    return summaries


def aggregate_csvs(
    input_paths: Sequence[str | Path],
    output_path: str | Path | None = None,
) -> list[dict[str, object]] | Path:
    """Aggregate compatible evaluation CSVs and optionally write a summary.

    Non-evaluation files fail with an explicit schema error so a training or
    validation log cannot silently enter paper statistics.
    """
    rows: list[dict[str, str]] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"evaluation CSV does not exist: {path}")
        try:
            source_label = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            source_label = path.as_posix()
        for row in _read_rows(path):
            row["__source_file"] = source_label
            rows.append(row)

    summaries = aggregate_rows(rows)
    if output_path is None:
        return summaries
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    return output


__all__ = ["SUMMARY_FIELDS", "aggregate_rows", "aggregate_csvs"]
