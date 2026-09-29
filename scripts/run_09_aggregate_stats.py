"""Aggregate completed fixed-instance evaluations into one CSV."""

from __future__ import annotations

from pathlib import Path

from _bootstrap import ROOT
from agent.evaluation.stats import aggregate_csvs


def _evaluation_inputs() -> list[Path]:
    # Keep discovery explicit and deterministic. Training/validation logs have
    # different schemas and must never be mixed into paper statistics.
    names = {
        "eval_results.csv", "generalization.csv", "structural_effects.csv",
        "ablation_results.csv", "sensitivity.csv",
    }
    return sorted(
        path for path in (ROOT / "result").rglob("*.csv")
        if path.name in names and path.is_file()
    )


def main() -> None:
    inputs = _evaluation_inputs()
    if not inputs:
        raise FileNotFoundError(
            "no evaluation CSV found under result/; run the fixed-test evaluation scripts first"
        )
    output = aggregate_csvs(inputs, ROOT / "result" / "stats_summary.csv")
    print(f"Statistics summary written: {output} ({len(inputs)} evaluation files)")


if __name__ == "__main__":
    main()
