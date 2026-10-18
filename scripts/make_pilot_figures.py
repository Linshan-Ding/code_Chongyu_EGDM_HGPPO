"""Render two publication-ready pilot-joint result figures from report CSV data.

The script intentionally uses only repeat=0, feasible final-policy evaluations.
Figure 1 compares the proposed policy with the registered baselines. Figure 2
shows paired full-versus-ablation improvements, where positive values mean that
the proposed policy has lower TWT than the ablation control.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SKILL_SCRIPTS = Path(r"C:/Users/28263/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


REPORT = Path(
    r"D:/newpytorch/advisor_latest_code_Chongyu_EGDM_HGPPO/result/"
    r"paper_sm100/dd3290755a81b120/pilot-joint/_reports/run01"
)
SOURCE = REPORT / "all-results.csv"
OUT = REPORT / "figures"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

DISPLAY = {
    "full": "EGDM-HGPPO",
    "Fixed-EDD": "Fixed-EDD",
    "Fixed-ATC": "Fixed-ATC",
    "Periodic-ATC": "Periodic-ATC",
    "RH-MILP": "RH-MILP",
    "RH-ALNS": "RH-ALNS",
    "Opportunity-ATC": "Opportunity-ATC",
    "mlp": "MLP",
    "shared_relation": "Shared relation",
    "periodic_gate": "Periodic gate",
    "no_rule_prior": "No rule prior",
}


def load_rows() -> list[dict[str, str]]:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if row["repeat"] == "0"
        and row["status"] == "success"
        and row["feasible"].lower() == "true"
    ]
    if not rows:
        raise RuntimeError(f"No usable final-policy rows found in {SOURCE}")
    return rows


def mean_by_scale(rows: list[dict[str, str]], methods: list[str]) -> dict[str, dict[str, tuple[float, int, float]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["method"] in methods and row["scale"] in ("S", "M"):
            values[row["scale"]][row["method"]].append(float(row["twt"]))
    return {
        scale: {
            method: (
                float(np.mean(items)),
                len(items),
                float(np.std(items, ddof=1)) if len(items) > 1 else 0.0,
            )
            for method, items in values[scale].items()
        }
        for scale in ("S", "M")
    }


def paired_ablation(rows: list[dict[str, str]], controls: list[str]) -> dict[str, dict[str, dict[str, float | int]]]:
    indexed = {
        (row["method"], row["seed"], row["instance_id"]): float(row["twt"])
        for row in rows
    }
    out: dict[str, dict[str, dict[str, float | int]]] = {"S": {}, "M": {}}
    for scale in ("S", "M"):
        full_rows = [r for r in rows if r["method"] == "full" and r["scale"] == scale]
        for control in controls:
            changes: list[float] = []
            wins = losses = ties = 0
            for row in full_rows:
                key = (control, row["seed"], row["instance_id"])
                if key not in indexed:
                    continue
                full = float(row["twt"])
                control_value = indexed[key]
                changes.append(100.0 * (control_value - full) / max(control_value, 1.0))
                if full < control_value - 1e-8:
                    wins += 1
                elif full > control_value + 1e-8:
                    losses += 1
                else:
                    ties += 1
            if not changes:
                raise RuntimeError(f"No paired rows for {control} at scale {scale}")
            out[scale][control] = {
                "improvement_pct": float(np.mean(changes)),
                "n": len(changes),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "sd": float(np.std(changes, ddof=1)) if len(changes) > 1 else 0.0,
            }
    return out


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    alignment_json = stem.with_suffix(".alignment.json")
    alignment_svg = stem.with_suffix(".alignment.svg")
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(alignment_json),
        overlay_svg=str(alignment_svg),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def baseline_figure(rows: list[dict[str, str]]) -> None:
    methods = [
        "Fixed-EDD",
        "Fixed-ATC",
        "Periodic-ATC",
        "RH-MILP",
        "RH-ALNS",
        "Opportunity-ATC",
        "full",
    ]
    summary = mean_by_scale(rows, methods)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), sharey=False)
    colors = ["#8FA1B2"] * 6 + ["#1F5A85"]
    for panel, (ax, scale) in enumerate(zip(axes, ("S", "M"))):
        values = [summary[scale][m][0] for m in methods]
        ns = [summary[scale][m][1] for m in methods]
        spreads = [summary[scale][m][2] for m in methods]
        y = np.arange(len(methods))
        for yi, value, spread, color in zip(y, values, spreads, colors):
            ax.errorbar(
                value,
                yi,
                xerr=spread,
                fmt="o",
                markersize=5.5,
                color=color,
                ecolor="#24313A",
                elinewidth=0.8,
                capsize=2.5,
                markeredgecolor="#24313A",
                markeredgewidth=0.45,
                zorder=3,
            )
        ax.set_title(f"{scale} instances")
        ax.text(0.0, 1.02, chr(97 + panel), transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
        ax.set_yticks(y, [f"{DISPLAY[m]} | {value:,.0f} (n={n})" for m, value, n in zip(methods, values, ns)])
        ax.invert_yaxis()
        ax.grid(axis="x", color="#D9DEE3", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        right = max(value + spread for value, spread in zip(values, spreads))
        ax.set_xlim(0, right * 1.12)
        ax.set_xlabel("Mean TWT")
        ax.tick_params(axis="y", labelsize=7, pad=4)
    axes[0].set_ylabel("Method | mean TWT | n")
    fig.suptitle("Proposed method versus baseline algorithms", y=0.995, fontsize=11, fontweight="bold")
    fig.text(0.5, 0.005, "Points show means; horizontal lines show ± SD across repeat=0 final-policy evaluation rows.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0.0, 0.08, 1, 0.94))
    save_figure(fig, OUT / "figure_baseline_comparison")


def ablation_figure(rows: list[dict[str, str]]) -> None:
    controls = ["mlp", "shared_relation", "periodic_gate", "no_rule_prior"]
    summary = paired_ablation(rows, controls)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=False)
    colors = ["#4C78A8", "#F2A65A", "#7AA974", "#C16E70"]
    for panel, (ax, scale) in enumerate(zip(axes, ("S", "M"))):
        values = [float(summary[scale][m]["improvement_pct"]) for m in controls]
        spreads = [float(summary[scale][m]["sd"]) for m in controls]
        y = np.arange(len(controls))
        for yi, value, spread, color in zip(y, values, spreads, colors):
            ax.errorbar(
                value,
                yi,
                xerr=spread,
                fmt="o",
                markersize=6,
                color=color,
                ecolor="#24313A",
                elinewidth=0.8,
                capsize=2.5,
                markeredgecolor="#24313A",
                markeredgewidth=0.45,
                zorder=3,
            )
        ax.axvline(0, color="#24313A", linewidth=0.8)
        ax.set_title(f"{scale} instances")
        ax.text(0.0, 1.02, chr(97 + panel), transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
        ax.set_yticks(y, [
            f"{DISPLAY[m]} | {value:+.1f}% ({summary[scale][m]['wins']}W/{summary[scale][m]['losses']}L)"
            for m, value in zip(controls, values)
        ])
        ax.invert_yaxis()
        ax.grid(axis="x", color="#D9DEE3", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        lim = max(12.0, max(abs(v) + spread for v, spread in zip(values, spreads)) * 1.35)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Full-method improvement (%)")
        ax.tick_params(axis="y", labelsize=7, pad=4)
    axes[0].set_ylabel("Ablation | improvement | W/L")
    fig.suptitle("Ablation analysis of the proposed method", y=1.03, fontsize=11, fontweight="bold")
    fig.text(0.5, 0.005, "Points show paired means; horizontal lines show ± SD. Positive values mean lower TWT for full.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0.0, 0.08, 1, 0.94))
    save_figure(fig, OUT / "figure_ablation_comparison")


def write_source_summary(rows: list[dict[str, str]]) -> None:
    baseline_methods = ["Fixed-EDD", "Fixed-ATC", "Periodic-ATC", "RH-MILP", "RH-ALNS", "Opportunity-ATC", "full"]
    ablation_methods = ["full", "mlp", "shared_relation", "periodic_gate", "no_rule_prior"]
    output = OUT / "figure_source_summary.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "scale", "method", "mean_twt", "n"])
        for figure_name, methods in (("baseline", baseline_methods), ("ablation", ablation_methods)):
            summary = mean_by_scale(rows, methods)
            for scale in ("S", "M"):
                for method in methods:
                    mean, n, _sd = summary[scale][method]
                    writer.writerow([figure_name, scale, method, f"{mean:.8f}", n])
    with (OUT / "figure_notes.txt").open("w", encoding="utf-8") as handle:
        handle.write("Source: report all-results.csv; repeat=0, success=True, feasible=True.\n")
        handle.write("Baseline bars are means over available final-policy seed-instance rows; n is reported on each bar.\n")
        handle.write("Ablation improvement = 100 * (TWT_control - TWT_full) / TWT_control, paired by instance_id and seed.\n")
        handle.write("Positive ablation values mean full has lower TWT; W/L annotations count paired wins/losses.\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    write_source_summary(rows)
    baseline_figure(rows)
    ablation_figure(rows)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
