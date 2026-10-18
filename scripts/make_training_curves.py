"""Plot training curves for EGDM-HGPPO and learned methods that outperform it.

The curves are the per-epoch mean TWT from three independent seeds. Fixed
heuristics are shown only as horizontal final-result reference lines because
they do not have a training process.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SKILL_SCRIPTS = Path(r"C:/Users/28263/.codex/skills/nature-figure/scripts")
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


METHODS = {
    "full": ("EGDM-HGPPO", "#1f4e79"),
    "gat": ("GAT", "#d97706"),
    "dan": ("DAN", "#2f855a"),
}
FIXED_METHODS = ("Periodic-ATC", "RH-ALNS", "Fixed-ATC")


def find_report_root(project: Path) -> Path:
    reports = list(project.glob("result/paper_sm100/*/*/_reports/run01"))
    if not reports:
        raise FileNotFoundError("No paper_sm100 report directory was found")
    return max(reports, key=lambda p: (p / "all-results.csv").stat().st_mtime)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window < 3:
        return values
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    report_root = find_report_root(project)
    suite_root = report_root.parent.parent
    output = report_root / "figures"
    output.mkdir(parents=True, exist_ok=True)

    raw: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for method in METHODS:
        for seed in range(3):
            log_path = suite_root / method / f"seed_{seed}" / "run01" / "log.csv"
            if not log_path.exists():
                raise FileNotFoundError(log_path)
            for row in read_csv(log_path):
                scale = row["instance_id"].split("_")[1]
                raw[(method, scale, int(row["epoch"]))].append(float(row["mean_twt"]))

    aggregate: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    for method in METHODS:
        for scale in ("S", "M"):
            points = []
            epochs = sorted(
                epoch for (m, s, epoch) in raw if m == method and s == scale
            )
            for epoch in epochs:
                values = np.asarray(raw[(method, scale, epoch)], dtype=float)
                points.append((epoch, float(values.mean()), float(values.std(ddof=1))))
            if not points:
                raise RuntimeError(f"No training points for {method}/{scale}")
            aggregate[(method, scale)] = points

    # Keep only fixed methods that are better than EGDM-HGPPO for that scale.
    summary_rows = read_csv(report_root / "summary.csv")
    fixed: dict[tuple[str, str], float] = {}
    for row in summary_rows:
        if row["method"] in FIXED_METHODS and row["split"] == "validation":
            fixed[(row["method"], row["scale"])] = float(row["mean_twt"])
    final_full = {
        row["scale"]: float(row["mean_twt"])
        for row in summary_rows
        if row["method"] == "full"
        and row["kind"] == "latest"
        and row["split"] == "validation"
    }
    fixed_better = {
        scale: [
            (method, value)
            for method in FIXED_METHODS
            if (method, scale) in fixed
            for value in [fixed[(method, scale)]]
            if value < final_full[scale]
        ]
        for scale in ("S", "M")
    }

    source_rows: list[dict[str, object]] = []
    for (method, scale), points in aggregate.items():
        for epoch, mean, std in points:
            source_rows.append(
                {
                    "kind": "training_curve",
                    "method": METHODS[method][0],
                    "scale": scale,
                    "epoch": epoch,
                    "mean_twt": mean,
                    "seed_sd": std,
                    "n_seeds": 3,
                }
            )
    for scale, entries in fixed_better.items():
        for method, value in entries:
            source_rows.append(
                {
                    "kind": "fixed_reference",
                    "method": method,
                    "scale": scale,
                    "epoch": "",
                    "mean_twt": value,
                    "seed_sd": "",
                    "n_seeds": 3,
                }
            )
    source_path = output / "training_curve_source.csv"
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_rows[0].keys())
        writer.writeheader()
        writer.writerows(source_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True)
    for idx, (ax, scale) in enumerate(zip(axes, ("S", "M"))):
        for method, (label, color) in METHODS.items():
            points = aggregate[(method, scale)]
            epochs = np.asarray([p[0] for p in points], dtype=float)
            means = np.asarray([p[1] for p in points], dtype=float)
            spread = np.asarray([p[2] for p in points], dtype=float)
            smooth = moving_average(means)
            ax.fill_between(epochs, means - spread, means + spread, color=color, alpha=0.10)
            ax.plot(epochs, smooth, color=color, linewidth=1.8, label=label)
            ax.plot(epochs, means, color=color, linewidth=0.45, alpha=0.30)

        for method, value in fixed_better[scale]:
            ax.axhline(
                value,
                color="#6b7280",
                linewidth=0.9,
                linestyle=(0, (4, 2)),
                alpha=0.80,
            )

        ax.text(-0.12, 1.04, "ab"[idx], transform=ax.transAxes, fontweight="bold")
        ax.set_title(f"{scale}-scale", fontsize=10, pad=8)
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Mean TWT" if idx == 0 else "")
        ax.set_xlim(1, 200)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.5, alpha=0.55)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Training curves of EGDM-HGPPO and stronger learned baselines",
        fontsize=10.5,
        y=0.99,
    )
    fig.text(
        0.5,
        0.005,
        "Mean ± seed SD (n=3); dashed lines are fixed-method final references, not training curves",
        ha="center",
        fontsize=7,
        color="#4b5563",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fixed_handles = [
        Line2D(
            [0],
            [0],
            color="#6b7280",
            linewidth=0.9,
            linestyle=(0, (4, 2)),
            label=f"{method} (fixed)",
        )
        for method in FIXED_METHODS
    ]
    fig.legend(
        handles + fixed_handles,
        labels + [handle.get_label() for handle in fixed_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.5,
    )
    fig.tight_layout(rect=[0.04, 0.08, 0.98, 0.84])
    fig.canvas.draw()
    stem = output / "figure_training_curves"
    require_matplotlib_panel_alignment(
        fig,
        json_out=f"{stem}.alignment.json",
        overlay_svg=f"{stem}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    print(f"Training curves written: {stem}.png")
    print(f"Source data written: {source_path}")
    print(f"Fixed references better than EGDM-HGPPO: {fixed_better}")


if __name__ == "__main__":
    main()
