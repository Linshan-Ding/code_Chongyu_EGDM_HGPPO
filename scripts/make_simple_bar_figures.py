"""Quick bar-chart previews for the pilot-joint report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from make_pilot_figures import OUT, DISPLAY, load_rows, mean_by_scale, paired_ablation  # noqa: E402


def baseline_bar(rows):
    methods = ["Fixed-EDD", "Fixed-ATC", "Periodic-ATC", "RH-MILP", "RH-ALNS", "Opportunity-ATC", "full"]
    summary = mean_by_scale(rows, methods)
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    s = [summary["S"][m][0] for m in methods]
    m = [summary["M"][m][0] for m in methods]
    bars_s = ax.bar(x - width / 2, s, width, label="S", color="#7FA6C9")
    bars_m = ax.bar(x + width / 2, m, width, label="M", color="#D98C83")
    ax.set_xticks(x, [DISPLAY[k] for k in methods], rotation=28, ha="right")
    ax.set_ylabel("Mean TWT (lower is better)")
    ax.set_title("Baseline comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for bars in (bars_s, bars_m):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():,.0f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure_baseline_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def ablation_bar(rows):
    controls = ["mlp", "shared_relation", "periodic_gate", "no_rule_prior"]
    summary = paired_ablation(rows, controls)
    x = np.arange(len(controls))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    s = [float(summary["S"][k]["improvement_pct"]) for k in controls]
    m = [float(summary["M"][k]["improvement_pct"]) for k in controls]
    bars_s = ax.bar(x - width / 2, s, width, label="S", color="#7FA6C9")
    bars_m = ax.bar(x + width / 2, m, width, label="M", color="#D98C83")
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(x, [DISPLAY[k] for k in controls], rotation=18, ha="right")
    ax.set_ylabel("Full-method improvement (%)")
    ax.set_title("Ablation comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for bars in (bars_s, bars_m):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + (0.7 if value >= 0 else -0.7),
                    f"{value:+.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure_ablation_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = load_rows()
    baseline_bar(rows)
    ablation_bar(rows)
    print(f"Wrote simple bar charts to {OUT}")
