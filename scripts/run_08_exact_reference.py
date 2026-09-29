"""Produce offline small-instance reference objectives for gap reporting."""

from _bootstrap import run

run(
    "-m", "agent.baselines.run_exact", "--scales", "S",
    "--output", "result/ref_exact.csv", "--minimum-dwell", "0",
)
