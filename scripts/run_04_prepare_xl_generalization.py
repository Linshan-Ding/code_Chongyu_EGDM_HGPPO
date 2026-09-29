from _bootstrap import run

run(
    "eval.py", "--prepare", "--prepare-only", "--methods", "Fixed-EDD", "--count", "1",
    "--scales", "XL", "--scenarios", "D1", "D2", "D3", "D4", "D5",
    "--load-ratios", "0.65", "0.80", "0.95",
    "--due-tightness", "tight", "medium", "loose",
)
