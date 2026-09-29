"""Evaluate available methods on the controlled S/M/R/J test suite."""

from _bootstrap import ROOT, formal_iterations, run
from agent.baselines.learned_variants import LEARNED_BASELINE_METHODS


formal_dir = ROOT / "result" / "scheme2_runs" / f"egdm_hgppo_scheme2_seed0_n{formal_iterations()}"
egdm_checkpoint = formal_dir / "checkpoints" / "best_model.pt"
if not egdm_checkpoint.is_file():
    raise SystemExit(f"missing validation-selected checkpoint: {egdm_checkpoint}")

methods = [
    "Fixed-EDD", "Periodic-ATC", "Threshold-EDD", "Bottleneck-Rule",
    "RH-MILP", "RH-ALNS", "EGDM-HGPPO",
]
baseline_args: list[str] = []
baseline_root = ROOT / "result" / "baseline_scheme2_runs"
for method in LEARNED_BASELINE_METHODS:
    safe = method.lower().replace("-", "_")
    checkpoint = baseline_root / f"{safe}_scheme2_seed0_n{formal_iterations()}" / "checkpoints" / "best_model.pt"
    if checkpoint.is_file():
        methods.append(method)
        baseline_args.extend(["--baseline-checkpoint", f"{method}={checkpoint}"])
command = [
    "eval.py",
    "--test-root", "data/instances/test_structural_effects_v2",
    "--methods", *methods,
    "--checkpoint", str(egdm_checkpoint),
    "--output", "result/structural_effects.csv",
    "--run-id", "scheme2_seed0_structural_effects",
    "--tier", "structural_effects",
    *baseline_args,
]
run(*command)
