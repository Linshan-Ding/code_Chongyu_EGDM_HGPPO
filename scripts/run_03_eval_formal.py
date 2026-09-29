"""Evaluate all available formal methods on the frozen test suite."""

from pathlib import Path

from _bootstrap import ROOT, formal_iterations, run


def _checkpoint(run_dir: Path) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None
    candidate = run_dir / "checkpoints" / "best_model.pt"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"validation-selected checkpoint is missing: {candidate}. "
            "Complete formal validation before evaluation."
        )
    return candidate


formal_dir = ROOT / "result" / "scheme2_runs" / f"egdm_hgppo_scheme2_seed0_n{formal_iterations()}"
egdm_checkpoint = _checkpoint(formal_dir)
methods = ["Fixed-EDD", "Periodic-ATC", "Threshold-EDD", "Bottleneck-Rule", "RH-MILP", "RH-ALNS"]
learned_checkpoints: dict[str, Path] = {}
if egdm_checkpoint is not None:
    methods.append("EGDM-HGPPO")

baseline_root = ROOT / "result" / "baseline_scheme2_runs"
for method in ("MLP-PPO", "GAT-PPO", "HGT-PPO-Flat", "HGT-MAPPO"):
    safe = method.lower().replace("-", "_")
    candidate = _checkpoint(baseline_root / f"{safe}_scheme2_seed0_n{formal_iterations()}")
    if candidate is not None:
        methods.append(method)
        learned_checkpoints[method] = candidate

command = [
    "eval.py", "--methods", *methods,
    "--scales", "S", "M", "L", "--scenarios", "D1", "D2", "D3", "D4", "D5",
    "--output", "result/eval_results.csv", "--run-id", "scheme2_seed0_formal", "--tier", "main",
]
if egdm_checkpoint is not None:
    command.extend(["--checkpoint", str(egdm_checkpoint)])
for method, checkpoint in learned_checkpoints.items():
    command.extend(["--baseline-checkpoint", f"{method}={checkpoint}"])
if (ROOT / "result" / "ref_exact.csv").is_file():
    command.extend(["--references", "result/ref_exact.csv"])

print(f"Evaluating methods: {', '.join(methods)}")
run(*command)
