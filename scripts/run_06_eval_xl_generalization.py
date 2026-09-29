"""Evaluate the frozen formal Scheme-2 checkpoint on XL cases."""

from _bootstrap import ROOT, formal_iterations, run


run_dir = ROOT / "result" / "scheme2_runs" / f"egdm_hgppo_scheme2_seed0_n{formal_iterations()}"
checkpoint = run_dir / "checkpoints" / "best_model.pt"
if not checkpoint.is_file():
    raise FileNotFoundError(
        f"validation-selected checkpoint is missing: {checkpoint}. "
        "Complete formal validation before XL evaluation."
    )

run(
    "eval.py", "--methods", "EGDM-HGPPO", "--checkpoint", str(checkpoint),
    "--scales", "XL", "--scenarios", "D1", "D2", "D3", "D4", "D5",
    "--output", "result/generalization.csv", "--run-id", "scheme2_seed0_xl",
    "--tier", "large",
)
