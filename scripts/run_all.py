"""Run the complete Scheme-2 experiment workflow in dependency order."""

import subprocess

from _bootstrap import run

run("scripts/run_00_hardware_probe.py")
run("scripts/run_00_preflight.py")
run("scripts/run_00_smoke.py")
run("scripts/run_01_prepare_data.py")
run("scripts/run_01b_prepare_structural_effects.py")
run("scripts/run_02_train_formal.py")
run("scripts/run_07_train_learned_baselines.py")
run("scripts/run_08_exact_reference.py")
run("scripts/run_03_eval_formal.py")
run("scripts/run_04_prepare_xl_generalization.py")
run("scripts/run_06_eval_xl_generalization.py")
run("scripts/run_10_sensitivity.py")
run("scripts/run_11_eval_structural_effects.py")
try:
    run("scripts/run_12_train_ablations.py")
except subprocess.CalledProcessError as exc:
    # The trainer returns nonzero when a variant failed; its manifest is still
    # authoritative, and the evaluator records missing checkpoints explicitly.
    print(
        "WARNING: one or more ablations did not finish successfully "
        f"(exit={exc.returncode}); continuing so evaluation and aggregation run.",
        flush=True,
    )
run("scripts/run_13_eval_ablations.py")
run("scripts/run_09_aggregate_stats.py")
