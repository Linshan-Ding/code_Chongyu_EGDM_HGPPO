"""Zero-argument Scheme-2 smoke test that always executes the current code."""

import shutil

from _bootstrap import ROOT, run


confirm_run = ROOT / "result" / "scheme2_runs" / "scheme2_confirm_seed0_n1"
# Smoke artifacts are deliberately disposable. Removing them prevents a stale
# checkpoint from making a newly edited package appear to pass without running
# the Scheme-2 rollout and PPO update.
if confirm_run.is_dir():
    shutil.rmtree(confirm_run)
run("scripts/scheme2.py", "--confirm")

# The evaluator smoke uses the same environment/reward contract and writes only
# under result/phase_k_smoke_test, so it is safe to repeat after a completed run.
run("eval.py", "--smoke")
