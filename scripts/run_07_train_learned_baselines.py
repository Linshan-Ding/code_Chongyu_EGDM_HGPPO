"""Train or resume the four learned comparison policies under Scheme-2."""

from _bootstrap import ROOT, formal_iterations, require_cuda, run


require_cuda()
budget = formal_iterations()
for method in ("MLP-PPO", "GAT-PPO", "HGT-PPO-Flat", "HGT-MAPPO"):
    safe = method.lower().replace("-", "_")
    run_dir = ROOT / "result" / "baseline_scheme2_runs" / f"{safe}_scheme2_seed0_n{budget}"
    final_model = run_dir / "checkpoints" / "final_model.pt"
    latest = run_dir / "checkpoints" / "latest.pt"
    if final_model.is_file():
        print(f"SKIP completed learned baseline: {method}")
        continue
    mode = "--resume" if latest.is_file() else "--run"
    print(f"{('RESUME' if latest.is_file() else 'START')} learned baseline: {method}")
    run(
        "scripts/baseline_scheme2.py", "--method", method, mode,
        "--iterations", str(budget),
    )
