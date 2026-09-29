"""Scheme-2 learned-baseline runner with 4060-safe execution overrides."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.baselines.learned_variants import LEARNED_BASELINE_METHODS
from agent.experiments.baseline_scheme2 import (
    baseline_scheme2_preflight,
    run_baseline_scheme2,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=LEARNED_BASELINE_METHODS, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true", help="cheap 8-env/512-event/1-epoch wiring test")
    mode.add_argument("--confirm", action="store_true", help="one full 32-env/8192-event/4-epoch iteration")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--replay-microbatch", type=int, default=512,
                   help="execution-only replay split; logical minibatch remains 512")
    p.add_argument("--cache-mib", type=int, default=2048,
                   help="execution-only CPU materialize-cache budget")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.preflight:
        r = baseline_scheme2_preflight(
            method=args.method,
            replay_microbatch=args.replay_microbatch,
            cache_mib=args.cache_mib,
        )
        print("Baseline Scheme 2 preflight: PASS")
        print(f"method: {r['method']}")
        print(f"training_seed: {r['training_seed']} (single seed); curriculum_enabled={r['curriculum_enabled']}")
        distribution_cells = r["training_distribution_cells"]
        distribution_label = (
            "range-sampled structural dimensions"
            if distribution_cells is None else f"{distribution_cells} declared cells"
        )
        print(
            f"distribution: {distribution_label}; structure_mode={r['training_structure_mode']}; "
            f"scales={r['training_scales']}; scenarios={r['training_scenarios']}; "
            f"rho={r['training_load_ratios']}; due={r['training_due_tightness']}"
        )
        print(
            f"formal budget: {r['parallel_envs']} envs, {r['rollout_events']} events/iter, "
            f"logical_mb={r['logical_minibatch']}, replay_microbatch={r['replay_microbatch']}, "
            f"PPO_epochs={r['ppo_epochs']}"
        )
        print(f"hardware_profile: {r['hardware_profile_label']}")
        print(
            f"fresh_instances_each_iteration={r['fresh_instances_each_iteration']}; "
            f"validation={r['validation_instances']}/{r['validation_pool_instances']} every "
            f"{r['validation_every_iterations']} iterations; checkpoint_selection={r['validation_checkpoint_selection']}"
        )
        print(
            f"multiprocess={r['multiprocess_rollout']}; cross_epoch_cache={r['cross_epoch_materialize_cache']}; "
            f"cache_mib={r['materialize_cache_max_mib']}; action_constraints={r['baseline_action_constraints']}"
        )
        return

    if args.smoke:
        mode, resume, label = "smoke", False, "SMOKE"
    elif args.confirm:
        mode, resume, label = "confirm", False, "CONFIRM"
    elif args.run:
        mode, resume, label = "formal", False, "FORMAL"
    else:
        mode, resume, label = "formal", True, "RESUME"

    result, stats = run_baseline_scheme2(
        method=args.method,
        mode=mode,
        iterations=args.iterations,
        device=args.device,
        resume=resume,
        replay_microbatch=args.replay_microbatch,
        cache_mib=args.cache_mib,
    )
    print(f"Baseline Scheme 2 {label} completed: {args.method}")
    print(f"run_dir: {result.run_dir}")
    print(f"iterations: {result.global_iterations}")
    print(f"latest_checkpoint: {result.latest_checkpoint}")
    print(f"best_model: {result.best_model}")
    print(f"final_model: {Path(result.run_dir) / 'checkpoints' / 'final_model.pt'}")
    if stats:
        print(
            f"last_iteration: wall={stats['wall_seconds']:.1f}s; rollout={stats['rollout_seconds']:.1f}s; "
            f"ppo={stats['ppo_seconds']:.1f}s; optimizer_steps={stats['optimizer_steps']}; "
            f"cuda_peak={stats['cuda_peak_alloc_mib']:.1f} MiB"
        )
        print(
            f"replay_runtime: configured={stats.get('configured_replay_microbatch', -1)}; "
            f"effective={stats.get('effective_replay_microbatch', -1)}; "
            f"oom_fallbacks={stats.get('oom_fallback_count', -1)}"
        )


if __name__ == "__main__":
    main()
