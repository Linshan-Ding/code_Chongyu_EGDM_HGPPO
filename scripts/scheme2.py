"""Teacher Scheme-2 runner: one seed + random joint training + fixed validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.experiments.scheme2 import run_scheme2, scheme2_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="audit the Scheme-2 protocol")
    mode.add_argument("--confirm", action="store_true", help="one isolated full-stack random-joint iteration")
    mode.add_argument("--pilot", action="store_true", help="short random-joint training pilot")
    mode.add_argument("--run", action="store_true", help="start formal Scheme-2 training from scratch")
    mode.add_argument("--resume", action="store_true", help="resume an interrupted formal Scheme-2 run")
    parser.add_argument("--iterations", type=int, default=None, help="pilot/formal fixed PPO-iteration budget")
    parser.add_argument(
        "--profile",
        default=None,
        help="budget profile from configs/scheme2.yaml (default: active_budget_profile)",
    )
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.preflight:
        report = scheme2_preflight(budget_profile=args.profile)
        print("Scheme 2 preflight: PASS")
        print(
            f"budget_profile: {report['budget_profile']} "
            f"(available={report['available_budget_profiles']})"
        )
        print(f"training_seed: {report['training_seed']} (single seed)")
        print(f"curriculum_enabled: {report['curriculum_enabled']}")
        distribution_cells = report["training_distribution_cells"]
        distribution_label = (
            "range-sampled structural dimensions"
            if distribution_cells is None else f"{distribution_cells} declared cells"
        )
        print(
            f"training_distribution: {distribution_label}; "
            f"structure_mode={report['training_structure_mode']}; "
            f"parameter_cases={report.get('training_parameter_cases')}; "
            f"scales={report['training_scales']}; scenarios={report['training_scenarios']}; "
            f"rho={report['training_load_ratios']}; due={report['training_due_tightness']}"
        )
        print(
            f"runtime: {report['parallel_envs']} envs, {report['rollout_events']} events/iter, "
            f"logical_minibatch={report['logical_minibatch']}, replay_microbatch={report['replay_microbatch']}, "
            f"replay_microbatch_min={report['replay_microbatch_min']}, ppo_epochs={report['ppo_epochs']}"
        )
        print(
            f"formal_budget: iterations={report['formal_iterations'] if 'formal_iterations' in report else 'n/a'}, "
            f"sampled_events={report['sampled_events_budget']}, "
            f"optimizer_steps={report['optimizer_steps_budget']}"
        )
        print(f"hardware_profile: {report['hardware_profile_label']}")
        print(
            f"L1.7.6 stack: multiprocess={report['multiprocess_rollout']}; "
            f"cross_epoch_cache={report['cross_epoch_materialize_cache']}; "
            f"cache_budget={report['materialize_cache_max_mib']} MiB; "
            f"tensorized_decoder={report['tensorized_decoder_scoring']}; "
            f"candidate_score_cache={report['replay_candidate_score_memoization']}"
        )
        print(
            f"fresh_instances_each_iteration: {report['fresh_instances_each_iteration']}; "
            f"config_mutated_by_oom: {report['microbatch_fallback_persists']}; "
            f"run_local_microbatch_safe_cap: {report['run_local_microbatch_safe_cap']}"
        )
        print(
            f"fixed_validation_monitor: {report['validation_instances']} / "
            f"pool {report['validation_pool_instances']}; subset={report.get('validation_monitor_subset', 'n/a')}; "
            f"every={report['validation_every_iterations']} iterations"
        )
        print(f"fixed_test_visible_to_training: {report['test_visible_to_training']}")
        return

    if args.confirm:
        result, stats = run_scheme2(
            mode="confirm", device=args.device, budget_profile=args.profile
        )
        label = "CONFIRM"
    elif args.pilot:
        result, stats = run_scheme2(
            mode="pilot", iterations=args.iterations, device=args.device,
            budget_profile=args.profile,
        )
        label = "PILOT"
    elif args.run:
        result, stats = run_scheme2(
            mode="formal", iterations=args.iterations, device=args.device,
            resume=False, budget_profile=args.profile,
        )
        label = "FORMAL"
    else:
        result, stats = run_scheme2(
            mode="formal", iterations=args.iterations, device=args.device,
            resume=True, budget_profile=args.profile,
        )
        label = "RESUME"

    print(f"Scheme 2 {label} completed")
    print(f"run_dir: {result.run_dir}")
    print(f"iterations: {result.global_iterations}")
    print(f"training_mode: {result.final_stage_name}")
    print(f"latest_checkpoint: {result.latest_checkpoint}")
    print(f"best_model: {result.best_model}")
    print(f"final_model: {Path(result.run_dir) / 'checkpoints' / 'final_model.pt'}")
    if stats:
        print(
            f"last_iteration: wall={stats['wall_seconds']:.1f}s; "
            f"rollout={stats['rollout_seconds']:.1f}s; ppo={stats['ppo_seconds']:.1f}s; "
            f"optimizer_steps={stats['optimizer_steps']}; cuda_peak={stats['cuda_peak_alloc_mib']:.1f} MiB"
        )
        print(
            "full_action_runtime_audit: "
            f"reconfiguration_fraction={stats.get('reconfiguration_fraction', float('nan')):.6f}; "
            f"max_worker_moves={stats.get('max_worker_moves', -1)}; "
            f"max_robot_moves={stats.get('max_robot_moves', -1)}"
        )
        print(
            "replay_runtime: "
            f"configured_microbatch={stats.get('configured_replay_microbatch', -1)}; "
            f"effective_microbatch={stats.get('effective_replay_microbatch', -1)}; "
            f"safe_cap={stats.get('safe_replay_microbatch_cap', -1)}; "
            f"oom_fallbacks={stats.get('oom_fallback_count', -1)}"
        )
        cache = stats.get("materialize_cache") or {}
        if cache:
            print(f"materialize_cache: {cache}")
    if label == "CONFIRM":
        print("NEXT: python scripts/run_01_prepare_data.py")
    elif label == "PILOT":
        print("NEXT: inspect train_log.csv + validation_log.csv, then freeze one formal --iterations N budget")


if __name__ == "__main__":
    main()
