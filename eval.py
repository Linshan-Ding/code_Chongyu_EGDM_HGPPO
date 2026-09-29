"""Phase K fixed-test evaluation entrypoint.

K1: four rule baselines + EGDM-HGPPO checkpoint inference.
K2: RH-MILP, RH-ALNS and four learned comparison architectures.  The offline
CP-SAT/MILP reference is run separately via ``python -m agent.baselines.run_exact`` so
its proof status / lower bound are not mixed with online episode diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.baselines.rules import RULE_METHODS, build_rule_policy
from agent.baselines.optimization import RollingHorizonMILPPolicy
from agent.baselines.alns import RollingHorizonALNSPolicy
from agent.baselines.learned_variants import LEARNED_BASELINE_METHODS
from configs.config import load_config
from agent.evaluation.config import load_eval_settings
from agent.evaluation.learned import EGDMHGPPOEvaluationPolicy
from agent.evaluation.learned_baselines import LearnedBaselineEvaluationPolicy
from agent.evaluation.references import load_reference_objectives
from agent.evaluation.runner import evaluate_records, write_eval_csv
from agent.evaluation.test_suite import ensure_fixed_test_suite, load_test_records
from agent.training.trainer import resolve_device


DEFAULT_CONFIGS = [
    "configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml", "configs/curriculum.yaml"
]
ONLINE_K2 = ("RH-MILP", "RH-ALNS", *LEARNED_BASELINE_METHODS)
IMPLEMENTED = (*RULE_METHODS, *ONLINE_K2, "EGDM-HGPPO")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EGDM-HGPPO Phase K fixed-test evaluator")
    p.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--eval-config", default="configs/eval.yaml")
    p.add_argument("--methods", nargs="+", default=list(RULE_METHODS))
    p.add_argument("--test-root", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--references", default=None, help="optional ref_exact.csv for optimality-gap columns")
    p.add_argument("--run-id", default="eval")
    p.add_argument("--tier", default="main")
    p.add_argument("--scales", nargs="+", default=None)
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--checkpoint", default=None, help="EGDM-HGPPO best/latest checkpoint")
    p.add_argument("--normalizer-checkpoint", default=None)
    p.add_argument(
        "--baseline-checkpoint", action="append", default=[], metavar="METHOD=PATH",
        help="repeat for learned K2 methods, e.g. GAT-PPO=result/.../best_model.pt",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--prepare", action="store_true", help="materialize selected formal test cells first")
    p.add_argument("--prepare-only", action="store_true", help="materialize fixed test cells and exit")
    p.add_argument("--count", type=int, default=None, help="instances per selected cell when --prepare")
    p.add_argument("--base-seed", type=int, default=400000)
    p.add_argument("--load-ratios", nargs="+", type=float, default=None)
    p.add_argument("--due-tightness", nargs="+", default=None)
    p.add_argument("--relocation-multiplier", type=float, default=1.0)
    p.add_argument("--worker-skill-density", type=float, default=None)
    p.add_argument("--robot-capability-density", type=float, default=None)
    p.add_argument("--prefix", default="test", help="fixed-instance filename prefix when preparing a suite")
    p.add_argument("--smoke", action="store_true")
    return p


def _checkpoint_map(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--baseline-checkpoint expects METHOD=PATH, got {item!r}")
        method, path = item.split("=", 1)
        method = method.strip(); path = path.strip()
        if method in out:
            raise SystemExit(f"duplicate baseline checkpoint for {method}")
        out[method] = path
    return out


def main() -> None:
    args = parser().parse_args()
    cfg = load_config(args.config)
    settings = load_eval_settings(args.eval_config)
    if args.smoke:
        from agent.evaluation.smoke import main as smoke_main
        smoke_main(); return

    unknown = [m for m in args.methods if m not in IMPLEMENTED]
    if "CP-SAT/MILP" in args.methods:
        raise SystemExit(
            "CP-SAT/MILP is an offline reference, not an online DecisionPolicy. "
            "Run: python -m agent.baselines.run_exact ... and pass its CSV with --references."
        )
    if unknown:
        raise SystemExit(f"unknown/unimplemented methods: {', '.join(unknown)}")

    root = args.test_root or settings.test_root
    scales = tuple(args.scales or settings.formal_scales)
    scenarios = tuple(args.scenarios or settings.formal_scenarios)
    if args.prepare:
        ensure_fixed_test_suite(
            cfg, root=root, base_seed=args.base_seed,
            scales=scales, scenarios=scenarios,
            load_ratios=tuple(args.load_ratios or settings.formal_load_ratios),
            due_tightness=tuple(args.due_tightness or settings.formal_due_tightness),
            instances_per_combination=int(args.count or settings.formal_instances_per_combination),
            instance_parameter_table_path=settings.formal_instance_parameter_table_path,
            one_per_parameter_case=True,
            prefix=args.prefix,
            relocation_multiplier=args.relocation_multiplier,
            worker_skill_density=args.worker_skill_density,
            robot_capability_density=args.robot_capability_density,
        )
        if args.prepare_only:
            print(f"Fixed test suite ready: {root}")
            return
    records = load_test_records(root, scales=scales, scenarios=scenarios)
    if not records:
        raise SystemExit("no fixed test records found; run with --prepare first")

    baseline_ckpt = _checkpoint_map(args.baseline_checkpoint)
    policies = []
    for name in args.methods:
        if name in RULE_METHODS:
            policies.append(build_rule_policy(name, settings.rule_config))
        elif name == "RH-MILP":
            policies.append(RollingHorizonMILPPolicy(settings.optimization_config, settings.rule_config))
        elif name == "RH-ALNS":
            policies.append(RollingHorizonALNSPolicy(settings.alns_config, settings.rule_config))
        elif name in LEARNED_BASELINE_METHODS:
            checkpoint = baseline_ckpt.get(name)
            if checkpoint is None:
                raise SystemExit(f"{name} evaluation requires --baseline-checkpoint {name}=PATH")
            policies.append(LearnedBaselineEvaluationPolicy(
                cfg, method=name, checkpoint=checkpoint,
                device=resolve_device(args.device),
                deterministic=settings.deterministic_learned_policy,
            ))
        elif name == "EGDM-HGPPO":
            if args.checkpoint is None:
                raise SystemExit("EGDM-HGPPO evaluation requires --checkpoint")
            policies.append(EGDMHGPPOEvaluationPolicy(
                cfg, checkpoint=args.checkpoint,
                normalizer_checkpoint=args.normalizer_checkpoint,
                device=resolve_device(args.device),
                deterministic=settings.deterministic_learned_policy,
            ))

    references = {} if args.references is None else load_reference_objectives(args.references)
    rows = evaluate_records(
        cfg, records=records, policies=policies, run_id=args.run_id,
        tier=args.tier, max_episode_decisions=settings.max_episode_decisions,
        references=references,
        learned_policy_batch_size=settings.learned_policy_batch_size,
    )
    output = write_eval_csv(args.output or settings.output_csv, rows)
    print(f"Phase K evaluation completed: {len(rows)} rows")
    print(output)


if __name__ == "__main__":
    main()
