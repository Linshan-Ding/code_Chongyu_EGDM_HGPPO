"""Cheap Phase K1 fixed-test + rule-baseline integration smoke."""

from __future__ import annotations

from pathlib import Path

from agent.baselines.rules import RULE_METHODS, build_rule_policy
from configs.config import load_config
from agent.evaluation.config import load_eval_settings
from agent.evaluation.runner import evaluate_records, write_eval_csv
from agent.evaluation.test_suite import ensure_fixed_test_suite


CONFIGS = (
    "configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml", "configs/curriculum.yaml"
)


def main() -> None:
    cfg = load_config(CONFIGS)
    settings = load_eval_settings("configs/eval.yaml")
    records = ensure_fixed_test_suite(
        cfg,
        root=settings.smoke_root,
        base_seed=settings.smoke_seed,
        scales=(settings.smoke_scale,),
        scenarios=(settings.smoke_scenario,),
        load_ratios=(settings.smoke_load_ratio,),
        due_tightness=(settings.smoke_due_tightness,),
        instances_per_combination=1,
        prefix="smoke",
    )
    policies = [build_rule_policy(name, settings.rule_config) for name in RULE_METHODS]
    rows = evaluate_records(
        cfg,
        records=records,
        policies=policies,
        run_id="phase_k_smoke",
        tier="smoke",
        max_episode_decisions=settings.max_episode_decisions,
    )
    out = write_eval_csv(settings.smoke_output_csv, rows)
    print("Phase K1 evaluation smoke: PASSED")
    print(f"fixed_instances: {len(records)}")
    print(f"methods: {', '.join(RULE_METHODS)}")
    for row in rows:
        print(
            f"{row.method}: TWT={row.twt:.6f}, tardy_ratio={row.tardy_ratio:.4f}, "
            f"reconfigs={row.reconfiguration_count}, reward_error={row.reward_identity_error:.3e}"
        )
    print(f"eval_csv: {out}")


if __name__ == "__main__":
    main()
