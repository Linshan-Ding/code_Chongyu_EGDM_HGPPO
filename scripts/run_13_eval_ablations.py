"""Evaluate every completed Table-10 ablation on the immutable fixed test set."""
from __future__ import annotations

import csv
import argparse
from pathlib import Path

from _bootstrap import ROOT, formal_iterations, require_cuda
from configs.config import load_config
from agent.evaluation.ablation import AblationEvaluationPolicy
from agent.evaluation.config import load_eval_settings
from agent.evaluation.learned import EGDMHGPPOEvaluationPolicy
from agent.evaluation.metrics import EVAL_FIELDS
from agent.evaluation.references import load_reference_objectives
from agent.evaluation.runner import evaluate_records
from agent.evaluation.test_suite import load_test_records
from agent.training.trainer import resolve_device
from agent.experiments.ablation_scheme2 import load_ablation_settings
from agent.experiments.scheme2 import load_scheme2_config


def main() -> None:
    argparse.ArgumentParser(
        description="Evaluate completed Table-10 ablations on the fixed test suite"
    ).parse_args()
    ablation_cfg = load_ablation_settings()
    require_cuda()
    s2 = load_scheme2_config()
    seed = s2.training_seed
    cfg = load_config(("configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml", "configs/curriculum.yaml"))
    settings = load_eval_settings("configs/eval.yaml")
    records = load_test_records(settings.test_root, scales=settings.formal_scales, scenarios=settings.formal_scenarios)
    if not records:
        raise SystemExit("no fixed test records found; run python scripts/run_04_prepare_xl_generalization.py first")
    references = {}
    ref_path = ROOT / "result" / "ref_exact.csv"
    if ref_path.is_file():
        references = load_reference_objectives(ref_path)
    rows = []
    device = resolve_device("auto")
    # The full formal model is the reference row for Table 10.  It is reused,
    # never retrained as an "ablation", and must come from fixed-validation
    # checkpoint selection under the exact active formal budget.
    total = formal_iterations()
    full_checkpoint = (
        ROOT / "result" / "scheme2_runs"
        / f"egdm_hgppo_scheme2_seed{seed}_n{total}"
        / "checkpoints" / "best_model.pt"
    )
    if full_checkpoint.is_file():
        full_policy = EGDMHGPPOEvaluationPolicy(
            cfg, checkpoint=full_checkpoint, device=device,
            deterministic=settings.deterministic_learned_policy,
        )
        metrics = evaluate_records(
            cfg, records=records, policies=(full_policy,),
            run_id=f"full_seed{seed}", tier="main",
            max_episode_decisions=settings.max_episode_decisions,
            references=references,
            learned_policy_batch_size=settings.learned_policy_batch_size,
        )
        rows.extend({"variant": "full", **metric.to_dict(), "status": "ok"} for metric in metrics)
        print(f"full: {len(metrics)} fixed-test rows", flush=True)
    else:
        rows.append({"variant": "full", "status": "missing_checkpoint"})

    for variant in ablation_cfg.variants:
        run_root = ROOT / ablation_cfg.run_root
        checkpoint = (
            run_root / f"ablation_{variant}_seed{seed}_n{total}"
            / "checkpoints" / "best_model.pt"
        )
        if not checkpoint.is_file():
            rows.append({"variant": variant, "status": "missing_checkpoint"})
            continue
        policy = AblationEvaluationPolicy(
            cfg, variant=variant, checkpoint=checkpoint, device=device,
            deterministic=settings.deterministic_learned_policy,
            periodic_gate_period=ablation_cfg.periodic_gate_period,
        )
        metrics = evaluate_records(
            cfg, records=records, policies=(policy,),
            run_id=f"ablation_{variant}_seed{seed}",
            tier="main", max_episode_decisions=settings.max_episode_decisions,
            references=references,
            learned_policy_batch_size=settings.learned_policy_batch_size,
        )
        rows.extend({"variant": variant, **metric.to_dict(), "status": "ok"} for metric in metrics)
        print(f"{variant}: {len(metrics)} fixed-test rows", flush=True)
    output = ROOT / ablation_cfg.eval_csv
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("variant",) + EVAL_FIELDS + ("status",)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
