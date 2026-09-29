from _bootstrap import run
from configs.config import load_config
from agent.experiments.scheme2 import load_scheme2_config
from agent.training.validation import ensure_fixed_validation_suite


s2 = load_scheme2_config()
cfg = load_config([
    "configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml", "configs/curriculum.yaml",
])
ensure_fixed_validation_suite(
    cfg,
    root=s2.validation_root,
    base_seed=s2.validation_base_seed,
    scales=s2.validation_scales,
    scenarios=s2.validation_scenarios,
    load_ratios=s2.validation_load_ratios,
    due_tightness=s2.validation_due_tightness,
    instances_per_combination=s2.validation_instances_per_combination,
    instance_parameter_table_path=s2.instance_parameter_table_path,
    one_per_parameter_case=True,
)
print(f"Fixed validation suite ready: {s2.validation_root}")

# Materialize the complete fixed Scheme-2 design for S/M/L. XL is prepared by
# the generalization script only, so it cannot leak into formal training.
run(
    "eval.py", "--prepare", "--prepare-only", "--methods", "Fixed-EDD", "--count", "1",
    "--scales", "S", "M", "L",
    "--scenarios", "D1", "D2", "D3", "D4", "D5",
    "--load-ratios", "0.65", "0.80", "0.95",
    "--due-tightness", "tight", "medium", "loose",
)
