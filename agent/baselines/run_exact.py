"""Run the Phase-K2 offline MILP reference on fixed test instances."""

from __future__ import annotations
import argparse
from dataclasses import replace

from configs.config import load_config
from data.io import load_instance_csv
from agent.evaluation.config import load_eval_settings
from agent.evaluation.references import write_reference_csv
from agent.evaluation.test_suite import load_test_records
from agent.baselines.optimization import OfflineMILPReferenceSolver

DEFAULT_CONFIGS=["configs/instance.yaml","configs/env.yaml","configs/algo.yaml","configs/curriculum.yaml"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--eval-config", default="configs/eval.yaml")
    p.add_argument("--test-root", default=None)
    p.add_argument("--scales", nargs="+", default=["S"])
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--max-operations", type=int, default=None)
    p.add_argument("--minimum-dwell", type=float, default=0.0,
                   help="Exactness flag for the reference model; full model is exact only at dwell=0")
    p.add_argument("--output", default="result/ref_exact.csv")
    args=p.parse_args()
    cfg=load_config(args.config); settings=load_eval_settings(args.eval_config)
    opt=settings.optimization_config
    if args.time_limit is not None: opt=replace(opt, offline_time_limit_seconds=args.time_limit)
    if args.max_operations is not None: opt=replace(opt, offline_max_operations=args.max_operations)
    solver=OfflineMILPReferenceSolver(opt, minimum_dwell_time=args.minimum_dwell)
    records=load_test_records(args.test_root or settings.test_root,
                              scales=tuple(args.scales), scenarios=None if args.scenarios is None else tuple(args.scenarios))
    if not records: raise SystemExit("no fixed test records found")
    rows=[]
    for i, rec in enumerate(records,1):
        result=solver.solve(load_instance_csv(rec.path)); rows.append(result)
        print(f"[{i}/{len(records)}] {rec.instance_id}: objective={result.objective_twt} "
              f"gap={result.mip_gap} optimal={result.optimal} dwell_exact={result.dwell_exact}")
    out=write_reference_csv(args.output, rows)
    print(out)

if __name__=="__main__": main()
