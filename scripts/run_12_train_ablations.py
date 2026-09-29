"""Train all configured Table-10 ablations independently."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.experiments.ablation_scheme2 import load_ablation_settings, run_ablation
from agent.experiments.scheme2 import load_scheme2_config
from _bootstrap import require_cuda

def main():
    p=argparse.ArgumentParser(); p.add_argument("--variant", default=None); p.add_argument("--iterations", type=int, default=None); p.add_argument("--profile", default=None); p.add_argument("--device", default=None); p.add_argument("--resume", action="store_true"); a=p.parse_args()
    require_cuda()
    ablation_cfg=load_ablation_settings()
    if a.variant is not None and a.variant not in ablation_cfg.variants:
        p.error(f"unknown or disabled variant: {a.variant}")
    names=[a.variant] if a.variant else list(ablation_cfg.variants)
    s2=load_scheme2_config(budget_profile=a.profile)
    total=a.iterations if a.iterations is not None else s2.formal_iterations
    out=Path(ablation_cfg.run_root); out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for name in names:
        run_dir=out/f"ablation_{name}_seed{s2.training_seed}_n{total}"; latest=run_dir/"checkpoints"/"latest.pt"; final=run_dir/"checkpoints"/"final_model.pt"
        if final.is_file():
            rows.append({"variant":name,"status":"skipped_completed","run_dir":str(run_dir),"checkpoint":str(run_dir/"checkpoints"/"best_model.pt")}); continue
        try:
            # Zero-argument reruns automatically resume interrupted variants;
            # no completed training budget is silently mistaken for complete
            # merely because an intermediate latest.pt exists.
            resume = bool(a.resume or latest.is_file())
            result,_=run_ablation(variant=name,iterations=total,device=a.device,budget_profile=a.profile,resume=resume)
            rows.append({"variant":name,"status":"completed","run_dir":str(result.run_dir),"checkpoint":str(result.best_model or result.latest_checkpoint)})
        except Exception as exc:
            rows.append({"variant":name,"status":"failed","run_dir":str(run_dir),"error":repr(exc)})
    path=out/"training_manifest.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["variant","status","run_dir","checkpoint","error"]); w.writeheader(); w.writerows(rows)
    print(path)
    failed = [row for row in rows if row.get("status") == "failed"]
    if failed:
        print(f"{len(failed)} ablation variant(s) failed; inspect {path}", file=sys.stderr)
        raise SystemExit(1)

if __name__=="__main__": main()
