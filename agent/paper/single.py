"""Single-case convergence diagnostic.

Learners are trained on one registered case and greedily validated on that same case,
next to every rule and search baseline on it. Training and evaluation case coincide, so
the outcome says whether a method can learn this case, not whether it generalises; the
suite is kept apart from pilot/formal evidence and never enters admission or claims.
"""
import math
import time
import numpy as np
from environment.env import AssemblyEnv
from .config import LEARNERS, RULES, ALIASES, project_config, scientific_options, source_manifest
from .data import rows, load
from .evaluate import evaluate
from .experiment import suite_root, local_root, case_identity
from .network import create_network
from .report import csv_file
from .storage import write_json, read_csv, json_digest, load_checkpoint
from .train import train

SUITE = "single"
TAIL_SHARE = 0.3


def convergence(curve):
    """Relative spread of the greedy development score over the last 30% of validation points."""
    values = [float(r["rel_twt_geomean"]) for r in curve if r.get("rel_twt_geomean") not in (None, "")]
    if len(values) < 2:
        return dict(points=len(values), tail_points=len(values), tail_relative_change=None, converged=False)
    tail = values[-max(2, math.ceil(TAIL_SHARE * len(values))):]
    change = (max(tail) - min(tail)) / max(abs(float(np.mean(tail))), 1e-9)
    return dict(points=len(values), tail_points=len(tail), tail_relative_change=float(change), converged=bool(change < 0.01),
                first=values[0], last=values[-1], best=min(values))


def run_single(options, *, instance_id, methods, seeds, resume=False):
    from .runtime import ensure_idle, exclusive_lease
    ensure_idle(options)
    row = next((r for r in rows(options) if r["instance_id"] == instance_id), None)
    if row is None:
        raise ValueError(f"Unknown registered case: {instance_id}")
    methods = [ALIASES.get(m, m) for m in methods]
    unknown = sorted(set(methods) - set(LEARNERS))
    if unknown:
        raise ValueError(f"Unknown learners: {unknown}")
    if options["instances_per_update"] != 1:
        raise ValueError("The single-case diagnostic trains one case per update")
    instance = load(row)
    root = suite_root(options, SUITE) / instance_id
    reports = root / "_reports" / options["run_id"]
    with exclusive_lease(local_root(options) / "execution.lease"):
        baselines = []
        for method in RULES:
            for seed in (options["seeds"] if method == "RH-ALNS" else [0]):
                out = root / method / f"seed_{seed}" / options["run_id"] / f"{instance_id}.json"
                result = evaluate(instance, options, out, method=method, seed=seed, timing_scope="single_case")
                baselines.append(dict(method=method, seed=seed, kind="fixed", epoch=None, twt=result["twt"], feasible=result["feasible"],
                                      gate_fraction=result["gate_fraction"], moves=result["moves"], travel=result["travel"],
                                      decision_p95_ms=result["decision_p95_ms"], failure=result["failure"]))
        anchor = next(r for r in baselines if r["method"] == "Periodic-ATC")
        if not anchor["feasible"]:
            raise RuntimeError(f"Periodic-ATC is infeasible on {instance_id}: {anchor['failure']}")
        reference = float(anchor["twt"])
        references = {instance_id: reference}
        for r in baselines:
            r["rel_twt"] = (r["twt"] / max(reference, 1.0)) if r["feasible"] else None
        learners, curves, convergences = [], [], []
        for method in methods:
            for seed in seeds:
                folder = root / method / f"seed_{seed}" / options["run_id"]
                signature = json_digest(dict(suite=SUITE, case=case_identity(row), options=scientific_options(options), method=method,
                                             seed=seed, baselines=references, source=source_manifest()["digest"]))
                write_json(folder / "baselines.json", references)
                write_json(folder / "single-case.json", dict(case=case_identity(row), method=method, seed=seed, signature=signature,
                           note="diagnostic: training and evaluation case coincide"))
                started = time.perf_counter()
                outcome = train(options, method, seed, [instance], [instance], references, folder, options["pilot_updates"], signature, resume=resume)
                env = AssemblyEnv(project_config(options))
                env.reset(instance)
                network = create_network(options, env.graph(), method)
                for kind in ("latest", "best"):
                    checkpoint = folder / f"{kind}.pt"
                    if not checkpoint.exists():
                        continue
                    payload = load_checkpoint(checkpoint, signature)
                    network.load_state_dict(payload["network"], strict=True)
                    out = folder / "final" / kind / f"{instance_id}.json"
                    result = evaluate(instance, options, out, method=method, seed=seed, network=network, baseline=reference, timing_scope="single_case")
                    learners.append(dict(method=method, seed=seed, kind=kind, epoch=payload["epoch"], twt=result["twt"], feasible=result["feasible"],
                                         gate_fraction=result["gate_fraction"], moves=result["moves"], travel=result["travel"],
                                         decision_p95_ms=result["decision_p95_ms"], failure=result["failure"],
                                         rel_twt=(result["twt"] / max(reference, 1.0)) if result["feasible"] else None,
                                         training_seconds=outcome["training_seconds"], wall_seconds=time.perf_counter() - started))
                curve = read_csv(folder / "validation_summary.csv")
                curves.extend(dict(method=method, seed=seed, **r) for r in curve)
                convergences.append(dict(method=method, seed=seed, **convergence(curve)))
        from .monitor import export
        figures = export(options)
        ranked = sorted(baselines + learners, key=lambda r: (r["twt"] is None, r["twt"] if r["twt"] is not None else 0.0))
        csv_file(reports / "baselines.csv", baselines)
        csv_file(reports / "learners.csv", learners)
        csv_file(reports / "validation-curves.csv", curves)
        csv_file(reports / "convergence.csv", convergences)
        csv_file(reports / "ranking.csv", ranked)
        number = lambda x: "missing" if x is None else f"{x:.3f}"
        lines = [f"# single-case diagnostic — {instance_id}", "",
                 f"Updates per learner/seed: {options['pilot_updates']} × {options['episodes_per_instance']} complete episodes; greedy validation every {options['validation_every_updates']} updates on the same case.",
                 f"Periodic-ATC TWT = {reference:.3f} (reference 1.0). Training and evaluation case coincide: not generalization evidence.", "",
                 "| Method | Policy | Seed | Update | TWT | TWT / Periodic-ATC |", "|---|---|---:|---:|---:|---:|"]
        for r in ranked:
            lines.append(f"| {r['method']} | {r['kind']} | {r['seed']} | {r['epoch'] if r['epoch'] is not None else '-'} | {number(r['twt'])} | {number(r['rel_twt'])} |")
        lines += ["", "| Learner | Seed | Validation points | Tail relative change | Converged (<1%) | First | Best | Last |", "|---|---:|---:|---:|---|---:|---:|---:|"]
        for c in convergences:
            lines.append(f"| {c['method']} | {c['seed']} | {c['points']} | {number(c.get('tail_relative_change'))} | {c['converged']} | {number(c.get('first'))} | {number(c.get('best'))} | {number(c.get('last'))} |")
        (reports / "summary.md").write_text("\n".join(lines) + "\n", "utf-8")
        summary = dict(suite=SUITE, case=case_identity(row), reference_twt=reference, baselines=baselines, learners=learners,
                       convergence=convergences, figures=figures, reports=str(reports),
                       note="single-case diagnostic; training and evaluation case coincide; not generalization evidence")
        write_json(reports / "summary.json", summary)
        return summary
