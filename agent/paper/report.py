"""Results stay visible even when failed, missing, stronger or not selected."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from .config import ROOT, LEARNERS
from .matrix import matrix
from .storage import read_json, write_json, read_csv, committed_bundles
from .experiment import suite_root
from .rollout import batch_summary


def csv_file(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", "utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired(full, controls):
    wins = ties = losses = missing = 0
    improvements = []
    for f, c in zip(full, controls):
        if not f or not c or not f.get("feasible") or not c.get("feasible"):
            missing += 1
            continue
        diff = c["twt"] - f["twt"]
        wins += diff > 1e-8
        losses += diff < -1e-8
        ties += abs(diff) <= 1e-8
        improvements.append(100 * diff / max(c["twt"], 1.))
    return dict(wins=int(wins), ties=int(ties), losses=int(losses), missing=missing,
                mean_improvement_pct=float(np.mean(improvements)) if improvements else None,
                win_rate=wins / len(improvements) if improvements else None)


def training_tail(options, job):
    """The registered last ten visits, never the last ten successful batches."""
    folder = Path(job['output'])
    cases = job['training']
    grouped = defaultdict(list)
    error = None
    try:
        for bundle in committed_bundles(folder):
            if not bundle['epoch']:
                continue
            epoch = bundle['epoch']
            expected = cases[(epoch - 1) % len(cases)]['instance_id']
            visits = (epoch - 1) // len(cases) + 1
            summary = bundle['summaries'][0]
            check = batch_summary(bundle['episodes'], options['environments'])
            if summary['instance_id'] != expected or summary['instance_epoch'] != visits or any(summary[k] != v for k, v in check.items()):
                raise ValueError('Training schedule or raw batch statistics mismatch')
            grouped[expected].append((visits, summary, bundle['episodes']))
    except (KeyError, ValueError, OSError, IndexError) as exc:
        error = str(exc)
    output = []
    for case in cases:
        iid = case['instance_id']
        target = job['case_batches'][iid]
        window = options['tail_epochs']
        available = grouped[iid]
        selected = [r for r in available if target - window < r[0] <= target]
        complete = not error and target >= window and [r[0] for r in available] == list(range(1, target + 1))
        raw = [ep['twt'] for _, _, eps in selected for ep in eps] if complete else []
        means = [s['mean_twt'] for _, s, _ in selected] if complete else []
        output.append(dict(method=job['method'], seed=job['seed'], instance_id=iid, scale=case['parameters']['scale'],
            target_instance_epochs=target, committed_instance_epochs=len(available), tail_start=target-window+1,
            tail_end=target, expected_tail_episodes=window * options['environments'], tail_episodes=len(raw),
            mean_twt=float(np.mean(means)) if means else None,
            std_epoch_mean=float(np.std(means)) if means else None,
            std_episode_twt=float(np.std(raw)) if raw else None,
            status='complete' if complete else 'invalid' if error else 'incomplete',
            failure=error, meaning='stochastic training tail across changing policies; not final-policy test'))
    return output


def report(options, suite):
    folder = suite_root(options, suite) / "_reports" / options["run_id"]
    folder.mkdir(parents=True, exist_ok=True)
    plan = matrix(options, suite)
    outcomes = {j["id"]: read_json(Path(j["output"]) / "job-status.json") for j in plan["jobs"] if (Path(j["output"]) / "job-status.json").exists()}
    training_jobs = {j["id"]: j for j in plan["jobs"] if j["kind"] == "train"}
    all_rows, job_rows, replays, attempt_failures = [], [], defaultdict(list), []
    for job in plan["jobs"]:
        for evidence in sorted((ROOT / job["output"] / "failures").glob("*.json")):
            failure = read_json(evidence)
            episodes = failure.get("episodes", [])
            attempt_failures.append(dict(job_id=job["id"], method=job.get("method"), seed=job.get("seed"),
                level="batch_attempt" if failure.get("attempt") else "job", epoch=failure.get("epoch"),
                committed_epoch=failure.get("committed_epoch"), returned_episodes=len(episodes),
                feasible_episodes=sum(r.get("feasible", False) for r in episodes),
                failure=failure.get("failure", str(failure)), evidence_path=str(evidence)))
        state = outcomes.get(job["id"], dict(status="missing", failure="not executed"))
        job_rows.append(dict(job_id=job["id"], kind=job["kind"], method=job.get("method"), seed=job.get("seed"),
                             budget_epochs=job.get("budget_epochs", 0), budget_episodes=job.get("budget_episodes", 0), **state))
        if job["kind"] not in ("replay", "rule"):
            continue
        path = ROOT / job["output"] / "results.json"
        existing = {r["instance_id"]: r for r in read_json(path)} if path.exists() else {}
        for case in job["evaluation"]:
            iid = case["instance_id"]
            r = existing.get(iid, dict(instance_id=iid, feasible=False, twt=None, gap=None,
                failure=state.get("failure", "missing result"), decision_p95_ms=None, wall_seconds=None))
            snapshot = Path(training_jobs[job["training_job"]]["output"]) / "progress.json" if job.get("training_job") else folder / "_absent"
            progress = read_json(snapshot) if snapshot.exists() else {}
            row = dict(job_id=job["id"], method=job["method"], seed=job["seed"], instance_id=iid,
                scale=case["parameters"]["scale"], scenario=case["parameters"]["scenario"],
                load=case["parameters"]["load_ratio"], migration=case["parameters"]["relocation_multiplier"],
                split=case["split"], kind=job.get("checkpoint_kind", "fixed"), repeat=job.get("repeat", 0),
                budget_epochs=job.get("budget_epochs", 0), committed_epochs=progress.get("epoch", 0), committed_events=progress.get("committed_events", 0),
                environments_per_epoch=options["environments"], budget_episodes=job.get("budget_episodes", 0),
                committed_episodes=progress.get("committed_episodes", 0), selected_episodes=r.get("selected_episodes"),
                selected_epoch=r.get("epoch"), selected_events=r.get("selected_events"), training_seconds=progress.get("training_seconds"),
                status="success" if r["feasible"] else "failed" if iid in existing else "missing",
                feasible=r["feasible"], twt=r["twt"], gap=r["gap"], decision_p95_ms=r["decision_p95_ms"],
                evaluation_seconds=r["wall_seconds"], failure=r["failure"], trace_sha256=r.get("trace_sha256"),
                gate_fraction=r.get("gate_fraction"), moves=r.get("moves"), travel=r.get("travel"),
                reference_kind="empirical_policy", baseline_twt=r.get("baseline_twt"),
                fallback_status=r.get("fallback_status", "not_applied"))
            all_rows.append(row)
            if row["kind"] == "latest":
                replays[(row["method"], row["seed"], iid)].append(row)
    consistency = []
    for key, rr in replays.items():
        ok = (len(rr) == options["replays"] and all(r["feasible"] for r in rr)
              and len({(r["twt"], r["trace_sha256"]) for r in rr}) == 1)
        consistency.append(dict(method=key[0], seed=key[1], instance_id=key[2], reproducible=ok))
    selected = [r for r in all_rows if r["repeat"] == 0]
    groups = defaultdict(list)
    for r in selected:
        groups[(r["method"], r["kind"], r["split"], r["scale"])].append(r)
    tables = []
    for (method, kind, split, scale), group in groups.items():
        good = [r for r in group if r["feasible"]]
        tables.append(dict(method=method, kind=kind, split=split, scale=scale, seeds=sorted({r["seed"] for r in group}),
            complete=len(good), failed=sum(r["status"] == "failed" for r in group), missing=sum(r["status"] == "missing" for r in group),
            mean_twt=float(np.mean([r["twt"] for r in good])) if good else None,
            mean_gap_pct=float(np.mean([r["gap"] for r in good if r["gap"] is not None])) if any(r["gap"] is not None for r in good) else None,
            # Mean of independently measured instance p95 values, not a pooled p95.
            mean_instance_p95_ms=float(np.mean([r["decision_p95_ms"] for r in good])) if good else None,
            min_committed_epochs=min(r["committed_epochs"] for r in group), max_committed_epochs=max(r["committed_epochs"] for r in group),
            min_committed_episodes=min(r["committed_episodes"] for r in group), max_committed_episodes=max(r["committed_episodes"] for r in group),
            min_committed_events=min(r["committed_events"] for r in group), max_committed_events=max(r["committed_events"] for r in group)))
    full = [r for r in selected if r["method"] == "full" and r["kind"] == "latest"]
    comparisons = []
    for method in [m for m in LEARNERS if m != "full"] + ["Fixed-EDD", "Fixed-ATC", "Periodic-ATC", "RH-MILP", "RH-ALNS", "Opportunity-ATC"]:
        controls = [r for r in selected if r["method"] == method and r["kind"] in ("latest", "fixed")]
        paired_controls = [next((c for c in controls if c["instance_id"] == f["instance_id"] and
            (c["seed"] == f["seed"] if method in LEARNERS or method == "RH-ALNS" else c["seed"] == 0)), None) for f in full]
        comparisons.append(dict(control=method, **paired(full, paired_controls)))
    complete = bool(job_rows) and all(r["status"] == "completed" for r in job_rows) and all(r["status"] != "missing" for r in all_rows)
    target = next(iter(training_jobs.values()))["budget_epochs"]
    scale_gate = {}
    for scale in ("S", "M"):
        g = [r for r in full if r["scale"] == scale]
        feasible = bool(g) and all(r["feasible"] for r in g)
        # Mean paired gap across five development cases and the training seeds.
        mean_gap = float(np.mean([r["gap"] for r in g])) if feasible else None
        scale_gate[scale] = dict(mean_gap_pct=mean_gap, pass_quality=feasible and mean_gap <= -5.)
    mlp = next(c for c in comparisons if c["control"] == "mlp")
    graph_noninferiority = mlp["missing"] == 0 and mlp["mean_improvement_pct"] is not None and mlp["mean_improvement_pct"] >= -1.
    learned_best = [r for r in selected if r["kind"] == "latest"]
    budget_pass = bool(learned_best) and all(r["committed_epochs"] == target and
        r["committed_episodes"] == target * options["environments"] for r in learned_best)
    replay_pass = bool(consistency) and all(r["reproducible"] for r in consistency)
    all_feasible = bool(all_rows) and all(r["feasible"] for r in all_rows)
    three_seeds = set(options["seeds"]) == {0, 1, 2}
    budget_pass = budget_pass and options['environments'] == 8 and all(
        set(j['case_batches'].values()) == {100} for j in training_jobs.values())
    conflict_paths = list((suite_root(options, suite) / "_conflicts" / options["run_id"]).glob("**/conflict.json"))
    complete = complete and not conflict_paths
    quality_pass = complete and all_feasible and three_seeds and all(v["pass_quality"] for v in scale_gate.values()) and graph_noninferiority and budget_pass and replay_pass
    admission = dict(scope="engineering_only" if suite == "smoke" else suite, passed=quality_pass,
                     scale_results=scale_gate, graph_noninferiority_pass=graph_noninferiority,
                     full_budget_pass=budget_pass, replay_pass=replay_pass, all_evaluations_feasible=all_feasible, three_seeds=three_seeds,
                     latency_target_ms=100, latency_pass=bool(full) and all(r["decision_p95_ms"] is not None and r["decision_p95_ms"] <= 100 for r in full),
                     human_review_required=True, formal_started=suite == "formal" and any(r["status"] == "completed" for r in job_rows))
    tails = [r for job in training_jobs.values() for r in training_tail(options, job)]
    result = dict(protocol_hash=plan["protocol_hash"], run_id=options["run_id"], conflicts=[str(p) for p in conflict_paths], training_tail=tails, protocol=options["protocol"], epoch_semantics=options["epoch_semantics"], suite=suite, complete=complete, admission=admission, tables=tables,
                  comparisons=comparisons, replay_checks=consistency, jobs=job_rows, all_results=all_rows,
                  aliases=plan["aliases"], attempt_failures=attempt_failures, notes=["Training/development evidence is not generalization evidence.",
                  "Training tail summaries are not frozen-policy or unseen-instance test results.",
                  "Missing/failed controls cannot be counted as wins. No automatic fallback replaces a learned result."])
    write_json(folder / "report.json", result)
    csv_file(folder / "training-tail.csv", tails)
    csv_file(folder / "all-results.csv", all_rows)
    csv_file(folder / "summary.csv", tables)
    csv_file(folder / "comparisons.csv", comparisons)
    csv_file(folder / "replay-checks.csv", consistency)
    csv_file(folder / "job-status.csv", job_rows)
    csv_file(folder / "attempt-failures.csv", attempt_failures)
    csv_file(folder / "main-results.csv", [r for r in selected if r["method"] in
        ("full", "mlp", "gat", "dan", "Fixed-EDD", "Fixed-ATC", "Periodic-ATC", "RH-MILP", "RH-ALNS") and r["split"] != "generalization"])
    csv_file(folder / "ablation-results.csv", [dict(r, role="no_graph / MLP-PPO (shared evidence)" if r["method"] == "mlp" else r["method"])
        for r in selected if r["method"] in ("full", "mlp", "shared_relation", "periodic_gate", "no_rule_prior")])
    csv_file(folder / "generalization-L.csv", [r for r in selected if r["split"] == "generalization"])
    csv_file(folder / "appendix-strong-reference.csv", [r for r in selected if r["method"] == "Opportunity-ATC"])
    for dimension in ("load", "migration"):
        grouped = defaultdict(list)
        for r in selected:
            grouped[(r["method"], r["kind"], r["split"], r[dimension])].append(r)
        csv_file(folder / f"effects-{dimension}.csv", [dict(method=k[0], kind=k[1], split=k[2], level=k[3],
            complete=sum(r["feasible"] for r in rr), expected=len(rr),
            mean_twt=float(np.mean([r["twt"] for r in rr if r["feasible"]])) if any(r["feasible"] for r in rr) else None) for k, rr in grouped.items()])
    csv_file(folder / "gate-behavior.csv", [{k: r[k] for k in ("method", "seed", "instance_id", "kind", "gate_fraction", "moves", "travel", "status")} for r in selected])
    claims = []
    for c, m in (("dynamic graph", "mlp"), ("relation distinction", "shared_relation"), ("learned event gate", "periodic_gate"), ("rule prior support", "no_rule_prior")):
        comparison = next(x for x in comparisons if x["control"] == m)
        ready = complete and comparison["missing"] == 0 and comparison["mean_improvement_pct"] is not None
        verdict = ("insufficient_evidence" if not ready else "directionally_supported" if
                   comparison["mean_improvement_pct"] > 0 and comparison["win_rate"] > .5 else "not_supported")
        claims.append(dict(claim=c, control=m, evidence=comparison, verdict=verdict,
            limitation="Descriptive paired direction only; not a statistical significance or novelty claim."))
    write_json(folder / "claims.json", claims)
    reference_jobs = [j for j in plan["jobs"] if j["kind"] == "reference"]
    reference = Path(reference_jobs[0]["output"]) / "results.json" if reference_jobs else folder / "_absent_reference"
    write_json(folder / "reference-status.json", read_json(reference) if reference.exists() else
               dict(status="not_run" if suite == "formal" else "not_applicable", reference_kind="missing"))
    csv_file(folder / "fallback-status.csv", [{k: r[k] for k in ("method", "seed", "instance_id", "kind", "twt", "baseline_twt", "fallback_status", "status")} for r in selected])
    lines = [f"# {suite} — measured results", "", f"Complete: {complete}; numerical admission: {quality_pass}. Human review required.",
             f"Budget per learner/seed: {target} same-instance batch epochs × {options['environments']} complete episodes = {target * options['environments']} episodes. Events are measured, not budgeted.",
             f"Preserved failure evidence files: {len(attempt_failures)} (batch and supervisor records may describe the same attempt; see attempt-failures.csv).", "",
             "| Method | Policy | Split/scale | TWT | Complete/failed/missing | Mean instance p95 ms |", "|---|---|---|---:|---|---:|"]
    number = lambda x: "missing" if x is None else f"{x:.3f}"
    for r in tables:
        lines.append(f"| {r['method']} | {r['kind']} | {r['split']}/{r['scale']} | {number(r['mean_twt'])} | {r['complete']}/{r['failed']}/{r['missing']} | {number(r['mean_instance_p95_ms'])} |")
    lines += ["", "No-graph and MLP-PPO share one result. Per-case/seed epoch and complete-episode budgets, measured events, runtime, failures and paired comparisons are in the adjacent CSV/JSON files.",
              "No source in these tables is automatically a proven optimum. Training tail statistics are separate from final-policy test results."]
    (folder / "report.md").write_text("\n".join(lines) + "\n", "utf-8")
    return result
