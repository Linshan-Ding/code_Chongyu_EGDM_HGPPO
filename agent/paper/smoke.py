"""Engineering acceptance only. Never launches the 200/3600-epoch experiment."""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
import torch
import yaml

from environment.env import AssemblyEnv
from .config import ROOT, LEARNERS, RULES, project_config, source_manifest
from .data import prepare, rows, load
from .network import create_network
from .train import train, signature
from .evaluate import evaluate
from .storage import write_json, read_json, load_checkpoint, committed_bundles
from .matrix import matrix
from .runtime import spawn, identity, alive, ensure_idle


def smoke(options, skip_display=False):
    ensure_idle(options)
    prepare(options)
    root = ROOT / options["result_root"] / "smoke" / str(time.time_ns())
    root.mkdir(parents=True)
    o = dict(options, result_root=str(root), workers=2, environments=8,
             update_threads=1, minibatch=32, microbatch=16)
    config = root / "options.yaml"
    config.write_text(yaml.safe_dump(o), "utf-8")
    torch.set_num_threads(1)
    status = dict(scope="engineering_only", passed=False, source=source_manifest(),
                  completed=[], failures=[], started=time.time(), result_root=str(root))
    def done(stage, **fields):
        status["completed"].append(dict(stage=stage, **fields))
        write_json(root / "acceptance.json", status)
        print(stage, flush=True)
    try:
        with (root / "unit-tests.log").open("w", encoding="utf-8") as f:
            p = subprocess.run([sys.executable, "-m", "unittest", "tests.test_paper", "tests.test_cleanup", "tests.test_sm100", "-v"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if p.returncode:
            raise RuntimeError(f"Unit tests failed: {root / 'unit-tests.log'}")
        done("unit tests")
        for suite in ("pilot-single", "pilot-joint", "formal"):
            plan = matrix(options, suite)
            write_json(root / f"matrix-{suite}.json", plan)
        done("complete matrices")
        # Exactly one eight-episode batch plus one PPO cycle at each scale.
        # Initial evaluation also verifies the complete anchor action trace.
        scale_options = dict(options, result_root=str(root))
        for row in rows(o, "pilot"):
            instance = load(row)
            torch.set_num_threads(scale_options["update_threads"])
            baseline = evaluate(instance, scale_options, root / "anchor" / f"{instance.instance_id}_rule.json", method="Periodic-ATC")
            folder = root / ("scale_" + row["parameters"]["scale"])
            bs = {instance.instance_id: baseline["twt"]}
            write_json(folder / "baselines.json", bs)
            sig = signature(scale_options, "full", 0, [row], [], bs)
            probe = AssemblyEnv(project_config(scale_options))
            probe.reset(instance)
            zero = create_network(scale_options, probe.graph(), "full")
            learned = evaluate(instance, scale_options, root / "anchor" / f"{instance.instance_id}_zero.json", method="full", network=zero, baseline=baseline["twt"])
            result = train(scale_options, "full", 0, [instance], [], bs, folder, 1, sig)
            bundles = committed_bundles(folder)
            if not (baseline["feasible"] and learned["feasible"] and baseline["twt"] == learned["twt"] and baseline["trace_sha256"] == learned["trace_sha256"]):
                raise AssertionError(dict(baseline=baseline, zero=learned))
            if result["epoch"] != 1 or result["completed_episodes"] != 8:
                raise AssertionError("Scale smoke must commit one batch of eight complete episodes")
            done("complete batch update and zero-residual trace " + row["parameters"]["scale"],
                 baseline_twt=baseline["twt"], trace=baseline["trace_sha256"], summary=bundles[1]["summaries"][0])
        tiny = rows(o, "tiny")[0]
        instance = load(tiny)
        b = evaluate(instance, o, root / "tiny_baseline.json", method="Periodic-ATC")
        if not b["feasible"]:
            raise RuntimeError(b)
        bs = {instance.instance_id: b["twt"]}
        for method in LEARNERS:
            folder = root / f"train_{method}"
            sig = signature(o, method, 0, [tiny], [], bs)
            write_json(folder / "baselines.json", bs)
            # Eight complete episodes, one PPO cycle with three internal passes.
            result = train(o, method, 0, [instance], [], bs, folder, 1, sig)
            if result["epoch"] != 1 or result["completed_episodes"] != 8:
                raise AssertionError("smoke epoch budget mismatch")
            before = read_json(folder / "progress.json")
            train(o, method, 0, [instance], [], bs, folder, 1, sig, resume=True)
            if read_json(folder / "progress.json") != before:
                raise AssertionError("Completed-budget resume unexpectedly changed state")
            repeated = []
            training_job = dict(output=str(folder), training=[tiny])
            for kind, repeats in (("latest", 3),):
                for repeat in range(repeats):
                    output = root / f"replay_{method}_{kind}_{repeat}"
                    job = dict(method=method, seed=0, checkpoint_kind=kind, repeat=repeat, output=str(output), evaluation=[tiny])
                    request = root / f"request_{method}_{kind}_{repeat}.json"
                    write_json(request, dict(job=job, training_job=training_job))
                    process = spawn(["_replay-file", "--config", str(config), "--request", str(request)], output / "process")
                    if process.wait() != 0:
                        raise RuntimeError(f"Fresh-process replay failed: {output / 'process.stderr.log'}")
                    outcome = read_json(output / "results.json")[0]
                    if not outcome["feasible"]:
                        raise AssertionError(outcome)
                    if kind == "latest":
                        repeated.append((outcome["twt"], outcome["trace_sha256"]))
            if len(set(repeated)) != 1:
                raise AssertionError("Non-deterministic repeated final-policy traces")
            done("update and fresh-process replay " + method, **result)
        for method in RULES:
            for seed in options["seeds"] if method == "RH-ALNS" else [0]:
                result = evaluate(instance, o, root / "rules" / f"{method}_{seed}.json", method=method, seed=seed, baseline=b["twt"])
                if not result["feasible"]:
                    raise AssertionError(result)
            done("baseline " + method)
        from .runtime import execute_job
        reference_job = dict(kind="reference", output=str(root / "reference-smoke"), evaluation=[tiny])
        execute_job(dict(o, reference_seconds=.2), reference_job, {}, False)
        reference_result = read_json(root / "reference-smoke/results.json")[0]
        if reference_result.get("failure"):
            raise RuntimeError(reference_result)
        done("micro reference status", reference_kind=reference_result["reference_kind"])
        from .monitor import start_server, monitor, export, figures
        if not skip_display:
            server = start_server(o, config)
            try:
                state = monitor(o, server["port"], once=True)
                if not state["connected"]:
                    raise RuntimeError(state)
                from .visdom_transport import client
                vis = client(server["port"])
                fig = figures(root / "train_full")[instance.instance_id + "_TWT"]
                received = json.loads(vis.get_window_data(win=instance.instance_id + "_TWT", env="paper-train_full"))
                if received["content"] != fig:
                    raise AssertionError("Visdom values/axes differ from durable training logs")
                write_json(root / "visdom-curve-check.json", dict(expected=fig, received=received))
                from unittest.mock import patch
                with patch("agent.paper.monitor.client", side_effect=ConnectionError("injected smoke disconnection")):
                    disconnected = monitor(o, server["port"], once=True)
                recovered = monitor(o, server["port"], once=True)
                replayed = json.loads(vis.get_window_data(win=instance.instance_id + "_TWT", env="paper-train_full"))
                if disconnected["connected"] or not recovered["connected"] or replayed["content"] != fig:
                    raise AssertionError("Visdom reconnect/backfill is not exact")
                write_json(root / "visdom-reconnect-check.json", dict(injected_disconnect=True,
                           disconnected=disconnected, recovered=recovered, exact_backfill=True))
                done("Visdom live transport", url=server["url"])
            finally:
                if alive(server["server"]):
                    import psutil
                    psutil.Process(server["server"]["pid"]).terminate()
        else:
            status["failures"].append("live display acceptance explicitly skipped")
        done("offline curves", figures=export(o))
        status["passed"] = not status["failures"]
    except Exception:
        status["failures"].append(traceback.format_exc())
    status["ended"] = time.time()
    write_json(root / "acceptance.json", status)
    write_json(ROOT / options["result_root"] / "latest-smoke.json", status)
    if not status["passed"]:
        raise RuntimeError(f"Engineering acceptance incomplete: {root / 'acceptance.json'}")
    return status
