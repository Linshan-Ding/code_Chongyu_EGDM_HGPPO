"""Current paper workflow. With no subcommand, print help and start nothing."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _configured_device(argv):
    """Read the execution device before torch is imported; CPU runs hide every accelerator."""
    import yaml
    path = ROOT / "configs/paper.yaml"
    if "--config" in argv:
        path = Path(argv[argv.index("--config") + 1])
    try:
        return str(yaml.safe_load(path.read_text("utf-8")).get("device", "cpu"))
    except Exception:
        return "cpu"


if _configured_device(sys.argv) != "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    commands = ("prepare", "matrix", "smoke", "pilot-single", "pilot-joint", "formal", "evaluate-final", "collect", "authorize-formal", "resume", "status", "monitor", "report", "_replay", "_replay-file", "_server", "_execute")
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument("--config", default=str(ROOT / "configs/paper.yaml"))
        p.add_argument("--suite", choices=("pilot-single", "pilot-joint", "formal"), default="pilot-joint")
        p.add_argument("--result-root")
        p.add_argument("--run-id")
        p.add_argument("--workers", type=int)
        p.add_argument("--methods", nargs="+")
        p.add_argument("--seeds", nargs="+", type=int)
        p.add_argument("--jobs", nargs="+")
        p.add_argument("--phase", choices=("train", "evaluate", "all"), default="train")
        if name == "collect":
            p.add_argument("--from", dest="source", required=True)
        if name == "authorize-formal":
            p.add_argument("--human-reviewed-pilot", action="store_true", required=True)
        if name == "matrix":
            p.add_argument("--dry-run", action="store_true", required=True)
        if name in ("pilot-single", "pilot-joint", "formal", "evaluate-final", "resume", "_execute"):
            p.add_argument("--human-reviewed-pilot", action="store_true")
            p.add_argument("--background", action="store_true")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--approval-file")
        if name == "_replay":
            p.add_argument("--job", required=True)
        if name == "_replay-file":
            p.add_argument("--request", required=True)
        if name in ("monitor", "_server"):
            p.add_argument("--port", type=int, default=8097)
        if name == "monitor":
            p.add_argument("--start-server", action="store_true")
            p.add_argument("--once", action="store_true")
            p.add_argument("--offline", action="store_true")
        if name == "smoke":
            p.add_argument("--skip-display", action="store_true", help="Explicitly leave live display acceptance unverified")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    import torch
    from agent.paper.config import settings
    from agent.paper import runtime
    from agent.paper.data import prepare
    from agent.paper.matrix import matrix
    from agent.paper.storage import read_json, write_json
    overrides = {k: getattr(args, k) for k in ("result_root", "run_id", "workers") if getattr(args, k) is not None}
    options = settings(args.config, **overrides)
    torch.set_num_threads(options["update_threads"])
    if args.command == "prepare":
        records = prepare(options)
        print(json.dumps(dict(instances=len(records), manifest=options["data_manifest"])))
    elif args.command == "matrix":
        plan = matrix(options, args.suite)
        from agent.paper.experiment import local_root
        from agent.paper.matrix import select_jobs
        plan["selected_jobs"] = [j["id"] for j in select_jobs(plan, phase=args.phase, methods=args.methods, seeds=args.seeds, job_ids=args.jobs)]
        path = local_root(options) / f"matrix-{args.suite}-preview.json"
        write_json(path, plan)
        print(json.dumps(dict(jobs=len(plan["jobs"]), epoch_semantics=options["epoch_semantics"],
            environments_per_epoch=options["environments"],
            training_epochs=sum(j.get("budget_epochs", 0) for j in plan["jobs"] if j["kind"] == "train"),
            training_episodes=sum(j.get("budget_episodes", 0) for j in plan["jobs"] if j["kind"] == "train"),
            output=str(path), aliases=plan["aliases"]), indent=2))
    elif args.command == "status":
        print(json.dumps(runtime.status(options), indent=2))
    elif args.command == "collect":
        from agent.paper.transfer import collect
        result = collect(options, args.source)
        print(json.dumps(result, indent=2))
        if result["conflicts"] or result["rejected"]:
            raise SystemExit(2)
    elif args.command == "authorize-formal":
        print(json.dumps(runtime.approve_formal(options, args.human_reviewed_pilot), indent=2))
    elif args.command == "smoke":
        from agent.paper.smoke import smoke
        print(json.dumps(smoke(options, skip_display=args.skip_display), indent=2))
    elif args.command in ("pilot-single", "pilot-joint", "formal", "evaluate-final", "resume", "_execute"):
        suite = args.command if args.command in ("pilot-single", "pilot-joint", "formal") else args.suite
        resume = args.command == "resume" or args.resume
        phase = "evaluate" if args.command == "evaluate-final" else args.phase
        if phase == "all":
            raise ValueError("Training and final evaluation must be explicitly separate commands")
        if args.background:
            argv = ["_execute", "--config", str(Path(args.config).resolve()), "--suite", suite]
            argv += ["--phase", phase]
            for flag, val in (("--result-root", args.result_root), ("--run-id", args.run_id), ("--workers", args.workers), ("--approval-file", args.approval_file)):
                if val is not None:
                    argv += [flag, str(val)]
            for flag, val in (("--methods", args.methods), ("--seeds", args.seeds), ("--jobs", args.jobs)):
                if val:
                    argv += [flag, *map(str, val)]
            if resume:
                argv.append("--resume")
            if args.human_reviewed_pilot:
                argv.append("--human-reviewed-pilot")
            from agent.paper.experiment import local_root
            p = runtime.spawn(argv, local_root(options) / suite / "supervisor", detached=True)
            print(json.dumps(runtime.identity(p.pid)))
        else:
            result = runtime.run_suite(options, suite, phase=phase, resume=resume, reviewed=args.human_reviewed_pilot,
                                       approval_file=args.approval_file, methods=args.methods, seeds=args.seeds, job_ids=args.jobs)
            print(json.dumps(dict(complete=result["complete"], admission=result["admission"]), indent=2))
    elif args.command == "_replay":
        plan = matrix(options, args.suite)
        job = next(j for j in plan["jobs"] if j["id"] == args.job)
        source = next(j for j in plan["jobs"] if j["id"] == job["training_job"])
        runtime.replay_job(options, job, source)
    elif args.command == "_replay-file":
        request = read_json(args.request)
        runtime.replay_job(options, request["job"], request["training_job"])
    elif args.command == "report":
        from agent.paper.report import report
        from agent.paper.monitor import export
        result = report(options, args.suite)
        export(options)
        print(json.dumps(dict(complete=result["complete"], admission=result["admission"]), indent=2))
    elif args.command == "monitor":
        from agent.paper.monitor import start_server, monitor, export
        if args.offline:
            print(json.dumps(dict(exported=export(options))))
        else:
            port = args.port
            if args.start_server:
                server = start_server(options, Path(args.config).resolve())
                port = server["port"]
                print(server["url"], flush=True)
            monitor(options, port, once=args.once)
    elif args.command == "_server":
        from visdom.server.app import tornado_settings
        from visdom.server.run_server import start_server
        tornado_settings["autoreload"] = False
        start_server(port=args.port, hostname="127.0.0.1", base_url="", bind_local=True,
                     env_path=str(__import__("agent.paper.experiment", fromlist=["local_root"]).local_root(options) / "visdom-data"))


if __name__ == "__main__":
    main()
