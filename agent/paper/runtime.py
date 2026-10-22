"""Sequential experiment jobs with process identity locks and explicit formal review."""
import os
import sys
import time
import json
import subprocess
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import psutil
import torch

from .config import ROOT, FORMAT, project_config, source_manifest
from .data import load, prepare
from .matrix import matrix, select_jobs
from .experiment import local_root, host_id, hardware, suite_root, protocol_record, protocol_hash
from .transfer import job_identity, seal, validate_training
from .evaluate import evaluate
from .train import train, signature
from .storage import write_json, read_json, json_digest, load_checkpoint, preserve_file

ENTRY = ROOT / "scripts/run_31_paper.py"


def identity(pid=None):
    p = psutil.Process(pid or os.getpid())
    return dict(host_id=host_id(), pid=p.pid, created_at=p.create_time(), command=p.cmdline())


def alive(record):
    try:
        if record.get("host_id") != host_id():
            return False
        return record == identity(record["pid"]) and psutil.Process(record["pid"]).is_running()
    except (psutil.Error, KeyError):
        return False


def ensure_idle(options):
    """Do not mistake a missing supervisor for a stopped sampling worker."""
    paths = set()
    for root in (local_root(options),):
        paths.update(root.glob("**/workers/*.json"))
        paths.update(root.glob("**/active-workers.json"))
    for progress in paths:
        for worker in read_json(progress).get("workers", []):
            if alive(worker.get("identity", {})):
                raise RuntimeError(f"Existing sampling worker: {worker['identity']}; inspect {progress}")
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        args = p.info["cmdline"] or []
        entry_index = next((i for i, arg in enumerate(args[1:], 1) if Path(arg).name == ENTRY.name), None)
        command = args[entry_index + 1] if entry_index is not None and entry_index + 1 < len(args) else None
        # A suite option such as `matrix --suite formal` is not a training command.
        if p.pid not in (os.getpid(), os.getppid()) and "python" in (p.info["name"] or "").lower() and command in (
            "_replay", "_replay-file", "_execute", "pilot-single", "pilot-joint", "formal", "resume", "evaluate-final", "pilot", "supervisor", "smoke"
        ):
            raise RuntimeError(f"Existing experiment process: {p.info}")


def status(options):
    root = ROOT / options["result_root"]
    out = dict(processes=[], runs=[])
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        args = p.info["cmdline"] or []
        if "python" in (p.info["name"] or "").lower() and any(Path(a).name == ENTRY.name for a in args[1:]):
            out["processes"].append(p.info)
    for path in local_root(options).glob("**/supervisor.json"):
        value = read_json(path)
        value["owner_alive"] = alive(value.get("owner", {}))
        value["path"] = str(path)
        out["runs"].append(value)
    out["progress"] = []
    workers_by_folder = {r['folder']: r for file in local_root(options).glob('workers/*.json')
                         if (r := read_json(file)).get('folder')}
    for p in root.glob("**/progress.json"):
        if '_conflicts' in p.parts:
            continue
        progress = read_json(p)
        active_state = workers_by_folder.get(str(p.parent))
        if active_state:
            progress["workers"] = active_state["workers"]
            progress["owner_alive"] = alive(active_state["owner"])
        attempts = sorted(p.parent.glob("attempts/epoch_*/started.json"))
        if attempts:
            started = read_json(attempts[-1])
            completed = [read_json(f) for f in (attempts[-1].parent / "environment-results").glob("*.json")]
            progress["latest_attempt"] = dict(epoch=started["epoch"], instances=started.get("instances", started.get("instance_id")),
                returned_environments=len(completed), feasible_environments=sum(r["feasible"] for r in completed),
                committed=started["epoch"] <= progress["epoch"], path=str(attempts[-1].parent))
        for worker in progress.get("workers", []):
            worker["identity_alive"] = alive(worker.get("identity", {}))
        out["progress"].append(dict(path=str(p), **progress))
    return out


def spawn(arguments, output, detached=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Child processes inherit the parent's accelerator visibility; replays and
    # the dashboard are CPU-only by construction of their own code paths.
    kwargs = dict(cwd=ROOT, stdin=subprocess.DEVNULL, close_fds=True,
                  env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        if detached:
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startup
    elif detached:
        kwargs["start_new_session"] = True
    preserve_file(output.with_suffix(".stdout.log"))
    preserve_file(output.with_suffix(".stderr.log"))
    with output.with_suffix(".stdout.log").open("wb") as out, output.with_suffix(".stderr.log").open("wb") as err:
        return subprocess.Popen([sys.executable, "-u", str(ENTRY), *arguments], stdout=out, stderr=err, **kwargs)


def baseline(options, row):
    folder = local_root(options) / "baselines" / row["instance_id"]
    sig = json_digest(dict(instance=row["sha256"], protocol_hash=protocol_hash(options)))
    output = folder / f"{sig}.json"
    if output.exists():
        result = read_json(output)
    else:
        result = evaluate(load(row), options, output, method="Periodic-ATC")
    if not result["feasible"]:
        raise RuntimeError(f"Invalid baseline: {output}: {result['failure']}")
    return result["twt"]


def replay_job(options, job, training_job):
    from environment.env import AssemblyEnv
    from .network import create_network
    folder = ROOT / training_job["output"]
    snapshot = read_json(folder / "config_snapshot.json")
    if snapshot["source"]["digest"] != source_manifest()["digest"]:
        raise ValueError("Replay source differs from training source")
    kind = job["checkpoint_kind"]
    if kind not in ("latest", "best"):
        raise ValueError("Only the final committed policy or the development-selected policy may be evaluated")
    if "budget_epochs" in training_job:
        payload = validate_training(options, training_job, folder, require_complete=True)
    else:  # Engineering fixtures use no registered long-training budget.
        payload = load_checkpoint(folder / "latest.pt", snapshot["signature"])
    if kind == "best":
        # Selected on the registered development split only, never on main/L.
        if not (folder / "best.pt").exists():
            raise FileNotFoundError("No development-selected checkpoint was written by this training job")
        best = load_checkpoint(folder / "best.pt", snapshot["signature"])
        if best.get("kind") != "best_validation_policy" or best["epoch"] > payload["epoch"]:
            raise ValueError("Invalid development-selected checkpoint")
        payload = dict(best, events=payload["events"])
    reference = load(training_job["training"][0])
    env = AssemblyEnv(project_config(options))
    env.reset(reference)
    network = create_network(options, env.graph(), job["method"])
    network.load_state_dict(payload["network"], strict=True)
    output = ROOT / job["output"]
    results = []
    for row in job["evaluation"]:
        b = baseline(options, row)
        result = evaluate(load(row), options, output / f"{row['instance_id']}.json", method=job["method"], seed=job["seed"],
                          network=network, baseline=b, timing_scope="independent_process")
        results.append(dict(result, instance=row, checkpoint_kind=kind,
                            selected_events=payload["events"], trained_events=read_json(folder / "progress.json")["committed_events"],
                            epoch=payload["epoch"], selected_episodes=payload["epoch"] * options["environments"],
                            trained_episodes=read_json(folder / "progress.json")["committed_episodes"],
                            selection=payload.get("validation")))
    preserve_file(output / "results.json")
    write_json(output / "results.json", results)


def execute_job(options, job, plan, resume):
    output = ROOT / job["output"]
    if job["kind"] == "train":
        cases = {r["instance_id"]: r for r in job["training"] + job.get("validation", [])}
        bs = {iid: baseline(options, row) for iid, row in cases.items()}
        sig = signature(options, job["method"], job["seed"], job["training"], job.get("validation", []), bs)
        write_json(output / "baselines.json", bs)
        result = train(options, job["method"], job["seed"], [load(r) for r in job["training"]],
            [load(r) for r in job.get("validation", [])], bs, output, job["budget_epochs"], sig, resume)
        write_json(output / "training-result.json", result)
    elif job["kind"] == "replay":
        # Each repeat is a fresh interpreter, running alone after its training job.
        import yaml
        work = local_root(options) / 'replays' / (job['id'] + '_' + str(time.time_ns()))
        work.mkdir(parents=True)
        config = work / 'options.yaml'
        config.write_text(yaml.safe_dump(options), 'utf-8')
        request = work / 'request.json'
        training_job = next(j for j in plan['jobs'] if j['id'] == job['training_job'])
        write_json(request, dict(job=job, training_job=training_job))
        p = spawn(['_replay-file', '--config', str(config), '--request', str(request)], output / 'process')
        write_json(work / 'process.json', identity(p.pid))
        if p.wait() != 0:
            raise RuntimeError(f"Independent replay failed: {output / 'process.stderr.log'}")
    elif job["kind"] == "rule":
        results = []
        for row in job["evaluation"]:
            result = evaluate(load(row), options, output / f"{row['instance_id']}.json", method=job["method"], seed=job["seed"],
                              baseline=baseline(options, row), timing_scope="sequential_job")
            results.append(dict(result, instance=row))
        preserve_file(output / "results.json")
        write_json(output / "results.json", results)

    elif job["kind"] == "reference":
        from agent.baselines.optimization import OfflineMILPReferenceSolver, OptimizationBaselineConfig
        results = []
        for row in job["evaluation"]:
            try:
                instance = load(row)
                cfg = OptimizationBaselineConfig(offline_time_limit_seconds=options["reference_seconds"])
                dwell = float(project_config(options).instance.minimum_dwell_time_minutes.training_default)
                result = asdict(OfflineMILPReferenceSolver(cfg, minimum_dwell_time=dwell).solve(instance))
                result["reference_kind"] = ("proven_optimum" if result["dwell_exact"] and result["optimal"] else
                    "solver_feasible_solution" if result["dwell_exact"] and result["feasible"] else "relaxation_with_lower_bound")
                results.append(dict(result, failure=""))
            except Exception:
                results.append(dict(instance_id=row["instance_id"], failure=traceback.format_exc(), reference_kind="missing"))
        preserve_file(output / "results.json")
        write_json(output / "results.json", results)


def approve_formal(options, reviewed=False):
    from .report import report
    if not reviewed:
        raise RuntimeError('Explicit --human-reviewed-pilot is required')
    evidence = report(options, 'pilot-joint')
    if not evidence['complete'] or not evidence['admission']['passed']:
        raise RuntimeError('A complete, numerically passed pilot report is required before formal authorization')
    approval = dict(protocol_hash=protocol_hash(options), run_id=options['run_id'], acknowledged=True,
                    report=evidence, report_digest=json_digest(evidence), time=time.time(), host=host_id())
    path = suite_root(options, 'formal') / '_approvals' / (options['run_id'] + '.json')
    write_json(path, approval)
    return dict(path=str(path), protocol_hash=approval['protocol_hash'])


def check_formal_approval(options, *, reviewed=False, approval_file=None):
    if reviewed:
        approve_formal(options, True)
    path = Path(approval_file) if approval_file else suite_root(options, 'formal') / '_approvals' / (options['run_id'] + '.json')
    if not path.exists():
        raise RuntimeError('Formal work requires a reviewed pilot authorization; use authorize-formal and transfer its --approval-file')
    a = read_json(path)
    if (a['protocol_hash'] != protocol_hash(options) or a['run_id'] != options['run_id'] or not a['acknowledged']
        or a['report_digest'] != json_digest(a['report']) or not a['report']['complete'] or not a['report']['admission']['passed']):
        raise ValueError('Invalid or incompatible human review evidence')
    return a


@contextmanager
def exclusive_lease(path):
    """An OS file lock prevents duplicate jobs even if process inspection is restricted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another experiment holds the local execution lease') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_suite(options, suite, **kwargs):
    with exclusive_lease(local_root(options) / 'execution.lease'):
        return _run_suite_locked(options, suite, **kwargs)


def _run_suite_locked(options, suite, *, phase='train', resume=False, reviewed=False, approval_file=None,
              methods=None, seeds=None, job_ids=None):
    from .report import report
    if suite == 'formal':
        check_formal_approval(options, reviewed=reviewed, approval_file=approval_file)
    ensure_idle(options)
    local = local_root(options)
    local.mkdir(parents=True, exist_ok=True)
    lock = local / 'experiment.lock'
    if lock.exists():
        owner = read_json(lock)
        if alive(owner) or not resume:
            raise RuntimeError('Local experiment lock exists; inspect status and explicitly resume')
        write_json(local / 'evidence' / f'stale-lock-{time.time_ns()}.json', owner)
        lock.unlink()
    with lock.open('x', encoding='utf-8') as f:
        json.dump(identity(), f)
    try:
        prepare(options)
        plan = matrix(options, suite)
        jobs = select_jobs(plan, phase=phase, methods=methods, seeds=seeds, job_ids=job_ids)
        write_json(local / suite / 'run-matrix.json', plan)
        write_json(local / suite / 'selection.json', dict(phase=phase, jobs=[j['id'] for j in jobs]))
        root = suite_root(options, suite)
        registered_protocol = root.parent / 'protocol.json'
        expected_protocol = dict(protocol_hash=plan['protocol_hash'], record=protocol_record(options))
        if registered_protocol.exists() and read_json(registered_protocol) != expected_protocol:
            raise ValueError('Protocol directory prefix collision; full SHA256 differs')
        write_json(registered_protocol, expected_protocol)
        if phase == 'evaluate':
            # This portable declaration pins every timed evaluation to one host.
            host_path = root.parent / '_aggregation_host.json'
            measured_host = hardware(options)
            if host_path.exists() and read_json(host_path) != measured_host:
                raise RuntimeError('Final evaluations must use the registered aggregation hardware/software configuration')
            write_json(host_path, measured_host)
        for job in jobs:
            output = Path(job['output'])
            state_path = output / 'job-status.json'
            state = read_json(state_path) if state_path.exists() else {}
            if state.get('status') == 'completed':
                if job['kind'] == 'train':
                    validate_training(options, job, output, require_complete=True)
                continue
            missing = [d for d in job['depends'] if read_json_or_missing(
                Path(next(j for j in plan['jobs'] if j['id'] == d)['output']) / 'job-status.json').get('status') != 'completed']
            if missing:
                write_json(state_path, dict(status='missing_dependency', dependencies=missing))
                continue
            registration = dict(format=FORMAT, protocol_hash=plan['protocol_hash'], job=job_identity(job))
            record_path = output / 'run.json'
            if record_path.exists() and read_json(record_path) != registration:
                raise ValueError('Run-ID collision with different registered job')
            write_json(record_path, registration)
            host_path = output / 'execution' / (host_id() + '.json')
            write_json(host_path, hardware(options))
            write_json(local / suite / 'supervisor.json', dict(owner=identity(), active=job['id'], phase=phase, status='running'))
            started = time.time()
            try:
                execute_job(options, job, plan, resume)
                status_record = dict(status='completed', started=started, ended=time.time(), host=host_id())
            except Exception:
                status_record = dict(status='failed', started=started, ended=time.time(), host=host_id(), failure=traceback.format_exc())
                write_json(output / 'failures' / f'{time.time_ns()}.json', status_record)
            preserve_file(state_path)
            write_json(state_path, status_record)
            if job['kind'] == 'train':
                seal(output)
        write_json(local / suite / 'supervisor.json', dict(owner=identity(), status='finished', active=None, phase=phase))
        return report(options, suite)
    finally:
        if lock.exists() and read_json(lock) == identity():
            lock.unlink()


def read_json_or_missing(path):
    return read_json(path) if Path(path).exists() else dict(status='missing')
