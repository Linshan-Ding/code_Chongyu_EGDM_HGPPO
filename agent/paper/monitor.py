"""Independent durable-log consumer. All primary axes are epoch / original TWT."""
import json
import socket
import time
from collections import defaultdict
from pathlib import Path
from .visdom_transport import client, send_figure, verify_service, number, horizontal
from .storage import read_csv, read_json, write_json, committed_bundles
from .rollout import batch_summary
from .config import ROOT
from .experiment import local_root
from .runtime import alive, spawn, identity
from .report import csv_file


def trace(rows, key, name, success=False, mode="lines+markers"):
    good = [r for r in rows if not success or str(r.get("feasible")).lower() == "true"]
    pairs = [(number(r.get("epoch")), number(r.get(key))) for r in good]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    return dict(type="scatter", mode=mode, name=name, x=[x for x, _ in pairs], y=[y for _, y in pairs])


def figures(folder):
    folder = Path(folder)
    if not (folder / "baselines.json").exists():
        return {}
    baselines = read_json(folder / "baselines.json")
    bundles = committed_bundles(folder)
    eps = [r for b in bundles for r in b["episodes"]]
    evals = [r for b in bundles for r in b["evaluations"]]
    logs = [r for b in bundles for r in b["summaries"]]
    # Audit every displayed batch against its raw N records, never partial means.
    for b in bundles:
        if b["summaries"]:
            summary = b["summaries"][0]
            recomputed = batch_summary(b["episodes"], summary["environments"])
            if any(summary[k] != v for k, v in recomputed.items()):
                raise ValueError("Committed curve differs from raw complete episodes")
    all_failures = [read_json(p) for p in sorted((folder / "failures").glob("*.json"))]
    failures = [r for r in all_failures if "instance_id" in r]
    job_failures = [r for r in all_failures if "instance_id" not in r]
    # Only instances represented in this training job; no cross-scale raw average.
    output = {}
    for iid in sorted(set(baselines) | {r["instance_id"] for r in eps + failures}):
        episodes = [r for r in eps if r["instance_id"] == iid]
        evaluations = [r for r in evals if r["instance_id"] == iid]
        means = [r for r in logs if r["instance_id"] == iid]
        failed = [r for r in failures if r["instance_id"] == iid]
        maximum = max([int(r["epoch"]) for r in episodes + evaluations + failed] + [0])
        b = baselines[iid]
        band = [dict(r, lower=r["mean_twt"] - r["std_twt"], upper=r["mean_twt"] + r["std_twt"]) for r in means]
        lower = dict(trace(band, "lower", "Mean - std (population)", mode="lines"), line=dict(width=0), showlegend=False)
        upper = dict(trace(band, "upper", "Training mean ± std (not CI)", mode="lines"),
                     fill="tonexty", fillcolor="rgba(31,119,180,0.18)", line=dict(width=0))
        mean_trace = trace(means, "mean_twt", "Batch mean TWT (pre-update frozen policy)")
        # A single epoch has no horizontal band area; retain its visible spread.
        if len(means) == 1:
            mean_trace["error_y"] = dict(type="data", array=[means[0]["std_twt"]], visible=True)
        traces = [lower, upper, trace(episodes, "twt", "Raw complete rollouts (pre-update)", True, "markers"),
                  mean_trace,
                  horizontal(b, "Periodic-ATC beta=1", maximum), horizontal(.95 * b, "5% improvement target", maximum)]
        failed += [r for r in evaluations if not r["feasible"]]
        if failed:
            traces.append(dict(type="scatter", mode="markers", name="FAILED (baseline marker, not objective)",
                x=[int(r["epoch"]) for r in failed], y=[b] * len(failed), text=[r.get("failure", "") for r in failed],
                marker=dict(symbol="x", color="red", size=12)))
        output[iid + "_TWT"] = dict(data=traces, layout=dict(title=f"{folder.name} | {iid}",
            xaxis=dict(title="Training epoch"), yaxis=dict(title="TWT"), showlegend=True))
        output[iid + "_auxiliary"] = dict(data=[trace(episodes, k, k, True) for k in
            ("gate_fraction", "moves", "travel", "anchor_fraction")],
            layout=dict(title=f"{iid} | behavior and gap", xaxis=dict(title="Training epoch"), yaxis=dict(title="Value")))
    output["diagnostics"] = dict(data=[trace(logs, k, k) for k in
        ("policy_loss", "value_loss", "kl", "entropy", "actor_grad_norm", "critic_grad_norm", "events_per_second", "failures")],
        layout=dict(title=f"{folder.name} | training diagnostics", xaxis=dict(title="Training epoch"), yaxis=dict(title="Value")))
    if job_failures:
        last_epoch = max([b["epoch"] for b in bundles] + [0])
        output["job_failures"] = dict(data=[dict(type="scatter", mode="markers", name="Job failure (not TWT)",
            x=[last_epoch] * len(job_failures), y=[1] * len(job_failures),
            text=[r.get("failure", "Unknown job failure") for r in job_failures], marker=dict(symbol="x", color="red"))],
            layout=dict(title=folder.name + " | job-level failures", xaxis=dict(title="Last committed epoch (marker)"),
                        yaxis=dict(title="Failure marker")))
    return output


def export(options, *, render=True):
    root = ROOT / options["result_root"]
    count = 0
    if render:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    for snapshot in root.glob("**/config_snapshot.json"):
        if "_conflicts" in snapshot.parts:
            continue
        folder = snapshot.parent
        for name, fig in figures(folder).items():
            output = folder / "curves" / name
            write_json(output.with_suffix(".json"), fig)
            points = [dict(series=t["name"], epoch=x, value=y) for t in fig["data"] for x, y in zip(t["x"], t["y"])]
            csv_file(output.with_suffix(".csv"), points)
            if render:
                chart, axis = plt.subplots(figsize=(10, 5))
                for i, t in enumerate(fig["data"]):
                    if t.get("fill") == "tonexty":
                        axis.fill_between(t["x"], fig["data"][i - 1]["y"], t["y"], alpha=.18, label=t["name"])
                        continue
                    if t.get("showlegend") is False:
                        continue
                    mode = t.get("mode", "lines")
                    style = "." if mode == "markers" else ".-" if "markers" in mode else "--" if t.get("line", {}).get("dash") else "-"
                    if t.get("error_y", {}).get("visible"):
                        axis.errorbar(t["x"], t["y"], yerr=t["error_y"]["array"], fmt=style, capsize=4, label=t["name"])
                    else:
                        axis.plot(t["x"], t["y"], style, label=t["name"])
                axis.set(xlabel="Training epoch", ylabel=fig["layout"]["yaxis"]["title"], title=fig["layout"]["title"])
                axis.legend(fontsize=7)
                chart.tight_layout()
                chart.savefig(output.with_suffix(".png"), dpi=150)
                chart.savefig(output.with_suffix(".pdf"))
                plt.close(chart)
            count += 1
    return count


def start_server(options, config_path):
    import yaml
    root = local_root(options)
    root.mkdir(parents=True, exist_ok=True)
    # Include command-line execution overrides in the spawned service too.
    effective_config = root / 'visdom-options.yaml'
    effective_config.write_text(yaml.safe_dump(options), 'utf-8')
    path = root / "visdom.json"
    if path.exists():
        prior = read_json(path)
        if alive(prior["server"]):
            verify_service(prior["port"], root)
            return prior
    for port in range(options["visdom_port"], options["visdom_port"] + 11):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        p = spawn(["_server", "--config", str(effective_config), "--port", str(port)], root / f"visdom-{time.time_ns()}", detached=True)
        record = identity(p.pid)
        errors = []
        for _ in range(45):
            if p.poll() is not None:
                raise RuntimeError("Visdom server exited; inspect its stderr log")
            try:
                url = verify_service(port, root)
                result = dict(url=url, port=port, server=record)
                write_json(path, result)
                return result
            except Exception as exc:
                errors.append(repr(exc))
                time.sleep(.5)
        write_json(root / f"visdom-start-failure-{time.time_ns()}.json", dict(server=record, port=port, errors=errors))
        if alive(record):
            p.terminate()
        raise RuntimeError(f"Visdom failed frontend/transport acceptance: {errors[-1] if errors else 'unknown'}")
    raise RuntimeError("No available Visdom port in the registered range")


def monitor(options, port, *, once=False):
    root = ROOT / options["result_root"]
    state_path = local_root(options) / "monitor.json"
    state = read_json(state_path) if state_path.exists() else dict(gaps=[], pushes=0)
    while True:
        try:
            vis = client(port)
            for snapshot in root.glob("**/config_snapshot.json"):
                if "_conflicts" in snapshot.parts:
                    continue
                folder = snapshot.parent
                env = "paper-" + folder.relative_to(root).as_posix().replace("/", "-")
                for name, fig in figures(folder).items():
                    send_figure(vis, fig, win=name, env=env)
                    write_json(folder / "curves" / (name + ".json"), fig)
                vis.save([env])
            if state.get("gap_started"):
                state["gaps"].append(dict(start=state.pop("gap_started"), end=time.time()))
            state.update(connected=True, last_success=time.time(), error=None, pushes=state["pushes"] + 1)
        except Exception as exc:
            state.setdefault("gap_started", time.time())
            state.update(connected=False, error=repr(exc))
        write_json(state_path, state)
        if once:
            return state
        time.sleep(options["monitor_seconds"])
