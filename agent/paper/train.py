import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from environment.env import AssemblyEnv
from .config import project_config, source_manifest, EPOCH_SEMANTICS, LEGACY_KEYS, scientific_options
from .network import create_network
from .rollout import Pool, batch_summary, update_plan
from .ppo import PPO
from .profiling import PeakMemory
from .trajectory import write_trajectory
from .storage import (write_json, json_digest, load_checkpoint, save_checkpoint,
                      rng_state, restore_rng, publish_commit, read_json)


def signature(options, method, seed, training_rows, validation_rows, baselines):
    from .experiment import case_identity
    allowed = options.get("validation_split", "validation")
    if any(r["split"] != allowed for r in validation_rows):
        raise ValueError("Training may only validate on the registered development split")
    return json_digest(dict(options=scientific_options(options), method=method, seed=seed,
                            train=[case_identity(r) for r in training_rows],
                            validation=[case_identity(r) for r in validation_rows], baselines=baselines,
                            source=source_manifest()["digest"]))


def validation_summary(epoch, rows, best):
    """Greedy development score: geometric mean of TWT / Periodic-ATC TWT over all cases."""
    feasible = bool(rows) and all(r["feasible"] for r in rows)
    relative = [float(r["rel_twt"]) for r in rows] if feasible else []
    geomean = float(np.exp(np.mean(np.log(np.maximum(relative, 1e-9))))) if relative else None
    improved = geomean is not None and (best is None or geomean < best["rel_twt_geomean"] - 1e-12)
    return dict(epoch=epoch, instances=len(rows), feasible_instances=sum(bool(r["feasible"]) for r in rows),
                all_feasible=feasible, rel_twt_mean=float(np.mean(relative)) if relative else None,
                rel_twt_geomean=geomean, mean_twt=float(np.mean([r["twt"] for r in rows])) if feasible else None,
                gate_rate=float(np.mean([r["gate_fraction"] for r in rows])) if feasible else None,
                moves_per_case=float(np.mean([r["moves"] for r in rows])) if feasible else None,
                mean_decision_p95_ms=float(np.mean([r["decision_p95_ms"] for r in rows])) if feasible else None,
                improved=improved,
                best_epoch=epoch if improved else (best["epoch"] if best else None),
                best_rel_twt_geomean=geomean if improved else (best["rel_twt_geomean"] if best else None))


def train(options, method, seed, instances, validation, baselines, folder, target, sig, resume=False):
    if options.get("epoch_semantics") != EPOCH_SEMANTICS or any(k in options for k in LEGACY_KEYS):
        raise ValueError("Incompatible training epoch protocol")
    validation = list(validation or [])
    missing = [i.instance_id for i in list(instances) + validation if i.instance_id not in baselines]
    if missing:
        raise ValueError(f"Anchor baselines missing for cases: {missing}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(options["update_threads"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = AssemblyEnv(project_config(options))
    env.reset(instances[0])
    network = create_network(options, env.graph(), method)
    # The learner may update on an accelerator; sampling workers and the vendored
    # DAN stay on CPU. Checkpoints and worker weights are always CPU tensors.
    device = torch.device("cuda" if options.get("device") == "cuda" and torch.cuda.is_available() and method != "dan" else "cpu")
    network.to(device)
    ppo = PPO(network, options, total_updates=target)
    events, epoch, update, elapsed = 0, 0, 0, 0.
    best, last_validation = None, None
    latest = folder / "latest.pt"
    state = None
    source = source_manifest()
    every = int(options["validation_every_updates"])

    def due(e):
        return bool(validation) and (e % every == 0 or e == target)

    if latest.exists():
        if not resume:
            raise FileExistsError("Existing checkpoint: use explicit resume after checking status")
        state = load_checkpoint(latest, sig)
        if state["source_digest"] != source["digest"] or state["scientific_options"] != scientific_options(options):
            raise ValueError("Checkpoint source/options changed")
        if state["instance_hashes"] != [json_digest(asdict(i)) for i in instances]:
            raise ValueError("Checkpoint instances changed")
        network.load_state_dict(state["network"], strict=True)
        ppo.load_state_dict(state)
        events, epoch, elapsed = (state[k] for k in ("events", "epoch", "training_seconds"))
        update = state["policy_version"]
        best = state.get("best")
        last_validation = state["progress"].get("last_validation")
        if update != epoch or state["progress"]["committed_episodes"] != epoch * options["environments"]:
            raise ValueError("Invalid batch epoch/checkpoint counters")
        if epoch and (len(state["terminal_states"]) != options["environments"] or
                      not all(s["terminal"] and s["epoch"] == epoch for s in state["terminal_states"])):
            raise ValueError("Nonterminal committed environment state")
        restore_rng(state["rng"])
        marker = folder / "committed.json"
        repair = not marker.exists() or read_json(marker)["epoch"] != epoch
        if repair:
            publish_commit(folder, state, rebuild=True)
        # Abrupt process exits cannot write exception handlers. Preserve every
        # completed environment result already emitted by the workers.
        for path in sorted((folder / "attempts").glob("epoch_*/started.json")):
            start = read_json(path)
            if start["epoch"] > epoch and not (path.parent / "failure.json").exists():
                evidence = dict(epoch=start["epoch"], instances=start.get("instances"),
                    sampling_policy_version=start["policy_version"], committed_epoch=epoch, attempt=str(path.parent),
                    failure="Interrupted before atomic epoch commit; rerun entire batch",
                    episodes=[read_json(p) for p in sorted((path.parent / "environment-results").glob("*.json"))])
                write_json(path.parent / "failure.json", evidence)
                write_json(folder / "failures" / f"interrupted-{time.time_ns()}.json", evidence)
        # No publication changes on a completed-budget no-op resume.
        if epoch >= target:
            return dict(events=events, epoch=epoch, completed_episodes=epoch * options["environments"],
                        training_seconds=elapsed, checkpoint_kind="latest", best=best)
        publish_commit(folder, state, rebuild=True)
    write_json(folder / "config_snapshot.json", dict(options=options, method=method, seed=seed, signature=sig,
               source=source, target_epochs=target, target_episodes=target * options["environments"],
               instance_order=[i.instance_id for i in instances], validation_order=[i.instance_id for i in validation]))

    def identity_fields():
        return dict(signature=sig, method=method, seed=seed, scientific_options=scientific_options(options),
                    source_digest=source["digest"], instance_hashes=[json_digest(asdict(i)) for i in instances])

    def network_state():
        return {k: v.detach().cpu() for k, v in network.state_dict().items()}

    def checkpoint(bundle, terminal_states, workers):
        progress = dict(epoch=epoch, committed_events=events, committed_episodes=epoch * options["environments"],
            target_epochs=target, target_episodes=target * options["environments"], environments_per_epoch=options["environments"],
            instances_per_update=options["instances_per_update"], episodes_per_instance=options["episodes_per_instance"],
            training_seconds=elapsed, policy_version=update, checkpoint_kind="latest",
            epoch_semantics=EPOCH_SEMANTICS, best_validation=best, last_validation=last_validation,
            last_summary=bundle["summaries"][0] if bundle["summaries"] else None)
        payload = dict(identity_fields(), network=network_state(), **ppo.state_dict(), terminal_states=terminal_states,
            rng=rng_state(), events=events, epoch=epoch, training_seconds=elapsed,
            policy_version=update, kind="latest_learning_policy", best=best, last_bundle=bundle, progress=progress)
        save_checkpoint(latest, payload)
        publish_commit(folder, payload)

    def save_best(at_epoch, summary):
        payload = dict(identity_fields(), network=network_state(), epoch=at_epoch, policy_version=update,
                       kind="best_validation_policy", validation=summary)
        save_checkpoint(folder / "best.pt", payload)

    def validate(pool, at_epoch):
        nonlocal best, last_validation
        rows = pool.validate(network, at_epoch, baselines, folder / "validation" / f"epoch_{at_epoch:06d}")
        summary = validation_summary(at_epoch, rows, best)
        if summary["improved"]:
            best = dict(epoch=at_epoch, rel_twt_geomean=summary["rel_twt_geomean"])
            save_best(at_epoch, summary)
        last_validation = summary["rel_twt_geomean"]
        return rows, [summary]

    with Pool(options, instances, method, seed, validation) as pool:
        from .runtime import identity
        workers = [dict(pid=p.pid, alive=p.is_alive(), identity=identity(p.pid)) for p, _ in pool.workers]
        # Live identities are separate from committed scientific progress.
        from .experiment import local_root
        live_path = local_root(options) / "workers" / (json_digest(str(folder.resolve())) + ".json")
        write_json(live_path, dict(owner=identity(), workers=workers, folder=str(folder)))
        if state is None:
            evaluations, summaries = validate(pool, 0) if due(0) else ([], [])
            checkpoint(dict(epoch=0, episodes=[], summaries=[], evaluations=evaluations, validation_summaries=summaries), [], workers)
        while epoch < target:
            next_epoch = epoch + 1
            plan = update_plan(options, seed, next_epoch, len(instances))
            planned = sorted({instances[s["instance_index"]].instance_id for s in plan})
            attempt = folder / "attempts" / f"epoch_{next_epoch:06d}_{time.time_ns()}"
            attempt.mkdir(parents=True)
            write_json(attempt / "started.json", dict(epoch=next_epoch, instances=planned, plan=plan,
                       policy_version=update, owner=identity(), workers=workers, started=time.time()))
            started = time.perf_counter()
            episodes, records = [], []
            try:
                with PeakMemory() as memory:
                    records, bootstrap, episodes = pool.collect(network, update, next_epoch, plan, baselines,
                                                               evidence_dir=str(attempt))
                    sample_seconds = time.perf_counter() - started
                    summary = batch_summary(episodes, options["environments"])
                    write_json(attempt / "episodes.json", episodes)
                    journal = write_trajectory(attempt / "events.jsonl.gz", records, sig)
                    write_json(attempt / "trajectory.json", journal)
                    if options.get("save_full_rollouts", False):
                        save_checkpoint(attempt / "rollout.pt", dict(signature=sig, epoch=next_epoch,
                            records=records, bootstrap=bootstrap, policy_version=update))
                    update_start = time.perf_counter()
                    metrics = ppo.update(records, bootstrap, episodes=episodes, step=update)
                    update_seconds = time.perf_counter() - update_start
                    next_events = events + len(records)
                    evaluation_start = time.perf_counter()
                    evaluations, validation_rows = validate(pool, next_epoch) if due(next_epoch) else ([], [])
                    evaluation_seconds = time.perf_counter() - evaluation_start
                iteration_seconds = time.perf_counter() - started
                summary.update(epoch=next_epoch, events=next_events,
                    sampling_policy_version=update, policy_version=update + 1, ppo_update_cycles=1,
                    ppo_update_epochs=options["ppo_update_epochs"], completed_episodes=next_epoch * options["environments"],
                    environments=options["environments"], trajectory_bytes=journal["compressed_bytes"], **metrics, sample_seconds=sample_seconds,
                    update_seconds=update_seconds, evaluation_seconds=evaluation_seconds, iteration_seconds=iteration_seconds,
                    events_per_second=len(records) / (sample_seconds + update_seconds),
                    training_seconds=elapsed + iteration_seconds, peak_tree_rss_mib=memory.peak_bytes / 2**20, failures=0,
                    validation_rel_twt_geomean=last_validation, best_rel_twt_geomean=best["rel_twt_geomean"] if best else None,
                    best_epoch=best["epoch"] if best else None)
                bundle = dict(epoch=next_epoch, attempt=attempt.relative_to(folder).as_posix(), episodes=episodes,
                              summaries=[summary], evaluations=evaluations, validation_summaries=validation_rows)
                write_json(attempt / "completed.json", bundle)
                # No epoch/counter publication before all terminal episodes AND the
                # PPO cycle succeed. The atomic latest checkpoint is the commit.
                epoch, update, events, elapsed = next_epoch, update + 1, next_events, elapsed + iteration_seconds
                checkpoint(bundle, pool.terminal_states, workers)
                del records
            except BaseException as exc:
                tensor_error = None
                if records and not (attempt / "rollout.pt").exists():
                    try:
                        save_checkpoint(attempt / "failed-rollout.pt", dict(signature=sig, epoch=next_epoch,
                            records=records, bootstrap=bootstrap, policy_version=records[0].version))
                    except Exception:
                        tensor_error = traceback.format_exc()
                evidence = dict(tensor_journal_failure=tensor_error, epoch=next_epoch, instances=planned,
                    sampling_policy_version=records[0].version if records else update, committed_epoch=(state["epoch"] if state else 0),
                    attempt=str(attempt), failure=traceback.format_exc(),
                    episodes=getattr(exc, "evidence", {}).get("episodes", episodes),
                    worker_evidence=getattr(exc, "evidence", None), elapsed_seconds=time.perf_counter() - started)
                # Read only metadata already published by the last successful commit.
                evidence["committed_epoch"] = read_json(folder / "committed.json")["epoch"]
                write_json(attempt / "failure.json", evidence)
                write_json(folder / "failures" / f"{time.time_ns()}.json", evidence)
                raise
    return dict(events=events, epoch=epoch, completed_episodes=epoch * options["environments"],
                training_seconds=elapsed, checkpoint_kind="latest", best=best)
