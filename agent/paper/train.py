import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from environment.env import AssemblyEnv
from .config import project_config, source_manifest, EPOCH_SEMANTICS, scientific_options
from .network import create_network
from .rollout import Pool, batch_summary
from .ppo import PPO
from .profiling import PeakMemory
from .trajectory import write_trajectory
from .storage import (write_json, json_digest, load_checkpoint, save_checkpoint,
                      rng_state, restore_rng, publish_commit, read_json)


def signature(options, method, seed, training_rows, validation_rows, baselines):
    from .experiment import case_identity
    if validation_rows:
        raise ValueError("Training cannot receive validation/test rows")
    return json_digest(dict(options=scientific_options(options), method=method, seed=seed,
                            train=[case_identity(r) for r in training_rows], baselines=baselines,
                            source=source_manifest()["digest"]))


def train(options, method, seed, instances, validation, baselines, folder, target, sig, resume=False):
    if options.get("epoch_semantics") != EPOCH_SEMANTICS or "complete_epochs_per_update" in options:
        raise ValueError("Incompatible training epoch protocol")
    if validation:
        raise ValueError("SM100 trains without any validation environment")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(options["update_threads"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = AssemblyEnv(project_config(options))
    env.reset(instances[0])
    network = create_network(options, env.graph(), method)
    ppo = PPO(network, options)
    events, epoch, update, elapsed = 0, 0, 0, 0.
    latest = folder / "latest.pt"
    state = None
    source = source_manifest()
    if latest.exists():
        if not resume:
            raise FileExistsError("Existing checkpoint: use explicit resume after checking status")
        state = load_checkpoint(latest, sig)
        if state["source_digest"] != source["digest"] or state["scientific_options"] != scientific_options(options):
            raise ValueError("Checkpoint source/options changed")
        if state["instance_hashes"] != [json_digest(asdict(i)) for i in instances]:
            raise ValueError("Checkpoint instances changed")
        network.load_state_dict(state["network"], strict=True)
        ppo.actor.load_state_dict(state["actor_optimizer"])
        ppo.critic.load_state_dict(state["critic_optimizer"])
        events, epoch, elapsed = (state[k] for k in ("events", "epoch", "training_seconds"))
        update = state["policy_version"]
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
                evidence = dict(epoch=start["epoch"], instance_id=start["instance_id"],
                    sampling_policy_version=start["policy_version"], committed_epoch=epoch, attempt=str(path.parent),
                    failure="Interrupted before atomic epoch commit; rerun entire batch",
                    episodes=[read_json(p) for p in sorted((path.parent / "environment-results").glob("*.json"))])
                write_json(path.parent / "failure.json", evidence)
                write_json(folder / "failures" / f"interrupted-{time.time_ns()}.json", evidence)
        # No publication changes on a completed-budget no-op resume.
        if epoch >= target:
            return dict(events=events, epoch=epoch, completed_episodes=epoch * options["environments"],
                        training_seconds=elapsed, checkpoint_kind="latest")
        publish_commit(folder, state, rebuild=True)
    write_json(folder / "config_snapshot.json", dict(options=options, method=method, seed=seed, signature=sig,
               source=source, target_epochs=target, target_episodes=target * options["environments"],
               instance_order=[i.instance_id for i in instances]))

    def checkpoint(bundle, terminal_states, workers):
        from dataclasses import asdict
        progress = dict(epoch=epoch, committed_events=events, committed_episodes=epoch * options["environments"],
            target_epochs=target, target_episodes=target * options["environments"], environments_per_epoch=options["environments"],
            training_seconds=elapsed, policy_version=update, checkpoint_kind="latest",
            epoch_semantics=EPOCH_SEMANTICS,
            last_summary=bundle["summaries"][0] if bundle["summaries"] else None)
        payload = dict(signature=sig, method=method, seed=seed, network=network.state_dict(), scientific_options=scientific_options(options),
            source_digest=source["digest"], instance_hashes=[json_digest(asdict(i)) for i in instances],
            actor_optimizer=ppo.actor.state_dict(), critic_optimizer=ppo.critic.state_dict(), terminal_states=terminal_states,
            rng=rng_state(), events=events, epoch=epoch, training_seconds=elapsed,
            policy_version=update, kind="latest_learning_policy", last_bundle=bundle, progress=progress)
        save_checkpoint(latest, payload)
        publish_commit(folder, payload)

    if state is None:
        checkpoint(dict(epoch=0, episodes=[], summaries=[], evaluations=[]), [], [])
    with Pool(options, instances, method, seed) as pool:
        from .runtime import identity
        workers = [dict(pid=p.pid, alive=p.is_alive(), identity=identity(p.pid)) for p, _ in pool.workers]
        # Live identities are separate from committed scientific progress.
        from .experiment import local_root
        live_path = local_root(options) / "workers" / (json_digest(str(folder.resolve())) + ".json")
        write_json(live_path, dict(owner=identity(), workers=workers, folder=str(folder)))
        while epoch < target:
            next_epoch, instance_index = epoch + 1, epoch % len(instances)
            attempt = folder / "attempts" / f"epoch_{next_epoch:06d}_{time.time_ns()}"
            attempt.mkdir(parents=True)
            write_json(attempt / "started.json", dict(epoch=next_epoch, instance_id=instances[instance_index].instance_id,
                       policy_version=update, owner=identity(), workers=workers, started=time.time()))
            started = time.perf_counter()
            episodes, records = [], []
            try:
                with PeakMemory() as memory:
                    records, bootstrap, episodes = pool.collect(network, update, next_epoch, instance_index, baselines,
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
                    metrics = ppo.update(records, bootstrap)
                    update_seconds = time.perf_counter() - update_start
                    next_events = events + len(records)
                iteration_seconds = time.perf_counter() - started
                summary.update(epoch=next_epoch, instance_id=instances[instance_index].instance_id, events=next_events,
                    sampling_policy_version=update, policy_version=update + 1, ppo_update_cycles=1,
                    ppo_update_epochs=options["ppo_update_epochs"], completed_episodes=next_epoch * options["environments"],
                    environments=options["environments"], trajectory_bytes=journal["compressed_bytes"], **metrics, sample_seconds=sample_seconds,
                    update_seconds=update_seconds, evaluation_seconds=0., instance_epoch=epoch // len(instances) + 1, iteration_seconds=iteration_seconds,
                    events_per_second=len(records) / (sample_seconds + update_seconds),
                    training_seconds=elapsed + iteration_seconds, peak_tree_rss_mib=memory.peak_bytes / 2**20, failures=0)
                bundle = dict(epoch=next_epoch, attempt=attempt.relative_to(folder).as_posix(), episodes=episodes,
                              summaries=[summary], evaluations=[])
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
                evidence = dict(tensor_journal_failure=tensor_error, epoch=next_epoch, instance_id=instances[instance_index].instance_id,
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
                training_seconds=elapsed, checkpoint_kind="latest")
