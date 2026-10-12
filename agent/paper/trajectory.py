"""Successful rollout journal: reconstruct graphs from immutable data + actions."""
import gzip
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
from environment.state import CompositeAction, ResourceAssignment, ScheduleAssignment


def physical_action(value):
    return CompositeAction(reconfigure=value["reconfigure"],
        worker_assignments=tuple(ResourceAssignment(**r) for r in value["worker_assignments"]),
        robot_assignments=tuple(ResourceAssignment(**r) for r in value["robot_assignments"]),
        schedule_assignments=tuple(ScheduleAssignment(**r) for r in value["schedule_assignments"]))


def write_trajectory(path, records, signature):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=3) as output:
        def emit(row):
            line = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            digest.update(line.encode())
            output.write(line)
        emit(dict(format="paper-event-journal-1", signature=signature,
                  graph_storage="reconstruct from immutable instance and physical actions"))
        for r in records:
            obs = r.observation
            if hasattr(obs, "prior"):
                candidates = dict(identifiers=obs.identifiers, prior=obs.prior.tolist(), groups=obs.groups.tolist(),
                    features=obs.features.tolist(), stages=obs.stages.tolist(), state=obs.state.tolist())
            else:
                candidates = dict(tokens=[dict(sources=t.sources, targets=t.targets, pairs=t.pairs.tolist(),
                                               identifiers=t.identifiers) for t in obs.tokens])
            emit(dict(epoch=r.episode, env_id=r.env_id, event=r.event, instance_id=r.instance_id,
                policy_version=r.version, action_index=r.action, logp=r.logp, value=r.value, reward=r.reward,
                done=r.done, choice=r.choice, details=r.details, candidates=candidates))
    return dict(events=len(records), content_sha256=digest.hexdigest(), compressed_bytes=path.stat().st_size)


def read_trajectory(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(next(f))
        if header["format"] != "paper-event-journal-1":
            raise ValueError("Incompatible event journal")
        yield from (json.loads(line) for line in f)
