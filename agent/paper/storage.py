import os
import shutil
import time
import torch
import csv
from pathlib import Path
import hashlib
import json
import random
import uuid
import numpy as np

def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Monitor and final reporter can publish the same curve concurrently.
    # Each writer needs its own temporary file before the atomic replacement.
    temp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    for attempt in range(100):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            # Windows can briefly hold the destination open during another
            # publisher's rename or a dashboard read. Retry without truncation.
            if os.name != "nt" or attempt == 99:
                raise
            time.sleep(0.01)

def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))

def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as f:
            names = next(csv.reader(f))
        if set(names) != set(row):
            raise ValueError(f"CSV schema changed: {path}")
    else:
        names = list(row)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()

def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
from .config import FORMAT, EPOCH_SEMANTICS


def preserve_file(path):
    """Archive a prior attempt before replacing an evaluation or process log."""
    path = Path(path)
    if path.exists():
        archive = path.parent / "attempts" / f"{time.time_ns()}_{path.name}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(dict(payload, format=FORMAT, epoch_semantics=EPOCH_SEMANTICS), temp)
    os.replace(temp, path)


def load_checkpoint(path, signature):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if (data.get("format") != FORMAT or data.get("epoch_semantics") != EPOCH_SEMANTICS
        or data.get("signature") != signature):
        raise ValueError("Incompatible paper checkpoint: format/source/data/options/baseline mismatch")
    return data


def committed_bundles(folder):
    """Dashboard reads atomic committed bundles, never an in-flight CSV append."""
    folder = Path(folder)
    marker = folder / "committed.json"
    if not marker.exists():
        return []
    epoch = read_json(marker)["epoch"]
    return [read_json(folder / "epochs" / f"{k:06d}.json") for k in range(epoch + 1)]


def publish_commit(folder, state, *, rebuild=False):
    """latest.pt is the authority; recover publication after a post-save interruption."""
    folder = Path(folder)
    bundle = state["last_bundle"]
    path = folder / "epochs" / f"{state['epoch']:06d}.json"
    if path.exists() and read_json(path) != bundle:
        preserve_file(path)
    write_json(path, bundle)
    mapping = {"episodes.csv": "episodes", "log.csv": "summaries", "evaluations.csv": "evaluations"}
    if rebuild:
        bundles = [read_json(folder / "epochs" / f"{k:06d}.json") for k in range(state["epoch"] + 1)]
        for filename, key in mapping.items():
            target = folder / filename
            rows = [r for b in bundles for r in b[key]]
            preserve_file(target)
            temp = target.with_suffix(".csv.tmp")
            with temp.open("w", newline="", encoding="utf-8") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            os.replace(temp, target)
    else:
        for filename, key in mapping.items():
            for row in bundle[key]:
                append_csv(folder / filename, row)
    progress = state["progress"]
    write_json(folder / "progress.json", progress)
    write_json(folder / "committed.json", dict(epoch=state["epoch"], episodes=progress["committed_episodes"],
               events=state["events"], policy_version=state["policy_version"], format=FORMAT))
