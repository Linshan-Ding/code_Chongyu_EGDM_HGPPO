"""Registered fixed data; restore missing files only when their original hash matches."""
import tempfile
from collections import Counter
from pathlib import Path

from data.generator import InstanceGenerator
from data.io import load_instance_csv, save_instance_csv
from .storage import read_json
from .config import ROOT, digest, project_config

COUNTS = dict(pilot=2, train=36, validation=10, main=24, generalization=12, tiny=8)
DATA_PROTOCOLS = ("paper-sm100-1", "paper-sm100-2")
PILOT = {
    "validation_S_D3_5fa810942029": "fb3b12ae526c956c8f841bdfcd2f7a3eb3562f28a40057ec966e0925fc54a82e",
    "validation_M_D3_1b424c3f1612": "6f3e9d8353b3023057f72ae0ce9ed4a5f6c70f0ff737f7a618667444539a90d0",
}


def registered_manifest(options):
    target = (ROOT / options["data_manifest"]).resolve()
    if not target.exists():
        raise FileNotFoundError("The registered fixed-data manifest is required; refuse to redraw a dataset")
    manifest = read_json(target)
    # The frozen case registry is shared by the training protocols that use it
    # unchanged; the registry file itself is never rewritten for a new protocol.
    if manifest["protocol"] not in DATA_PROTOCOLS or options["protocol"] not in DATA_PROTOCOLS:
        raise ValueError("Manifest protocol mismatch")
    records = manifest["instances"]
    if Counter(r["split"] for r in records) != COUNTS or manifest["counts"] != COUNTS:
        raise ValueError("Fixed design cardinality changed")
    if len({r["instance_id"] for r in records}) != 92 or len({r["path"] for r in records}) != 92:
        raise ValueError("Duplicate fixed instances")
    if {r["instance_id"]: r["sha256"] for r in records if r["split"] == "pilot"} != PILOT:
        raise ValueError("Pilot instance hashes changed")
    if manifest.get("inactive_instances", []):
        raise ValueError("Retired L data must be registered as removed, not inactive")
    removed = manifest.get("removed_instances", [])
    if (len(removed) != 36 or len({r['instance_id'] for r in removed}) != 36
            or Counter(r['parameters']['scale'] for r in removed) != dict(L=24, XL=12)
            or {r['instance_id'] for r in records} & {r['instance_id'] for r in removed}
            or {r['path'] for r in records} & {r['path'] for r in removed}):
        raise ValueError("Removed-data audit registration changed")
    for r in records + removed:
        path = (ROOT / r["path"]).resolve()
        if path != target.parent / r.get("storage_split", r["split"]) / (r["instance_id"] + ".csv"):
            raise ValueError("Fixed data must remain inside the manifest role directories")
    for r in records:
        scale = r['parameters']['scale']
        if scale == 'XL' or (r['split'] == 'generalization' and scale != 'L') or (r['split'] not in ('generalization', 'tiny') and scale not in ('S', 'M')):
            raise ValueError("Scale isolation violated")
    return manifest


def restore_registered_file(row, generator):
    destination = ROOT / row["path"]
    if destination.exists():
        if digest(destination) != row["sha256"]:
            raise ValueError(f"Frozen data changed: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        instance = generator.sample(**row["parameters"], seed=row["seed"], instance_id=row["instance_id"])
        generated = save_instance_csv(instance, Path(temporary) / "instance.csv")
        if digest(generated) != row["sha256"]:
            raise ValueError(f"Deterministic regeneration differs from registered hash: {row['instance_id']}")
        generated.replace(destination)


def prepare(options):
    manifest = registered_manifest(options)
    cfg = project_config(options)
    if manifest["generator_config"] != cfg.instance.to_dict():
        raise ValueError("Generator configuration changed; refuse to modify frozen instances")
    generator = InstanceGenerator(cfg)
    for row in manifest["instances"]:
        restore_registered_file(row, generator)
    return manifest["instances"]


def rows(options, split=None):
    records = registered_manifest(options)["instances"]
    return [r for r in records if split is None or r["split"] == split]


def load(row):
    path = ROOT / row["path"]
    if digest(path) != row["sha256"]:
        raise ValueError(f"Data hash mismatch: {path}")
    return load_instance_csv(path)
