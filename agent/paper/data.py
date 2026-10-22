"""Registered fixed data; restore missing files only when their original hash matches."""
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from data.generator import InstanceGenerator
from data.io import load_instance_csv, save_instance_csv
from .storage import read_json, write_json
from .config import ROOT, digest, project_config

COUNTS = dict(pilot=2, train=36, validation=10, main=24, generalization=12, tiny=8)
DATA_PROTOCOLS = ("paper-sm100-1", "paper-sm100-2")
# Development-pilot cases (paper-sm100-2 registry): id -> sha256 of the frozen file.
PILOT = {
    "S_D3_rho72p5_tau100": "483e7ebc6a2a199e34ff60893a84f81ca558655a870f953afb30f25c47758d8e",
    "M_D3_rho72p5_tau100": "906cb59dc204f7b2f5a0cb63a7abb86f68e4584ad0a7870bd5a1634ace7b46cd",
}


def _percent(value):
    """0.7 -> '70', 0.725 -> '72p5', 1.25 -> '125': percent, decimal point written as 'p'."""
    x = float(value) * 100.0
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}".replace(".", "p")


def case_id(parameters, split):
    """Paper-citable case name derived from the design parameters.

    ``<scale>_<scenario>_rho<load ratio x100>_tau<relocation multiplier x100>``; engineering
    fixtures are ``tiny_S<stages>_J<orders>_rho<load x100>``. Every registered design unit has
    distinct parameters, so the name is unique and says what the case is.
    """
    p = parameters
    rho = _percent(p["load_ratio"])
    if split == "tiny":
        ip = p["instance_parameters"]
        return f"tiny_S{ip['S']}_J{ip['J']}_rho{rho}"
    return f"{p['scale']}_{p['scenario']}_rho{rho}_tau{_percent(p['relocation_multiplier'])}"


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
    if any(r["instance_id"] != case_id(r["parameters"], r["split"]) for r in records):
        raise ValueError("Registered case ids must be derived from the design parameters")
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


def rename_registry(options):
    """One-time migration: give every active case its parameter-derived name.

    Each case is regenerated from its registered parameters and seed under the new id,
    compared field by field with the frozen file (only the id may differ), written to
    the new path and re-hashed. Removed-data audit records keep their historical ids.
    """
    target = (ROOT / options["data_manifest"]).resolve()
    manifest = read_json(target)
    cfg = project_config(options)
    if manifest["generator_config"] != cfg.instance.to_dict():
        raise ValueError("Generator configuration changed; refuse to regenerate frozen instances")
    generator = InstanceGenerator(cfg)
    records, old_files, renamed = [], [], {}
    for r in manifest["instances"]:
        new_id = case_id(r["parameters"], r["split"])
        storage = r.get("storage_split", r["split"])
        new_path = target.parent / storage / f"{new_id}.csv"
        old_file = (ROOT / r["path"]).resolve()
        old = asdict(load_instance_csv(old_file))
        regenerated = generator.sample(**r["parameters"], seed=r["seed"], instance_id=new_id)
        fresh = asdict(regenerated)
        differing = {k for k in old if old[k] != fresh[k]}
        if not differing <= {"instance_id", "generation_meta"}:
            raise ValueError(f"{r['instance_id']}: regeneration changed {sorted(differing)}")
        meta_old, meta_new = old.get("generation_meta") or {}, fresh.get("generation_meta") or {}
        if {k for k in set(meta_old) | set(meta_new) if meta_old.get(k) != meta_new.get(k)} - {"instance_id"}:
            raise ValueError(f"{r['instance_id']}: generation metadata changed beyond the id")
        if new_path != old_file:
            save_instance_csv(regenerated, new_path)
            old_files.append(old_file)
        record = dict(r, instance_id=new_id, path=new_path.relative_to(ROOT).as_posix(), sha256=digest(new_path))
        records.append(record)
        renamed[r["instance_id"]] = new_id
    if len({r["instance_id"] for r in records}) != len(records):
        raise ValueError("Parameter-derived case ids collide")
    for old_file in old_files:
        if old_file.exists():
            old_file.unlink()
    manifest.update(protocol="paper-sm100-2", instances=records,
                    naming="<scale>_<scenario>_rho<load x100>_tau<relocation multiplier x100>; tiny_S<stages>_J<orders>_rho<load x100>; 'p' is the decimal point")
    write_json(target, manifest)
    return dict(renamed=renamed, pilot={r["instance_id"]: r["sha256"] for r in records if r["split"] == "pilot"})
