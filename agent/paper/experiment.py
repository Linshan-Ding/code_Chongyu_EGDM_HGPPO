"""Portable scientific identities; local process state is never imported."""
import platform
import re
import socket
import sys
from pathlib import Path
import psutil
import torch
from .config import ROOT, FORMAT, scientific_options, source_manifest
from .data import registered_manifest
from .storage import json_digest


def token(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value) or value in (".", ".."):
        raise ValueError("IDs must use letters, digits, underscore, hyphen or dot")
    return value


def host_id():
    return re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())


def local_root(options):
    return ROOT / options["result_root"] / "_local" / host_id()


def case_identity(row):
    return {k: row[k] for k in ("instance_id", "split", "seed", "parameters", "sha256")}


def protocol_record(options):
    return dict(format=FORMAT, options=scientific_options(options), source=source_manifest(),
                data=[case_identity(r) for r in registered_manifest(options)["instances"]])


def protocol_hash(options):
    return json_digest(protocol_record(options))


def suite_root(options, suite):
    # Keep nested atomic files within Windows path limits. Full SHA256 is still
    # checked in every registration, checkpoint and portable review record.
    return ROOT / options["result_root"] / protocol_hash(options)[:16] / suite


def hardware(options):
    import numpy
    return dict(host_id=host_id(), platform=platform.platform(), processor=platform.processor(),
                logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                ram_bytes=psutil.virtual_memory().total, python=sys.version, torch=torch.__version__,
                numpy=numpy.__version__, device=options.get("device", "cpu"), workers=options["workers"],
                environments=options["environments"], update_threads=options["update_threads"])
