"""Pinned official DAN class loader; no upstream command-line side effects."""
import ast
import hashlib
import importlib.util
import json
import torch
from torch import nn
from .config import ROOT


def official_classes():
    root = ROOT / "vendor/dan_official"
    info = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    for name, digest in info["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"DAN provenance mismatch: {name}")
    namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional}
    for name in ("attention_layer", "sub_layers"):
        spec = importlib.util.spec_from_file_location(
            f"egdm_official_{name}", root / f"model/{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace.update(
            {
                key: value
                for key, value in vars(module).items()
                if not key.startswith("_")
            }
        )

    # The official class has absolute script imports. Load its class definitions
    # into this isolated namespace without importing upstream CLI / argparse.
    def nonzero_averaging(x):
        return x.sum(dim=1) / (x.abs().sum(dim=-1) > 0).sum(
            dim=1, keepdim=True
        ).clamp_min(1)

    namespace["nonzero_averaging"] = nonzero_averaging
    path = root / "model/main_model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 -- hash-verified, pinned upstream module
    return namespace
