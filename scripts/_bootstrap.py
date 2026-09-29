"""Shared zero-argument runner helpers for the Scheme-2 experiment scripts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(*args: str) -> None:
    command = [sys.executable, *args]
    subprocess.run(command, cwd=ROOT, check=True)


def budget_profile() -> str:
    """Read the active Scheme-2 budget profile from the config."""
    config_path = ROOT / "configs" / "scheme2.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    scheme2 = payload["scheme2"]
    return str(scheme2.get("active_budget_profile", "legacy"))


def formal_iterations() -> int:
    """Read the declared formal budget from the active Scheme-2 profile."""
    config_path = ROOT / "configs" / "scheme2.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    scheme2 = payload["scheme2"]
    profile = budget_profile()
    profiles = scheme2.get("budget_profiles") or {}
    if profiles:
        return int(profiles[profile]["formal_iterations"])
    return int(scheme2["formal_iterations"])


def require_cuda() -> None:
    """Refuse long training jobs when the selected interpreter cannot use CUDA."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "FAIL: formal training requires CUDA, but this Python interpreter "
            "reports torch.cuda.is_available() == False. Run "
            "python scripts/run_00_hardware_probe.py after installing a CUDA-enabled PyTorch build."
        )
