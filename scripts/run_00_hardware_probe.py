"""Verify that the active Python interpreter can train on the NVIDIA GPU."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

from _bootstrap import ROOT


def _memory_gb() -> float | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(float(status.total_phys) / 1024**3, 2)
    try:
        return round(float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1024**3, 2)
    except (AttributeError, OSError, ValueError):
        return None


def _nvidia_smi() -> tuple[list[dict[str, str]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return [], str(exc)
    if completed.returncode != 0:
        return [], (completed.stderr.strip() or f"nvidia-smi exited {completed.returncode}")
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append({"name": parts[0], "memory_total_mib": parts[1], "driver_version": parts[2]})
    return rows, None


def main() -> None:
    smi, smi_error = _nvidia_smi()
    cuda_available = bool(torch.cuda.is_available())
    report = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_logical_cores": os.cpu_count(),
        "system_memory_gb": _memory_gb(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": (int(torch.cuda.device_count()) if cuda_available else 0),
        "torch_active_device": (int(torch.cuda.current_device()) if cuda_available else None),
        "torch_gpu": (torch.cuda.get_device_name(0) if cuda_available else None),
        "torch_compute_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_available else None
        ),
        "torch_vram_gb": (
            round(float(torch.cuda.get_device_properties(0).total_memory) / 1024**3, 2)
            if cuda_available else None
        ),
        "bf16_supported": (bool(torch.cuda.is_bf16_supported()) if cuda_available else False),
        "nvidia_smi": smi,
        "nvidia_smi_error": smi_error,
    }
    output = ROOT / "result" / "hardware_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Hardware report written: {output.relative_to(ROOT)}")
    if smi and not cuda_available:
        raise SystemExit(
            "FAIL: NVIDIA GPU is visible but this Python uses CPU-only PyTorch. "
            "Install a CUDA-enabled PyTorch build before training."
        )
    if not cuda_available:
        raise SystemExit("FAIL: CUDA is unavailable to PyTorch; formal training would run on CPU.")
    print("Hardware probe: PASS")


if __name__ == "__main__":
    main()
