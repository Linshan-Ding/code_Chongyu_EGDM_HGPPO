"""Best-effort hardware telemetry for reproducible training logs.

Telemetry is observational only.  Missing NVML/``nvidia-smi`` support is
reported as NaN instead of fabricating a utilization value or changing the
training path.
"""

from __future__ import annotations

import math
import subprocess

import torch


def gpu_runtime_snapshot(
    device,
    *,
    peak_alloc_bytes: int | None = None,
    peak_reserved_bytes: int | None = None,
) -> dict[str, float]:
    """Return utilization and PyTorch peak allocated/reserved memory.

    ``nvidia-smi`` reports a point-in-time utilization sample, while the memory
    value comes from PyTorch's peak allocator counter for the current update.
    Both values are NaN when CUDA telemetry is unavailable.
    """

    if getattr(device, "type", str(device)) != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_util": math.nan,
            "gpu_mem_gb": math.nan,
            "gpu_mem_reserved_gb": math.nan,
        }
    if peak_alloc_bytes is None:
        peak_alloc_bytes = int(torch.cuda.max_memory_allocated(device))
    if peak_reserved_bytes is None:
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    memory_gb = float(peak_alloc_bytes) / float(1024**3)
    reserved_gb = float(peak_reserved_bytes) / float(1024**3)
    utilization = math.nan
    try:
        completed = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        value = completed.stdout.strip().splitlines()[0].strip()
        utilization = float(value)
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError, OSError):
        pass
    return {
        "gpu_util": utilization,
        "gpu_mem_gb": memory_gb,
        "gpu_mem_reserved_gb": reserved_gb,
    }


__all__ = ["gpu_runtime_snapshot"]
