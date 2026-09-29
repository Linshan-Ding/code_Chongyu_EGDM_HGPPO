"""Diagnostic-only PyTorch operator profiling for one PPO update."""

from __future__ import annotations

import json
import gzip
from pathlib import Path
import shutil
from typing import Callable, TypeVar

import torch


T = TypeVar("T")


def _first_step_only(step: int) -> torch.profiler.ProfilerAction:
    return (
        torch.profiler.ProfilerAction.RECORD_AND_SAVE
        if int(step) == 0
        else torch.profiler.ProfilerAction.NONE
    )


def _warmup_then_profile_one_step(step: int) -> torch.profiler.ProfilerAction:
    if int(step) == 0:
        return torch.profiler.ProfilerAction.WARMUP
    if int(step) == 1:
        return torch.profiler.ProfilerAction.RECORD_AND_SAVE
    return torch.profiler.ProfilerAction.NONE


def _operator_table(averages, sort_keys: tuple[str, ...], *, row_limit: int = 30) -> tuple[str, str]:
    errors: list[str] = []
    for sort_key in sort_keys:
        try:
            return sort_key, averages.table(sort_by=sort_key, row_limit=row_limit)
        except (AttributeError, RuntimeError, TypeError) as exc:
            errors.append(f"{sort_key}: {exc}")
    raise RuntimeError("no supported profiler table key: " + "; ".join(errors))


def run_profiled_ppo_update(
    call: Callable[[], T],
    *,
    output_dir: str | Path,
    use_cuda: bool,
    with_stack: bool = False,
    set_step_callback: Callable[[Callable[[], None] | None], None] | None = None,
) -> tuple[T, dict[str, object]]:
    """Run one update while tracing only its first logical optimizer step."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA operator profiling requested without CUDA")
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=activities,
        schedule=(
            _warmup_then_profile_one_step
            if set_step_callback is not None else _first_step_only
        ),
        record_shapes=True,
        profile_memory=True,
        # Python stacks multiply trace memory for this ragged autoregressive
        # replay path. Keep them opt-in; operator names/shapes/memory and the
        # Chrome timeline remain available in the default zero-argument run.
        with_stack=bool(with_stack),
        acc_events=True,
    ) as profiler:
        if set_step_callback is not None:
            set_step_callback(profiler.step)
        try:
            result = call()
            if set_step_callback is None:
                profiler.step()
        finally:
            if set_step_callback is not None:
                set_step_callback(None)

    if use_cuda:
        torch.cuda.synchronize()
    averages = profiler.key_averages(group_by_input_shape=False)
    cpu_key, cpu_table = _operator_table(averages, ("self_cpu_time_total",))
    cpu_path = output_dir / "top_self_cpu_time.txt"
    cpu_path.write_text(cpu_table, encoding="utf-8")

    files = {"top_self_cpu_time": str(cpu_path)}
    sort_keys = {"cpu": cpu_key}
    if use_cuda:
        cuda_key, cuda_table = _operator_table(
            averages,
            ("self_cuda_time_total", "self_device_time_total"),
        )
        cuda_path = output_dir / "top_self_cuda_time.txt"
        cuda_path.write_text(cuda_table, encoding="utf-8")
        files["top_self_cuda_time"] = str(cuda_path)
        sort_keys["cuda"] = cuda_key

    memory_sections: list[str] = []
    cpu_memory_key, cpu_memory_table = _operator_table(
        averages,
        ("self_cpu_memory_usage",),
    )
    memory_sections.extend((f"SORT: {cpu_memory_key}", cpu_memory_table))
    sort_keys["cpu_memory"] = cpu_memory_key
    if use_cuda:
        cuda_memory_key, cuda_memory_table = _operator_table(
            averages,
            ("self_cuda_memory_usage", "self_device_memory_usage"),
        )
        memory_sections.extend((f"SORT: {cuda_memory_key}", cuda_memory_table))
        sort_keys["cuda_memory"] = cuda_memory_key
    memory_path = output_dir / "top_memory.txt"
    memory_path.write_text("\n\n".join(memory_sections), encoding="utf-8")
    files["top_memory"] = str(memory_path)

    trace_path = output_dir / "ppo_trace.json"
    profiler.export_chrome_trace(str(trace_path))
    compressed_trace_path = output_dir / "ppo_trace.json.gz"
    with trace_path.open("rb") as source, gzip.open(
        compressed_trace_path, "wb", compresslevel=6
    ) as destination:
        shutil.copyfileobj(source, destination)
    trace_path.unlink()
    files["chrome_trace"] = str(compressed_trace_path)

    metadata = {
        "diagnostic_only": True,
        "scope": "second logical optimizer step after one warmup step within a complete Scheme-2 PPO update",
        "complete_update_executed": True,
        "warmup_logical_optimizer_steps": 1 if set_step_callback is not None else 0,
        "profiled_logical_optimizer_steps": 1,
        "activities": [activity.name for activity in activities],
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": bool(with_stack),
        "row_limit": 30,
        "sort_keys": sort_keys,
        "files": files,
    }
    metadata_path = output_dir / "operator_profile.json"
    files["metadata"] = str(metadata_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, metadata


__all__ = ["run_profiled_ppo_update"]
