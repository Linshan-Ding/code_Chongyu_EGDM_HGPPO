"""Online Scheme-2 training-instance sampler.

Formal Scheme-2 samples structural dimensions from ``configs/instance.yaml``
scale ranges.  A discrete parameter table remains supported for controlled or
legacy experiments, but is not the formal training distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from data.generator import InstanceGenerator, load_parameter_table


@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample_index: int
    instance_seed: int
    scale: str
    scenario: str
    load_ratio: float
    due_tightness: str
    instance_id: str
    instance_parameters: dict | None = None


class OnlineInstanceSampler:
    def __init__(
        self,
        cfg,
        *,
        seed: int,
        scale_pool,
        scenario_pool,
        load_ratio_pool=None,
        due_tightness_pool=None,
        instance_parameter_table_path=None,
    ):
        self.cfg = cfg
        self.generator = InstanceGenerator(cfg)
        self.rng = np.random.default_rng(int(seed))

        self.scale_pool = tuple(str(x) for x in scale_pool)
        self.scenario_pool = tuple(str(x) for x in scenario_pool)

        self.load_ratio_pool = tuple(
            float(x) for x in (
                cfg.instance.load_ratio_choices
                if load_ratio_pool is None else load_ratio_pool
            )
        )

        self.due_tightness_pool = tuple(
            str(x) for x in (
                ("tight", "medium", "loose")
                if due_tightness_pool is None else due_tightness_pool
            )
        )

        self.instance_parameter_table = None
        if instance_parameter_table_path is not None:
            self.instance_parameter_table = load_parameter_table(instance_parameter_table_path)

        self.sample_count = 0
        self.last_record = None

    def sample(self):

        instance_seed = int(
            self.rng.integers(0, 2**32 - 1, dtype=np.uint64)
        )

        index = self.sample_count

        if self.instance_parameter_table:
            names = [
                key for key, value in self.instance_parameter_table.items()
                if str(value.get("scale")) in self.scale_pool
            ]
            if not names:
                raise ValueError(
                    "instance parameter table has no cases matching the requested scale_pool"
                )
            name = names[
                int(self.rng.integers(0, len(names)))
            ]

            params = dict(self.instance_parameter_table[name])
            params["instance_name"] = name

            scale = str(params.get("scale", "M"))
            # Dynamic factors are sampled independently from the declared pools;
            # they are experimental perturbations, not hidden dimensions of M/A/R/J.
            scenario = self.scenario_pool[
                int(self.rng.integers(0, len(self.scenario_pool)))
            ]
            load_ratio = self.load_ratio_pool[
                int(self.rng.integers(0, len(self.load_ratio_pool)))
            ]
            due_tightness = self.due_tightness_pool[
                int(self.rng.integers(0, len(self.due_tightness_pool)))
            ]

        else:
            params = None

            scale = self.scale_pool[
                int(self.rng.integers(0, len(self.scale_pool)))
            ]
            scenario = self.scenario_pool[
                int(self.rng.integers(0, len(self.scenario_pool)))
            ]
            load_ratio = self.load_ratio_pool[
                int(self.rng.integers(0, len(self.load_ratio_pool)))
            ]
            due_tightness = self.due_tightness_pool[
                int(self.rng.integers(0, len(self.due_tightness_pool)))
            ]

        instance_id = (
            f"train_{index:07d}_{scale}_{scenario}_{instance_seed}"
        )

        instance = self.generator.sample(
            scale=scale,
            scenario=scenario,
            seed=instance_seed,
            load_ratio=load_ratio,
            due_tightness=due_tightness,
            instance_id=instance_id,
            instance_parameters=params,
        )

        self.sample_count += 1

        self.last_record = SampleRecord(
            sample_index=index,
            instance_seed=instance_seed,
            scale=scale,
            scenario=scenario,
            load_ratio=float(load_ratio),
            due_tightness=due_tightness,
            instance_id=instance_id,
            instance_parameters=params,
        )

        return instance


__all__ = ["OnlineInstanceSampler", "SampleRecord"]
