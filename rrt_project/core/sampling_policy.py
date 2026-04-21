from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from rrt_project.algorithms.base_planner import Sampler


@dataclass
class MixedSampler(Sampler):
    def sample(self, state: Dict[str, Any]) -> np.ndarray:
        rng: np.random.Generator = state["rng"]
        scene = state["scene"]
        goal = np.asarray(state["goal"], dtype=np.float64)
        goal_bias: float = float(state.get("goal_bias", 0.05))
        bridge_bias: float = float(state.get("bridge_bias", 0.0))
        alpha: float = float(state.get("alpha", 1.0))
        bridge_sampler = state.get("bridge_sampler")

        counts = state.setdefault("sample_source_counts", {"goal": 0, "bridge": 0, "uniform": 0})
        r = float(rng.random())
        if r < goal_bias:
            state["last_sample_source"] = "goal"
            counts["goal"] += 1
            return goal.copy()

        if r < goal_bias + bridge_bias and bridge_sampler is not None and bridge_sampler.has_candidates():
            b = bridge_sampler.sample_candidate(rng)
            if b is not None:
                state["last_sample_source"] = "bridge"
                counts["bridge"] += 1
                return b

        state["last_sample_source"] = "uniform"
        counts["uniform"] += 1
        return scene.sample_free(rng, alpha=alpha)
