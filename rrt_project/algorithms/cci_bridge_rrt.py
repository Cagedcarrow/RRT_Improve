from __future__ import annotations

from typing import Sequence

import numpy as np

from rrt_project.algorithms.base_planner import PlannerConfig, PlannerResult
from rrt_project.algorithms.rrt_star import RRTStarPlanner
from rrt_project.core.bridge_sampler import BridgeSampler
from rrt_project.core.cci import AlphaScheduler


class CCIBridgeRRTPlanner(RRTStarPlanner):
    def __init__(self, collision_backend):
        super().__init__(collision_backend=collision_backend)
        self.name = "cci_bridge_rrt"

    def plan(self, scene, start: Sequence[float], goal: Sequence[float], config: PlannerConfig) -> PlannerResult:
        schedule = config.alpha_schedule if config.enable_cci else [(0.0, 1.0), (1.0, 1.0)]
        scheduler = AlphaScheduler(schedule=schedule)

        bridge_sampler = None
        if config.enable_bridge and config.bridge_bias > 0.0:
            rng = np.random.default_rng(config.seed)
            bridge_sampler = BridgeSampler(
                num_samples=config.bridge_num_samples,
                perturbation_delta=config.bridge_delta,
                cluster_eps=config.bridge_cluster_eps,
                min_cluster_size=config.bridge_cluster_min_pts,
            )
            # Bridge candidates are generated in full environment to keep narrow-passage bias.
            bridge_sampler.generate(scene=scene, rng=rng, alpha=1.0)

        return self._run(
            scene=scene,
            start=np.asarray(start, dtype=np.float64),
            goal=np.asarray(goal, dtype=np.float64),
            config=config,
            alpha_provider=scheduler.alpha_at,
            bridge_sampler=bridge_sampler,
        )
