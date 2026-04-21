from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from rrt_project.algorithms.base_planner import PlannerConfig, PlannerResult
from rrt_project.algorithms.rrt_star import RRTStarPlanner
from rrt_project.core.bridge_sampler import BridgeSampler
from rrt_project.core.cci import AlphaScheduler, build_leaf_sequence


class CCIBridgeRRTPlanner(RRTStarPlanner):
    def __init__(self, collision_backend):
        super().__init__(collision_backend=collision_backend)
        self.name = "cci_bridge_rrt"

    @staticmethod
    def _apply_ablation(config: PlannerConfig) -> PlannerConfig:
        mode = (config.ablation_mode or "baseline").lower()
        if mode == "baseline":
            return replace(config, enable_cci=False, enable_bridge=False, bridge_bias=0.0)
        if mode == "cci_only":
            return replace(config, enable_cci=True, enable_bridge=False, bridge_bias=0.0)
        if mode == "bridge_only":
            return replace(config, enable_cci=False, enable_bridge=True)
        if mode == "cci_bridge":
            return replace(config, enable_cci=True, enable_bridge=True)
        return config

    def plan(self, scene, start: Sequence[float], goal: Sequence[float], config: PlannerConfig) -> PlannerResult:
        cfg = self._apply_ablation(config)
        schedule = cfg.alpha_schedule if cfg.enable_cci else [(0.0, 1.0), (1.0, 1.0)]
        scheduler = AlphaScheduler(schedule=schedule)

        leaf_plan = build_leaf_sequence(scene.obstacles)

        bridge_sampler = None
        if cfg.enable_bridge and cfg.bridge_bias > 0.0:
            rng = np.random.default_rng(cfg.seed)
            bridge_sampler = BridgeSampler(
                num_samples=cfg.bridge_num_samples,
                perturbation_delta=cfg.bridge_delta,
                cluster_eps=cfg.bridge_cluster_eps,
                min_cluster_size=cfg.bridge_cluster_min_pts,
                clustering_mode=cfg.bridge_clustering,
            )
            bridge_sampler.generate(scene=scene, rng=rng, alpha=1.0)

        result = self._run(
            scene=scene,
            start=np.asarray(start, dtype=np.float64),
            goal=np.asarray(goal, dtype=np.float64),
            config=cfg,
            alpha_provider=scheduler.alpha_at,
            bridge_sampler=bridge_sampler,
        )

        result.meta["leaf_plan_batches"] = float(len(leaf_plan.batches))
        result.meta["leaf_plan_init_objects"] = float(len(leaf_plan.init_indices))
        result.meta["ablation_mode"] = cfg.ablation_mode
        return result
