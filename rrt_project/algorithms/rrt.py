from __future__ import annotations

from time import perf_counter
from typing import Sequence

import numpy as np

from rrt_project.algorithms.base_planner import BasePlanner, PlannerConfig, PlannerResult
from rrt_project.core.sampling_policy import MixedSampler


class RRTPlanner(BasePlanner):
    def __init__(self, collision_backend):
        super().__init__(name="rrt", collision_backend=collision_backend)

    def plan(
        self,
        scene,
        start: Sequence[float],
        goal: Sequence[float],
        config: PlannerConfig,
    ) -> PlannerResult:
        start_t = perf_counter()
        self.collision_backend.reset_counters()
        rng = np.random.default_rng(config.seed)
        sampler = MixedSampler()

        start_arr = np.asarray(start, dtype=np.float64)
        goal_arr = np.asarray(goal, dtype=np.float64)

        nodes = [start_arr]
        parents = [-1]
        goal_iter = -1
        success = False
        sample_counts = {"goal": 0, "bridge": 0, "uniform": 0}

        for i in range(config.max_iters):
            if perf_counter() - start_t > config.max_time_sec:
                break
            state = {
                "rng": rng,
                "scene": scene,
                "goal": goal_arr,
                "goal_bias": config.goal_bias,
                "bridge_bias": 0.0,
                "alpha": 1.0,
                "bridge_sampler": None,
                "sample_source_counts": sample_counts,
            }
            q_rand = sampler.sample(state)
            near_idx = self._nearest_index(nodes, q_rand)
            q_near = nodes[near_idx]
            q_new = self._steer(q_near, q_rand, config.step_size)

            if not self._is_segment_free(q_near, q_new, scene, alpha=1.0, config=config):
                continue

            nodes.append(q_new)
            parents.append(near_idx)
            new_idx = len(nodes) - 1

            if np.linalg.norm(q_new - goal_arr) <= config.goal_tolerance and self._is_segment_free(
                q_new, goal_arr, scene, alpha=1.0, config=config
            ):
                nodes.append(goal_arr)
                parents.append(new_idx)
                goal_iter = i
                success = True
                break

        if success:
            path = self._reconstruct_path(nodes, parents, len(nodes) - 1)
        else:
            idx = self._nearest_index(nodes, goal_arr)
            path = self._reconstruct_path(nodes, parents, idx)

        total_samples = max(1, sum(sample_counts.values()))
        meta = {
            "alpha_final": 1.0,
            "bridge_usage_ratio": 0.0,
            "sample_source_counts": sample_counts,
            "alpha_trace": [1.0],
            "alpha_trace_stats": {"alpha_min": 1.0, "alpha_mean": 1.0, "alpha_max": 1.0},
            "bridge_candidate_stats": {},
            "tree_nodes": np.asarray(nodes, dtype=np.float64).tolist(),
            "tree_parents": [int(x) for x in parents],
            "sample_goal_ratio": float(sample_counts["goal"] / total_samples),
        }
        success, path = self._enforce_hard_validation(
            success=success,
            path=path,
            scene=scene,
            config=config,
            meta=meta,
        )

        return self._finalize(
            success=success,
            path=path,
            start_t=start_t,
            scene_id=scene.scene_id,
            config=config,
            iters_used=i + 1,
            goal_iter=goal_iter,
            meta=meta,
        )
