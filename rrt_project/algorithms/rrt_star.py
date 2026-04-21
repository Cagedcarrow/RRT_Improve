from __future__ import annotations

from time import perf_counter
from typing import Sequence

import numpy as np

from rrt_project.algorithms.base_planner import BasePlanner, PlannerConfig, PlannerResult
from rrt_project.core.sampling_policy import MixedSampler


class RRTStarPlanner(BasePlanner):
    def __init__(self, collision_backend):
        super().__init__(name="rrt_star", collision_backend=collision_backend)

    def _run(self, scene, start, goal, config, alpha_provider, bridge_sampler=None) -> PlannerResult:
        start_t = perf_counter()
        self.collision_backend.reset_counters()
        rng = np.random.default_rng(config.seed)
        sampler = MixedSampler()

        start_arr = np.asarray(start, dtype=np.float64)
        goal_arr = np.asarray(goal, dtype=np.float64)

        nodes = [start_arr]
        parents = [-1]
        costs = [0.0]

        goal_idx = -1
        goal_iter = -1
        bridge_hits = 0
        sample_count = 0

        for i in range(config.max_iters):
            if perf_counter() - start_t > config.max_time_sec:
                break
            progress = i / max(1, config.max_iters - 1)
            alpha = alpha_provider(progress)

            state = {
                "rng": rng,
                "scene": scene,
                "goal": goal_arr,
                "goal_bias": config.goal_bias,
                "bridge_bias": config.bridge_bias if bridge_sampler is not None and config.enable_bridge else 0.0,
                "alpha": alpha,
                "bridge_sampler": bridge_sampler,
            }
            q_rand = sampler.sample(state)
            sample_count += 1
            if state.get("last_sample_source") == "bridge":
                bridge_hits += 1

            near_idx = self._nearest_index(nodes, q_rand)
            q_near = nodes[near_idx]
            q_new = self._steer(q_near, q_rand, config.step_size)

            if not self._is_segment_free(q_near, q_new, scene, alpha=alpha, config=config):
                continue

            d = np.linalg.norm(np.asarray(nodes) - q_new, axis=1)
            near_ids = np.where(d <= config.rewire_radius)[0]
            parent = near_idx
            best_cost = costs[near_idx] + float(np.linalg.norm(q_new - q_near))

            for j in near_ids:
                c = costs[j] + float(np.linalg.norm(nodes[j] - q_new))
                if c < best_cost and self._is_segment_free(nodes[j], q_new, scene, alpha=alpha, config=config):
                    parent = int(j)
                    best_cost = c

            nodes.append(q_new)
            parents.append(parent)
            costs.append(best_cost)
            new_idx = len(nodes) - 1

            for j in near_ids:
                if j == new_idx:
                    continue
                c_new = costs[new_idx] + float(np.linalg.norm(nodes[j] - q_new))
                if c_new + 1e-9 < costs[j] and self._is_segment_free(q_new, nodes[j], scene, alpha=alpha, config=config):
                    parents[j] = new_idx
                    costs[j] = c_new

            if np.linalg.norm(q_new - goal_arr) <= config.goal_tolerance and self._is_segment_free(
                q_new, goal_arr, scene, alpha=alpha, config=config
            ):
                nodes.append(goal_arr)
                parents.append(new_idx)
                costs.append(costs[new_idx] + float(np.linalg.norm(goal_arr - q_new)))
                goal_idx = len(nodes) - 1
                goal_iter = i
                break

        success = goal_idx >= 0
        if success:
            path = self._reconstruct_path(nodes, parents, goal_idx)
        else:
            idx = self._nearest_index(nodes, goal_arr)
            path = self._reconstruct_path(nodes, parents, idx)

        alpha_final = alpha_provider(min(1.0, (i + 1) / max(1, config.max_iters)))
        return self._finalize(
            success=success,
            path=path,
            start_t=start_t,
            scene_id=scene.scene_id,
            config=config,
            iters_used=i + 1,
            goal_iter=goal_iter,
            meta={
                "alpha_final": float(alpha_final),
                "bridge_usage_ratio": float(bridge_hits / max(1, sample_count)),
            },
        )

    def plan(self, scene, start: Sequence[float], goal: Sequence[float], config: PlannerConfig) -> PlannerResult:
        return self._run(
            scene=scene,
            start=start,
            goal=goal,
            config=config,
            alpha_provider=lambda _progress: 1.0,
            bridge_sampler=None,
        )
