from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Sequence

import numpy as np

from rrt_project.algorithms.base_planner import BasePlanner, PlannerConfig, PlannerResult
from rrt_project.core.sampling_policy import MixedSampler


class RRTStarPlanner(BasePlanner):
    def __init__(self, collision_backend):
        super().__init__(name="rrt_star", collision_backend=collision_backend)

    def _prepare_run(self, start, goal, config: PlannerConfig):
        self.collision_backend.reset_counters()
        rng = np.random.default_rng(config.seed)
        start_arr = np.asarray(start, dtype=np.float64)
        goal_arr = np.asarray(goal, dtype=np.float64)
        nodes = [start_arr]
        parents = [-1]
        costs = [0.0]
        stats = {
            "sample_source_counts": {"goal": 0, "bridge": 0, "uniform": 0},
            "alpha_trace": [],
            "bridge_usage_ratio": 0.0,
        }
        return rng, start_arr, goal_arr, nodes, parents, costs, stats

    def _sample_state(
        self,
        sampler: MixedSampler,
        rng: np.random.Generator,
        scene,
        goal_arr: np.ndarray,
        config: PlannerConfig,
        alpha: float,
        bridge_sampler,
        stats: Dict[str, Any],
    ) -> np.ndarray:
        state = {
            "rng": rng,
            "scene": scene,
            "goal": goal_arr,
            "goal_bias": config.goal_bias,
            "bridge_bias": config.bridge_bias if bridge_sampler is not None and config.enable_bridge else 0.0,
            "alpha": alpha,
            "bridge_sampler": bridge_sampler,
            "sample_source_counts": stats["sample_source_counts"],
        }
        return sampler.sample(state)

    def _choose_parent(self, nodes, costs, q_new, near_ids, near_idx, scene, alpha, config):
        parent = near_idx
        best_cost = costs[near_idx] + float(np.linalg.norm(q_new - nodes[near_idx]))
        for j in near_ids:
            c = costs[j] + float(np.linalg.norm(nodes[j] - q_new))
            if c < best_cost and self._is_segment_free(nodes[j], q_new, scene, alpha=alpha, config=config):
                parent = int(j)
                best_cost = c
        return parent, best_cost

    def _rewire(self, nodes, parents, costs, new_idx, near_ids, scene, alpha, config) -> None:
        for j in near_ids:
            if j == new_idx:
                continue
            c_new = costs[new_idx] + float(np.linalg.norm(nodes[j] - nodes[new_idx]))
            if c_new + 1e-9 < costs[j] and self._is_segment_free(nodes[new_idx], nodes[j], scene, alpha=alpha, config=config):
                parents[j] = new_idx
                costs[j] = c_new

    def _run(self, scene, start, goal, config, alpha_provider, bridge_sampler=None) -> PlannerResult:
        start_t = perf_counter()
        rng, start_arr, goal_arr, nodes, parents, costs, stats = self._prepare_run(start, goal, config)
        sampler = MixedSampler()

        goal_idx = -1
        goal_iter = -1

        for i in range(config.max_iters):
            if perf_counter() - start_t > config.max_time_sec:
                break
            progress = i / max(1, config.max_iters - 1)
            alpha = float(alpha_provider(progress))
            stats["alpha_trace"].append(alpha)

            q_rand = self._sample_state(
                sampler=sampler,
                rng=rng,
                scene=scene,
                goal_arr=goal_arr,
                config=config,
                alpha=alpha,
                bridge_sampler=bridge_sampler,
                stats=stats,
            )

            near_idx = self._nearest_index(nodes, q_rand)
            q_near = nodes[near_idx]
            q_new = self._steer(q_near, q_rand, config.step_size)
            if not self._is_segment_free(q_near, q_new, scene, alpha=alpha, config=config):
                continue

            d = np.linalg.norm(np.asarray(nodes) - q_new, axis=1)
            near_ids = np.where(d <= config.rewire_radius)[0]
            parent, best_cost = self._choose_parent(nodes, costs, q_new, near_ids, near_idx, scene, alpha, config)

            nodes.append(q_new)
            parents.append(parent)
            costs.append(best_cost)
            new_idx = len(nodes) - 1
            self._rewire(nodes, parents, costs, new_idx, near_ids, scene, alpha, config)

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

        sample_counts = stats["sample_source_counts"]
        total_samples = max(1, sum(sample_counts.values()))
        stats["bridge_usage_ratio"] = float(sample_counts.get("bridge", 0) / total_samples)

        alpha_trace = np.asarray(stats["alpha_trace"], dtype=np.float64) if stats["alpha_trace"] else np.asarray([1.0])
        alpha_stats = {
            "alpha_min": float(alpha_trace.min()),
            "alpha_mean": float(alpha_trace.mean()),
            "alpha_max": float(alpha_trace.max()),
        }

        bridge_stats = {}
        if bridge_sampler is not None and getattr(bridge_sampler, "diagnostics", None) is not None:
            bridge_stats = bridge_sampler.diagnostics.to_dict()

        meta = {
            "alpha_final": float(alpha_trace[-1]),
            "bridge_usage_ratio": float(stats["bridge_usage_ratio"]),
            "sample_source_counts": sample_counts,
            "alpha_trace": [float(x) for x in stats["alpha_trace"][:: max(1, len(stats["alpha_trace"]) // 80)]],
            "alpha_trace_stats": alpha_stats,
            "bridge_candidate_stats": bridge_stats,
            "tree_nodes": np.asarray(nodes, dtype=np.float64).tolist(),
            "tree_parents": [int(x) for x in parents],
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

    def plan(self, scene, start: Sequence[float], goal: Sequence[float], config: PlannerConfig) -> PlannerResult:
        return self._run(
            scene=scene,
            start=start,
            goal=goal,
            config=config,
            alpha_provider=lambda _progress: 1.0,
            bridge_sampler=None,
        )
