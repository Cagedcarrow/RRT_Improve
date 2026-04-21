from __future__ import annotations

from time import perf_counter
from typing import Sequence, Tuple

import numpy as np

from rrt_project.algorithms.base_planner import BasePlanner, PlannerConfig, PlannerResult
from rrt_project.core.sampling_policy import MixedSampler


class RRTConnectPlanner(BasePlanner):
    def __init__(self, collision_backend):
        super().__init__(name="rrt_connect", collision_backend=collision_backend)

    def _extend(self, nodes, parents, target, scene, config) -> Tuple[bool, int]:
        near_idx = self._nearest_index(nodes, target)
        q_near = nodes[near_idx]
        q_new = self._steer(q_near, target, config.step_size)
        if not self._is_segment_free(q_near, q_new, scene, alpha=1.0, config=config):
            return False, near_idx
        nodes.append(q_new)
        parents.append(near_idx)
        return True, len(nodes) - 1

    def plan(self, scene, start: Sequence[float], goal: Sequence[float], config: PlannerConfig) -> PlannerResult:
        start_t = perf_counter()
        self.collision_backend.reset_counters()
        rng = np.random.default_rng(config.seed)
        sampler = MixedSampler()

        a_nodes = [np.asarray(start, dtype=np.float64)]
        a_par = [-1]
        b_nodes = [np.asarray(goal, dtype=np.float64)]
        b_par = [-1]

        success = False
        goal_iter = -1
        meet_a = -1
        meet_b = -1
        sample_counts = {"goal": 0, "bridge": 0, "uniform": 0}

        for i in range(config.max_iters):
            if perf_counter() - start_t > config.max_time_sec:
                break
            state = {
                "rng": rng,
                "scene": scene,
                "goal": np.asarray(goal, dtype=np.float64),
                "goal_bias": config.goal_bias,
                "bridge_bias": 0.0,
                "alpha": 1.0,
                "bridge_sampler": None,
                "sample_source_counts": sample_counts,
            }
            q_rand = sampler.sample(state)
            ok, new_idx = self._extend(a_nodes, a_par, q_rand, scene, config)
            if not ok:
                a_nodes, b_nodes = b_nodes, a_nodes
                a_par, b_par = b_par, a_par
                continue

            q_new = a_nodes[new_idx]
            connect_ok = True
            last_idx = -1
            while connect_ok:
                connect_ok, last_idx = self._extend(b_nodes, b_par, q_new, scene, config)
                if connect_ok:
                    q_try = b_nodes[last_idx]
                    if np.linalg.norm(q_try - q_new) <= config.goal_tolerance:
                        success = True
                        meet_a = new_idx
                        meet_b = last_idx
                        goal_iter = i
                        break
            if success:
                break
            a_nodes, b_nodes = b_nodes, a_nodes
            a_par, b_par = b_par, a_par

        if success:
            path_a = self._reconstruct_path(a_nodes, a_par, meet_a)
            path_b = self._reconstruct_path(b_nodes, b_par, meet_b)
            path_b.reverse()
            if path_a and path_b and path_a[-1] == path_b[0]:
                path_b = path_b[1:]
            path = path_a + path_b
        else:
            path = self._reconstruct_path(a_nodes, a_par, self._nearest_index(a_nodes, np.asarray(goal, dtype=np.float64)))

        merged_nodes = np.asarray(a_nodes + b_nodes, dtype=np.float64).tolist()
        offset = len(a_nodes)
        merged_parents = [int(x) for x in a_par] + [int(x + offset) if x >= 0 else -1 for x in b_par]
        meta = {
            "alpha_final": 1.0,
            "bridge_usage_ratio": 0.0,
            "sample_source_counts": sample_counts,
            "alpha_trace": [1.0],
            "alpha_trace_stats": {"alpha_min": 1.0, "alpha_mean": 1.0, "alpha_max": 1.0},
            "bridge_candidate_stats": {},
            "tree_nodes": merged_nodes,
            "tree_parents": merged_parents,
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
