from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PlannerConfig:
    step_size: float
    max_iters: int
    goal_bias: float
    bridge_bias: float
    cuda_enabled: bool
    alpha_schedule: List[Tuple[float, float]]
    seed: int = 0
    goal_tolerance: float = 0.08
    rewire_radius: float = 0.2
    max_time_sec: float = 8.0
    bridge_num_samples: int = 300
    bridge_delta: float = 0.05
    bridge_cluster_eps: float = 0.12
    bridge_cluster_min_pts: int = 3
    bridge_clustering: str = "radius"
    cci_eta: float = 18.0
    enable_cci: bool = True
    enable_bridge: bool = True
    ablation_mode: str = "baseline"  # baseline | cci_only | bridge_only | cci_bridge
    final_hard_validation: bool = True
    hard_validation_alpha: float = 1.0
    obstacle_min_scale: float = 1.0


@dataclass
class PlannerResult:
    success: bool
    path: List[List[float]]
    time_ms: float
    path_len: float
    collision_checks: int
    seed: int
    scene_id: str
    iters_used: int = 0
    goal_reached_iter: int = -1
    meta: Dict[str, Any] = field(default_factory=dict)


class Sampler(ABC):
    @abstractmethod
    def sample(self, state: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


class BasePlanner(ABC):
    def __init__(self, name: str, collision_backend: Any):
        self.name = name
        self.collision_backend = collision_backend

    @abstractmethod
    def plan(
        self,
        scene: Any,
        start: Sequence[float],
        goal: Sequence[float],
        config: PlannerConfig,
    ) -> PlannerResult:
        raise NotImplementedError

    @staticmethod
    def _path_length(path: Sequence[Sequence[float]]) -> float:
        if len(path) < 2:
            return 0.0
        arr = np.asarray(path, dtype=np.float64)
        seg = arr[1:] - arr[:-1]
        return float(np.linalg.norm(seg, axis=1).sum())

    @staticmethod
    def _nearest_index(nodes: Sequence[np.ndarray], q: np.ndarray) -> int:
        data = np.asarray(nodes)
        d = np.linalg.norm(data - q, axis=1)
        return int(np.argmin(d))

    @staticmethod
    def _steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
        direction = q_to - q_from
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return q_from.copy()
        step = min(step_size, float(norm))
        return q_from + direction / norm * step

    @staticmethod
    def _reconstruct_path(
        nodes: Sequence[np.ndarray], parents: Sequence[int], idx: int
    ) -> List[List[float]]:
        rev: List[List[float]] = []
        cur = idx
        while cur >= 0:
            rev.append(nodes[cur].astype(float).tolist())
            cur = parents[cur]
        rev.reverse()
        return rev

    def _is_segment_free(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        scene: Any,
        alpha: float,
        config: PlannerConfig,
    ) -> bool:
        collided = self.collision_backend.segment_collides(
            p1=p1,
            p2=p2,
            scene_state=scene,
            alpha=alpha,
            cuda_enabled=config.cuda_enabled,
            obstacle_min_scale=config.obstacle_min_scale,
        )
        return not collided

    def _hard_validate_path(
        self,
        path: Sequence[Sequence[float]],
        scene: Any,
        config: PlannerConfig,
    ) -> tuple[bool, int]:
        if len(path) < 2:
            return True, -1
        for idx in range(len(path) - 1):
            hit = self.collision_backend.segment_collides(
                p1=path[idx],
                p2=path[idx + 1],
                scene_state=scene,
                alpha=config.hard_validation_alpha,
                cuda_enabled=config.cuda_enabled,
                obstacle_min_scale=1.0,
            )
            if hit:
                return False, idx
        return True, -1

    def _hard_safe_path(
        self,
        path: Sequence[Sequence[float]],
        scene: Any,
        config: PlannerConfig,
    ) -> tuple[List[List[float]], int]:
        if len(path) < 2:
            return [list(np.asarray(p, dtype=float)) for p in path], -1

        safe: List[List[float]] = [list(np.asarray(path[0], dtype=float))]
        for idx in range(len(path) - 1):
            hit = self.collision_backend.segment_collides(
                p1=path[idx],
                p2=path[idx + 1],
                scene_state=scene,
                alpha=config.hard_validation_alpha,
                cuda_enabled=config.cuda_enabled,
                obstacle_min_scale=1.0,
            )
            if hit:
                return safe, idx
            safe.append(list(np.asarray(path[idx + 1], dtype=float)))
        return safe, -1

    def _enforce_hard_validation(
        self,
        success: bool,
        path: Sequence[Sequence[float]],
        scene: Any,
        config: PlannerConfig,
        meta: Dict[str, Any],
    ) -> tuple[bool, List[List[float]]]:
        path_out = [list(np.asarray(p, dtype=float)) for p in path]
        meta["planned_success"] = bool(success)
        if not config.final_hard_validation:
            meta["hard_validation_passed"] = bool(success)
            meta["hard_validation_collision_idx"] = -1.0
            meta["hard_validated_path"] = path_out
            return success, path_out

        safe_path, collision_idx = self._hard_safe_path(path=path, scene=scene, config=config)
        passed = collision_idx < 0
        path_out = safe_path

        meta["hard_validation_passed"] = bool(passed)
        meta["hard_validation_collision_idx"] = float(collision_idx)
        meta["hard_validated_path"] = path_out

        if success and not passed:
            meta["hard_validation_failed"] = True
        return bool(success and passed), path_out

    def _finalize(
        self,
        success: bool,
        path: List[List[float]],
        start_t: float,
        scene_id: str,
        config: PlannerConfig,
        iters_used: int,
        goal_iter: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> PlannerResult:
        elapsed_ms = (perf_counter() - start_t) * 1000.0
        return PlannerResult(
            success=success,
            path=path,
            time_ms=elapsed_ms,
            path_len=self._path_length(path),
            collision_checks=int(self.collision_backend.segment_checks),
            seed=config.seed,
            scene_id=scene_id,
            iters_used=iters_used,
            goal_reached_iter=goal_iter,
            meta=meta or {},
        )

    @staticmethod
    def config_with_seed(cfg: PlannerConfig, seed: int) -> PlannerConfig:
        return replace(cfg, seed=seed)
