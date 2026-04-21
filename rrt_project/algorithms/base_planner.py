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
    enable_cci: bool = True
    enable_bridge: bool = True


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
    meta: Dict[str, float] = field(default_factory=dict)


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
        )
        return not collided

    def _finalize(
        self,
        success: bool,
        path: List[List[float]],
        start_t: float,
        scene_id: str,
        config: PlannerConfig,
        iters_used: int,
        goal_iter: int,
        meta: Optional[Dict[str, float]] = None,
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
