from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class BridgeSampler:
    num_samples: int
    perturbation_delta: float
    cluster_eps: float
    min_cluster_size: int
    max_candidates: int = 256

    def __post_init__(self) -> None:
        self._candidates: np.ndarray = np.empty((0, 2), dtype=np.float64)

    def candidates(self) -> np.ndarray:
        return self._candidates

    def has_candidates(self) -> bool:
        return self._candidates.shape[0] > 0

    def sample_candidate(self, rng: np.random.Generator) -> Optional[np.ndarray]:
        if self._candidates.shape[0] == 0:
            return None
        i = int(rng.integers(0, self._candidates.shape[0]))
        return self._candidates[i].copy()

    def generate(self, scene, rng: np.random.Generator, alpha: float = 1.0) -> np.ndarray:
        dim = scene.dimension
        raw: List[np.ndarray] = []

        for _ in range(self.num_samples):
            x = scene.random_point(rng)
            p = scene.random_point(rng)
            if scene.is_point_free(x, alpha=alpha) or scene.is_point_free(p, alpha=alpha):
                continue
            m = 0.5 * (x + p)
            if not scene.is_point_free(m, alpha=alpha):
                continue

            dirs = self._orthogonal_dirs(x, p, dim)
            free_count = 0
            for d in dirs:
                mk = m + self.perturbation_delta * d
                if scene.is_point_free(mk, alpha=alpha):
                    free_count += 1

            if free_count == 1:
                raw.append(m)

        if not raw:
            self._candidates = np.empty((0, dim), dtype=np.float64)
            return self._candidates

        raw_arr = np.asarray(raw, dtype=np.float64)
        clusters = self._radius_cluster(raw_arr, self.cluster_eps)
        centers = []
        for c in clusters:
            if len(c) >= self.min_cluster_size:
                centers.append(raw_arr[c].mean(axis=0))

        if not centers:
            self._candidates = np.empty((0, dim), dtype=np.float64)
        else:
            out = np.asarray(centers, dtype=np.float64)
            if out.shape[0] > self.max_candidates:
                idx = rng.choice(out.shape[0], size=self.max_candidates, replace=False)
                out = out[idx]
            self._candidates = out
        return self._candidates

    @staticmethod
    def _radius_cluster(points: np.ndarray, eps: float) -> List[List[int]]:
        n = points.shape[0]
        remaining = set(range(n))
        clusters: List[List[int]] = []
        while remaining:
            seed = next(iter(remaining))
            queue = [seed]
            cluster = []
            remaining.remove(seed)
            while queue:
                i = queue.pop()
                cluster.append(i)
                pi = points[i]
                take = []
                for j in list(remaining):
                    if np.linalg.norm(points[j] - pi) <= eps:
                        take.append(j)
                for j in take:
                    remaining.remove(j)
                    queue.append(j)
            clusters.append(cluster)
        return clusters

    @staticmethod
    def _orthogonal_dirs(x: np.ndarray, p: np.ndarray, dim: int) -> List[np.ndarray]:
        v = p - x
        n = np.linalg.norm(v)
        if n < 1e-9:
            v = np.ones(dim)
            n = np.linalg.norm(v)
        u = v / n
        dirs: List[np.ndarray] = []
        if dim == 2:
            d = np.array([-u[1], u[0]], dtype=np.float64)
            dirs.extend([d, -d])
        else:
            base = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(base, u)) > 0.9:
                base = np.array([0.0, 1.0, 0.0])
            d1 = base - np.dot(base, u) * u
            d1 /= max(np.linalg.norm(d1), 1e-9)
            d2 = np.cross(u, d1)
            d2 /= max(np.linalg.norm(d2), 1e-9)
            dirs.extend([d1, -d1, d2, -d2])
        return dirs
