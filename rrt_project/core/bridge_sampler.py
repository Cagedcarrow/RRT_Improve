from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BridgeDiagnostics:
    total_trials: int = 0
    accepted_candidates: int = 0
    raw_midpoints: int = 0
    clusters_total: int = 0
    clusters_valid: int = 0
    corner_like_rejected: int = 0

    def to_dict(self) -> Dict[str, float]:
        density = self.accepted_candidates / max(1, self.total_trials)
        corner_rate = self.corner_like_rejected / max(1, self.total_trials)
        cluster_validity = self.clusters_valid / max(1, self.clusters_total)
        return {
            "total_trials": float(self.total_trials),
            "accepted_candidates": float(self.accepted_candidates),
            "candidate_density": float(density),
            "corner_false_positive_rate": float(corner_rate),
            "cluster_total": float(self.clusters_total),
            "cluster_validity": float(cluster_validity),
        }


@dataclass
class BridgeSampler:
    num_samples: int
    perturbation_delta: float
    cluster_eps: float
    min_cluster_size: int
    clustering_mode: str = "radius"  # radius | dbscan
    max_candidates: int = 256
    _candidates: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    diagnostics: BridgeDiagnostics = field(default_factory=BridgeDiagnostics)

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
        diag = BridgeDiagnostics(total_trials=self.num_samples)

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

            # From improved bridge sampling: keep single-channel interior candidates.
            if free_count == 1:
                raw.append(m)
            else:
                diag.corner_like_rejected += 1

        diag.raw_midpoints = len(raw)

        if not raw:
            self._candidates = np.empty((0, dim), dtype=np.float64)
            self.diagnostics = diag
            return self._candidates

        raw_arr = np.asarray(raw, dtype=np.float64)
        if self.clustering_mode == "dbscan":
            clusters = self._dbscan_cluster(raw_arr, eps=self.cluster_eps, min_pts=self.min_cluster_size)
        else:
            clusters = self._radius_cluster(raw_arr, eps=self.cluster_eps)

        diag.clusters_total = len(clusters)
        centers = []
        for c in clusters:
            if len(c) >= self.min_cluster_size:
                diag.clusters_valid += 1
                centers.append(raw_arr[c].mean(axis=0))

        if not centers:
            self._candidates = np.empty((0, dim), dtype=np.float64)
        else:
            out = np.asarray(centers, dtype=np.float64)
            if out.shape[0] > self.max_candidates:
                idx = rng.choice(out.shape[0], size=self.max_candidates, replace=False)
                out = out[idx]
            self._candidates = out

        diag.accepted_candidates = int(self._candidates.shape[0])
        self.diagnostics = diag
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
    def _dbscan_cluster(points: np.ndarray, eps: float, min_pts: int) -> List[List[int]]:
        """Lightweight DBSCAN cluster extraction (core+reachable points).

        Returns clusters as index lists.
        """
        n = points.shape[0]
        labels = np.full(n, -2, dtype=int)  # -2 unvisited, -1 noise

        def region_query(i: int) -> List[int]:
            d = np.linalg.norm(points - points[i], axis=1)
            return np.where(d <= eps)[0].tolist()

        cluster_id = 0
        for i in range(n):
            if labels[i] != -2:
                continue
            neighbors = region_query(i)
            if len(neighbors) < min_pts:
                labels[i] = -1
                continue
            labels[i] = cluster_id
            seeds = [p for p in neighbors if p != i]
            while seeds:
                j = seeds.pop()
                if labels[j] == -1:
                    labels[j] = cluster_id
                if labels[j] != -2:
                    continue
                labels[j] = cluster_id
                n2 = region_query(j)
                if len(n2) >= min_pts:
                    for q in n2:
                        if labels[q] in (-2, -1):
                            seeds.append(q)
            cluster_id += 1

        clusters: List[List[int]] = []
        for cid in range(cluster_id):
            ids = np.where(labels == cid)[0].tolist()
            if ids:
                clusters.append(ids)
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
