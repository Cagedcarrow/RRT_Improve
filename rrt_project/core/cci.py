from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


def shaping_function(x: float | np.ndarray, eta: float) -> float | np.ndarray:
    """Convex non-decreasing shaping function used by CCI interpolation.

    f(x) = (exp(eta*x)-1)/eta, with clipping for numeric stability.
    """
    arr = np.asarray(x, dtype=np.float64)
    z = np.clip(eta * arr, -60.0, 60.0)
    out = (np.exp(z) - 1.0) / max(eta, 1e-9)
    if np.isscalar(x):
        return float(out)
    return out


def interpolate_sdf(sdf_a: float, sdf_b: float, alpha: float, eta: float) -> float:
    a = float(np.clip(alpha, 0.0, 1.0))
    return float((1.0 - a) * shaping_function(sdf_a, eta) + a * shaping_function(sdf_b, eta))


@dataclass
class AlphaScheduler:
    schedule: List[Tuple[float, float]]

    def __post_init__(self) -> None:
        if not self.schedule:
            self.schedule = [(0.0, 1.0), (1.0, 1.0)]
        self.schedule = sorted(self.schedule, key=lambda x: x[0])

    def alpha_at(self, progress: float) -> float:
        p = max(0.0, min(1.0, progress))
        if p <= self.schedule[0][0]:
            return float(self.schedule[0][1])
        if p >= self.schedule[-1][0]:
            return float(self.schedule[-1][1])
        for i in range(1, len(self.schedule)):
            p0, a0 = self.schedule[i - 1]
            p1, a1 = self.schedule[i]
            if p0 <= p <= p1:
                if p1 - p0 < 1e-12:
                    return float(a1)
                r = (p - p0) / (p1 - p0)
                return float(a0 + r * (a1 - a0))
        return float(self.schedule[-1][1])


@dataclass
class AlphaExecutor:
    scheduler: AlphaScheduler
    trace: List[float] = field(default_factory=list)

    def at(self, progress: float) -> float:
        a = self.scheduler.alpha_at(progress)
        self.trace.append(float(a))
        return a

    def summary(self) -> Dict[str, float]:
        if not self.trace:
            return {"alpha_min": 1.0, "alpha_mean": 1.0, "alpha_max": 1.0}
        arr = np.asarray(self.trace, dtype=np.float64)
        return {
            "alpha_min": float(arr.min()),
            "alpha_mean": float(arr.mean()),
            "alpha_max": float(arr.max()),
        }


@dataclass
class LeafBatch:
    leaf_indices: List[int]
    parent_map: Dict[int, int]


@dataclass
class LeafSequencePlan:
    init_indices: List[int]
    batches: List[LeafBatch]


def _aabb_intersects(center_a: np.ndarray, ext_a: np.ndarray, center_b: np.ndarray, ext_b: np.ndarray, margin: float) -> bool:
    d = np.abs(center_a - center_b)
    lim = ext_a + ext_b + margin
    return bool(np.all(d <= lim))


def build_leaf_sequence(obstacles: Sequence, margin: float = 1e-6) -> LeafSequencePlan:
    """Approximate leaf-set sequence generation inspired by CCI paper.

    Conditions:
    1) each leaf intersects with exactly one remaining object (parent)
    2) leafs in same batch do not intersect each other
    """
    remaining = list(range(len(obstacles)))
    batches: List[LeafBatch] = []

    def intersects(i: int, j: int) -> bool:
        oi = obstacles[i]
        oj = obstacles[j]
        return _aabb_intersects(
            np.asarray(oi.center, dtype=np.float64),
            np.asarray(oi.half_extents, dtype=np.float64),
            np.asarray(oj.center, dtype=np.float64),
            np.asarray(oj.half_extents, dtype=np.float64),
            margin,
        )

    while len(remaining) > 2:
        selected: List[int] = []
        parent_map: Dict[int, int] = {}
        for idx in list(remaining):
            parents = [j for j in remaining if j != idx and intersects(idx, j)]
            if len(parents) != 1:
                continue
            # non-intersection between leaves in same batch
            if any(intersects(idx, s) for s in selected):
                continue
            selected.append(idx)
            parent_map[idx] = parents[0]

        if not selected:
            break

        batches.append(LeafBatch(leaf_indices=selected, parent_map=parent_map))
        remaining = [i for i in remaining if i not in selected]

    # Reverse order as described in CCI leaf-set sequence generation
    batches.reverse()
    return LeafSequencePlan(init_indices=remaining, batches=batches)


@dataclass
class CCIInterpolator:
    eta: float = 18.0

    def blended_sdf(self, sdf_parent: float, sdf_leaf: float, alpha: float) -> float:
        return interpolate_sdf(sdf_parent, sdf_leaf, alpha, self.eta)

    def point_sdf(self, point: Sequence[float], obstacle, alpha: float, min_scale: float = 0.25) -> float:
        """Strategy-compatible obstacle SDF query.

        Currently uses scaled object SDF as robust baseline.
        """
        return float(obstacle.sdf(point, alpha=alpha, min_scale=min_scale))
