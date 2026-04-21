from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def winding_number(path: Sequence[Sequence[float]], obstacle_point: Sequence[float]) -> float:
    arr = np.asarray(path, dtype=np.float64)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return 0.0
    op = np.asarray(obstacle_point, dtype=np.float64)
    total = 0.0
    for i in range(arr.shape[0] - 1):
        v1 = arr[i, :2] - op[:2]
        v2 = arr[i + 1, :2] - op[:2]
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        d = a2 - a1
        if d > np.pi:
            d -= 2.0 * np.pi
        elif d < -np.pi:
            d += 2.0 * np.pi
        total += d
    return float(total / (2.0 * np.pi))


def l_value_proxy(path: Sequence[Sequence[float]], obstacle_points: Iterable[Sequence[float]]) -> float:
    vals = [winding_number(path, o) for o in obstacle_points]
    if not vals:
        return 0.0
    return float(np.linalg.norm(np.asarray(vals, dtype=np.float64), ord=2))
