from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class CollisionBackend:
    samples_per_segment: int = 24
    safety_margin: float = 0.0

    def __post_init__(self) -> None:
        self.segment_checks: int = 0

    @staticmethod
    def _device(cuda_enabled: bool) -> torch.device:
        if cuda_enabled and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def reset_counters(self) -> None:
        self.segment_checks = 0

    def batch_segment_check(
        self,
        segments: Sequence[Sequence[Sequence[float]]],
        scene_state,
        alpha: float = 1.0,
        cuda_enabled: bool = True,
    ) -> torch.BoolTensor:
        dev = self._device(cuda_enabled)
        seg = torch.as_tensor(segments, dtype=torch.float32, device=dev)
        if seg.ndim != 3:
            raise ValueError("segments must have shape [B, 2, D]")
        batch = seg.shape[0]
        self.segment_checks += int(batch)

        p0 = seg[:, 0, :]
        p1 = seg[:, 1, :]
        t = torch.linspace(0.0, 1.0, self.samples_per_segment, device=dev)
        pts = p0[:, None, :] + (p1 - p0)[:, None, :] * t[None, :, None]

        # Bounds collision check first
        lo = torch.as_tensor(scene_state.bounds[:, 0], dtype=torch.float32, device=dev)
        hi = torch.as_tensor(scene_state.bounds[:, 1], dtype=torch.float32, device=dev)
        out_of_bounds = (pts < lo).any(dim=-1) | (pts > hi).any(dim=-1)
        collided = out_of_bounds.any(dim=1)

        min_scale = 0.25
        scale = min_scale + (1.0 - min_scale) * float(np.clip(alpha, 0.0, 1.0))

        for obs in scene_state.obstacles:
            c = torch.as_tensor(obs.center, dtype=torch.float32, device=dev)
            e = torch.as_tensor(obs.half_extents * scale, dtype=torch.float32, device=dev)
            q = torch.abs(pts - c) - e
            outside = torch.clamp(q, min=0.0)
            outside_dist = torch.linalg.norm(outside, dim=-1)
            inside_dist = torch.minimum(torch.max(q, dim=-1).values, torch.zeros_like(outside_dist))
            sdf = outside_dist + inside_dist
            obs_coll = (sdf <= self.safety_margin).any(dim=1)
            collided = collided | obs_coll

        return collided

    def segment_collides(
        self,
        p1: Sequence[float],
        p2: Sequence[float],
        scene_state,
        alpha: float = 1.0,
        cuda_enabled: bool = True,
    ) -> bool:
        hit = self.batch_segment_check(
            segments=np.asarray([[p1, p2]], dtype=np.float32),
            scene_state=scene_state,
            alpha=alpha,
            cuda_enabled=cuda_enabled,
        )
        return bool(hit.detach().cpu().numpy()[0])
