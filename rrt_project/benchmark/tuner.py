from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from typing import Dict, List, Tuple

from rrt_project.algorithms.base_planner import PlannerConfig


def tune_from_failures(config: PlannerConfig, failures: List[str]) -> Tuple[PlannerConfig, Dict[str, float]]:
    c = Counter(f for f in failures if f)
    total = max(1, sum(c.values()))

    next_cfg = config
    change_log: Dict[str, float] = {}

    if c.get("collision_stuck", 0) / total > 0.2:
        next_cfg = replace(
            next_cfg,
            bridge_bias=min(0.6, next_cfg.bridge_bias + 0.05),
            step_size=max(0.02, next_cfg.step_size * 0.9),
        )
        change_log["bridge_bias"] = next_cfg.bridge_bias
        change_log["step_size"] = next_cfg.step_size

    if c.get("timeout", 0) / total > 0.2:
        next_cfg = replace(next_cfg, goal_bias=min(0.30, next_cfg.goal_bias + 0.02))
        change_log["goal_bias"] = next_cfg.goal_bias

        schedule = []
        for p, a in next_cfg.alpha_schedule:
            schedule.append((p, min(1.0, a + (0.05 if p <= 0.6 else 0.0))))
        next_cfg = replace(next_cfg, alpha_schedule=schedule)
        change_log["alpha_schedule_updated"] = 1.0

    if c.get("overlong_path", 0) / total > 0.2:
        next_cfg = replace(next_cfg, step_size=max(0.02, next_cfg.step_size * 0.9), rewire_radius=min(0.5, next_cfg.rewire_radius + 0.02))
        change_log["step_size"] = next_cfg.step_size
        change_log["rewire_radius"] = next_cfg.rewire_radius

    if c.get("unstable", 0) / total > 0.2:
        next_cfg = replace(next_cfg, step_size=max(0.02, next_cfg.step_size * 0.95))
        change_log["step_size"] = next_cfg.step_size

    if not change_log:
        change_log["no_change"] = 1.0

    return next_cfg, change_log


def config_to_dict(cfg: PlannerConfig) -> dict:
    return asdict(cfg)
