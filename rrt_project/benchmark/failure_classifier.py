from __future__ import annotations

from typing import Dict


def classify_failure(row: dict, baseline_len: Dict[str, float], unstable_threshold: float = 0.2) -> str:
    planned_success = bool(row.get("planned_success", row.get("success", False)))
    hard_pass = bool(row.get("hard_validation_passed", row.get("success", False)))

    if row.get("success"):
        scene = row.get("scene_id", "")
        ref = baseline_len.get(scene)
        if ref and row.get("path_len", 0.0) > 1.5 * ref:
            return "overlong_path"
        return ""

    if planned_success and not hard_pass:
        return "hard_validation_failed"

    max_iters = max(1, int(row.get("max_iters", 1)))
    iters = int(row.get("iters_used", 0))
    checks = max(1, int(row.get("collision_checks", 0)))

    if iters >= int(0.95 * max_iters):
        return "timeout"

    collision_pressure = checks / max(1, iters)
    if collision_pressure > 2.0:
        return "collision_stuck"

    bridge_ratio = float(row.get("bridge_usage_ratio", 0.0))
    if bridge_ratio < unstable_threshold:
        return "unstable"
    return "timeout"
