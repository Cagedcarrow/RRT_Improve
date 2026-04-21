from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Dict, Iterable, List


def summarize_records(records: Iterable[dict]) -> dict:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    failure_counts: Dict[str, int] = defaultdict(int)

    for r in records:
        grouped[r["planner"]].append(r)
        ft = r.get("failure_type", "")
        if ft:
            failure_counts[ft] += 1

    planners = {}
    for name, rows in grouped.items():
        success_rows = [x for x in rows if x["success"]]
        planners[name] = {
            "runs": len(rows),
            "success_rate": sum(1 for x in rows if x["success"]) / max(1, len(rows)),
            "median_time_ms": median([x["time_ms"] for x in rows]) if rows else 0.0,
            "median_path_len": median([x["path_len"] for x in success_rows]) if success_rows else 0.0,
            "median_collision_checks": median([x["collision_checks"] for x in rows]) if rows else 0.0,
        }

    return {
        "planners": planners,
        "failure_counts": dict(failure_counts),
    }


def baseline_length_by_scene(records: Iterable[dict], planner_name: str = "rrt_star") -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if r.get("planner") == planner_name and r.get("success"):
            grouped[r["scene_id"]].append(float(r["path_len"]))
    out = {}
    for k, vals in grouped.items():
        out[k] = median(vals)
    return out
