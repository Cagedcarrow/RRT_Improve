from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Dict, Iterable, List, Tuple


def _group_key(row: dict) -> str:
    return f"{row.get('planner','unknown')}::{row.get('ablation_mode','baseline')}"


def summarize_records(records: Iterable[dict]) -> dict:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    failure_counts: Dict[str, int] = defaultdict(int)

    for r in records:
        grouped[_group_key(r)].append(r)
        ft = r.get("failure_type", "")
        if ft:
            failure_counts[ft] += 1

    groups = {}
    for name, rows in grouped.items():
        success_rows = [x for x in rows if x["success"]]
        planned_rows = [x for x in rows if x.get("planned_success", x["success"])]
        groups[name] = {
            "runs": len(rows),
            "planned_success_rate": len(planned_rows) / max(1, len(rows)),
            "success_rate": sum(1 for x in rows if x["success"]) / max(1, len(rows)),
            "hard_validation_pass_rate": sum(1 for x in rows if x.get("hard_validation_passed", x["success"]))
            / max(1, len(rows)),
            "median_time_ms": median([x["time_ms"] for x in rows]) if rows else 0.0,
            "median_path_len": median([x["path_len"] for x in success_rows]) if success_rows else 0.0,
            "median_collision_checks": median([x["collision_checks"] for x in rows]) if rows else 0.0,
        }

    planners = {}
    for full, stats in groups.items():
        planner, ablation = full.split("::", 1)
        planners.setdefault(planner, {})[ablation] = stats

    return {
        "planner_ablation_groups": groups,
        "planners": planners,
        "failure_counts": dict(failure_counts),
    }


def baseline_length_by_scene(records: Iterable[dict], planner_name: str = "rrt_star") -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if (
            r.get("planner") == planner_name
            and r.get("ablation_mode", "baseline") == "baseline"
            and r.get("success")
        ):
            grouped[r["scene_id"]].append(float(r["path_len"]))
    out = {}
    for k, vals in grouped.items():
        out[k] = median(vals)
    return out


def improvement_against_baseline(records: Iterable[dict], target_planner: str = "cci_bridge_rrt") -> Dict[str, float]:
    by_scene: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("success"):
            key = f"{r.get('planner')}::{r.get('ablation_mode','baseline')}"
            by_scene[r.get("scene_id", "")][key].append(1.0)

    improve = {}
    for scene, kv in by_scene.items():
        base = kv.get("rrt::baseline", [])
        tgt = kv.get(f"{target_planner}::cci_bridge", [])
        if base and tgt:
            improve[scene] = float((len(tgt) / len(tgt)) - (len(base) / len(base)))
    return improve
