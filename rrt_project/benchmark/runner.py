from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from rrt_project.algorithms.base_planner import PlannerConfig
from rrt_project.algorithms.cci_bridge_rrt import CCIBridgeRRTPlanner
from rrt_project.algorithms.rrt import RRTPlanner
from rrt_project.algorithms.rrt_connect import RRTConnectPlanner
from rrt_project.algorithms.rrt_star import RRTStarPlanner
from rrt_project.analysis.report import write_markdown_report
from rrt_project.benchmark.failure_classifier import classify_failure
from rrt_project.benchmark.metrics import baseline_length_by_scene, summarize_records
from rrt_project.benchmark.rate_limiter import RateLimiter
from rrt_project.benchmark.tuner import config_to_dict, tune_from_failures
from rrt_project.core.collision_backend import CollisionBackend
from rrt_project.env.scene_2d import build_scenes_2d
from rrt_project.env.scene_3d import build_scenes_3d

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PACKAGE_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
REPORT_DIR = RESULTS_DIR / "reports"
RECORDS_PATH = RESULTS_DIR / "records.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
STATE_PATH = CHECKPOINT_DIR / "stage_state.json"
RATE_STATE_PATH = CHECKPOINT_DIR / "rate_state.json"

CSV_FIELDS = [
    "run_id",
    "stage",
    "round_id",
    "seed",
    "scene_id",
    "difficulty",
    "planner",
    "success",
    "time_ms",
    "path_len",
    "collision_checks",
    "iters_used",
    "goal_reached_iter",
    "cuda_enabled",
    "device_name",
    "alpha_final",
    "bridge_usage_ratio",
    "failure_type",
    "max_iters",
]


class BenchmarkRunner:
    def __init__(self, base_config: PlannerConfig, limit_per_minute: int = 40) -> None:
        self.base_config = base_config
        self.limiter = RateLimiter(limit_per_minute=limit_per_minute, state_path=RATE_STATE_PATH)
        self._snapshot_t0 = time.time()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        planners: Sequence,
        scenes: Sequence,
        seeds: Sequence[int],
        stage: str = "adhoc",
        round_id: int = 0,
        config_override: Optional[PlannerConfig] = None,
    ) -> List[dict]:
        cfg = config_override or self.base_config
        rows: List[dict] = []

        existing = load_records(RECORDS_PATH)
        baseline_len = baseline_length_by_scene(existing)

        for scene in scenes:
            for planner in planners:
                for seed in seeds:
                    self.limiter.acquire(label=f"{stage}:{planner.name}")
                    run_cfg = replace(cfg, seed=int(seed))
                    row = self._execute_with_retry(
                        planner=planner,
                        scene=scene,
                        config=run_cfg,
                        stage=stage,
                        round_id=round_id,
                        baseline_len=baseline_len,
                    )
                    rows.append(row)
                    append_record(RECORDS_PATH, row)

                    if time.time() - self._snapshot_t0 > 600:
                        snap = summarize_records(load_records(RECORDS_PATH))
                        snap_path = REPORT_DIR / f"snapshot_{stage}_{int(time.time())}.json"
                        with snap_path.open("w", encoding="utf-8") as f:
                            json.dump(snap, f, ensure_ascii=False, indent=2)
                        self._snapshot_t0 = time.time()

        return rows

    def _execute_with_retry(
        self,
        planner,
        scene,
        config: PlannerConfig,
        stage: str,
        round_id: int,
        baseline_len: Dict[str, float],
    ) -> dict:
        backoff = [10, 30, 60]
        err = None
        for i in range(4):
            try:
                result = planner.plan(scene, scene.start, scene.goal, config)
                device_name = "cuda" if config.cuda_enabled else "cpu"
                if config.cuda_enabled:
                    try:
                        import torch

                        if torch.cuda.is_available():
                            device_name = torch.cuda.get_device_name(0)
                    except Exception:
                        device_name = "cuda"

                row = {
                    "run_id": f"{stage}_{planner.name}_{scene.scene_id}_{config.seed}_{int(time.time()*1000)}",
                    "stage": stage,
                    "round_id": int(round_id),
                    "seed": int(config.seed),
                    "scene_id": scene.scene_id,
                    "difficulty": scene.difficulty,
                    "planner": planner.name,
                    "success": bool(result.success),
                    "time_ms": float(result.time_ms),
                    "path_len": float(result.path_len),
                    "collision_checks": int(result.collision_checks),
                    "iters_used": int(result.iters_used),
                    "goal_reached_iter": int(result.goal_reached_iter),
                    "cuda_enabled": bool(config.cuda_enabled),
                    "device_name": device_name,
                    "alpha_final": float(result.meta.get("alpha_final", 1.0)),
                    "bridge_usage_ratio": float(result.meta.get("bridge_usage_ratio", 0.0)),
                    "failure_type": "",
                    "max_iters": int(config.max_iters),
                }
                row["failure_type"] = classify_failure(row, baseline_len=baseline_len)
                return row
            except Exception as ex:  # noqa: BLE001
                err = ex
                if i < 3:
                    time.sleep(backoff[i])
                    continue
        return {
            "run_id": f"{stage}_{planner.name}_{scene.scene_id}_{config.seed}_{int(time.time()*1000)}",
            "stage": stage,
            "round_id": int(round_id),
            "seed": int(config.seed),
            "scene_id": scene.scene_id,
            "difficulty": scene.difficulty,
            "planner": planner.name,
            "success": False,
            "time_ms": 0.0,
            "path_len": 0.0,
            "collision_checks": 0,
            "iters_used": 0,
            "goal_reached_iter": -1,
            "cuda_enabled": bool(config.cuda_enabled),
            "device_name": "unknown",
            "alpha_final": 1.0,
            "bridge_usage_ratio": 0.0,
            "failure_type": "unstable",
            "max_iters": int(config.max_iters),
            "error": str(err) if err else "unknown",
        }


def append_record(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        filtered = {k: row.get(k, "") for k in CSV_FIELDS}
        writer.writerow(filtered)


def load_records(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = dict(r)
            row["success"] = str(row.get("success", "")).lower() in {"true", "1", "yes"}
            for key in [
                "time_ms",
                "path_len",
                "alpha_final",
                "bridge_usage_ratio",
            ]:
                row[key] = float(row.get(key, 0.0) or 0.0)
            for key in [
                "seed",
                "collision_checks",
                "iters_used",
                "goal_reached_iter",
                "round_id",
                "max_iters",
            ]:
                row[key] = int(float(row.get(key, 0) or 0))
            rows.append(row)
    return rows


def load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "completed_stages": [],
            "tuning_history": [],
            "current_config": None,
            "last_updated": "",
        }
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: Path, state: dict) -> None:
    state["last_updated"] = datetime.utcnow().isoformat() + "Z"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def default_config() -> PlannerConfig:
    cfg = load_config(PACKAGE_ROOT / "configs" / "default.yaml")
    planner = cfg["planner"]
    return PlannerConfig(
        step_size=float(planner["step_size"]),
        max_iters=int(planner["max_iters"]),
        goal_bias=float(planner["goal_bias"]),
        bridge_bias=float(planner["bridge_bias"]),
        cuda_enabled=bool(planner["cuda_enabled"]),
        alpha_schedule=[tuple(x) for x in planner["alpha_schedule"]],
        goal_tolerance=float(planner.get("goal_tolerance", 0.08)),
        rewire_radius=float(planner.get("rewire_radius", 0.22)),
        max_time_sec=float(cfg["benchmark"].get("timeout_sec", 8)),
        bridge_num_samples=int(planner.get("bridge_num_samples", 300)),
        bridge_delta=float(planner.get("bridge_delta", 0.06)),
        bridge_cluster_eps=float(planner.get("bridge_cluster_eps", 0.12)),
        bridge_cluster_min_pts=int(planner.get("bridge_cluster_min_pts", 3)),
        enable_cci=True,
        enable_bridge=True,
    )


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def planners_factory() -> dict:
    return {
        "rrt": RRTPlanner(CollisionBackend()),
        "rrt_connect": RRTConnectPlanner(CollisionBackend()),
        "rrt_star": RRTStarPlanner(CollisionBackend()),
        "cci_bridge_rrt": CCIBridgeRRTPlanner(CollisionBackend()),
    }


def run_full_pipeline(resume: bool, limit_per_minute: int) -> None:
    cfg = default_config()
    cfg_file = load_config(PACKAGE_ROOT / "configs" / "default.yaml")
    seeds = cfg_file["benchmark"]["seeds"]

    if not resume:
        if RECORDS_PATH.exists():
            RECORDS_PATH.unlink()
        if SUMMARY_PATH.exists():
            SUMMARY_PATH.unlink()

    state = load_state(STATE_PATH)
    if state.get("current_config"):
        cc = state["current_config"]
        cfg = PlannerConfig(**cc)

    runner = BenchmarkRunner(base_config=cfg, limit_per_minute=limit_per_minute)
    plans = planners_factory()

    scenes_2d = build_scenes_2d()
    scenes_3d = build_scenes_3d()
    scenes_all = scenes_2d + scenes_3d
    scenes_simple = [s for s in scenes_all if s.difficulty == "simple"]
    scenes_complex = [s for s in scenes_all if s.difficulty == "complex"]

    completed = set(state.get("completed_stages", []))

    if "M1" not in completed:
        cfg_m1 = replace(cfg, enable_cci=False, enable_bridge=False, bridge_bias=0.0)
        runner.run(
            planners=[plans["rrt"], plans["rrt_connect"], plans["rrt_star"]],
            scenes=scenes_simple,
            seeds=seeds,
            stage="M1",
            config_override=cfg_m1,
        )
        completed.add("M1")
        state["completed_stages"] = sorted(completed)
        save_state(STATE_PATH, state)

    if "M2" not in completed:
        cfg_m2 = replace(cfg, enable_cci=True, enable_bridge=False, bridge_bias=0.0)
        runner.run(
            planners=[plans["cci_bridge_rrt"]],
            scenes=scenes_complex,
            seeds=seeds,
            stage="M2",
            config_override=cfg_m2,
        )
        completed.add("M2")
        state["completed_stages"] = sorted(completed)
        save_state(STATE_PATH, state)

    if "M3" not in completed:
        cfg_m3 = replace(cfg, enable_cci=True, enable_bridge=True)
        runner.run(
            planners=[plans["cci_bridge_rrt"]],
            scenes=scenes_complex,
            seeds=seeds,
            stage="M3",
            config_override=cfg_m3,
        )
        completed.add("M3")
        state["completed_stages"] = sorted(completed)
        save_state(STATE_PATH, state)

    if "M4" not in completed:
        cfg_base = replace(cfg, enable_cci=False, enable_bridge=False, bridge_bias=0.0)
        runner.run(
            planners=[plans["rrt"], plans["rrt_connect"], plans["rrt_star"]],
            scenes=scenes_all,
            seeds=seeds,
            stage="M4_baseline",
            config_override=cfg_base,
        )
        runner.run(
            planners=[plans["cci_bridge_rrt"]],
            scenes=scenes_all,
            seeds=seeds,
            stage="M4_cci",
            config_override=replace(cfg, enable_cci=True, enable_bridge=True),
        )
        completed.add("M4")
        state["completed_stages"] = sorted(completed)
        save_state(STATE_PATH, state)

    # M5 requires at least two rounds
    tuning_history = state.get("tuning_history", [])
    m5_cfg = cfg
    if state.get("current_config"):
        m5_cfg = PlannerConfig(**state["current_config"])

    for rid in [1, 2]:
        stage = f"M5_R{rid}"
        if stage in completed:
            continue
        rows = runner.run(
            planners=[plans["cci_bridge_rrt"]],
            scenes=scenes_complex,
            seeds=seeds,
            stage=stage,
            round_id=rid,
            config_override=replace(m5_cfg, enable_cci=True, enable_bridge=True),
        )
        failures = [r.get("failure_type", "") for r in rows]
        next_cfg, changes = tune_from_failures(m5_cfg, failures)
        tuning_history.append(
            {
                "round": rid,
                "changes": changes,
                "config_after": config_to_dict(next_cfg),
            }
        )
        m5_cfg = next_cfg
        completed.add(stage)
        state["completed_stages"] = sorted(completed)
        state["tuning_history"] = tuning_history
        state["current_config"] = config_to_dict(m5_cfg)
        save_state(STATE_PATH, state)

    # final summary
    all_rows = load_records(RECORDS_PATH)
    summary = summarize_records(all_rows)
    summary["tuning_history"] = tuning_history
    summary["completed_stages"] = sorted(completed)
    summary["rate_limit_target_per_min"] = limit_per_minute
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_markdown_report(SUMMARY_PATH, REPORT_DIR / "summary_report.md")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RRT benchmark pipeline runner")
    p.add_argument("--pipeline", choices=["full"], default="full")
    p.add_argument("--resume", action="store_true", help="resume from stage checkpoint")
    p.add_argument("--limit-per-minute", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_full_pipeline(resume=args.resume, limit_per_minute=args.limit_per_minute)


if __name__ == "__main__":
    main()
