from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rrt_project.algorithms.base_planner import PlannerConfig
from rrt_project.algorithms.cci_bridge_rrt import CCIBridgeRRTPlanner
from rrt_project.algorithms.rrt import RRTPlanner
from rrt_project.algorithms.rrt_connect import RRTConnectPlanner
from rrt_project.algorithms.rrt_star import RRTStarPlanner
from rrt_project.analysis.report import write_markdown_report
from rrt_project.analysis.visualizer import render_collage, render_run
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
IMAGES_DIR = PACKAGE_ROOT / "images"
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
    "ablation_mode",
    "planned_success",
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
    "hard_validation_passed",
    "hard_validation_collision_idx",
    "max_iters",
    "sample_source_counts",
    "alpha_trace",
    "alpha_trace_stats",
    "bridge_candidate_stats",
    "leaf_plan_batches",
    "cpu_gpu_consistency",
    "image_path",
]


class BenchmarkRunner:
    def __init__(self, base_config: PlannerConfig, limit_per_minute: int = 40, viz_cfg: Optional[dict] = None) -> None:
        self.base_config = base_config
        self.limiter = RateLimiter(limit_per_minute=limit_per_minute, state_path=RATE_STATE_PATH)
        self._snapshot_t0 = time.time()
        self._consistency_cache: Dict[str, bool] = {}
        self.viz_cfg = viz_cfg or {}

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    def run_matrix(
        self,
        planners: Sequence,
        scenes: Sequence,
        seeds: Sequence[int],
        ablations: Sequence[str],
        stage: str,
        round_id: int = 0,
        config_override: Optional[PlannerConfig] = None,
    ) -> List[dict]:
        cfg = config_override or self.base_config
        rows: List[dict] = []

        existing = load_records(RECORDS_PATH)
        baseline_len = baseline_length_by_scene(existing)

        for scene in scenes:
            for planner in planners:
                for ablation in ablations:
                    for seed in seeds:
                        self.limiter.acquire(label=f"{stage}:{planner.name}:{ablation}")
                        run_cfg = replace(cfg, seed=int(seed), ablation_mode=str(ablation))
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

    def _cpu_gpu_consistency(self, scene, config: PlannerConfig) -> bool:
        key = f"{scene.scene_id}:{config.ablation_mode}"
        if key in self._consistency_cache:
            return self._consistency_cache[key]

        if not config.cuda_enabled:
            self._consistency_cache[key] = True
            return True

        try:
            import torch

            if not torch.cuda.is_available():
                self._consistency_cache[key] = True
                return True
        except Exception:
            self._consistency_cache[key] = True
            return True

        backend = CollisionBackend(samples_per_segment=24)
        p1 = scene.start
        p2 = scene.goal
        cpu = backend.segment_collides(p1, p2, scene_state=scene, alpha=1.0, cuda_enabled=False)
        gpu = backend.segment_collides(p1, p2, scene_state=scene, alpha=1.0, cuda_enabled=True)
        ok = bool(cpu == gpu)
        self._consistency_cache[key] = ok
        return ok

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

                consistency = self._cpu_gpu_consistency(scene, config)
                sample_source = result.meta.get("sample_source_counts", {"goal": 0, "bridge": 0, "uniform": 0})
                alpha_trace = result.meta.get("alpha_trace", [])
                alpha_stats = result.meta.get("alpha_trace_stats", {})
                bridge_stats = result.meta.get("bridge_candidate_stats", {})
                leaf_batches = float(result.meta.get("leaf_plan_batches", 0.0))

                image_path = IMAGES_DIR / scene.scene_id / planner.name / f"{config.ablation_mode}_seed_{config.seed}.png"
                if self.viz_cfg.get("enable", True):
                    render_run(
                        scene=scene,
                        planner=f"{planner.name}:{config.ablation_mode}",
                        seed=config.seed,
                        path=result.path,
                        tree_nodes=result.meta.get("tree_nodes", []),
                        tree_parents=result.meta.get("tree_parents", []),
                        out_path=image_path,
                        dpi=int(self.viz_cfg.get("image_dpi", 140)),
                        max_tree_nodes_to_draw=int(self.viz_cfg.get("max_tree_nodes_to_draw", 1400)),
                    )

                row = {
                    "run_id": f"{stage}_{planner.name}_{scene.scene_id}_{config.ablation_mode}_{config.seed}_{int(time.time()*1000)}",
                    "stage": stage,
                    "round_id": int(round_id),
                    "seed": int(config.seed),
                    "scene_id": scene.scene_id,
                    "difficulty": scene.difficulty,
                    "planner": planner.name,
                    "ablation_mode": config.ablation_mode,
                    "planned_success": bool(result.meta.get("planned_success", result.success)),
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
                    "hard_validation_passed": bool(result.meta.get("hard_validation_passed", result.success)),
                    "hard_validation_collision_idx": int(float(result.meta.get("hard_validation_collision_idx", -1))),
                    "max_iters": int(config.max_iters),
                    "sample_source_counts": json.dumps(sample_source, ensure_ascii=False),
                    "alpha_trace": json.dumps(alpha_trace, ensure_ascii=False),
                    "alpha_trace_stats": json.dumps(alpha_stats, ensure_ascii=False),
                    "bridge_candidate_stats": json.dumps(bridge_stats, ensure_ascii=False),
                    "leaf_plan_batches": leaf_batches,
                    "cpu_gpu_consistency": bool(consistency),
                    "image_path": str(image_path.relative_to(PACKAGE_ROOT)),
                }
                row["failure_type"] = classify_failure(row, baseline_len=baseline_len)
                return row
            except Exception as ex:  # noqa: BLE001
                err = ex
                if i < 3:
                    time.sleep(backoff[i])
                    continue

        return {
            "run_id": f"{stage}_{planner.name}_{scene.scene_id}_{config.ablation_mode}_{config.seed}_{int(time.time()*1000)}",
            "stage": stage,
            "round_id": int(round_id),
            "seed": int(config.seed),
            "scene_id": scene.scene_id,
            "difficulty": scene.difficulty,
            "planner": planner.name,
            "ablation_mode": config.ablation_mode,
            "planned_success": False,
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
            "hard_validation_passed": False,
            "hard_validation_collision_idx": -1,
            "max_iters": int(config.max_iters),
            "sample_source_counts": json.dumps({"goal": 0, "bridge": 0, "uniform": 0}),
            "alpha_trace": "[]",
            "alpha_trace_stats": "{}",
            "bridge_candidate_stats": "{}",
            "leaf_plan_batches": 0.0,
            "cpu_gpu_consistency": False,
            "image_path": "",
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
            row["planned_success"] = str(row.get("planned_success", "")).lower() in {"true", "1", "yes"}
            row["cuda_enabled"] = str(row.get("cuda_enabled", "")).lower() in {"true", "1", "yes"}
            row["cpu_gpu_consistency"] = str(row.get("cpu_gpu_consistency", "")).lower() in {"true", "1", "yes"}
            row["hard_validation_passed"] = str(row.get("hard_validation_passed", "")).lower() in {"true", "1", "yes"}
            for key in ["time_ms", "path_len", "alpha_final", "bridge_usage_ratio", "leaf_plan_batches"]:
                row[key] = float(row.get(key, 0.0) or 0.0)
            for key in [
                "seed",
                "collision_checks",
                "iters_used",
                "goal_reached_iter",
                "round_id",
                "max_iters",
                "hard_validation_collision_idx",
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


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return __import__("yaml").safe_load(f)


def default_config() -> tuple[PlannerConfig, dict]:
    cfg = load_config(PACKAGE_ROOT / "configs" / "default.yaml")
    planner = cfg["planner"]
    p = PlannerConfig(
        step_size=float(planner["step_size"]),
        max_iters=int(planner["max_iters"]),
        goal_bias=float(planner["goal_bias"]),
        bridge_bias=float(planner["bridge_bias"]),
        cuda_enabled=bool(planner["cuda_enabled"]),
        alpha_schedule=[tuple(x) for x in planner["alpha_schedule"]],
        goal_tolerance=float(planner.get("goal_tolerance", 0.08)),
        rewire_radius=float(planner.get("rewire_radius", 0.22)),
        max_time_sec=float(planner.get("max_time_sec", cfg["benchmark"].get("timeout_sec", 3.0))),
        bridge_num_samples=int(planner.get("bridge_num_samples", 260)),
        bridge_delta=float(planner.get("bridge_delta", 0.05)),
        bridge_cluster_eps=float(planner.get("bridge_cluster_eps", 0.10)),
        bridge_cluster_min_pts=int(planner.get("bridge_cluster_min_pts", 3)),
        bridge_clustering=str(planner.get("bridge_clustering", "dbscan")),
        cci_eta=float(planner.get("cci_eta", 18.0)),
        enable_cci=True,
        enable_bridge=True,
        ablation_mode="baseline",
        final_hard_validation=bool(planner.get("final_hard_validation", True)),
        hard_validation_alpha=float(planner.get("hard_validation_alpha", 1.0)),
        obstacle_min_scale=float(planner.get("obstacle_min_scale", 0.25)),
    )
    return p, cfg


def planners_factory() -> dict:
    return {
        "rrt": RRTPlanner(CollisionBackend()),
        "rrt_connect": RRTConnectPlanner(CollisionBackend()),
        "rrt_star": RRTStarPlanner(CollisionBackend()),
        "cci_bridge_rrt": CCIBridgeRRTPlanner(CollisionBackend()),
    }


def _scene_lookup() -> dict:
    scenes = build_scenes_2d() + build_scenes_3d()
    return {s.scene_id: s for s in scenes}


def _build_collages(viz_cfg: dict) -> None:
    images = sorted(IMAGES_DIR.rglob("*.png"))
    if not images:
        return
    render_collage(
        images,
        out_path=REPORT_DIR / "collage_overview.png",
        max_images=int(viz_cfg.get("collage_max_images", 48)),
        cols=6,
        dpi=int(viz_cfg.get("image_dpi", 120)),
    )


def run_full_pipeline(resume: bool, limit_per_minute: int) -> None:
    cfg, raw_cfg = default_config()
    seeds = raw_cfg["benchmark"]["seeds"]
    ablations = raw_cfg["benchmark"]["ablations"]
    planner_names = raw_cfg["benchmark"]["planners"]
    viz_cfg = raw_cfg.get("visualization", {})

    if not resume:
        for path in [RECORDS_PATH, SUMMARY_PATH, STATE_PATH, RATE_STATE_PATH]:
            if path.exists():
                path.unlink()
        if IMAGES_DIR.exists():
            shutil.rmtree(IMAGES_DIR)

    state = load_state(STATE_PATH)
    if state.get("current_config"):
        cfg = PlannerConfig(**state["current_config"])

    runner = BenchmarkRunner(base_config=cfg, limit_per_minute=limit_per_minute, viz_cfg=viz_cfg)
    plans = planners_factory()
    scene_map = _scene_lookup()
    scenes_all = [scene_map[k] for k in sorted(scene_map)]
    scenes_complex = [s for s in scenes_all if s.difficulty == "complex"]

    selected_planners = [plans[n] for n in planner_names]

    completed = set(state.get("completed_stages", []))

    if "MATRIX" not in completed:
        runner.run_matrix(
            planners=selected_planners,
            scenes=scenes_all,
            seeds=seeds,
            ablations=ablations,
            stage="MATRIX",
            config_override=cfg,
        )
        completed.add("MATRIX")
        state["completed_stages"] = sorted(completed)
        save_state(STATE_PATH, state)

    tuning_history = state.get("tuning_history", [])
    m5_cfg = cfg
    if state.get("current_config"):
        m5_cfg = PlannerConfig(**state["current_config"])

    for rid in [1, 2]:
        stage = f"TUNE_R{rid}"
        if stage in completed:
            continue
        rows = runner.run_matrix(
            planners=[plans["cci_bridge_rrt"]],
            scenes=scenes_complex,
            seeds=seeds,
            ablations=["cci_bridge"],
            stage=stage,
            round_id=rid,
            config_override=replace(m5_cfg, ablation_mode="cci_bridge"),
        )
        failures = [r.get("failure_type", "") for r in rows]
        next_cfg, changes = tune_from_failures(m5_cfg, failures)
        tuning_history.append({"round": rid, "changes": changes, "config_after": config_to_dict(next_cfg)})
        m5_cfg = next_cfg

        completed.add(stage)
        state["completed_stages"] = sorted(completed)
        state["tuning_history"] = tuning_history
        state["current_config"] = config_to_dict(m5_cfg)
        save_state(STATE_PATH, state)

    all_rows = load_records(RECORDS_PATH)
    summary = summarize_records(all_rows)
    summary["tuning_history"] = tuning_history
    summary["completed_stages"] = sorted(completed)
    summary["rate_limit_target_per_min"] = limit_per_minute
    summary["matrix"] = {
        "seeds": seeds,
        "ablations": ablations,
        "planners": planner_names,
        "scene_count": len(scenes_all),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _build_collages(viz_cfg)
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
