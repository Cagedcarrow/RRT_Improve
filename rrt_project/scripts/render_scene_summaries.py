from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from rrt_project.benchmark.runner import default_config, planners_factory
from rrt_project.env.scene_2d import build_scenes_2d
from rrt_project.env.scene_3d import build_scenes_3d


def _cube_faces(center: np.ndarray, ext: np.ndarray):
    cx, cy, cz = center
    ex, ey, ez = ext
    v = np.array(
        [
            [cx - ex, cy - ey, cz - ez],
            [cx + ex, cy - ey, cz - ez],
            [cx + ex, cy + ey, cz - ez],
            [cx - ex, cy + ey, cz - ez],
            [cx - ex, cy - ey, cz + ez],
            [cx + ex, cy - ey, cz + ez],
            [cx + ex, cy + ey, cz + ez],
            [cx - ex, cy + ey, cz + ez],
        ]
    )
    return [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[4], v[7], v[3], v[0]],
    ]


def _planner_specs():
    # Keep the panel compact: one representative mode per planner.
    return [
        ("rrt", "baseline", "RRT"),
        ("rrt_connect", "baseline", "RRT-Connect"),
        ("rrt_star", "baseline", "RRT*"),
        ("cci_bridge_rrt", "cci_bridge", "CCI-Bridge-RRT"),
    ]


def _scaled_config_for_scene(cfg, scene):
    span = scene.bounds[:, 1] - scene.bounds[:, 0]
    scale = float(np.mean(span))
    if int(scene.dimension) != 3 or scale <= 2.0:
        return cfg
    # Scale-sensitive tuning for large 3D cube scenes.
    return replace(
        cfg,
        step_size=max(cfg.step_size * scale, 0.45),
        goal_tolerance=max(cfg.goal_tolerance * scale, 0.8),
        rewire_radius=max(cfg.rewire_radius * scale, 1.2),
        bridge_delta=max(cfg.bridge_delta * scale, 0.25),
        max_iters=max(cfg.max_iters, 1400),
        max_time_sec=max(cfg.max_time_sec, 8.0),
    )


def _draw_panel_2d(ax, scene, path, tree_nodes, tree_parents, title: str) -> None:
    for obs in scene.obstacles:
        c = np.asarray(obs.center, dtype=float)
        e = np.asarray(obs.half_extents, dtype=float)
        rect = plt.Rectangle(
            (c[0] - e[0], c[1] - e[1]),
            2.0 * e[0],
            2.0 * e[1],
            facecolor="#5c677d",
            edgecolor="#1f2937",
            linewidth=1.4,
            alpha=1.0,
        )
        ax.add_patch(rect)

    arr_nodes = np.asarray(tree_nodes, dtype=float) if tree_nodes else np.empty((0, 2), dtype=float)
    parents = list(tree_parents or [])
    if arr_nodes.shape[0] and parents:
        for i, p in enumerate(parents[: arr_nodes.shape[0]]):
            if p < 0 or p >= arr_nodes.shape[0]:
                continue
            ax.plot(
                [arr_nodes[p, 0], arr_nodes[i, 0]],
                [arr_nodes[p, 1], arr_nodes[i, 1]],
                c="#60a5fa",
                alpha=0.28,
                linewidth=0.65,
                zorder=1,
            )
    if arr_nodes.shape[0] > 1200:
        arr_nodes = arr_nodes[:: max(1, arr_nodes.shape[0] // 1200)]
    if arr_nodes.size:
        ax.scatter(arr_nodes[:, 0], arr_nodes[:, 1], s=2, c="lightsteelblue", alpha=0.45)

    arr_path = np.asarray(path, dtype=float) if path else np.empty((0, 2), dtype=float)
    if arr_path.shape[0] >= 2:
        ax.plot(arr_path[:, 0], arr_path[:, 1], c="#ff0000", linewidth=3.4, zorder=20)

    start = scene.bounds[:, 0]
    goal = scene.bounds[:, 1]
    ax.scatter([start[0]], [start[1]], c="#10b981", s=52, marker="o")
    ax.scatter([goal[0]], [goal[1]], c="#7c3aed", s=66, marker="*")
    ax.set_xlim(scene.bounds[0, 0], scene.bounds[0, 1])
    ax.set_ylim(scene.bounds[1, 0], scene.bounds[1, 1])
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_title(title, fontsize=10)


def _draw_panel_3d(ax, scene, path, tree_nodes, tree_parents, title: str) -> None:
    for obs in scene.obstacles:
        faces = _cube_faces(np.asarray(obs.center, dtype=float), np.asarray(obs.half_extents, dtype=float))
        mesh = Poly3DCollection(
            faces,
            alpha=1.0,
            facecolor="#94a3b8",
            edgecolor="#0f172a",
            linewidth=0.8,
        )
        ax.add_collection3d(mesh)

    arr_nodes = np.asarray(tree_nodes, dtype=float) if tree_nodes else np.empty((0, 3), dtype=float)
    parents = list(tree_parents or [])
    if arr_nodes.shape[0] and parents:
        for i, p in enumerate(parents[: arr_nodes.shape[0]]):
            if p < 0 or p >= arr_nodes.shape[0]:
                continue
            ax.plot(
                [arr_nodes[p, 0], arr_nodes[i, 0]],
                [arr_nodes[p, 1], arr_nodes[i, 1]],
                [arr_nodes[p, 2], arr_nodes[i, 2]],
                c="#60a5fa",
                alpha=0.22,
                linewidth=0.65,
                zorder=1,
            )
    if arr_nodes.shape[0] > 900:
        arr_nodes = arr_nodes[:: max(1, arr_nodes.shape[0] // 900)]
    if arr_nodes.size:
        ax.scatter(arr_nodes[:, 0], arr_nodes[:, 1], arr_nodes[:, 2], s=1.5, c="lightsteelblue", alpha=0.35)

    arr_path = np.asarray(path, dtype=float) if path else np.empty((0, 3), dtype=float)
    if arr_path.shape[0] >= 2:
        ax.plot(arr_path[:, 0], arr_path[:, 1], arr_path[:, 2], c="#ff0000", linewidth=3.4, zorder=20)

    bounds_min = scene.bounds[:, 0]
    bounds_max = scene.bounds[:, 1]
    ax.scatter([bounds_min[0]], [bounds_min[1]], [bounds_min[2]], c="#10b981", s=52, marker="o", label="start")
    ax.scatter([bounds_max[0]], [bounds_max[1]], [bounds_max[2]], c="#7c3aed", s=72, marker="*", label="goal")

    # Draw scene bounding cube wireframe to make corner locations explicit.
    wire = np.array(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_max[1], bounds_max[2]],
            [bounds_min[0], bounds_max[1], bounds_max[2]],
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [wire[i, 0], wire[j, 0]],
            [wire[i, 1], wire[j, 1]],
            [wire[i, 2], wire[j, 2]],
            color="#111827",
            linewidth=1.0,
            alpha=0.85,
        )

    ax.set_xlim(bounds_min[0], bounds_max[0])
    ax.set_ylim(bounds_min[1], bounds_max[1])
    ax.set_zlim(bounds_min[2], bounds_max[2])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    # Left-bottom to right-top diagonal is more intuitive under this viewpoint.
    ax.view_init(elev=22, azim=-58)
    ax.set_title(title, fontsize=10)


def render_scene_seed_summary(scene, seed: int, out_path: Path) -> None:
    planners = planners_factory()
    cfg, _raw = default_config()
    cfg = _scaled_config_for_scene(cfg, scene)
    specs = _planner_specs()

    dim = int(scene.dimension)
    fig = plt.figure(figsize=(20, 16), dpi=220)

    for idx, (planner_name, ablation, label) in enumerate(specs, start=1):
        planner = planners[planner_name]
        run_cfg = replace(cfg, seed=int(seed), ablation_mode=ablation)
        start = scene.bounds[:, 0].astype(np.float64)
        goal = scene.bounds[:, 1].astype(np.float64)
        result = planner.plan(scene, start, goal, run_cfg)
        tag = "OK" if result.success else "FAIL"
        panel_title = f"{label} [{ablation}] | {tag} | {result.time_ms:.0f}ms"

        if dim == 2:
            ax = fig.add_subplot(2, 2, idx)
            _draw_panel_2d(
                ax,
                scene,
                result.path,
                result.meta.get("tree_nodes", []),
                result.meta.get("tree_parents", []),
                panel_title,
            )
        else:
            ax = fig.add_subplot(2, 2, idx, projection="3d")
            _draw_panel_3d(
                ax,
                scene,
                result.path,
                result.meta.get("tree_nodes", []),
                result.meta.get("tree_parents", []),
                panel_title,
            )

    fig.suptitle(
        f"{scene.scene_id} | seed={seed} | start={tuple(scene.bounds[:, 0])} -> goal={tuple(scene.bounds[:, 1])} | hard-obstacle validated",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Obstacles are rendered opaque; planner output is accepted only after hard obstacle validation.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#334155",
    )
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render compact scene summary figures.")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--count", type=int, default=5, help="how many summary images per scene")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "images" / "scene_summaries",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="remove old summary images before rendering",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if args.clean and out_dir.exists():
        for p in out_dir.rglob("*.png"):
            p.unlink()

    scenes = build_scenes_2d() + build_scenes_3d()
    seeds = list(range(args.seed_start, args.seed_start + args.count))

    for scene in scenes:
        for seed in seeds:
            out_path = out_dir / scene.scene_id / f"{scene.scene_id}_seed_{seed}.png"
            render_scene_seed_summary(scene=scene, seed=seed, out_path=out_path)
            print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
