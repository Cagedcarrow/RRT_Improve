from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


def _draw_aabb_2d(ax, obs, alpha: float = 1.0) -> None:
    c = np.asarray(obs.center, dtype=float)
    e = np.asarray(obs.half_extents, dtype=float)
    rect = Rectangle(
        (c[0] - e[0], c[1] - e[1]),
        2 * e[0],
        2 * e[1],
        facecolor="#5c677d",
        edgecolor="#1f2937",
        linewidth=1.5,
        alpha=alpha,
    )
    ax.add_patch(rect)


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


def render_run(
    scene,
    planner: str,
    seed: int,
    path: Sequence[Sequence[float]],
    tree_nodes: Sequence[Sequence[float]],
    out_path: Path,
    tree_parents: Sequence[int] | None = None,
    dpi: int = 140,
    max_tree_nodes_to_draw: int = 1400,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dim = int(scene.dimension)
    arr_path = np.asarray(path, dtype=float) if path else np.empty((0, dim), dtype=float)
    arr_nodes = np.asarray(tree_nodes, dtype=float) if tree_nodes else np.empty((0, dim), dtype=float)
    parents = list(tree_parents or [])

    def _iter_edge_indices(n: int):
        if not parents:
            return []
        take_every = max(1, n // max_tree_nodes_to_draw) if n > max_tree_nodes_to_draw else 1
        selected = set(range(0, n, take_every))
        edges = []
        for i, p in enumerate(parents[:n]):
            if i not in selected:
                continue
            if 0 <= p < n and p in selected:
                edges.append((p, i))
        return edges

    if arr_nodes.shape[0] > max_tree_nodes_to_draw:
        step = max(1, arr_nodes.shape[0] // max_tree_nodes_to_draw)
        arr_nodes = arr_nodes[::step]

    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        for obs in scene.obstacles:
            _draw_aabb_2d(ax, obs)

        full_nodes = np.asarray(tree_nodes, dtype=float) if tree_nodes else np.empty((0, 2), dtype=float)
        for p, c in _iter_edge_indices(full_nodes.shape[0]):
            ax.plot(
                [full_nodes[p, 0], full_nodes[c, 0]],
                [full_nodes[p, 1], full_nodes[c, 1]],
                c="#60a5fa",
                alpha=0.35,
                linewidth=0.8,
                zorder=2,
            )

        if arr_nodes.size:
            ax.scatter(arr_nodes[:, 0], arr_nodes[:, 1], s=4, c="lightsteelblue", alpha=0.5, label="tree")

        if arr_path.shape[0] >= 2:
            ax.plot(arr_path[:, 0], arr_path[:, 1], c="#ef4444", linewidth=2.8, label="path")

        ax.scatter([scene.start[0]], [scene.start[1]], c="tab:green", s=70, marker="o", label="start")
        ax.scatter([scene.goal[0]], [scene.goal[1]], c="tab:purple", s=70, marker="*", label="goal")
        ax.set_xlim(scene.bounds[0, 0], scene.bounds[0, 1])
        ax.set_ylim(scene.bounds[1, 0], scene.bounds[1, 1])
        ax.set_title(f"{scene.scene_id} | {planner} | seed={seed}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=8)
        fig.text(
            0.5,
            0.01,
            "Obstacle rendering and final validity use hard constraints (non-penetrable obstacles).",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    # 3D: one image with two viewpoints
    fig = plt.figure(figsize=(11, 5), dpi=dpi)
    views = [(22, 36), (20, 132)]
    for i, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        for obs in scene.obstacles:
            faces = _cube_faces(np.asarray(obs.center, dtype=float), np.asarray(obs.half_extents, dtype=float))
            mesh = Poly3DCollection(
                faces,
                alpha=1.0,
                facecolor="#94a3b8",
                edgecolor="#0f172a",
                linewidth=0.9,
            )
            ax.add_collection3d(mesh)

        full_nodes = np.asarray(tree_nodes, dtype=float) if tree_nodes else np.empty((0, 3), dtype=float)
        for p, c in _iter_edge_indices(full_nodes.shape[0]):
            ax.plot(
                [full_nodes[p, 0], full_nodes[c, 0]],
                [full_nodes[p, 1], full_nodes[c, 1]],
                [full_nodes[p, 2], full_nodes[c, 2]],
                c="#60a5fa",
                alpha=0.30,
                linewidth=0.8,
            )

        if arr_nodes.size:
            ax.scatter(arr_nodes[:, 0], arr_nodes[:, 1], arr_nodes[:, 2], s=3, c="lightsteelblue", alpha=0.4)
        if arr_path.shape[0] >= 2:
            ax.plot(arr_path[:, 0], arr_path[:, 1], arr_path[:, 2], c="#ef4444", linewidth=2.8)

        ax.scatter([scene.start[0]], [scene.start[1]], [scene.start[2]], c="tab:green", s=42, marker="o")
        ax.scatter([scene.goal[0]], [scene.goal[1]], [scene.goal[2]], c="tab:purple", s=48, marker="*")
        ax.set_xlim(scene.bounds[0, 0], scene.bounds[0, 1])
        ax.set_ylim(scene.bounds[1, 0], scene.bounds[1, 1])
        ax.set_zlim(scene.bounds[2, 0], scene.bounds[2, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"view {i}: elev={elev}, azim={azim}")

    fig.suptitle(f"{scene.scene_id} | {planner} | seed={seed}")
    fig.text(
        0.5,
        0.01,
        "Obstacle rendering and final validity use hard constraints (non-penetrable obstacles).",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#334155",
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_collage(image_paths: Sequence[Path], out_path: Path, max_images: int = 48, cols: int = 6, dpi: int = 120) -> None:
    picks = list(image_paths[:max_images])
    if not picks:
        return
    n = len(picks)
    cols = max(1, cols)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.5 * rows), dpi=dpi)
    axes = np.asarray(axes).reshape(rows, cols)
    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        if idx < n:
            img = plt.imread(picks[idx])
            ax.imshow(img)
            ax.set_title(picks[idx].stem, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
