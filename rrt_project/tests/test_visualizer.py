from pathlib import Path

from rrt_project.analysis.visualizer import render_run
from rrt_project.env.scene_2d import build_scenes_2d


def test_visualizer_outputs_image(tmp_path: Path):
    scene = build_scenes_2d()[0]
    out = tmp_path / "img.png"
    path = [scene.start.tolist(), scene.goal.tolist()]
    tree = [scene.start.tolist(), ((scene.start + scene.goal) * 0.5).tolist(), scene.goal.tolist()]
    render_run(scene, planner="rrt", seed=0, path=path, tree_nodes=tree, out_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
