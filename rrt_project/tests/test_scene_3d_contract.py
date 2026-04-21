import numpy as np

from rrt_project.env.scene_3d import build_scenes_3d


def test_scene_3d_cube_contract():
    scenes = build_scenes_3d()
    assert len(scenes) == 2
    for scene in scenes:
        assert scene.dimension == 3
        assert np.allclose(scene.bounds, np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]))
        assert np.allclose(scene.start, np.array([0.0, 0.0, 0.0]))
        assert np.allclose(scene.goal, np.array([10.0, 10.0, 10.0]))

