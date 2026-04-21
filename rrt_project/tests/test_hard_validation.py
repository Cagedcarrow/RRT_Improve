import numpy as np

from rrt_project.algorithms.base_planner import BasePlanner, PlannerConfig
from rrt_project.core.collision_backend import CollisionBackend
from rrt_project.env.obstacles import AABBObstacle
from rrt_project.env.scene_state import SceneState


class _DummyPlanner(BasePlanner):
    def plan(self, scene, start, goal, config):  # pragma: no cover
        raise NotImplementedError


def _scene():
    return SceneState(
        scene_id="hard_validate_test",
        dimension=2,
        difficulty="complex",
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        start=np.array([0.1, 0.5], dtype=np.float64),
        goal=np.array([0.9, 0.5], dtype=np.float64),
        obstacles=[AABBObstacle(np.array([0.5, 0.5]), np.array([0.12, 0.12]), "block")],
    )


def test_hard_validation_rejects_colliding_path_and_truncates_output():
    planner = _DummyPlanner(name="dummy", collision_backend=CollisionBackend(samples_per_segment=64))
    scene = _scene()
    cfg = PlannerConfig(
        step_size=0.08,
        max_iters=10,
        goal_bias=0.0,
        bridge_bias=0.0,
        cuda_enabled=False,
        alpha_schedule=[(0.0, 1.0), (1.0, 1.0)],
        final_hard_validation=True,
        hard_validation_alpha=1.0,
        obstacle_min_scale=1.0,
    )

    # This segment intersects the true obstacle and must be rejected.
    path = [[0.35, 0.56], [0.65, 0.56]]
    assert planner.collision_backend.segment_collides(
        path[0], path[1], scene, alpha=1.0, cuda_enabled=False, obstacle_min_scale=1.0
    ) is True

    meta = {}
    validated_success, validated_path = planner._enforce_hard_validation(
        success=True,
        path=path,
        scene=scene,
        config=cfg,
        meta=meta,
    )
    assert validated_success is False
    assert len(validated_path) == 1
    assert np.allclose(validated_path[0], np.array([0.35, 0.56]))
    assert meta["planned_success"] is True
    assert meta["hard_validation_passed"] is False
    assert meta["hard_validation_collision_idx"] == 0.0
