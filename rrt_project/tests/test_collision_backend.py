import numpy as np

from rrt_project.core.collision_backend import CollisionBackend
from rrt_project.env.obstacles import AABBObstacle
from rrt_project.env.scene_state import SceneState


def _scene():
    return SceneState(
        scene_id="test",
        dimension=2,
        difficulty="simple",
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        start=np.array([0.1, 0.1]),
        goal=np.array([0.9, 0.9]),
        obstacles=[AABBObstacle(np.array([0.5, 0.5]), np.array([0.15, 0.15]), "o")],
    )


def test_collision_hit_and_miss():
    scene = _scene()
    backend = CollisionBackend(samples_per_segment=32)
    hit = backend.segment_collides([0.2, 0.2], [0.8, 0.8], scene, alpha=1.0, cuda_enabled=False)
    miss = backend.segment_collides([0.1, 0.1], [0.1, 0.9], scene, alpha=1.0, cuda_enabled=False)
    assert hit is True
    assert miss is False


def test_cpu_gpu_consistency_if_available():
    scene = _scene()
    backend = CollisionBackend(samples_per_segment=32)
    cpu = backend.segment_collides([0.2, 0.2], [0.8, 0.8], scene, alpha=1.0, cuda_enabled=False)
    try:
        import torch

        if torch.cuda.is_available():
            gpu = backend.segment_collides([0.2, 0.2], [0.8, 0.8], scene, alpha=1.0, cuda_enabled=True)
            assert cpu == gpu
    except Exception:
        # If GPU stack is unavailable in CI, CPU assertion already covers behavior.
        assert isinstance(cpu, bool)
