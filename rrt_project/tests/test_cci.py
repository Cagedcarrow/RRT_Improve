import numpy as np

from rrt_project.core.cci import (
    AlphaExecutor,
    AlphaScheduler,
    build_leaf_sequence,
    interpolate_sdf,
    shaping_function,
)
from rrt_project.env.obstacles import AABBObstacle


def test_shaping_function_monotonic():
    xs = np.linspace(-0.5, 0.5, 200)
    ys = shaping_function(xs, eta=18.0)
    assert np.all(np.diff(ys) >= -1e-12)


def test_interpolate_sdf_boundary_consistency():
    a, b = -0.2, 0.3
    assert abs(interpolate_sdf(a, b, alpha=0.0, eta=20.0) - shaping_function(a, 20.0)) < 1e-9
    assert abs(interpolate_sdf(a, b, alpha=1.0, eta=20.0) - shaping_function(b, 20.0)) < 1e-9


def test_alpha_scheduler_and_executor():
    sched = AlphaScheduler(schedule=[(0.0, 0.0), (0.5, 0.4), (1.0, 1.0)])
    ex = AlphaExecutor(scheduler=sched)
    vals = [ex.at(i / 10.0) for i in range(11)]
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i - 1] - 1e-12
    s = ex.summary()
    assert s["alpha_min"] >= 0.0
    assert s["alpha_max"] <= 1.0


def test_leaf_set_sequence_generation():
    obs = [
        AABBObstacle(np.array([0.2, 0.2]), np.array([0.12, 0.12]), "o1"),
        AABBObstacle(np.array([0.45, 0.2]), np.array([0.12, 0.12]), "o2"),
        AABBObstacle(np.array([0.7, 0.2]), np.array([0.12, 0.12]), "o3"),
        AABBObstacle(np.array([0.45, 0.5]), np.array([0.10, 0.10]), "o4"),
    ]
    plan = build_leaf_sequence(obs)
    assert isinstance(plan.init_indices, list)
    assert isinstance(plan.batches, list)
    for b in plan.batches:
        assert set(b.parent_map.keys()) == set(b.leaf_indices)
