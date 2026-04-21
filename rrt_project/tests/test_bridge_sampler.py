import numpy as np

from rrt_project.core.bridge_sampler import BridgeSampler


def test_radius_cluster_basic():
    points = np.array([[0.0, 0.0], [0.01, 0.01], [0.9, 0.9]])
    clusters = BridgeSampler._radius_cluster(points, eps=0.05)
    sizes = sorted([len(c) for c in clusters])
    assert sizes == [1, 2]


def test_orthogonal_dirs_2d():
    d = BridgeSampler._orthogonal_dirs(np.array([0.0, 0.0]), np.array([1.0, 0.0]), dim=2)
    assert len(d) == 2
    assert abs(float(np.dot(d[0], np.array([1.0, 0.0])))) < 1e-8
