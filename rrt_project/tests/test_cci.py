from rrt_project.core.cci import AlphaScheduler


def test_alpha_scheduler_monotonic():
    sched = AlphaScheduler(schedule=[(0.0, 0.0), (0.5, 0.4), (1.0, 1.0)])
    vals = [sched.alpha_at(x / 10.0) for x in range(11)]
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i - 1] - 1e-12
    assert abs(vals[0] - 0.0) < 1e-9
    assert abs(vals[-1] - 1.0) < 1e-9
