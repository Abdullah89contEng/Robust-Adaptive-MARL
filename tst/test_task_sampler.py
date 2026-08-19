from utils.task_sampler import PhysicsTaskSampler, UniformRange


def test_seeded_sampler_is_reproducible():
    first = PhysicsTaskSampler(seed=7).sample_batch(3)
    second = PhysicsTaskSampler(seed=7).sample_batch(3)
    assert first == second


def test_sample_respects_configured_ranges():
    sampler = PhysicsTaskSampler(
        drag=UniformRange(0.2, 0.2),
        linear_friction=UniformRange(0.1, 0.1),
        angular_friction=UniformRange(0.05, 0.05),
        agent_mass=UniformRange(1.5, 1.5),
    )
    task = sampler.sample()
    assert task.drag == 0.2
    assert task.linear_friction == 0.1
    assert task.angular_friction == 0.05
    assert task.agent_mass == 1.5
