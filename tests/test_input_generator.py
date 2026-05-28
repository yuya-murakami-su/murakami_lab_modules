import pytest
import torch

from murakami_lab_modules.input_generator import InputGenerator


def test_random_generator_can_cache_samples():
    generator = InputGenerator(
        n_samples=5,
        input_range=((0.0, 1.0), (10.0, 20.0)),
        sampling='random',
        resample=False,
        random_seed=1,
    )

    first = generator()
    second = generator()

    assert first.shape == (5, 2)
    assert first.requires_grad
    assert torch.allclose(first, second)
    assert torch.all((0.0 <= first[:, 0]) & (first[:, 0] <= 1.0))
    assert torch.all((10.0 <= first[:, 1]) & (first[:, 1] <= 20.0))


def test_random_generator_resamples_when_requested():
    generator = InputGenerator(
        n_samples=5,
        input_range=((0.0, 1.0),),
        sampling='random',
        resample=True,
        random_seed=1,
    )

    assert not torch.allclose(generator(), generator())


def test_sobol_generator_is_reproducible_with_same_seed():
    kwargs = {
        'n_samples': 8,
        'input_range': ((0.0, 1.0), (0.0, 2.0)),
        'sampling': 'sobol',
        'random_seed': 10,
    }

    first = InputGenerator(**kwargs)()
    second = InputGenerator(**kwargs)()

    assert torch.allclose(first, second)
    assert torch.all((0.0 <= first[:, 0]) & (first[:, 0] <= 1.0))
    assert torch.all((0.0 <= first[:, 1]) & (first[:, 1] <= 2.0))


def test_log_scale_samples_within_original_range():
    generator = InputGenerator(
        n_samples=16,
        input_range=((0.0, 100.0), (1.0, 10.0)),
        sampling='sobol',
        scale=('log', 'linear'),
        log_epsilon=1e-2,
        random_seed=1,
    )

    samples = generator()

    assert torch.all((0.0 <= samples[:, 0]) & (samples[:, 0] <= 100.0))
    assert torch.all((1.0 <= samples[:, 1]) & (samples[:, 1] <= 10.0))


def test_filter_func_rejects_candidates():
    generator = InputGenerator(
        n_samples=20,
        input_range=((-1.0, 1.0), (-1.0, 1.0)),
        sampling='sobol',
        filter_func=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2 <= 1.0,
        oversample_factor=2,
        max_attempts=10,
        random_seed=1,
    )

    samples = generator()

    assert samples.shape == (20, 2)
    assert torch.all(samples[:, 0] ** 2 + samples[:, 1] ** 2 <= 1.0)


def test_filter_func_failure_has_clear_error():
    generator = InputGenerator(
        n_samples=3,
        input_range=((0.0, 1.0),),
        filter_func=lambda x: torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
        max_attempts=2,
    )

    with pytest.raises(RuntimeError, match='failed to collect enough samples'):
        generator()


def test_fixed_inputs_and_requires_grad_false():
    generator = InputGenerator(
        n_samples=2,
        sampling='fixed',
        inputs=[[1.0, 2.0], [3.0, 4.0]],
        requires_grad=False,
    )

    samples = generator()

    assert torch.allclose(samples, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert not samples.requires_grad
