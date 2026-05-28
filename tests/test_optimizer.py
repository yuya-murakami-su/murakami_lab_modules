import numpy as np
import pytest
import torch

from murakami_lab_modules.optimizer import (
    ConstantLROptimizer,
    CosineAnnealingOptimizer,
    ExponentialDecayOptimizer,
    InverseTimeDecayOptimizer,
    PolynomialDecayOptimizer,
    StepDecayOptimizer,
    WarmupDecayOptimizer,
    WarmupOptimizer,
)


def _lrs(optimizer, epochs):
    lr_function = optimizer.get_lr_function()
    return [lr_function(epoch) for epoch in epochs]


def test_constant_lr_optimizer():
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-2, momentum=0.9)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    optimizer.set_parameters([parameter])

    assert _lrs(optimizer, [0, 10]) == [1e-2, 1e-2]
    assert optimizer.optimizer.param_groups[0]['momentum'] == 0.9


def test_constant_lr_optimizer_skips_per_step_lr_updates_and_uses_set_to_none():
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-2)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer.set_parameters([parameter])
    calls = []

    def fail_lr_function(epoch):
        calls.append(epoch)
        raise AssertionError('constant lr should not be recalculated during step.')

    optimizer.lr_function = fail_lr_function
    parameter.grad = torch.ones_like(parameter)
    optimizer.zero_grad()
    optimizer.step(epoch=1)

    assert parameter.grad is None
    assert calls == []


def test_scheduled_optimizer_updates_lr_once_per_epoch():
    optimizer = WarmupOptimizer(
        init_lr=0.0,
        warmup_epochs=10,
        final_lr=1.0,
        scale='linear',
        algorithm=torch.optim.SGD,
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer.set_parameters([parameter])
    calls = []
    original_lr_function = optimizer.lr_function

    def counted_lr_function(epoch):
        calls.append(epoch)
        return original_lr_function(epoch)

    optimizer.lr_function = counted_lr_function
    optimizer.step(epoch=0)
    optimizer.step(epoch=1)
    optimizer.step(epoch=1)

    assert calls == [1]


def test_warmup_optimizer_supports_linear_scale():
    optimizer = WarmupOptimizer(init_lr=0.0, warmup_epochs=10, final_lr=1.0, scale='linear')

    assert _lrs(optimizer, [0, 5, 10, 20]) == [0.0, 0.5, 1.0, 1.0]


def test_warmup_decay_optimizer():
    optimizer = WarmupDecayOptimizer(
        init_lr=0.0,
        warmup_epochs=10,
        peak_lr=1.0,
        total_epochs=20,
        final_lr=0.0,
        warmup_scale='linear',
        decay_scale='linear',
    )

    assert _lrs(optimizer, [0, 5, 10, 15, 20]) == [0.0, 0.5, 1.0, 0.5, 0.0]


def test_common_decay_schedules():
    exponential = ExponentialDecayOptimizer(initial_lr=1.0, gamma=0.5, decay_steps=2)
    step = StepDecayOptimizer(initial_lr=1.0, step_size=2, gamma=0.1)
    inverse = InverseTimeDecayOptimizer(initial_lr=1.0, decay_steps=10, min_lr=0.25)
    polynomial = PolynomialDecayOptimizer(initial_lr=1.0, total_epochs=10, final_lr=0.1, power=1.0)

    assert np.allclose(_lrs(exponential, [0, 2, 4]), [1.0, 0.5, 0.25])
    assert np.allclose(_lrs(step, [0, 1, 2, 4]), [1.0, 1.0, 0.1, 0.01])
    assert np.allclose(_lrs(inverse, [0, 10, 100]), [1.0, 0.5, 0.25])
    assert np.allclose(_lrs(polynomial, [0, 5, 10]), [1.0, 0.55, 0.1])


def test_cosine_annealing_schedule():
    optimizer = CosineAnnealingOptimizer(initial_lr=1.0, total_epochs=10, min_lr=0.0)

    assert np.allclose(_lrs(optimizer, [0, 5, 10]), [1.0, 0.5, 0.0])


def test_invalid_schedule_parameters_raise_clear_errors():
    with pytest.raises(ValueError, match='warmup_epochs'):
        WarmupOptimizer(init_lr=1e-4, warmup_epochs=0)
    with pytest.raises(ValueError, match='total_epochs must be greater'):
        WarmupDecayOptimizer(init_lr=1e-4, warmup_epochs=10, peak_lr=1e-3, total_epochs=10, final_lr=1e-5)
    with pytest.raises(ValueError, match='exponential learning-rate interpolation'):
        WarmupOptimizer(init_lr=0.0, warmup_epochs=10, final_lr=1e-3, scale='exponential').get_lr_function()(1)
