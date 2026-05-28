import numpy as np
import pytest
import torch

from murakami_lab_modules.optimizer import (
    Optimizer,
    cosine_annealing_lr,
    exponential_decay_lr,
    inverse_time_decay_lr,
    linear_warmup_lr,
    polynomial_decay_lr,
    step_decay_lr,
    warmup_decay_lr,
)


def _lrs(schedule, epochs):
    return [schedule(epoch) for epoch in epochs]


def test_optimizer_uses_constant_lr_and_optimizer_kwargs():
    optimizer = Optimizer(torch.optim.SGD, lr=1e-2, momentum=0.9)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    optimizer.set_parameters([parameter])

    assert _lrs(optimizer.get_lr_function(), [0, 10]) == [1e-2, 1e-2]
    assert optimizer.optimizer.param_groups[0]['momentum'] == 0.9
    assert optimizer.optimizer.param_groups[0]['weight_decay'] == 0.0


def test_optimizer_supports_explicit_weight_decay():
    optimizer = Optimizer(torch.optim.SGD, lr=1e-2, weight_decay=1e-4)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    optimizer.set_parameters([parameter])

    assert optimizer.optimizer.param_groups[0]['weight_decay'] == 1e-4


def test_constant_lr_optimizer_skips_per_step_lr_updates_and_uses_set_to_none():
    optimizer = Optimizer(torch.optim.SGD, lr=1e-2)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer.set_parameters([parameter])
    calls = []

    def fail_schedule(epoch):
        calls.append(epoch)
        raise AssertionError('constant lr should not be recalculated during step.')

    optimizer.lr_schedule = fail_schedule
    parameter.grad = torch.ones_like(parameter)
    optimizer.zero_grad()
    optimizer.step(epoch=1)

    assert parameter.grad is None
    assert calls == []


def test_scheduled_optimizer_updates_lr_once_per_epoch():
    schedule = linear_warmup_lr(initial_lr=0.0, warmup_epochs=10, final_lr=1.0)
    optimizer = Optimizer(torch.optim.SGD, lr_schedule=schedule)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer.set_parameters([parameter])
    calls = []
    original_schedule = optimizer.lr_schedule

    def counted_schedule(epoch):
        calls.append(epoch)
        return original_schedule(epoch)

    optimizer.lr_schedule = counted_schedule
    optimizer.step(epoch=0)
    optimizer.step(epoch=1)
    optimizer.step(epoch=1)

    assert calls == [1]


def test_warmup_schedule_supports_linear_scale():
    schedule = linear_warmup_lr(initial_lr=0.0, warmup_epochs=10, final_lr=1.0)

    assert _lrs(schedule, [0, 5, 10, 20]) == [0.0, 0.5, 1.0, 1.0]


def test_warmup_decay_schedule():
    schedule = warmup_decay_lr(
        initial_lr=0.0,
        warmup_epochs=10,
        peak_lr=1.0,
        total_epochs=20,
        final_lr=0.0,
    )

    assert _lrs(schedule, [0, 5, 10, 15, 20]) == [0.0, 0.5, 1.0, 0.5, 0.0]


def test_common_decay_schedules():
    exponential = exponential_decay_lr(initial_lr=1.0, gamma=0.5, decay_steps=2, staircase=True)
    step = step_decay_lr(initial_lr=1.0, step_size=2, gamma=0.1)
    inverse = inverse_time_decay_lr(initial_lr=1.0, decay_steps=10, min_lr=0.25)
    polynomial = polynomial_decay_lr(initial_lr=1.0, total_epochs=10, final_lr=0.1, power=1.0)

    assert np.allclose(_lrs(exponential, [0, 2, 4]), [1.0, 0.5, 0.25])
    assert np.allclose(_lrs(step, [0, 1, 2, 4]), [1.0, 1.0, 0.1, 0.01])
    assert np.allclose(_lrs(inverse, [0, 10, 100]), [1.0, 0.5, 0.25])
    assert np.allclose(_lrs(polynomial, [0, 5, 10]), [1.0, 0.55, 0.1])


def test_cosine_annealing_schedule():
    schedule = cosine_annealing_lr(initial_lr=1.0, total_epochs=10, min_lr=0.0)

    assert np.allclose(_lrs(schedule, [0, 5, 10]), [1.0, 0.5, 0.0])


def test_invalid_schedule_parameters_raise_clear_errors():
    with pytest.raises(ValueError, match='warmup_epochs'):
        linear_warmup_lr(initial_lr=1e-4, warmup_epochs=0, final_lr=1e-3)
    with pytest.raises(ValueError, match='total_epochs must be greater'):
        warmup_decay_lr(initial_lr=1e-4, warmup_epochs=10, peak_lr=1e-3, total_epochs=10, final_lr=1e-5)
    with pytest.raises(ValueError, match='exponential learning-rate interpolation'):
        linear_warmup_lr(initial_lr=0.0, warmup_epochs=10, final_lr=1e-3, scale='exponential')(1)
