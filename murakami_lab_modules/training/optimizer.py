"""Optimizer wrapper and learning-rate schedule factories."""

from collections.abc import Callable, Iterable

import numpy as np
import torch

from .. import utils

__all__ = [
    'Optimizer',
    'constant_lr',
    'linear_warmup_lr',
    'warmup_decay_lr',
    'inverse_time_decay_lr',
    'exponential_decay_lr',
    'step_decay_lr',
    'cosine_annealing_lr',
    'polynomial_decay_lr',
]


LearningRateSchedule = Callable[[int], float]


def _merge_optimizer_params(
        optimizer_params: dict[str, object] = None,
        extra_params: dict[str, object] = None,
        weight_decay: float = 0.0
) -> dict[str, object]:
    _validate_non_negative_float(weight_decay, 'weight_decay')
    params = dict(optimizer_params or {})
    params.setdefault('weight_decay', weight_decay)
    params.update(extra_params or {})
    return params


def _validate_positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive int. {value} was given.')


def _validate_non_negative_float(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f'{name} must be non-negative. {value} was given.')


def _interpolate_lr(start_lr: float, end_lr: float, progress: float, scale: str) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    if scale == 'linear':
        return (end_lr - start_lr) * progress + start_lr
    if scale in {'exponential', 'log'}:
        if start_lr <= 0 or end_lr <= 0:
            raise ValueError('exponential learning-rate interpolation requires positive learning rates.')
        return float(np.exp(np.log(end_lr / start_lr) * progress) * start_lr)
    raise ValueError(f"scale must be 'linear' or 'exponential'. {scale} was given.")


def constant_lr(lr: float) -> LearningRateSchedule:
    """Return a schedule that keeps the learning rate fixed."""

    _validate_non_negative_float(lr, 'lr')
    return lambda _: lr


def linear_warmup_lr(
        initial_lr: float,
        warmup_epochs: int,
        final_lr: float,
        scale: str = 'linear',
) -> LearningRateSchedule:
    """Linearly or exponentially warm up from ``initial_lr`` to ``final_lr``."""

    _validate_positive_int(warmup_epochs, 'warmup_epochs')
    _validate_non_negative_float(initial_lr, 'initial_lr')
    _validate_non_negative_float(final_lr, 'final_lr')

    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return _interpolate_lr(initial_lr, final_lr, epoch / warmup_epochs, scale)
        return final_lr

    return schedule


def warmup_decay_lr(
        initial_lr: float,
        warmup_epochs: int,
        peak_lr: float,
        total_epochs: int,
        final_lr: float,
        warmup_scale: str = 'linear',
        decay_scale: str = 'linear',
) -> LearningRateSchedule:
    """Warm up to ``peak_lr`` and then decay to ``final_lr``."""

    _validate_positive_int(warmup_epochs, 'warmup_epochs')
    _validate_positive_int(total_epochs, 'total_epochs')
    if total_epochs <= warmup_epochs:
        raise ValueError('total_epochs must be greater than warmup_epochs.')
    for name, lr in [('initial_lr', initial_lr), ('peak_lr', peak_lr), ('final_lr', final_lr)]:
        _validate_non_negative_float(lr, name)

    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return _interpolate_lr(initial_lr, peak_lr, epoch / warmup_epochs, warmup_scale)
        if epoch < total_epochs:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return _interpolate_lr(peak_lr, final_lr, progress, decay_scale)
        return final_lr

    return schedule


def inverse_time_decay_lr(
        initial_lr: float,
        decay_steps: int,
        decay_rate: float = 1.0,
        min_lr: float = None,
) -> LearningRateSchedule:
    """Return an inverse-time decay schedule."""

    _validate_positive_int(decay_steps, 'decay_steps')
    _validate_non_negative_float(initial_lr, 'initial_lr')
    _validate_non_negative_float(decay_rate, 'decay_rate')
    if min_lr is not None:
        _validate_non_negative_float(min_lr, 'min_lr')

    def schedule(epoch: int) -> float:
        lr = initial_lr / (1 + decay_rate * epoch / decay_steps)
        return max(lr, min_lr) if min_lr is not None else lr

    return schedule


def exponential_decay_lr(
        initial_lr: float,
        gamma: float,
        decay_steps: int = 1,
        staircase: bool = False,
        min_lr: float = None,
) -> LearningRateSchedule:
    """Return an exponential decay schedule."""

    _validate_positive_int(decay_steps, 'decay_steps')
    _validate_non_negative_float(initial_lr, 'initial_lr')
    _validate_non_negative_float(gamma, 'gamma')
    if min_lr is not None:
        _validate_non_negative_float(min_lr, 'min_lr')

    def schedule(epoch: int) -> float:
        exponent = epoch // decay_steps if staircase else epoch / decay_steps
        lr = initial_lr * gamma ** exponent
        return max(lr, min_lr) if min_lr is not None else lr

    return schedule


def step_decay_lr(
        initial_lr: float,
        step_size: int,
        gamma: float = 0.1,
        min_lr: float = None,
) -> LearningRateSchedule:
    """Return a staircase schedule that decays every ``step_size`` epochs."""

    _validate_positive_int(step_size, 'step_size')
    _validate_non_negative_float(initial_lr, 'initial_lr')
    _validate_non_negative_float(gamma, 'gamma')
    if min_lr is not None:
        _validate_non_negative_float(min_lr, 'min_lr')

    def schedule(epoch: int) -> float:
        lr = initial_lr * gamma ** (epoch // step_size)
        return max(lr, min_lr) if min_lr is not None else lr

    return schedule


def cosine_annealing_lr(
        initial_lr: float,
        total_epochs: int,
        min_lr: float = 0.0,
) -> LearningRateSchedule:
    """Return cosine annealing from ``initial_lr`` to ``min_lr``."""

    _validate_positive_int(total_epochs, 'total_epochs')
    _validate_non_negative_float(initial_lr, 'initial_lr')
    _validate_non_negative_float(min_lr, 'min_lr')

    def schedule(epoch: int) -> float:
        progress = min(max(epoch / total_epochs, 0.0), 1.0)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))

    return schedule


def polynomial_decay_lr(
        initial_lr: float,
        total_epochs: int,
        final_lr: float = 0.0,
        power: float = 1.0,
) -> LearningRateSchedule:
    """Return polynomial decay from ``initial_lr`` to ``final_lr``."""

    _validate_positive_int(total_epochs, 'total_epochs')
    _validate_non_negative_float(initial_lr, 'initial_lr')
    _validate_non_negative_float(final_lr, 'final_lr')
    if power < 0:
        raise ValueError(f'power must be non-negative. {power} was given.')

    def schedule(epoch: int) -> float:
        progress = min(max(epoch / total_epochs, 0.0), 1.0)
        return (initial_lr - final_lr) * (1 - progress) ** power + final_lr

    return schedule


class Optimizer:
    """Small wrapper around a PyTorch optimizer and optional LR schedule.

    ``ModelHandler`` calls ``set_parameters`` once after the model is moved to
    the target device, then calls ``zero_grad`` and ``step(epoch)`` during
    training. This class intentionally does not hide the underlying PyTorch
    optimizer; ``state_dict`` and ``load_state_dict`` delegate to it directly.
    """

    def __init__(
            self,
            algorithm: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
            lr: float = 1e-3,
            lr_schedule: LearningRateSchedule = None,
            optimizer_params: dict[str, object] = None,
            weight_decay: float = 0.0,
            zero_grad_set_to_none: bool = True,
            **optimizer_kwargs
    ):
        _validate_non_negative_float(lr, 'lr')
        self.locals = utils.get_local_dict(locals())
        self.algorithm = algorithm
        self.lr = lr
        self._constant_lr = lr_schedule is None
        self.lr_schedule = lr_schedule or constant_lr(lr)
        self.optimizer_params = _merge_optimizer_params(
            optimizer_params=optimizer_params,
            extra_params=optimizer_kwargs,
            weight_decay=weight_decay
        )
        self.weight_decay = weight_decay
        self.zero_grad_set_to_none = zero_grad_set_to_none
        self.optimizer = None
        self._last_lr_epoch = None

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'algorithm': self.algorithm,
            'lr': self.lr,
            'lr_schedule': self.lr_schedule,
            'optimizer_params': self.optimizer_params,
            'weight_decay': self.weight_decay,
            'zero_grad_set_to_none': self.zero_grad_set_to_none,
        })

    def set_parameters(self, parameters: Iterable) -> None:
        self.optimizer = self.algorithm(parameters, lr=self.lr_schedule(0), **self.optimizer_params)
        self._last_lr_epoch = 0

    def get_lr_function(self) -> LearningRateSchedule:
        return self.lr_schedule

    def update_lr(self, epoch: int) -> None:
        if self._constant_lr:
            return
        if epoch == self._last_lr_epoch:
            return
        lr = self.lr_schedule(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self._last_lr_epoch = epoch

    def step(self, epoch: int) -> None:
        self.update_lr(epoch)
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=self.zero_grad_set_to_none)

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.optimizer.load_state_dict(state_dict)
        self._last_lr_epoch = None
