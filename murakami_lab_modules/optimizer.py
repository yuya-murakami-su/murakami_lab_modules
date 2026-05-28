from collections.abc import Callable, Iterable

import numpy as np
import torch

from . import utils

__all__ = [
    'OptimizerBase',
    'ConstantLROptimizer',
    'WarmupOptimizer',
    'WarmupDecayOptimizer',
    'InverseTimeDecayOptimizer',
    'ExponentialDecayOptimizer',
    'StepDecayOptimizer',
    'CosineAnnealingOptimizer',
    'PolynomialDecayOptimizer',
]


def _merge_optimizer_params(optimizer_params: dict[str, object] = None, extra_params: dict[str, object] = None):
    params = dict(optimizer_params or {})
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


class OptimizerBase:
    def __init__(
            self,
            algorithm: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_params: dict[str, object] = None,
            zero_grad_set_to_none: bool = True,
            **schedule_params
    ):
        self.locals = utils.get_local_dict(locals())
        self.algorithm = algorithm
        self.optimizer_params = dict(optimizer_params or {})
        self.zero_grad_set_to_none = zero_grad_set_to_none
        self.schedule_params = schedule_params
        self.lr_function = self.get_lr_function()
        self.optimizer = None
        self._last_lr_epoch = None

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'algorithm': self.algorithm,
            'optimizer_params': self.optimizer_params,
            'zero_grad_set_to_none': self.zero_grad_set_to_none,
            **self.schedule_params,
        })

    def set_parameters(self, parameters: Iterable):
        self.optimizer = self.algorithm(parameters, lr=self.lr_function(0), **self.optimizer_params)
        self._last_lr_epoch = 0

    def get_lr_function(self) -> Callable[[int], float]:
        raise NotImplementedError

    def update_lr(self, epoch: int):
        if epoch == self._last_lr_epoch:
            return
        lr = self.lr_function(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self._last_lr_epoch = epoch

    def step(self, epoch: int):
        self.update_lr(epoch)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=self.zero_grad_set_to_none)

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        self._last_lr_epoch = None


class ConstantLROptimizer(OptimizerBase):
    def __init__(
            self,
            algorithm: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
            lr: float = 1e-3,
            optimizer_params: dict[str, object] = None,
            zero_grad_set_to_none: bool = True,
            **optimizer_kwargs
    ):
        _validate_non_negative_float(lr, 'lr')
        self.lr = lr
        super().__init__(
            algorithm=algorithm,
            optimizer_params=_merge_optimizer_params(optimizer_params, optimizer_kwargs),
            zero_grad_set_to_none=zero_grad_set_to_none,
            lr=lr,
        )

    def get_lr_function(self) -> Callable[[int], float]:
        def lr_function(_: int):
            return self.lr
        return lr_function

    def update_lr(self, epoch: int):
        return


class WarmupOptimizer(OptimizerBase):
    def __init__(
            self,
            init_lr: float,
            warmup_epochs: int,
            final_lr: float = 1e-3,
            scale: str = 'exponential',
            algorithm: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_params: dict[str, object] = None,
            zero_grad_set_to_none: bool = True,
            **optimizer_kwargs
    ):
        _validate_positive_int(warmup_epochs, 'warmup_epochs')
        _validate_non_negative_float(init_lr, 'init_lr')
        _validate_non_negative_float(final_lr, 'final_lr')
        self.init_lr = init_lr
        self.warmup_epochs = warmup_epochs
        self.final_lr = final_lr
        self.scale = scale
        super().__init__(
            algorithm=algorithm,
            optimizer_params=_merge_optimizer_params(optimizer_params, optimizer_kwargs),
            zero_grad_set_to_none=zero_grad_set_to_none,
            init_lr=init_lr,
            warmup_epochs=warmup_epochs,
            final_lr=final_lr,
            scale=scale,
        )

    def get_lr_function(self) -> Callable[[int], float]:
        def lr_function(epoch: int):
            if epoch < self.warmup_epochs:
                return _interpolate_lr(self.init_lr, self.final_lr, epoch / self.warmup_epochs, self.scale)
            return self.final_lr
        return lr_function


class WarmupDecayOptimizer(OptimizerBase):
    def __init__(
            self,
            init_lr: float,
            warmup_epochs: int,
            peak_lr: float,
            total_epochs: int,
            final_lr: float,
            warmup_scale: str = 'exponential',
            decay_scale: str = 'exponential',
            algorithm: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_params: dict[str, object] = None,
            zero_grad_set_to_none: bool = True,
            **optimizer_kwargs
    ):
        _validate_positive_int(warmup_epochs, 'warmup_epochs')
        _validate_positive_int(total_epochs, 'total_epochs')
        if total_epochs <= warmup_epochs:
            raise ValueError('total_epochs must be greater than warmup_epochs.')
        for name, lr in [('init_lr', init_lr), ('peak_lr', peak_lr), ('final_lr', final_lr)]:
            _validate_non_negative_float(lr, name)
        self.init_lr = init_lr
        self.warmup_epochs = warmup_epochs
        self.peak_lr = peak_lr
        self.total_epochs = total_epochs
        self.final_lr = final_lr
        self.warmup_scale = warmup_scale
        self.decay_scale = decay_scale
        super().__init__(
            algorithm=algorithm,
            optimizer_params=_merge_optimizer_params(optimizer_params, optimizer_kwargs),
            zero_grad_set_to_none=zero_grad_set_to_none,
            init_lr=init_lr,
            warmup_epochs=warmup_epochs,
            peak_lr=peak_lr,
            total_epochs=total_epochs,
            final_lr=final_lr,
            warmup_scale=warmup_scale,
            decay_scale=decay_scale,
        )

    def get_lr_function(self) -> Callable[[int], float]:
        def lr_function(epoch: int):
            if epoch < self.warmup_epochs:
                return _interpolate_lr(
                    self.init_lr,
                    self.peak_lr,
                    epoch / self.warmup_epochs,
                    self.warmup_scale
                )
            if epoch < self.total_epochs:
                return _interpolate_lr(
                    self.peak_lr,
                    self.final_lr,
                    (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs),
                    self.decay_scale
                )
            return self.final_lr
        return lr_function


class InverseTimeDecayOptimizer(OptimizerBase):
    def __init__(
            self,
            initial_lr: float,
            decay_steps: int,
            decay_rate: float = 1.0,
            min_lr: float = None,
            algorithm: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_params: dict[str, object] = None,
            zero_grad_set_to_none: bool = True,
            **optimizer_kwargs
    ):
        _validate_positive_int(decay_steps, 'decay_steps')
        _validate_non_negative_float(initial_lr, 'initial_lr')
        _validate_non_negative_float(decay_rate, 'decay_rate')
        if min_lr is not None:
            _validate_non_negative_float(min_lr, 'min_lr')
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        super().__init__(
            algorithm=algorithm,
            optimizer_params=_merge_optimizer_params(optimizer_params, optimizer_kwargs),
            zero_grad_set_to_none=zero_grad_set_to_none,
            initial_lr=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            min_lr=min_lr,
        )

    def get_lr_function(self) -> Callable[[int], float]:
        def lr_function(epoch: int):
            lr = self.initial_lr / (1 + self.decay_rate * epoch / self.decay_steps)
            if self.min_lr is not None:
                lr = max(lr, self.min_lr)
            return lr
        return lr_function


class ExponentialDecayOptimizer(OptimizerBase):
    def __init__(
            self,
            initial_lr: float,
            gamma: float,
            decay_steps: int = 1,
            staircase: bool = False,
            min_lr: float = None,
            algorithm: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_params: dict[str, object] = None,
            zero_grad_set_to_none: bool = True,
            **optimizer_kwargs
    ):
        _validate_positive_int(decay_steps, 'decay_steps')
        _validate_non_negative_float(initial_lr, 'initial_lr')
        _validate_non_negative_float(gamma, 'gamma')
        if min_lr is not None:
            _validate_non_negative_float(min_lr, 'min_lr')
        self.initial_lr = initial_lr
        self.gamma = gamma
        self.decay_steps = decay_steps
        self.staircase = staircase
        self.min_lr = min_lr
        super().__init__(
            algorithm=algorithm,
            optimizer_params=_merge_optimizer_params(optimizer_params, optimizer_kwargs),
            zero_grad_set_to_none=zero_grad_set_to_none,
            initial_lr=initial_lr,
            gamma=gamma,
            decay_steps=decay_steps,
            staircase=staircase,
            min_lr=min_lr,
        )

    def get_lr_function(self) -> Callable[[int], float]:
        def lr_function(epoch: int):
            exponent = epoch // self.decay_steps if self.staircase else epoch / self.decay_steps
            lr = self.initial_lr * self.gamma ** exponent
            if self.min_lr is not None:
                lr = max(lr, self.min_lr)
            return lr
        return lr_function


class StepDecayOptimizer(OptimizerBase):
    def __init__(
            self,
            initial_lr: float,
            step_size: int,
            gamma: float = 0.1,
            min_lr: float = None,
            algorithm: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_params: dict[str, object] = None,
            zero_grad_set_to_none: bool = True,
            **optimizer_kwargs
    ):
        _validate_positive_int(step_size, 'step_size')
        _validate_non_negative_float(initial_lr, 'initial_lr')
        _validate_non_negative_float(gamma, 'gamma')
        if min_lr is not None:
            _validate_non_negative_float(min_lr, 'min_lr')
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(
            algorithm=algorithm,
            optimizer_params=_merge_optimizer_params(optimizer_params, optimizer_kwargs),
            zero_grad_set_to_none=zero_grad_set_to_none,
            initial_lr=initial_lr,
            step_size=step_size,
            gamma=gamma,
            min_lr=min_lr,
        )

    def get_lr_function(self) -> Callable[[int], float]:
        def lr_function(epoch: int):
            lr = self.initial_lr * self.gamma ** (epoch // self.step_size)
            if self.min_lr is not None:
                lr = max(lr, self.min_lr)
            return lr
        return lr_function


class CosineAnnealingOptimizer(OptimizerBase):
    def __init__(
            self,
            initial_lr: float,
            total_epochs: int,
            min_lr: float = 0.0,
            algorithm: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_params: dict[str, object] = None,
            zero_grad_set_to_none: bool = True,
            **optimizer_kwargs
    ):
        _validate_positive_int(total_epochs, 'total_epochs')
        _validate_non_negative_float(initial_lr, 'initial_lr')
        _validate_non_negative_float(min_lr, 'min_lr')
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(
            algorithm=algorithm,
            optimizer_params=_merge_optimizer_params(optimizer_params, optimizer_kwargs),
            zero_grad_set_to_none=zero_grad_set_to_none,
            initial_lr=initial_lr,
            total_epochs=total_epochs,
            min_lr=min_lr,
        )

    def get_lr_function(self) -> Callable[[int], float]:
        def lr_function(epoch: int):
            progress = min(max(epoch / self.total_epochs, 0.0), 1.0)
            return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        return lr_function


class PolynomialDecayOptimizer(OptimizerBase):
    def __init__(
            self,
            initial_lr: float,
            total_epochs: int,
            final_lr: float = 0.0,
            power: float = 1.0,
            algorithm: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_params: dict[str, object] = None,
            zero_grad_set_to_none: bool = True,
            **optimizer_kwargs
    ):
        _validate_positive_int(total_epochs, 'total_epochs')
        _validate_non_negative_float(initial_lr, 'initial_lr')
        _validate_non_negative_float(final_lr, 'final_lr')
        if power < 0:
            raise ValueError(f'power must be non-negative. {power} was given.')
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.final_lr = final_lr
        self.power = power
        super().__init__(
            algorithm=algorithm,
            optimizer_params=_merge_optimizer_params(optimizer_params, optimizer_kwargs),
            zero_grad_set_to_none=zero_grad_set_to_none,
            initial_lr=initial_lr,
            total_epochs=total_epochs,
            final_lr=final_lr,
            power=power,
        )

    def get_lr_function(self) -> Callable[[int], float]:
        def lr_function(epoch: int):
            progress = min(max(epoch / self.total_epochs, 0.0), 1.0)
            return (self.initial_lr - self.final_lr) * (1 - progress) ** self.power + self.final_lr
        return lr_function
