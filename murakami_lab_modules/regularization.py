from collections.abc import Callable

import torch

from . import differential, utils
from .data_handler import DataHandler
from .input_generator import InputGenerator
from .neural_network import AbstractNeuralNetwork

__all__ = ['Regularization']


class Regularization:
    _different_n_points_warned = False

    grad = staticmethod(differential.grad)
    partial = staticmethod(differential.partial)
    partial2 = staticmethod(differential.partial2)
    second_partial = staticmethod(differential.second_partial)
    jacobian = staticmethod(differential.jacobian)
    hessian_diag = staticmethod(differential.hessian_diag)
    laplacian = staticmethod(differential.laplacian)

    def __init__(
            self,
            input_generators: list[InputGenerator] | tuple[InputGenerator, ...],
            reg_weights: list[float],
            reg_func_name: str = 'regularization',
            reg_names: list[str] = None,
            reg_criteria: Callable[[torch.Tensor], torch.Tensor] = torch.square,
            use_reg_prod: bool = False,
            reg_min: float = None
    ):
        self.locals = utils.get_local_dict(locals())
        self.input_generators = input_generators
        self.reg_func_name = reg_func_name
        self.reg_weights = reg_weights
        self.reg_names = reg_names
        self.reg_criteria = reg_criteria
        self.use_reg_prod = use_reg_prod
        self.reg_min_value = reg_min
        self.reg_min = reg_min

        self.n_generator = len(input_generators)
        self.device = input_generators[0].device
        self.device_name = input_generators[0].device_name

        if not hasattr(self, f'{reg_func_name}'):
            raise ValueError(f'{self.__class__.__name__} does not have a method named {reg_func_name}.')

        self.reg_func = getattr(self, f'{reg_func_name}')

        self.n_reg = len(reg_weights)
        self.reg_weights = torch.tensor(reg_weights, device=self.device, dtype=torch.float32)
        if self.use_reg_prod:
            self.reg_mean_pow = torch.tensor(1 / self.n_reg, dtype=torch.float, device=self.device)

        if reg_names is None:
            self.reg_names = [f'Reg{i}' for i in range(self.n_reg)]
        elif len(reg_names) != self.n_reg:
            raise ValueError(f'Inconsistent length of reg_names: '
                             f'len(_{reg_func_name}()) = {self.n_reg}, len(reg_names) = {len(reg_names)}.')

        if reg_min is None:
            self.reg_min = torch.zeros([self.n_reg], device=self.device, dtype=torch.float32)
        else:
            self.reg_min = torch.full([self.n_reg], reg_min, device=self.device, dtype=torch.float32)

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'input_generators': [input_generator.config_dict() for input_generator in self.input_generators],
            'reg_weights': self.reg_weights.detach().cpu().tolist(),
            'reg_func_name': self.reg_func_name,
            'reg_names': self.reg_names,
            'reg_criteria': self.reg_criteria,
            'use_reg_prod': self.use_reg_prod,
            'reg_min': self.reg_min_value
        })

    def regularization(self, data_handler: DataHandler, nn: AbstractNeuralNetwork):
        raise NotImplementedError

    def _validate_regularization_outputs(self, regs) -> list[torch.Tensor] | tuple[torch.Tensor, ...]:
        if not isinstance(regs, (list, tuple)):
            raise TypeError(
                f'{self.reg_func_name}() must return list or tuple of torch.Tensor. '
                f'{type(regs)} was returned.'
            )
        if len(regs) != self.n_reg:
            raise ValueError(
                f'Inconsistent number of regularization terms: '
                f'len({self.reg_func_name}()) = {len(regs)}, len(reg_weights) = {self.n_reg}.'
            )

        for idx, reg in enumerate(regs):
            if not torch.is_tensor(reg):
                raise TypeError(
                    f'{self.reg_func_name}()[{idx}] must be torch.Tensor. {type(reg)} was returned.'
                )
            if reg.numel() == 0:
                raise ValueError(f'{self.reg_func_name}()[{idx}] is empty.')
        n_points = [reg.shape[0] for reg in regs if reg.ndim > 0]
        if len(set(n_points)) > 1 and not self.__class__._different_n_points_warned:
            utils.logging(
                f'[Warning] Regularization terms have different n_points: {n_points}. '
                f'Each term is averaged independently before applying reg_weights.'
            )
            self.__class__._different_n_points_warned = True
        return regs

    def _get_regularization_mean(self, regs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        reg_means = []
        full_regs = []
        for reg in regs:
            full_reg = self.reg_criteria(reg)
            is_finite = torch.isfinite(full_reg)
            if not is_finite.all():
                if not is_finite.any():
                    torch.save(full_regs + [full_reg], 'invalid_regularization.pth')
                    raise ValueError(f'Too many invalid value was encountered during regularization.')
                utils.logging(f'Invalid value was encountered during regularization.')
                full_reg = torch.where(is_finite, full_reg, 0.0)
                reg_mean = full_reg.sum() / is_finite.sum()
            else:
                reg_mean = full_reg.mean()

            full_regs.append(full_reg)
            reg_means.append(reg_mean)

        return torch.stack(reg_means)

    def get_regularization_value(
            self,
            nn: AbstractNeuralNetwork,
            data_handler: DataHandler = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        regs = self._validate_regularization_outputs(self.reg_func(data_handler=data_handler, nn=nn))
        reg_mean = self._get_regularization_mean(regs)

        if self.use_reg_prod:
            reg_mean.add_(self.reg_min).pow_(self.reg_weights)
            reg_value = reg_mean.prod()
        else:
            reg_mean.mul_(self.reg_weights).add_(self.reg_min)
            reg_value = reg_mean.sum()

        return reg_mean, reg_value
