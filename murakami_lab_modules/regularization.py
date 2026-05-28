from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch

from . import differential, utils
from .data_handler import DataHandler
from .input_generator import InputGenerator
from .neural_network import AbstractNeuralNetwork

__all__ = [
    'Regularization',
    'RegularizationWeightPolicy',
    'StaticRegWeights',
    'TargetTotalRegWeight',
    'MatchDataLossRegWeight',
]


class RegularizationWeightPolicy:
    def initialize(
            self,
            raw_reg_mean: torch.Tensor,
            data_loss: float | torch.Tensor = None,
            regularization: 'Regularization' = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def update(
            self,
            epoch: int,
            raw_reg_mean: torch.Tensor,
            current_weights: torch.Tensor,
            data_loss: float | torch.Tensor = None,
            regularization: 'Regularization' = None
    ) -> torch.Tensor:
        return current_weights

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {})

    @staticmethod
    def _as_tensor(
            value: float | list[float] | tuple[float, ...] | torch.Tensor,
            raw_reg_mean: torch.Tensor
    ) -> torch.Tensor:
        return torch.as_tensor(value, dtype=raw_reg_mean.dtype, device=raw_reg_mean.device)

    @staticmethod
    def _scalar_value(value: float | torch.Tensor, name: str) -> float:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f'{name} must be scalar. shape={tuple(value.shape)} was given.')
            return float(value.detach().cpu().item())
        if value is None:
            raise ValueError(f'{name} must be given.')
        return float(value)

    @staticmethod
    def _validate_factors(factors: torch.Tensor, raw_reg_mean: torch.Tensor) -> torch.Tensor:
        if factors.ndim == 0:
            factors = factors.expand_as(raw_reg_mean)
        if factors.shape != raw_reg_mean.shape:
            raise ValueError(
                f'factors must have the same shape as raw_reg_mean. '
                f'factors.shape={tuple(factors.shape)}, raw_reg_mean.shape={tuple(raw_reg_mean.shape)}.'
            )
        if torch.lt(factors, 0).any():
            raise ValueError('factors must be non-negative.')
        if torch.le(factors.sum(), 0).item():
            raise ValueError('At least one factor must be positive.')
        return factors


class StaticRegWeights(RegularizationWeightPolicy):
    def __init__(self, weights: list[float] | tuple[float, ...] | torch.Tensor):
        self.weights = weights

    def initialize(
            self,
            raw_reg_mean: torch.Tensor,
            data_loss: float | torch.Tensor = None,
            regularization: 'Regularization' = None
    ) -> torch.Tensor:
        weights = self._as_tensor(self.weights, raw_reg_mean)
        if weights.shape != raw_reg_mean.shape:
            raise ValueError(
                f'weights must have the same shape as raw_reg_mean. '
                f'weights.shape={tuple(weights.shape)}, raw_reg_mean.shape={tuple(raw_reg_mean.shape)}.'
            )
        return weights

    def config_dict(self) -> dict[str, object]:
        weights = self.weights.detach().cpu().tolist() if torch.is_tensor(self.weights) else list(self.weights)
        return utils.make_object_config(self, {'weights': weights})


class TargetTotalRegWeight(RegularizationWeightPolicy):
    def __init__(
            self,
            target_total: float,
            factors: list[float] | tuple[float, ...] | torch.Tensor = None,
            epsilon: float = 1e-12
    ):
        self.target_total = target_total
        self.factors = factors
        self.epsilon = epsilon

    def initialize(
            self,
            raw_reg_mean: torch.Tensor,
            data_loss: float | torch.Tensor = None,
            regularization: 'Regularization' = None
    ) -> torch.Tensor:
        factors = self._get_factors(raw_reg_mean)
        target_terms = float(self.target_total) * factors / factors.sum()
        return target_terms / raw_reg_mean.detach().clamp_min(self.epsilon)

    def _get_factors(self, raw_reg_mean: torch.Tensor) -> torch.Tensor:
        if self.factors is None:
            factors = torch.ones_like(raw_reg_mean)
        else:
            factors = self._as_tensor(self.factors, raw_reg_mean)
        return self._validate_factors(factors, raw_reg_mean)

    def config_dict(self) -> dict[str, object]:
        factors = self.factors.detach().cpu().tolist() if torch.is_tensor(self.factors) else self.factors
        return utils.make_object_config(self, {
            'target_total': self.target_total,
            'factors': factors,
            'epsilon': self.epsilon,
        })


class MatchDataLossRegWeight(TargetTotalRegWeight):
    def __init__(
            self,
            alpha: float,
            factors: list[float] | tuple[float, ...] | torch.Tensor = None,
            epsilon: float = 1e-12
    ):
        super().__init__(target_total=1.0, factors=factors, epsilon=epsilon)
        self.alpha = alpha

    def initialize(
            self,
            raw_reg_mean: torch.Tensor,
            data_loss: float | torch.Tensor = None,
            regularization: 'Regularization' = None
    ) -> torch.Tensor:
        data_loss_value = self._scalar_value(data_loss, 'data_loss')
        self.target_total = data_loss_value * float(self.alpha)
        return super().initialize(
            raw_reg_mean=raw_reg_mean,
            data_loss=data_loss,
            regularization=regularization
        )

    def config_dict(self) -> dict[str, object]:
        factors = self.factors.detach().cpu().tolist() if torch.is_tensor(self.factors) else self.factors
        return utils.make_object_config(self, {
            'alpha': self.alpha,
            'factors': factors,
            'epsilon': self.epsilon,
        })


class Regularization:
    _different_n_points_warned = False
    _validation_modes = {'always', 'once', 'never'}

    grad = staticmethod(differential.grad)
    partial = staticmethod(differential.partial)
    partial2 = staticmethod(differential.partial2)
    jacobian = staticmethod(differential.jacobian)
    hessian_diag = staticmethod(differential.hessian_diag)
    laplacian = staticmethod(differential.laplacian)

    def __init__(
            self,
            input_generators: list[InputGenerator] | tuple[InputGenerator, ...],
            reg_weights: list[float] = None,
            reg_weight_policy: RegularizationWeightPolicy = None,
            reg_func_name: str = 'regularization',
            reg_names: list[str] = None,
            reg_criteria: Callable[[torch.Tensor], torch.Tensor] = torch.square,
            use_reg_prod: bool = False,
            reg_min: float = None,
            validation: str = 'once'
    ):
        self.locals = utils.get_local_dict(locals())
        self.input_generators = input_generators
        self.reg_func_name = reg_func_name
        self.reg_weight_policy = reg_weight_policy
        self.reg_names = reg_names
        self.reg_criteria = reg_criteria
        self.use_reg_prod = use_reg_prod
        self.reg_min_value = reg_min
        self.reg_min = reg_min
        if validation not in self._validation_modes:
            raise ValueError(f'validation must be one of {sorted(self._validation_modes)}. {validation} was given.')
        self.validation = validation
        self._regularization_outputs_validated = False
        self.weight_report = None

        self.n_generator = len(input_generators)
        self.device = input_generators[0].device
        self.device_name = input_generators[0].device_name

        if not hasattr(self, f'{reg_func_name}'):
            raise ValueError(f'{self.__class__.__name__} does not have a method named {reg_func_name}.')

        self.reg_func = getattr(self, f'{reg_func_name}')

        if reg_weight_policy is None:
            if reg_weights is None:
                raise ValueError('Either reg_weights or reg_weight_policy must be given.')
            self.reg_weight_policy = StaticRegWeights(reg_weights)
            initial_weights = list(reg_weights)
            self.is_weight_calibrated = True
        else:
            initial_weights = reg_weights
            self.is_weight_calibrated = initial_weights is not None

        if initial_weights is not None:
            self.n_reg = len(initial_weights)
            self.reg_weights = torch.tensor(initial_weights, device=self.device, dtype=torch.float32)
        elif reg_names is not None:
            self.n_reg = len(reg_names)
            self.reg_weights = torch.ones([self.n_reg], device=self.device, dtype=torch.float32)
        else:
            raise ValueError('reg_names must be given when reg_weights is omitted.')

        if reg_names is None:
            self.reg_names = [f'Reg{i}' for i in range(self.n_reg)]
        elif len(reg_names) != self.n_reg:
            raise ValueError(f'Inconsistent length of reg_names: '
                             f'len(_{reg_func_name}()) = {self.n_reg}, len(reg_names) = {len(reg_names)}.')

        if reg_min is None:
            self.reg_min = torch.zeros([self.n_reg], device=self.device, dtype=torch.float32)
        else:
            self.reg_min = torch.full([self.n_reg], reg_min, device=self.device, dtype=torch.float32)

        if self.use_reg_prod:
            self.reg_mean_pow = torch.tensor(1 / self.n_reg, dtype=torch.float, device=self.device)

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'input_generators': [input_generator.config_dict() for input_generator in self.input_generators],
            'reg_weights': self.reg_weights.detach().cpu().tolist(),
            'reg_weight_policy': self.reg_weight_policy.config_dict(),
            'reg_func_name': self.reg_func_name,
            'reg_names': self.reg_names,
            'reg_criteria': self.reg_criteria,
            'use_reg_prod': self.use_reg_prod,
            'reg_min': self.reg_min_value,
            'validation': self.validation
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
                f'len({self.reg_func_name}()) = {len(regs)}, expected n_reg = {self.n_reg}.'
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

    def get_raw_regularization_mean(
            self,
            nn: AbstractNeuralNetwork,
            data_handler: DataHandler = None
    ) -> torch.Tensor:
        regs = self.reg_func(data_handler=data_handler, nn=nn)
        if self._should_validate_regularization_outputs():
            regs = self._validate_regularization_outputs(regs)
            self._regularization_outputs_validated = True
        return self._get_regularization_mean(regs)

    def _should_validate_regularization_outputs(self) -> bool:
        if self.validation == 'always':
            return True
        if self.validation == 'once':
            return not self._regularization_outputs_validated
        return False

    def combine_regularization_terms(
            self,
            raw_reg_mean: torch.Tensor,
            epoch: int = None,
            data_loss: float | torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_weight_calibrated:
            self.calibrate_weights_from_raw(raw_reg_mean=raw_reg_mean, data_loss=data_loss)
        elif epoch is not None:
            self.reg_weights = self.reg_weight_policy.update(
                epoch=epoch,
                raw_reg_mean=raw_reg_mean.detach(),
                current_weights=self.reg_weights,
                data_loss=data_loss,
                regularization=self
            ).to(device=self.device, dtype=torch.float32)

        if self.use_reg_prod:
            weighted_terms = (raw_reg_mean + self.reg_min).pow(self.reg_weights)
            reg_value = weighted_terms.prod()
        else:
            weighted_terms = raw_reg_mean * self.reg_weights + self.reg_min
            reg_value = weighted_terms.sum()

        return weighted_terms, reg_value

    def calibrate_weights(
            self,
            nn: AbstractNeuralNetwork,
            data_handler: DataHandler = None,
            data_loss: float | torch.Tensor = None
    ) -> pd.DataFrame:
        raw_reg_mean = self.get_raw_regularization_mean(nn=nn, data_handler=data_handler)
        return self.calibrate_weights_from_raw(raw_reg_mean=raw_reg_mean, data_loss=data_loss)

    def calibrate_weights_from_raw(
            self,
            raw_reg_mean: torch.Tensor,
            data_loss: float | torch.Tensor = None
    ) -> pd.DataFrame:
        if self.use_reg_prod and not isinstance(self.reg_weight_policy, StaticRegWeights):
            raise ValueError('Automatic regularization weight calibration is not supported with use_reg_prod=True.')

        new_weights = self.reg_weight_policy.initialize(
            raw_reg_mean=raw_reg_mean.detach(),
            data_loss=data_loss,
            regularization=self
        ).to(device=self.device, dtype=torch.float32)
        if new_weights.shape != self.reg_weights.shape:
            raise ValueError(
                f'Calibrated weights shape mismatch: '
                f'{tuple(new_weights.shape)} != {tuple(self.reg_weights.shape)}.'
            )
        self.reg_weights = new_weights
        self.is_weight_calibrated = True
        weighted_terms, total = self.combine_regularization_terms(raw_reg_mean=raw_reg_mean.detach())
        self.weight_report = self._make_weight_report(
            raw_reg_mean=raw_reg_mean.detach(),
            weighted_terms=weighted_terms.detach(),
            total=total.detach() if torch.is_tensor(total) else total,
            data_loss=data_loss
        )
        return self.weight_report

    def _make_weight_report(
            self,
            raw_reg_mean: torch.Tensor,
            weighted_terms: torch.Tensor,
            total: torch.Tensor | float,
            data_loss: float | torch.Tensor = None
    ) -> pd.DataFrame:
        data_loss_value = None
        if data_loss is not None:
            data_loss_value = RegularizationWeightPolicy._scalar_value(data_loss, 'data_loss')
        total_value = float(total.detach().cpu().item()) if torch.is_tensor(total) else float(total)
        return pd.DataFrame({
            'name': self.reg_names,
            'raw_mean': raw_reg_mean.detach().cpu().tolist(),
            'weight': self.reg_weights.detach().cpu().tolist(),
            'weighted_mean': weighted_terms.detach().cpu().tolist(),
            'total_regularization': [total_value] * self.n_reg,
            'data_loss': [data_loss_value] * self.n_reg,
            'weight_policy': [self.reg_weight_policy.__class__.__name__] * self.n_reg,
        })

    def weight_summary_df(self) -> pd.DataFrame:
        if self.weight_report is None:
            return pd.DataFrame({
                'name': self.reg_names,
                'weight': self.reg_weights.detach().cpu().tolist(),
                'weight_policy': [self.reg_weight_policy.__class__.__name__] * self.n_reg,
            })
        return self.weight_report.copy()

    def save_weight_report(self, path: str | Path, **to_csv_kwargs) -> None:
        self.weight_summary_df().to_csv(path, index=False, **to_csv_kwargs)

    def get_regularization_value(
            self,
            nn: AbstractNeuralNetwork,
            data_handler: DataHandler = None,
            epoch: int = None,
            data_loss: float | torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_reg_mean = self.get_raw_regularization_mean(nn=nn, data_handler=data_handler)
        return self.combine_regularization_terms(
            raw_reg_mean=raw_reg_mean,
            epoch=epoch,
            data_loss=data_loss
        )
