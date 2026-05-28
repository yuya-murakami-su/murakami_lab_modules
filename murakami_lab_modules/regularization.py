from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch

from . import differential, utils
from .data_handler import DataHandler
from .input_generator import InputGenerator
from .neural_network import BaseNeuralNetwork

__all__ = [
    'Regularization',
    'RegularizationWeightPolicy',
    'StaticRegularizationWeights',
    'TargetTotalRegularizationWeight',
    'MatchDataLossRegularizationWeight',
]

logger = utils.get_logger(__name__)


class RegularizationWeightPolicy:
    def initialize(
            self,
            raw_term_means: torch.Tensor,
            data_loss: float | torch.Tensor = None,
            regularization: 'Regularization' = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def update(
            self,
            epoch: int,
            raw_term_means: torch.Tensor,
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
            raw_term_means: torch.Tensor
    ) -> torch.Tensor:
        return torch.as_tensor(value, dtype=raw_term_means.dtype, device=raw_term_means.device)

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
    def _validate_factors(factors: torch.Tensor, raw_term_means: torch.Tensor) -> torch.Tensor:
        if factors.ndim == 0:
            factors = factors.expand_as(raw_term_means)
        if factors.shape != raw_term_means.shape:
            raise ValueError(
                f'factors must have the same shape as raw_term_means. '
                f'factors.shape={tuple(factors.shape)}, raw_term_means.shape={tuple(raw_term_means.shape)}.'
            )
        if torch.lt(factors, 0).any():
            raise ValueError('factors must be non-negative.')
        if torch.le(factors.sum(), 0).item():
            raise ValueError('At least one factor must be positive.')
        return factors


class StaticRegularizationWeights(RegularizationWeightPolicy):
    def __init__(self, weights: list[float] | tuple[float, ...] | torch.Tensor):
        self.weights = weights

    def initialize(
            self,
            raw_term_means: torch.Tensor,
            data_loss: float | torch.Tensor = None,
            regularization: 'Regularization' = None
    ) -> torch.Tensor:
        weights = self._as_tensor(self.weights, raw_term_means)
        if weights.shape != raw_term_means.shape:
            raise ValueError(
                f'weights must have the same shape as raw_term_means. '
                f'weights.shape={tuple(weights.shape)}, raw_term_means.shape={tuple(raw_term_means.shape)}.'
            )
        return weights

    def config_dict(self) -> dict[str, object]:
        weights = self.weights.detach().cpu().tolist() if torch.is_tensor(self.weights) else list(self.weights)
        return utils.make_object_config(self, {'weights': weights})


class TargetTotalRegularizationWeight(RegularizationWeightPolicy):
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
            raw_term_means: torch.Tensor,
            data_loss: float | torch.Tensor = None,
            regularization: 'Regularization' = None
    ) -> torch.Tensor:
        factors = self._get_factors(raw_term_means)
        target_terms = float(self.target_total) * factors / factors.sum()
        return target_terms / raw_term_means.detach().clamp_min(self.epsilon)

    def _get_factors(self, raw_term_means: torch.Tensor) -> torch.Tensor:
        if self.factors is None:
            factors = torch.ones_like(raw_term_means)
        else:
            factors = self._as_tensor(self.factors, raw_term_means)
        return self._validate_factors(factors, raw_term_means)

    def config_dict(self) -> dict[str, object]:
        factors = self.factors.detach().cpu().tolist() if torch.is_tensor(self.factors) else self.factors
        return utils.make_object_config(self, {
            'target_total': self.target_total,
            'factors': factors,
            'epsilon': self.epsilon,
        })


class MatchDataLossRegularizationWeight(TargetTotalRegularizationWeight):
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
            raw_term_means: torch.Tensor,
            data_loss: float | torch.Tensor = None,
            regularization: 'Regularization' = None
    ) -> torch.Tensor:
        data_loss_value = self._scalar_value(data_loss, 'data_loss')
        self.target_total = data_loss_value * float(self.alpha)
        return super().initialize(
            raw_term_means=raw_term_means,
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
            weights: list[float] = None,
            weight_policy: RegularizationWeightPolicy = None,
            method_name: str = 'regularization',
            term_names: list[str] = None,
            penalty_fn: Callable[[torch.Tensor], torch.Tensor] = torch.square,
            combine_by_product: bool = False,
            min_value: float = None,
            validation: str = 'once'
    ):
        self.locals = utils.get_local_dict(locals())
        self.input_generators = input_generators
        self.method_name = method_name
        self.weight_policy = weight_policy
        self.term_names = term_names
        self.penalty_fn = penalty_fn
        self.combine_by_product = combine_by_product
        self.min_value = min_value
        self.min_values = min_value
        if validation not in self._validation_modes:
            raise ValueError(f'validation must be one of {sorted(self._validation_modes)}. {validation} was given.')
        self.validation = validation
        self._regularization_outputs_validated = False
        self.weight_report = None

        self.n_generator = len(input_generators)
        self.device = input_generators[0].device
        self.device_name = input_generators[0].device_name

        if not hasattr(self, f'{method_name}'):
            raise ValueError(f'{self.__class__.__name__} does not have a method named {method_name}.')

        self.regularization_method = getattr(self, f'{method_name}')

        if weight_policy is None:
            if weights is None:
                raise ValueError('Either weights or weight_policy must be given.')
            self.weight_policy = StaticRegularizationWeights(weights)
            initial_weights = list(weights)
            self.is_weight_calibrated = True
        else:
            initial_weights = weights
            self.is_weight_calibrated = initial_weights is not None

        if initial_weights is not None:
            self.n_terms = len(initial_weights)
            self.weights = torch.tensor(initial_weights, device=self.device, dtype=torch.float32)
        elif term_names is not None:
            self.n_terms = len(term_names)
            self.weights = torch.ones([self.n_terms], device=self.device, dtype=torch.float32)
        else:
            raise ValueError('term_names must be given when weights is omitted.')

        if term_names is None:
            self.term_names = [f'term_{i}' for i in range(self.n_terms)]
        elif len(term_names) != self.n_terms:
            raise ValueError(f'Inconsistent length of term_names: '
                             f'len({method_name}()) = {self.n_terms}, len(term_names) = {len(term_names)}.')

        if min_value is None:
            self.min_values = torch.zeros([self.n_terms], device=self.device, dtype=torch.float32)
        else:
            self.min_values = torch.full([self.n_terms], min_value, device=self.device, dtype=torch.float32)

        if self.combine_by_product:
            self.mean_power = torch.tensor(1 / self.n_terms, dtype=torch.float, device=self.device)

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'input_generators': [input_generator.config_dict() for input_generator in self.input_generators],
            'weights': self.weights.detach().cpu().tolist(),
            'weight_policy': self.weight_policy.config_dict(),
            'method_name': self.method_name,
            'term_names': self.term_names,
            'penalty_fn': self.penalty_fn,
            'combine_by_product': self.combine_by_product,
            'min_value': self.min_value,
            'validation': self.validation
        })

    def regularization(self, data_handler: DataHandler, nn: BaseNeuralNetwork):
        raise NotImplementedError

    def _validate_regularization_outputs(self, regs) -> list[torch.Tensor] | tuple[torch.Tensor, ...]:
        if not isinstance(regs, (list, tuple)):
            raise TypeError(
                f'{self.method_name}() must return list or tuple of torch.Tensor. '
                f'{type(regs)} was returned.'
            )
        if len(regs) != self.n_terms:
            raise ValueError(
                f'Inconsistent number of regularization terms: '
                f'len({self.method_name}()) = {len(regs)}, expected n_terms = {self.n_terms}.'
            )

        for idx, reg in enumerate(regs):
            if not torch.is_tensor(reg):
                raise TypeError(
                    f'{self.method_name}()[{idx}] must be torch.Tensor. {type(reg)} was returned.'
                )
            if reg.numel() == 0:
                raise ValueError(f'{self.method_name}()[{idx}] is empty.')
        n_points = [reg.shape[0] for reg in regs if reg.ndim > 0]
        if len(set(n_points)) > 1 and not self.__class__._different_n_points_warned:
            logger.warning(
                'Regularization terms have different n_points: %s. '
                'Each term is averaged independently before applying weights.',
                n_points
            )
            self.__class__._different_n_points_warned = True
        return regs

    def _get_regularization_mean(self, regs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        term_means = []
        penalized_terms = []
        for reg in regs:
            penalized_term = self.penalty_fn(reg)
            is_finite = torch.isfinite(penalized_term)
            if not is_finite.all():
                if not is_finite.any():
                    torch.save(penalized_terms + [penalized_term], 'invalid_regularization.pth')
                    raise ValueError('Too many invalid values were encountered during regularization.')
                logger.warning('Invalid value was encountered during regularization.')
                penalized_term = torch.where(is_finite, penalized_term, 0.0)
                term_mean = penalized_term.sum() / is_finite.sum()
            else:
                term_mean = penalized_term.mean()

            penalized_terms.append(penalized_term)
            term_means.append(term_mean)

        return torch.stack(term_means)

    def compute_raw_term_means(
            self,
            nn: BaseNeuralNetwork,
            data_handler: DataHandler = None
    ) -> torch.Tensor:
        regs = self.regularization_method(data_handler=data_handler, nn=nn)
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
            raw_term_means: torch.Tensor,
            epoch: int = None,
            data_loss: float | torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_weight_calibrated:
            self.calibrate_weights_from_raw(raw_term_means=raw_term_means, data_loss=data_loss)
        elif epoch is not None:
            self.weights = self.weight_policy.update(
                epoch=epoch,
                raw_term_means=raw_term_means.detach(),
                current_weights=self.weights,
                data_loss=data_loss,
                regularization=self
            ).to(device=self.device, dtype=torch.float32)

        if self.combine_by_product:
            weighted_terms = (raw_term_means + self.min_values).pow(self.weights)
            regularization_loss = weighted_terms.prod()
        else:
            weighted_terms = raw_term_means * self.weights + self.min_values
            regularization_loss = weighted_terms.sum()

        return weighted_terms, regularization_loss

    def calibrate_weights(
            self,
            nn: BaseNeuralNetwork,
            data_handler: DataHandler = None,
            data_loss: float | torch.Tensor = None
    ) -> pd.DataFrame:
        raw_term_means = self.compute_raw_term_means(nn=nn, data_handler=data_handler)
        return self.calibrate_weights_from_raw(raw_term_means=raw_term_means, data_loss=data_loss)

    def calibrate_weights_from_raw(
            self,
            raw_term_means: torch.Tensor,
            data_loss: float | torch.Tensor = None
    ) -> pd.DataFrame:
        if self.combine_by_product and not isinstance(self.weight_policy, StaticRegularizationWeights):
            raise ValueError('Automatic regularization weight calibration is not supported with combine_by_product=True.')

        new_weights = self.weight_policy.initialize(
            raw_term_means=raw_term_means.detach(),
            data_loss=data_loss,
            regularization=self
        ).to(device=self.device, dtype=torch.float32)
        if new_weights.shape != self.weights.shape:
            raise ValueError(
                f'Calibrated weights shape mismatch: '
                f'{tuple(new_weights.shape)} != {tuple(self.weights.shape)}.'
            )
        self.weights = new_weights
        self.is_weight_calibrated = True
        weighted_terms, total = self.combine_regularization_terms(raw_term_means=raw_term_means.detach())
        self.weight_report = self._make_weight_report(
            raw_term_means=raw_term_means.detach(),
            weighted_terms=weighted_terms.detach(),
            total=total.detach() if torch.is_tensor(total) else total,
            data_loss=data_loss
        )
        return self.weight_report

    def _make_weight_report(
            self,
            raw_term_means: torch.Tensor,
            weighted_terms: torch.Tensor,
            total: torch.Tensor | float,
            data_loss: float | torch.Tensor = None
    ) -> pd.DataFrame:
        data_loss_value = None
        if data_loss is not None:
            data_loss_value = RegularizationWeightPolicy._scalar_value(data_loss, 'data_loss')
        total_value = float(total.detach().cpu().item()) if torch.is_tensor(total) else float(total)
        return pd.DataFrame({
            'name': self.term_names,
            'raw_mean': raw_term_means.detach().cpu().tolist(),
            'weight': self.weights.detach().cpu().tolist(),
            'weighted_mean': weighted_terms.detach().cpu().tolist(),
            'total_regularization': [total_value] * self.n_terms,
            'data_loss': [data_loss_value] * self.n_terms,
            'weight_policy': [self.weight_policy.__class__.__name__] * self.n_terms,
        })

    def weight_summary_df(self) -> pd.DataFrame:
        if self.weight_report is None:
            return pd.DataFrame({
                'name': self.term_names,
                'weight': self.weights.detach().cpu().tolist(),
                'weight_policy': [self.weight_policy.__class__.__name__] * self.n_terms,
            })
        return self.weight_report.copy()

    def save_weight_report(self, path: str | Path, **to_csv_kwargs) -> None:
        self.weight_summary_df().to_csv(path, index=False, **to_csv_kwargs)

    def get_regularization_value(
            self,
            nn: BaseNeuralNetwork,
            data_handler: DataHandler = None,
            epoch: int = None,
            data_loss: float | torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_term_means = self.compute_raw_term_means(nn=nn, data_handler=data_handler)
        return self.combine_regularization_terms(
            raw_term_means=raw_term_means,
            epoch=epoch,
            data_loss=data_loss
        )
