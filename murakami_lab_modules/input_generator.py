from collections.abc import Callable

import torch
from torch.quasirandom import SobolEngine

from . import utils

__all__ = ['InputGenerator']


class InputGenerator:
    def __init__(
            self,
            size_of_generated_inputs: int,
            input_range: tuple[tuple[float, float], ...] = None,
            device_name: str = 'cpu',
            sampling: str = 'random',
            scale: str | tuple[str, ...] | list[str] = 'linear',
            resample: bool = True,
            requires_grad: bool = True,
            filter_func: Callable[[torch.Tensor], torch.Tensor] = None,
            oversample_factor: int = 10,
            max_attempts: int = 100,
            inputs: torch.Tensor | list | tuple = None,
            random_seed: int = 2025,
            log_epsilon: float = 0.0,
            scramble: bool = True,
    ):
        self.locals = utils.get_local_dict(locals())

        self.size_of_generated_inputs = size_of_generated_inputs
        self.input_range = input_range
        self.device_name = device_name
        self.device = utils.get_device(device_name)
        self.sampling = sampling
        self.scale = scale
        self.resample = resample
        self.requires_grad = requires_grad
        self.filter_func = filter_func
        self.oversample_factor = oversample_factor
        self.max_attempts = max_attempts
        self.inputs = inputs
        self.random_seed = random_seed
        self.log_epsilon = log_epsilon
        self.scramble = scramble

        self._validate_common_settings()
        self._cached_inputs = None
        self._random_generator = torch.Generator(device='cpu')
        self._random_generator.manual_seed(random_seed)

        if self.sampling == 'fixed':
            self._fixed_inputs = self._prepare_fixed_inputs(inputs)
            self.n_input = self._fixed_inputs.shape[1]
            self._sobol_engine = None
        else:
            self._prepare_range()
            self._sobol_engine = self._prepare_sobol_engine()

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'size_of_generated_inputs': self.size_of_generated_inputs,
            'input_range': self.input_range,
            'device_name': self.device_name,
            'sampling': self.sampling,
            'scale': self.scale,
            'resample': self.resample,
            'requires_grad': self.requires_grad,
            'filter_func': self.filter_func,
            'oversample_factor': self.oversample_factor,
            'max_attempts': self.max_attempts,
            'inputs': self.inputs,
            'random_seed': self.random_seed,
            'log_epsilon': self.log_epsilon,
            'scramble': self.scramble,
        })

    def _validate_common_settings(self) -> None:
        if type(self.size_of_generated_inputs) is not int or self.size_of_generated_inputs <= 0:
            raise ValueError('size_of_generated_inputs must be a positive int.')
        if self.sampling not in {'random', 'sobol', 'fixed'}:
            raise ValueError("sampling must be one of 'random', 'sobol', or 'fixed'.")
        if type(self.oversample_factor) is not int or self.oversample_factor < 1:
            raise ValueError('oversample_factor must be an int >= 1.')
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError('max_attempts must be an int >= 1.')
        if self.sampling == 'fixed' and self.inputs is None:
            raise ValueError("inputs must be given when sampling='fixed'.")
        if self.sampling != 'fixed' and self.input_range is None:
            raise ValueError("input_range must be given when sampling is 'random' or 'sobol'.")

    def _prepare_range(self) -> None:
        input_range = torch.tensor(self.input_range, dtype=torch.float32)
        if input_range.ndim != 2 or input_range.shape[1] != 2:
            raise ValueError('input_range must be a tuple/list with shape (n_input, 2).')
        if torch.le(input_range[:, 1], input_range[:, 0]).any():
            raise ValueError('Each upper bound in input_range must be larger than the lower bound.')

        self.n_input = input_range.shape[0]
        scales = self._normalize_scale(self.scale, self.n_input)
        self.scales = scales
        log_mask = torch.tensor([scale == 'log' for scale in scales], dtype=torch.bool)
        if log_mask.any() and torch.le(input_range[log_mask] + self.log_epsilon, 0).any():
            raise ValueError('log-scaled input_range values must be positive after adding log_epsilon.')

        transformed_range = input_range.clone()
        transformed_range[log_mask] = torch.log(transformed_range[log_mask] + self.log_epsilon)
        self._log_mask = log_mask.to(self.device)
        self._min_range = transformed_range[:, 0].view(1, -1).to(self.device)
        self._d_range = (transformed_range[:, 1] - transformed_range[:, 0]).view(1, -1).to(self.device)

    @staticmethod
    def _normalize_scale(scale: str | tuple[str, ...] | list[str], n_input: int) -> tuple[str, ...]:
        if isinstance(scale, str):
            scales = (scale,) * n_input
        elif isinstance(scale, (tuple, list)):
            scales = tuple(scale)
        else:
            raise TypeError(f'scale must be str, tuple[str, ...], or list[str]. {type(scale)} was given.')

        if len(scales) != n_input:
            raise ValueError(f'len(scale) must be {n_input}. {len(scales)} was given.')
        for scale_ in scales:
            if scale_ not in {'linear', 'log'}:
                raise ValueError("scale must contain only 'linear' or 'log'.")
        return scales

    def _prepare_sobol_engine(self) -> SobolEngine | None:
        if self.sampling != 'sobol':
            return None
        return SobolEngine(
            dimension=self.n_input,
            scramble=self.scramble,
            seed=self.random_seed,
        )

    def _prepare_fixed_inputs(self, inputs: torch.Tensor | list | tuple) -> torch.Tensor:
        fixed_inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        if fixed_inputs.ndim == 1:
            fixed_inputs = fixed_inputs.reshape(-1, 1)
        if fixed_inputs.ndim != 2:
            raise ValueError(f'inputs must be 1D or 2D. inputs.shape={tuple(fixed_inputs.shape)}.')
        if fixed_inputs.shape[0] != self.size_of_generated_inputs:
            raise ValueError(
                f'len(inputs) must be equal to size_of_generated_inputs. '
                f'{fixed_inputs.shape[0]} != {self.size_of_generated_inputs}.'
            )
        return fixed_inputs.detach()

    def _draw_unit(self, n_samples: int) -> torch.Tensor:
        if self.sampling == 'random':
            unit = torch.rand(
                [n_samples, self.n_input],
                dtype=torch.float32,
                generator=self._random_generator,
                device='cpu'
            )
        elif self.sampling == 'sobol':
            unit = self._sobol_engine.draw(n_samples).to(dtype=torch.float32)
        else:
            raise ValueError(f'_draw_unit cannot be used when sampling={self.sampling}.')
        return unit.to(self.device)

    def _sample_candidates(self, n_samples: int) -> torch.Tensor:
        transformed = self._draw_unit(n_samples) * self._d_range + self._min_range
        if not self._log_mask.any():
            return transformed

        samples = transformed.clone()
        samples[:, self._log_mask] = torch.exp(samples[:, self._log_mask]) - self.log_epsilon
        return samples

    def _apply_filter(self, candidates: torch.Tensor) -> torch.Tensor:
        if self.filter_func is None:
            return candidates

        with torch.no_grad():
            mask = self.filter_func(candidates)
        if not torch.is_tensor(mask):
            raise TypeError(f'filter_func must return torch.Tensor. {type(mask)} was returned.')
        mask = mask.to(device=candidates.device)
        if mask.ndim == 2 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if mask.ndim != 1 or mask.shape[0] != candidates.shape[0]:
            raise ValueError(
                f'filter_func must return bool mask with shape ({candidates.shape[0]},). '
                f'mask.shape={tuple(mask.shape)} was returned.'
            )
        if mask.dtype != torch.bool:
            raise TypeError(f'filter_func must return bool tensor. mask.dtype={mask.dtype}.')
        return candidates[mask]

    def _build_sampled_inputs(self) -> torch.Tensor:
        if self.filter_func is None:
            return self._sample_candidates(self.size_of_generated_inputs).detach()

        accepted = []
        n_accepted = 0
        for attempt in range(1, self.max_attempts + 1):
            n_remaining = self.size_of_generated_inputs - n_accepted
            n_candidates = max(n_remaining * self.oversample_factor, n_remaining)
            candidates = self._sample_candidates(n_candidates)
            filtered = self._apply_filter(candidates).detach()
            if filtered.numel() > 0:
                accepted.append(filtered)
                n_accepted += filtered.shape[0]
            if n_accepted >= self.size_of_generated_inputs:
                return torch.vstack(accepted)[:self.size_of_generated_inputs].detach()

        raise RuntimeError(
            f'InputGenerator failed to collect enough samples after {self.max_attempts} attempts. '
            f'accepted={n_accepted}, required={self.size_of_generated_inputs}, '
            f'oversample_factor={self.oversample_factor}.'
        )

    def _get_base_inputs(self) -> torch.Tensor:
        if self.sampling == 'fixed':
            return self._fixed_inputs.clone()

        if self.resample:
            return self._build_sampled_inputs()

        if self._cached_inputs is None:
            self._cached_inputs = self._build_sampled_inputs()
        return self._cached_inputs.clone()

    def __call__(self) -> torch.Tensor:
        inputs = self._get_base_inputs().detach()
        return inputs.requires_grad_(self.requires_grad)
