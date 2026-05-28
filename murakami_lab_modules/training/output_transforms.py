"""Output transforms for latent-output regression.

Output transforms describe model structure that sits outside the neural
network. They are useful when the network should learn a latent quantity
``z = N(x)`` while observations are generated as ``y = f(x, z)``.
"""

from collections.abc import Sequence

import torch

from .. import utils

__all__ = [
    'BaseOutputTransform',
    'IdentityOutputTransform',
    'InputProductOutputTransform',
]


class BaseOutputTransform:
    """Base class for observation-to-latent output transforms.

    Subclasses implement two inverse directions:

    - ``to_latent(x, y)`` maps raw inputs and raw observations to latent targets.
    - ``to_observed(x, z)`` maps raw inputs and raw latent predictions back to
      raw observations.

    The class mirrors the normalizer serialization API so transforms can be
    stored with a saved model.
    """

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> 'BaseOutputTransform':
        """Fit transform state from raw training inputs and outputs."""

        return self

    def to_latent(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return latent targets from raw inputs and raw observations."""

        raise NotImplementedError

    def to_observed(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return raw observations from raw inputs and raw latent values."""

        raise NotImplementedError

    def state_dict(self) -> dict[str, object]:
        """Return fitted transform state."""

        return {}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Load fitted transform state."""

        pass

    def to(self, device: torch.device | str) -> 'BaseOutputTransform':
        """Move tensor state to ``device``."""

        return self

    def config_dict(self) -> dict[str, object]:
        """Return a serializable class reference and transform state."""

        return {
            'class': utils.get_object_path(self.__class__),
            'state_dict': self.state_dict()
        }

    @classmethod
    def from_config_dict(cls, config: dict[str, object]) -> 'BaseOutputTransform':
        """Reconstruct a transform from ``config_dict`` output."""

        transform_cls = utils.import_object(config['class'])
        transform = transform_cls()
        transform.load_state_dict(config['state_dict'])
        return transform


class IdentityOutputTransform(BaseOutputTransform):
    """Pass-through transform where the network latent is the observation."""

    def to_latent(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y

    def to_observed(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return z


class InputProductOutputTransform(BaseOutputTransform):
    """Represent observations as an input product times a latent quantity.

    This transform covers models such as ``y = x_i * N(x)`` and
    ``y = x_i * x_j * N(x)``. For a more complex observed variable, subclass
    this class and override ``observed_to_base`` / ``base_to_observed``.

    Parameters
    ----------
    input_indices:
        Input feature columns whose product forms ``g(x)``.
    min_abs_factor:
        Optional lower absolute bound used only when dividing by ``g(x)`` in
        ``to_latent``. ``to_observed`` uses the raw factor so exact boundary
        behavior such as ``g(x)=0 -> y=0`` is preserved.
    """

    def __init__(
            self,
            input_indices: int | Sequence[int] = 0,
            min_abs_factor: float | None = 1e-12,
    ):
        self.input_indices = self._normalize_input_indices(input_indices)
        self.min_abs_factor = min_abs_factor

    @staticmethod
    def _normalize_input_indices(input_indices: int | Sequence[int]) -> tuple[int, ...]:
        if type(input_indices) is int:
            return (input_indices,)
        indices = tuple(int(idx) for idx in input_indices)
        if len(indices) == 0:
            raise ValueError('input_indices must contain at least one index.')
        return indices

    def _factor(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f'{self.__class__.__name__} expects x with shape [N, features].')
        n_features = x.shape[1]
        normalized_indices = []
        for idx in self.input_indices:
            if not -n_features <= idx < n_features:
                raise IndexError(f'input index {idx} is out of range for {n_features} input features.')
            normalized_indices.append(idx % n_features)
        return x[:, normalized_indices].prod(dim=1, keepdim=True)

    def _safe_factor(self, factor: torch.Tensor) -> torch.Tensor:
        if self.min_abs_factor is None:
            return factor
        min_abs = torch.as_tensor(self.min_abs_factor, dtype=factor.dtype, device=factor.device)
        sign = torch.where(factor >= 0, torch.ones_like(factor), -torch.ones_like(factor))
        return torch.where(factor.abs() < min_abs, sign * min_abs, factor)

    def observed_to_base(self, y: torch.Tensor) -> torch.Tensor:
        """Map observed outputs to the product scale before division."""

        return y

    def base_to_observed(self, base: torch.Tensor) -> torch.Tensor:
        """Map product-scale predictions back to observed outputs."""

        return base

    def to_latent(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        factor = self._safe_factor(self._factor(x))
        return self.observed_to_base(y) / factor

    def to_observed(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.base_to_observed(self._factor(x) * z)

    def state_dict(self) -> dict[str, object]:
        return {
            'input_indices': self.input_indices,
            'min_abs_factor': self.min_abs_factor,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.input_indices = self._normalize_input_indices(state_dict['input_indices'])
        self.min_abs_factor = state_dict['min_abs_factor']
