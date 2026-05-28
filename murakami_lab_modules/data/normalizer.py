"""Normalizer classes used by :class:`murakami_lab_modules.data.DataHandler`.

Normalizers follow a small ``fit`` / ``transform`` / ``inverse_transform``
interface and carry enough state to be serialized with a saved model.
"""

import torch
from .. import utils

logger = utils.get_logger(__name__)


class BaseNormalizer:
    """Abstract base class for data normalizers.

    Subclasses should implement ``fit``, ``transform``, and
    ``inverse_transform``. They may also override ``state_dict`` and
    ``load_state_dict`` when fitted state needs to be saved.
    """

    def fit(self, data: torch.Tensor) -> 'BaseNormalizer':
        raise NotImplementedError

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        pass

    def to(self, device: torch.device | str) -> 'BaseNormalizer':
        return self

    def config_dict(self) -> dict[str, object]:
        """Return a serializable class reference and fitted state."""

        return {
            'class': utils.get_object_path(self.__class__),
            'state_dict': self.state_dict()
        }

    @classmethod
    def from_config_dict(cls, config: dict[str, object]) -> 'BaseNormalizer':
        """Reconstruct a normalizer from ``config_dict`` output."""

        normalizer_cls = utils.import_object(config['class'])
        normalizer = normalizer_cls()
        normalizer.load_state_dict(config['state_dict'])
        return normalizer


class IdentityNormalizer(BaseNormalizer):
    """Pass-through normalizer for already-scaled data or class targets."""

    def fit(self, data: torch.Tensor) -> 'IdentityNormalizer':
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data


class StandardNormalizer(BaseNormalizer):
    """Standardize data by subtracting mean and dividing by standard deviation.

    Statistics are computed over the sample axis and broadcast over remaining
    dimensions. ``exclude_indices`` can be used for feature columns that should
    remain unchanged.
    """

    _std_warned = False

    def __init__(self, exclude_indices: list[int] = None, epsilon: float = 1e-5):
        self.exclude_indices = exclude_indices
        self.epsilon = epsilon
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor) -> 'StandardNormalizer':
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True, unbiased=False)

        if torch.lt(self.std, self.epsilon).any():
            self.std = torch.where(torch.lt(self.std, self.epsilon), 1, self.std)
            if not self.__class__._std_warned:
                logger.warning('Standard deviation below epsilon was found during normalization.')
                self.__class__._std_warned = True

        if self.exclude_indices is not None and data.ndim < 2:
            raise ValueError(f'{self.__class__.__name__} expects at least 2D data when exclude_indices is used.')
        n_features = data.shape[1] if data.ndim >= 2 else 1
        exclude_indices = self._normalized_exclude_indices(n_features)
        if len(exclude_indices) > 0:
            self.mean[:, exclude_indices] = 0
            self.std[:, exclude_indices] = 1

        return self

    def _normalized_exclude_indices(self, n_features: int) -> list[int]:
        if self.exclude_indices is None:
            return []
        normalized = []
        for idx in self.exclude_indices:
            if not -n_features <= idx < n_features:
                raise IndexError(f'exclude_indices contains {idx}, which is out of range for {n_features} features.')
            normalized.append(idx % n_features)
        return normalized

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if data.numel() == 0:
            return data
        self._validate_fitted()
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        if data.numel() == 0:
            return data
        self._validate_fitted()
        return data * self.std + self.mean

    def _validate_fitted(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError(f'{self.__class__.__name__} must be fitted before transform is called.')

    def state_dict(self) -> dict[str, object]:
        return {
            'exclude_indices': self.exclude_indices,
            'epsilon': self.epsilon,
            'mean': self.mean,
            'std': self.std
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.exclude_indices = state_dict['exclude_indices']
        self.epsilon = state_dict['epsilon']
        self.mean = state_dict['mean']
        self.std = state_dict['std']

    def to(self, device: torch.device | str) -> 'StandardNormalizer':
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self


class LogStandardNormalizer(StandardNormalizer):
    """Apply ``log(data + log_epsilon)`` before standardization.

    This is useful for positive quantities that vary over orders of magnitude.
    ``exclude_indices`` keeps selected feature columns in the original scale.
    """

    def __init__(
            self,
            exclude_indices: list[int] = None,
            epsilon: float = 1e-5,
            log_epsilon: float = 1e-12
    ):
        super().__init__(exclude_indices=exclude_indices, epsilon=epsilon)
        self.log_epsilon = log_epsilon

    def fit(self, data: torch.Tensor) -> 'LogStandardNormalizer':
        return super().fit(self._forward_transform(data))

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return super().transform(self._forward_transform(data))

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self._inverse_forward_transform(super().inverse_transform(data))

    def _forward_transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.exclude_indices is None:
            return torch.log(data + self.log_epsilon)
        transformed = data.clone()
        transform_indices = self._transform_indices(data)
        if len(transform_indices) > 0:
            transformed[:, transform_indices] = torch.log(transformed[:, transform_indices] + self.log_epsilon)
        return transformed

    def _inverse_forward_transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.exclude_indices is None:
            return torch.exp(data) - self.log_epsilon
        transformed = data.clone()
        transform_indices = self._transform_indices(data)
        if len(transform_indices) > 0:
            transformed[:, transform_indices] = torch.exp(transformed[:, transform_indices]) - self.log_epsilon
        return transformed

    def _transform_indices(self, data: torch.Tensor) -> list[int]:
        if data.ndim < 2:
            raise ValueError(f'{self.__class__.__name__} expects 2D data when exclude_indices is used.')
        excluded = set(self._normalized_exclude_indices(data.shape[1]))
        return [idx for idx in range(data.shape[1]) if idx not in excluded]

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state['log_epsilon'] = self.log_epsilon
        return state

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        super().load_state_dict(state_dict)
        self.log_epsilon = state_dict['log_epsilon']
