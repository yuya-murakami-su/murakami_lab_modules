"""Data-loss adapters used by the training loop.

``DataFitting`` separates "how to call the model and compute data loss" from
``ModelHandler``. Subclass it when a project needs ODE integration, label-aware
prediction, multiple data-loss terms, or non-standard model inputs.
"""

from collections.abc import Callable

import torch

from .. import utils
from ..data.data_handler import DataHandler
from ..data.normalizer import BaseNormalizer, StandardNormalizer
from ..models.neural_network import BaseNeuralNetwork
from .output_transforms import BaseOutputTransform, IdentityOutputTransform

__all__ = [
    'DataFitting',
    'LatentOutputFitting',
    'MultiClassClassificationFitting',
    'BinaryClassificationFitting',
]


class DataFitting:
    """Compute the data-fitting loss for regression-style training.

    The default implementation calls the network with ``nn(x=x)`` and falls
    back to ``nn(x)`` for ordinary PyTorch modules that do not accept keyword
    arguments.
    """

    def __init__(
            self,
            data_handler: DataHandler,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.MSELoss(),
    ):
        self.locals = utils.get_local_dict(locals())
        self.data_handler = data_handler
        self.loss_fn = loss_fn
        self._nn_call_styles: dict[int, str] = {}

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'loss_fn': self.loss_fn,
        })

    def to_observed_target(self, y: torch.Tensor) -> torch.Tensor:
        """Convert a batch target to the observed output scale."""

        return self.data_handler.undo_normalize_y(y)

    def to_observed_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Convert model predictions to the observed output scale."""

        return self.data_handler.undo_normalize_y(y_pred)

    def compute_loss(
            self,
            nn: BaseNeuralNetwork,
            x: torch.Tensor,
            y: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> dict[str, object]:
        """Return a scalar loss and optional term dictionary for one batch.

        Subclasses should keep the returned ``total`` value scalar so
        ``backward()`` can be called safely by ``ModelHandler``.
        """

        y_pred = self.predict(nn=nn, x=x, label=label, phase=phase, epoch=epoch)
        loss = self.loss_fn(y, y_pred)
        self._validate_loss(loss)
        return {
            'total': loss,
            'terms': {
                'data': loss
            },
            'y_pred': y_pred
        }

    def predict(
            self,
            nn: BaseNeuralNetwork,
            x: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> torch.Tensor:
        """Return model predictions for one batch.

        Override this method when prediction requires labels, time integration,
        or structured inputs.
        """

        return self._call_nn(nn, x)

    def _call_nn(self, nn: BaseNeuralNetwork, x: torch.Tensor) -> torch.Tensor:
        call_key = id(nn)
        call_style = self._nn_call_styles.get(call_key)
        if call_style == 'keyword':
            return nn(x=x)
        if call_style == 'positional':
            return nn(x)

        try:
            y = nn(x=x)
        except TypeError as e:
            if "unexpected keyword argument 'x'" not in str(e):
                raise
            self._nn_call_styles[call_key] = 'positional'
            return nn(x)
        self._nn_call_styles[call_key] = 'keyword'
        return y

    @staticmethod
    def _validate_loss(loss: torch.Tensor) -> None:
        if not torch.is_tensor(loss):
            raise TypeError(f'loss_fn must return torch.Tensor. {type(loss)} was returned.')
        if loss.numel() != 1:
            raise ValueError(
                f'loss_fn must return a scalar tensor. '
                f'loss.shape={tuple(loss.shape)} was returned.'
            )


class LatentOutputFitting(DataFitting):
    """Data fitting for models that predict a latent output quantity.

    The network is expected to return normalized latent values ``z_norm``.
    ``latent_normalizer`` maps between ``z`` and ``z_norm``. ``output_transform``
    maps raw inputs plus raw latent values to raw observations.

    By default the loss is computed in observed ``y`` space:

    ``NN(x_norm) -> z_norm -> z -> output_transform.to_observed(x_raw, z)``.
    """

    def __init__(
            self,
            data_handler: DataHandler,
            output_transform: BaseOutputTransform = None,
            latent_normalizer: BaseNormalizer = None,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.MSELoss(),
            loss_space: str = 'observed',
            observed_loss_weight: float = 1.0,
            latent_loss_weight: float = 1.0,
    ):
        super().__init__(data_handler=data_handler, loss_fn=loss_fn)
        if loss_space not in {'observed', 'latent', 'both'}:
            raise ValueError("loss_space must be one of 'observed', 'latent', or 'both'.")
        if observed_loss_weight < 0 or latent_loss_weight < 0:
            raise ValueError('loss weights must be non-negative.')

        self.output_transform = output_transform or IdentityOutputTransform()
        self.latent_normalizer = latent_normalizer or StandardNormalizer()
        self.loss_space = loss_space
        self.observed_loss_weight = float(observed_loss_weight)
        self.latent_loss_weight = float(latent_loss_weight)
        self._fit_latent_pipeline()
        self.locals = utils.get_local_dict(locals())

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'output_transform': self.output_transform.config_dict(),
            'latent_normalizer': self.latent_normalizer.config_dict(),
            'loss_fn': self.loss_fn,
            'loss_space': self.loss_space,
            'observed_loss_weight': self.observed_loss_weight,
            'latent_loss_weight': self.latent_loss_weight,
        })

    def _fit_latent_pipeline(self) -> None:
        x_train = self.data_handler.undo_normalize_x(self.data_handler.train.inputs)
        y_train = self.data_handler.undo_normalize_y(self.data_handler.train.outputs)
        self.output_transform.fit(x_train, y_train)
        z_train = self.output_transform.to_latent(x_train, y_train)
        self.latent_normalizer.fit(z_train)
        self.output_transform.to(self.data_handler.device)
        self.latent_normalizer.to(self.data_handler.device)

    def to_observed_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred

    def predict_latent_normalized(
            self,
            nn: BaseNeuralNetwork,
            x: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> torch.Tensor:
        """Return normalized latent predictions from the network."""

        return self._call_nn(nn, x)

    def predict_latent(
            self,
            nn: BaseNeuralNetwork,
            x: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> torch.Tensor:
        """Return latent predictions in the raw latent scale."""

        z_norm = self.predict_latent_normalized(nn=nn, x=x, label=label, phase=phase, epoch=epoch)
        return self.latent_normalizer.inverse_transform(z_norm)

    def predict(
            self,
            nn: BaseNeuralNetwork,
            x: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> torch.Tensor:
        x_raw = self.data_handler.undo_normalize_x(x)
        z = self.predict_latent(nn=nn, x=x, label=label, phase=phase, epoch=epoch)
        return self.output_transform.to_observed(x_raw, z)

    def compute_loss(
            self,
            nn: BaseNeuralNetwork,
            x: torch.Tensor,
            y: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> dict[str, object]:
        x_raw = self.data_handler.undo_normalize_x(x)
        y_raw = self.to_observed_target(y)
        z_pred_norm = self.predict_latent_normalized(nn=nn, x=x, label=label, phase=phase, epoch=epoch)
        z_pred = self.latent_normalizer.inverse_transform(z_pred_norm)
        y_pred = self.output_transform.to_observed(x_raw, z_pred)

        terms = {}
        total = None
        if self.loss_space in {'observed', 'both'}:
            observed_loss = self.loss_fn(y_raw, y_pred)
            self._validate_loss(observed_loss)
            terms['observed'] = observed_loss
            total = observed_loss * self.observed_loss_weight
        if self.loss_space in {'latent', 'both'}:
            z_target = self.output_transform.to_latent(x_raw, y_raw)
            z_target_norm = self.latent_normalizer.transform(z_target)
            latent_loss = self.loss_fn(z_target_norm, z_pred_norm)
            self._validate_loss(latent_loss)
            terms['latent'] = latent_loss
            weighted_latent_loss = latent_loss * self.latent_loss_weight
            total = weighted_latent_loss if total is None else total + weighted_latent_loss

        if total is None:
            raise RuntimeError('No loss term was computed.')
        self._validate_loss(total)
        terms['data'] = total
        return {
            'total': total,
            'terms': terms,
            'y_pred': y_pred,
            'z_pred': z_pred,
            'z_pred_norm': z_pred_norm,
        }


class MultiClassClassificationFitting(DataFitting):
    """Data fitting for multi-class classification with class-index targets."""

    def __init__(
            self,
            data_handler: DataHandler,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(
            data_handler=data_handler,
            loss_fn=loss_fn or torch.nn.CrossEntropyLoss()
        )

    def compute_loss(
            self,
            nn: BaseNeuralNetwork,
            x: torch.Tensor,
            y: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> dict[str, object]:
        logits = self.predict(nn=nn, x=x, label=label, phase=phase, epoch=epoch)
        target = self._prepare_target(y)
        loss = self.loss_fn(logits, target)
        self._validate_loss(loss)
        return {
            'total': loss,
            'terms': {
                'data': loss
            },
            'y_pred': logits
        }

    @staticmethod
    def _prepare_target(y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 2 and y.shape[1] == 1:
            return y[:, 0].long()
        if y.ndim == 1:
            return y.long()
        raise ValueError(
            'MultiClassClassificationFitting expects class-index targets with shape [N] or [N, 1]. '
            f'y.shape={tuple(y.shape)} was given.'
        )


class BinaryClassificationFitting(DataFitting):
    """Data fitting for binary classification with ``BCEWithLogitsLoss``."""

    def __init__(
            self,
            data_handler: DataHandler,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(
            data_handler=data_handler,
            loss_fn=loss_fn or torch.nn.BCEWithLogitsLoss()
        )

    def compute_loss(
            self,
            nn: BaseNeuralNetwork,
            x: torch.Tensor,
            y: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> dict[str, object]:
        logits = self.predict(nn=nn, x=x, label=label, phase=phase, epoch=epoch)
        target = self._prepare_target(y, logits)
        loss = self.loss_fn(logits, target)
        self._validate_loss(loss)
        return {
            'total': loss,
            'terms': {
                'data': loss
            },
            'y_pred': logits
        }

    @staticmethod
    def _prepare_target(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        target = y.to(dtype=logits.dtype)
        if target.shape == logits.shape:
            return target
        if logits.ndim == 2 and logits.shape[1] == 1 and target.ndim == 1:
            return target.reshape(-1, 1)
        if logits.ndim == 1 and target.ndim == 2 and target.shape[1] == 1:
            return target[:, 0]
        raise ValueError(
            'BinaryClassificationFitting target shape must match logits, allowing [N] <-> [N, 1]. '
            f'y.shape={tuple(y.shape)}, logits.shape={tuple(logits.shape)} were given.'
        )
