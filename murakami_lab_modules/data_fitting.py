from collections.abc import Callable

import torch

from . import utils
from .data_handler import DataHandler
from .neural_network import AbstractNeuralNetwork

__all__ = [
    'DataFitting',
    'MultiClassClassificationFitting',
    'BinaryClassificationFitting',
]


class DataFitting:
    def __init__(
            self,
            data_handler: DataHandler,
            loss_criteria: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.MSELoss(),
    ):
        self.locals = utils.get_local_dict(locals())
        self.data_handler = data_handler
        self.loss_criteria = loss_criteria
        self._nn_call_styles: dict[int, str] = {}

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'loss_criteria': self.loss_criteria,
        })

    def compute_loss(
            self,
            nn: AbstractNeuralNetwork,
            x: torch.Tensor,
            y: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> dict[str, object]:
        y_pred = self.predict(nn=nn, x=x, label=label, phase=phase, epoch=epoch)
        loss = self.loss_criteria(y, y_pred)
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
            nn: AbstractNeuralNetwork,
            x: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> torch.Tensor:
        return self._call_nn(nn, x)

    def _call_nn(self, nn: AbstractNeuralNetwork, x: torch.Tensor) -> torch.Tensor:
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
            raise TypeError(f'loss_criteria must return torch.Tensor. {type(loss)} was returned.')
        if loss.numel() != 1:
            raise ValueError(
                f'loss_criteria must return a scalar tensor. '
                f'loss.shape={tuple(loss.shape)} was returned.'
            )


class MultiClassClassificationFitting(DataFitting):
    def __init__(
            self,
            data_handler: DataHandler,
            loss_criteria: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(
            data_handler=data_handler,
            loss_criteria=loss_criteria or torch.nn.CrossEntropyLoss()
        )

    def compute_loss(
            self,
            nn: AbstractNeuralNetwork,
            x: torch.Tensor,
            y: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> dict[str, object]:
        logits = self.predict(nn=nn, x=x, label=label, phase=phase, epoch=epoch)
        target = self._prepare_target(y)
        loss = self.loss_criteria(logits, target)
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
    def __init__(
            self,
            data_handler: DataHandler,
            loss_criteria: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(
            data_handler=data_handler,
            loss_criteria=loss_criteria or torch.nn.BCEWithLogitsLoss()
        )

    def compute_loss(
            self,
            nn: AbstractNeuralNetwork,
            x: torch.Tensor,
            y: torch.Tensor,
            label=None,
            phase: str = None,
            epoch: int = None
    ) -> dict[str, object]:
        logits = self.predict(nn=nn, x=x, label=label, phase=phase, epoch=epoch)
        target = self._prepare_target(y, logits)
        loss = self.loss_criteria(logits, target)
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
