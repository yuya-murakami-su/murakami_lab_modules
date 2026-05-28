from collections.abc import Callable

import torch

from . import utils
from .data_handler import DataHandler
from .neural_network import AbstractNeuralNetwork

__all__ = ['DataFitting']


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
