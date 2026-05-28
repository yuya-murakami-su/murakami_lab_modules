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
            check_test: bool = False
    ):
        self.locals = utils.get_local_dict(locals())
        self.data_handler = data_handler
        self.loss_criteria = loss_criteria
        self.check_test = check_test

        if data_handler.n_data['test'] == 0 and self.check_test:
            utils.logging(f'[Warning] check_test was set to True while no test data available.')
            self.check_test = False

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'loss_criteria': self.loss_criteria,
            'check_test': self.check_test
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
        y_pred = self._call_nn(nn, x)
        loss = self.loss_criteria(y, y_pred)
        self._validate_loss(loss)
        return {
            'total': loss,
            'terms': {
                'data': loss
            },
            'y_pred': y_pred
        }

    @staticmethod
    def _call_nn(nn: AbstractNeuralNetwork, x: torch.Tensor) -> torch.Tensor:
        try:
            return nn(x=x)
        except TypeError as e:
            if "unexpected keyword argument 'x'" not in str(e):
                raise
            return nn(x)

    @staticmethod
    def _validate_loss(loss: torch.Tensor) -> None:
        if not torch.is_tensor(loss):
            raise TypeError(f'loss_criteria must return torch.Tensor. {type(loss)} was returned.')
        if loss.numel() != 1:
            raise ValueError(
                f'loss_criteria must return a scalar tensor. '
                f'loss.shape={tuple(loss.shape)} was returned.'
            )
