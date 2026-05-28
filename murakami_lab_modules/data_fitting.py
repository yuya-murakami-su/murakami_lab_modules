from collections.abc import Callable

import torch

from . import utils
from .data_handler import DataHandler

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
