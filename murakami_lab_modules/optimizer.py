import torch
import numpy as np
from typing import Iterable
from . import utils


class AbstractOptimizer:
    def __init__(
            self,
            algorithm: callable = torch.optim.Adam,
            **kwargs
    ):
        self.locals = utils.get_local_dict(locals())
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.lr_function = self.get_lr_function()
        self.optimizer = None

    def set_parameters(self, parameters: Iterable):
        if 'optimizer_params' in self.kwargs.keys():
            self.optimizer = self.algorithm(parameters, lr=self.lr_function(0), **self.kwargs['optimizer_params'])
        else:
            self.optimizer = self.algorithm(parameters, lr=self.lr_function(0))

    def get_lr_function(self) -> callable:
        raise NotImplementedError

    def update_lr(self, epoch: int):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_function(epoch)

    def step(self, epoch: int):
        self.update_lr(epoch)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class Optimizer(AbstractOptimizer):
    def __init__(
            self,
            algorithm: callable = torch.optim.Adam,
            lr: float = 1e-3,
            **kwargs
    ):
        self.lr = lr

        super().__init__(
            algorithm=algorithm,
            lr=lr,
            **kwargs
        )

    def get_lr_function(self) -> callable:
        def lr_function(_: int):
            return self.lr
        return lr_function

class OptimizerWithWarmup(AbstractOptimizer):
    def __init__(
            self,
            init_lr: float,
            init_epoch: int,
            final_lr: float = 1e-3,
            log_scale: bool = True,
            algorithm: callable = torch.optim.Adam,
            **kwargs
    ):
        self.init_lr = init_lr
        self.init_epoch = init_epoch
        self.final_lr = final_lr
        self.log_scale = log_scale

        super().__init__(
            algorithm=algorithm,
            init_lr=init_lr,
            init_epoch=init_epoch,
            final_lr=final_lr,
            loc_scale=log_scale,
            **kwargs
        )


    def get_lr_function(self) -> callable:
        if self.log_scale:
            def lr_function(epoch: int):
                if epoch < self.init_epoch:
                    return np.exp((np.log(self.final_lr / self.init_lr)) * epoch / self.init_epoch) * self.init_lr
                else:
                    return self.final_lr
        else:
            def lr_function(epoch: int):
                if epoch < self.init_epoch:
                    return (self.final_lr - self.init_lr) * epoch / self.init_epoch + self.init_lr
                else:
                    return self.final_lr
        return lr_function


class OptimizerWithWarmupAndDecay(AbstractOptimizer):
    def __init__(
            self,
            init_lr: float,
            mid_epoch: int,
            mid_lr: float,
            final_epoch: int,
            final_lr: float,
            log_scale: bool = True,
            algorithm: callable = torch.optim.Adam,
            **kwargs
    ):
        self.init_lr = init_lr
        self.mid_epoch = mid_epoch
        self.mid_lr = mid_lr
        self.final_epoch = final_epoch
        self.final_lr = final_lr
        self.log_scale = log_scale

        super().__init__(
            algorithm=algorithm,
            init_lr=init_lr,
            init_epoch=mid_epoch,
            final_lr=final_lr,
            log_scale=log_scale,
            **kwargs
        )

    def get_lr_function(self):
        if self.log_scale:
            def lr_function(epoch: int):
                if epoch < self.mid_epoch:
                    return np.exp((np.log(self.mid_lr / self.init_lr)) * epoch / self.mid_epoch) * self.init_lr
                elif epoch < self.final_epoch:
                    return np.exp((np.log(self.final_lr / self.mid_lr)) *
                                  (epoch - self.mid_epoch) / (self.final_epoch - self.mid_epoch)) * self.mid_lr
                else:
                    return self.final_lr
        else:
            def lr_function(epoch: int):
                if epoch < self.mid_epoch:
                    return (self.mid_lr - self.init_lr) * epoch / self.mid_epoch + self.init_lr
                elif epoch < self.final_epoch:
                    return ((self.final_lr - self.mid_lr) * (epoch - self.mid_epoch)
                            / (self.final_epoch - self.mid_epoch) + self.mid_lr)
                else:
                    return self.final_lr
        return lr_function


class OptimizerWithInverseDecay(AbstractOptimizer):
    def __init__(
            self,
            init_lr: float,
            half_epoch: int,
            final_lr: float = None,
            algorithm: callable = torch.optim.Adam,
            **kwargs
    ):
        self.init_lr = init_lr
        self.half_epoch = half_epoch
        self.final_lr = final_lr

        super().__init__(
            algorithm=algorithm,
            init_lr=init_lr,
            inverse_rate=half_epoch,
            final_lr=final_lr,
            **kwargs
        )

    def get_lr_function(self):
        if self.final_lr is None:
            def lr_function(epoch: int):
                return self.init_lr / (1 + epoch / self.half_epoch)
        else:
            epoch_limit = (self.init_lr / self.final_lr - 1) * self.half_epoch
            def lr_function(epoch: int):
                if epoch_limit < epoch:
                    return self.final_lr
                else:
                    return self.init_lr / (1 + epoch / self.half_epoch)
        return lr_function