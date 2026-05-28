import time
import glob
import os
import pandas as pd
import numpy as np
import torch
import copy
from collections.abc import Callable
from .neural_network import AbstractNeuralNetwork
from .data_handler import DataHandler
from .input_generator import InputGenerator
from .optimizer import AbstractOptimizer
from .plotter import Plotter
from . import utils


def get_relative_error(epsilon: float = 1e-10, as_loss_function: bool = False):
    if as_loss_function:
        def relative_error(y_true: torch.Tensor, y_calc: torch.Tensor):
            return (y_true - y_calc).abs() / (y_true.abs() + epsilon).mean()
    else:
        def relative_error(y_true: torch.Tensor, y_calc: torch.Tensor):
            return ((y_true - y_calc).abs() / (y_true.abs() + epsilon)).mean(dim=1, keepdim=True)
    return relative_error


def get_mean_squared_error(as_loss_function: bool = False):
    if as_loss_function:
        mse_func = torch.nn.MSELoss()

        def mse(y_true: torch.Tensor, y_calc: torch.Tensor):
            return mse_func(y_true, y_calc).mean()
    else:
        mse_func = torch.nn.MSELoss(reduction='none')

        def mse(y_true: torch.Tensor, y_calc: torch.Tensor):
            return mse_func(y_true, y_calc).mean(dim=1, keepdim=True)
    return mse


def get_absolute_error(as_loss_function: bool = False):
    if as_loss_function:
        def absolute_error(y_true: torch.Tensor, y_calc: torch.Tensor):
            return (y_true - y_calc).abs().mean()
    else:
        def absolute_error(y_true: torch.Tensor, y_calc: torch.Tensor):
            return (y_true - y_calc).abs().mean(dim=1, keepdim=True)
    return absolute_error


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


class Regularization:
    _different_n_points_warned = False

    def __init__(
            self,
            input_generators: list[InputGenerator] | tuple[InputGenerator, ...],
            reg_weights: list[float],
            reg_func_name: str = 'regularization',
            reg_names: list[str] = None,
            reg_criteria: Callable[[torch.Tensor], torch.Tensor] = torch.square,
            use_reg_prod: bool = False,
            reg_min: float = None
    ):
        self.locals = utils.get_local_dict(locals())
        self.input_generators = input_generators
        self.reg_func_name = reg_func_name
        self.reg_weights = reg_weights
        self.reg_names = reg_names
        self.reg_criteria = reg_criteria
        self.use_reg_prod = use_reg_prod
        self.reg_min = reg_min

        self.n_generator = len(input_generators)
        self.device = input_generators[0].device
        self.device_name = input_generators[0].device_name

        if not hasattr(self, f'{reg_func_name}'):
            raise ValueError(f'{self.__class__.__name__} does not have a method named {reg_func_name}.')

        self.reg_func = getattr(self, f'{reg_func_name}')

        self.n_reg = len(reg_weights)
        self.reg_weights = torch.tensor(reg_weights, device=self.device, dtype=torch.float32)
        if self.use_reg_prod:
            self.reg_mean_pow = torch.tensor(1 / self.n_reg, dtype=torch.float, device=self.device)

        if reg_names is None:
            self.reg_names = [f'Reg{i}' for i in range(self.n_reg)]
        elif len(reg_names) != self.n_reg:
            raise ValueError(f'Inconsistent length of reg_names: '
                             f'len(_{reg_func_name}()) = {self.n_reg}, len(reg_names) = {len(reg_names)}.')

        if reg_min is None:
            self.reg_min = torch.zeros([self.n_reg], device=self.device, dtype=torch.float32)
        else:
            self.reg_min = torch.full([self.n_reg], reg_min, device=self.device, dtype=torch.float32)

    def regularization(self, data_handler: DataHandler, nn: AbstractNeuralNetwork):
        raise NotImplementedError

    def _validate_regularization_outputs(self, regs) -> list[torch.Tensor] | tuple[torch.Tensor, ...]:
        if not isinstance(regs, (list, tuple)):
            raise TypeError(
                f'{self.reg_func_name}() must return list or tuple of torch.Tensor. '
                f'{type(regs)} was returned.'
            )
        if len(regs) != self.n_reg:
            raise ValueError(
                f'Inconsistent number of regularization terms: '
                f'len({self.reg_func_name}()) = {len(regs)}, len(reg_weights) = {self.n_reg}.'
            )

        for idx, reg in enumerate(regs):
            if not torch.is_tensor(reg):
                raise TypeError(
                    f'{self.reg_func_name}()[{idx}] must be torch.Tensor. {type(reg)} was returned.'
                )
            if reg.numel() == 0:
                raise ValueError(f'{self.reg_func_name}()[{idx}] is empty.')
        n_points = [reg.shape[0] for reg in regs if reg.ndim > 0]
        if len(set(n_points)) > 1 and not self.__class__._different_n_points_warned:
            utils.logging(
                f'[Warning] Regularization terms have different n_points: {n_points}. '
                f'Each term is averaged independently before applying reg_weights.'
            )
            self.__class__._different_n_points_warned = True
        return regs

    def _get_regularization_mean(self, regs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        reg_means = []
        full_regs = []
        for reg in regs:
            full_reg = self.reg_criteria(reg)
            is_finite = torch.isfinite(full_reg)
            if not is_finite.all():
                if not is_finite.any():
                    torch.save(full_regs + [full_reg], 'invalid_regularization.pth')
                    raise ValueError(f'Too many invalid value was encountered during regularization.')
                utils.logging(f'Invalid value was encountered during regularization.')
                full_reg = torch.where(is_finite, full_reg, 0.0)
                reg_mean = full_reg.sum() / is_finite.sum()
            else:
                reg_mean = full_reg.mean()

            full_regs.append(full_reg)
            reg_means.append(reg_mean)

        return torch.stack(reg_means)

    def get_regularization_value(
            self,
            nn: AbstractNeuralNetwork,
            data_handler: DataHandler = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        regs = self._validate_regularization_outputs(self.reg_func(data_handler=data_handler, nn=nn))
        reg_mean = self._get_regularization_mean(regs)

        if self.use_reg_prod:
            reg_mean.add_(self.reg_min).pow_(self.reg_weights)
            reg_value = reg_mean.prod()
        else:
            reg_mean.mul_(self.reg_weights).add_(self.reg_min)
            reg_value = reg_mean.sum()

        return reg_mean, reg_value

    @staticmethod
    def _normalize_indices(indices: int | list[int] | tuple[int, ...] | None, size: int, name: str) -> list[int]:
        if indices is None:
            return list(range(size))
        if type(indices) is int:
            indices = (indices,)
        elif not isinstance(indices, (list, tuple)):
            raise TypeError(f'{name} must be int, list[int], tuple[int, ...], or None. {type(indices)} was given.')

        normalized = []
        for idx in indices:
            if type(idx) is not int:
                raise TypeError(f'All elements of {name} must be int. {type(idx)} was given.')
            if not -size <= idx < size:
                raise IndexError(f'{name} contains {idx}, which is out of range for size {size}.')
            normalized.append(idx % size)
        return normalized

    @staticmethod
    def _normalize_y_indices(y: torch.Tensor, y_indices: int | list[int] | tuple[int, ...] | None) -> list[int | None]:
        if y.ndim == 1:
            if y_indices is None:
                return [None]
            normalized = Regularization._normalize_indices(y_indices, 1, 'y_indices')
            return [None for _ in normalized]
        if y.ndim < 1:
            raise ValueError(f'y must have at least 1 dimension. y.shape={tuple(y.shape)}.')
        return Regularization._normalize_indices(y_indices, y.shape[1], 'y_indices')

    @staticmethod
    def _normalize_x_indices(x: torch.Tensor, x_indices: int | list[int] | tuple[int, ...] | None) -> list[int]:
        if x.ndim < 2:
            raise ValueError(f'x_indices requires x.ndim >= 2. x.shape={tuple(x.shape)}.')
        return Regularization._normalize_indices(x_indices, x.shape[1], 'x_indices')

    @staticmethod
    def grad(
            y: torch.Tensor,
            x: torch.Tensor,
            x_idx: int = None,
            y_idx: int = None,
            zero_if_unused: bool = False,
            keepdim: bool = False
    ):
        if not torch.is_tensor(y):
            raise TypeError(f'y must be torch.Tensor. {type(y)} was given.')
        if not torch.is_tensor(x):
            raise TypeError(f'x must be torch.Tensor. {type(x)} was given.')
        if not x.requires_grad:
            raise ValueError(f'x must require grad. x.requires_grad={x.requires_grad}.')

        if y_idx is not None:
            if y.ndim < 2:
                raise ValueError(f'y_idx requires y.ndim >= 2. y.shape={tuple(y.shape)}.')
            if not -y.shape[1] <= y_idx < y.shape[1]:
                raise IndexError(f'y_idx={y_idx} is out of range for y.shape={tuple(y.shape)}.')

        if x_idx is not None:
            if x.ndim < 2:
                raise ValueError(f'x_idx requires x.ndim >= 2. x.shape={tuple(x.shape)}.')
            if not -x.shape[1] <= x_idx < x.shape[1]:
                raise IndexError(f'x_idx={x_idx} is out of range for x.shape={tuple(x.shape)}.')
            normalized_x_idx = x_idx % x.shape[1]

        def select_x_idx(dy_dx: torch.Tensor):
            if x_idx is None:
                return dy_dx
            elif keepdim:
                return dy_dx[:, normalized_x_idx:normalized_x_idx + 1]
            else:
                return dy_dx[:, x_idx]

        if not y.requires_grad:
            if zero_if_unused:
                return select_x_idx(x * 0.0)
            raise ValueError(f'y must require grad. y.requires_grad={y.requires_grad}.')

        if y_idx is None:
            grad_outputs = torch.ones_like(y)
        else:
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, y_idx] = 1.0

        try:
            dy_dx = torch.autograd.grad(
                inputs=x,
                outputs=y,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=True,
                allow_unused=zero_if_unused
            )[0]
        except RuntimeError as e:
            if 'not have been used in the graph' not in str(e):
                raise
            raise RuntimeError(
                f'Failed to compute grad(y, x) because y does not depend on x. '
                f'y.shape={tuple(y.shape)}, x.shape={tuple(x.shape)}, '
                f'x_idx={x_idx}, y_idx={y_idx}. If y is intentionally independent of x, '
                f'set zero_if_unused=True.'
            ) from e

        if dy_dx is None:
            dy_dx = x * 0.0
        elif zero_if_unused and not dy_dx.requires_grad:
            dy_dx = dy_dx + x * 0.0

        return select_x_idx(dy_dx)

    @staticmethod
    def partial(
            y: torch.Tensor,
            x: torch.Tensor,
            x_idx: int,
            y_idx: int = None,
            zero_if_unused: bool = False,
            keepdim: bool = False
    ) -> torch.Tensor:
        if x_idx is None:
            raise ValueError('x_idx must be given for partial().')
        return Regularization.grad(
            y=y,
            x=x,
            x_idx=x_idx,
            y_idx=y_idx,
            zero_if_unused=zero_if_unused,
            keepdim=keepdim
        )

    @staticmethod
    def partial2(
            y: torch.Tensor,
            x: torch.Tensor,
            x_idx: int,
            y_idx: int = None,
            zero_if_unused: bool = False,
            keepdim: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dy_dx = Regularization.partial(
            y=y,
            x=x,
            x_idx=x_idx,
            y_idx=y_idx,
            zero_if_unused=zero_if_unused,
            keepdim=keepdim
        )
        d2y_dx2 = Regularization.partial(
            y=dy_dx,
            x=x,
            x_idx=x_idx,
            zero_if_unused=zero_if_unused or not dy_dx.requires_grad,
            keepdim=keepdim
        )
        return dy_dx, d2y_dx2

    @staticmethod
    def second_partial(
            y: torch.Tensor,
            x: torch.Tensor,
            x_idx: int,
            y_idx: int = None,
            zero_if_unused: bool = False,
            keepdim: bool = False
    ) -> torch.Tensor:
        return Regularization.partial2(
            y=y,
            x=x,
            x_idx=x_idx,
            y_idx=y_idx,
            zero_if_unused=zero_if_unused,
            keepdim=keepdim
        )[1]

    @staticmethod
    def jacobian(
            y: torch.Tensor,
            x: torch.Tensor,
            y_indices: int | list[int] | tuple[int, ...] = None,
            x_indices: int | list[int] | tuple[int, ...] = None,
            zero_if_unused: bool = False
    ) -> torch.Tensor:
        y_indices = Regularization._normalize_y_indices(y, y_indices)
        x_indices = Regularization._normalize_x_indices(x, x_indices)
        jacobian = []
        for y_idx in y_indices:
            row = [
                Regularization.partial(
                    y=y,
                    x=x,
                    x_idx=x_idx,
                    y_idx=y_idx,
                    zero_if_unused=zero_if_unused,
                    keepdim=True
                )
                for x_idx in x_indices
            ]
            jacobian.append(torch.cat(row, dim=1))
        return torch.stack(jacobian, dim=1)

    @staticmethod
    def hessian_diag(
            y: torch.Tensor,
            x: torch.Tensor,
            y_indices: int | list[int] | tuple[int, ...] = None,
            x_indices: int | list[int] | tuple[int, ...] = None,
            zero_if_unused: bool = False,
            return_first: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        y_indices = Regularization._normalize_y_indices(y, y_indices)
        x_indices = Regularization._normalize_x_indices(x, x_indices)
        jacobian = []
        hessian_diag = []
        for y_idx in y_indices:
            jacobian_row = []
            hessian_diag_row = []
            for x_idx in x_indices:
                dy_dx, d2y_dx2 = Regularization.partial2(
                    y=y,
                    x=x,
                    x_idx=x_idx,
                    y_idx=y_idx,
                    zero_if_unused=zero_if_unused,
                    keepdim=True
                )
                jacobian_row.append(dy_dx)
                hessian_diag_row.append(d2y_dx2)
            jacobian.append(torch.cat(jacobian_row, dim=1))
            hessian_diag.append(torch.cat(hessian_diag_row, dim=1))

        jacobian = torch.stack(jacobian, dim=1)
        hessian_diag = torch.stack(hessian_diag, dim=1)
        if return_first:
            return jacobian, hessian_diag
        return hessian_diag

    @staticmethod
    def laplacian(
            y: torch.Tensor,
            x: torch.Tensor,
            y_indices: int | list[int] | tuple[int, ...] = None,
            x_indices: int | list[int] | tuple[int, ...] = None,
            zero_if_unused: bool = False,
            keepdim: bool = False
    ) -> torch.Tensor:
        laplacian = Regularization.hessian_diag(
            y=y,
            x=x,
            y_indices=y_indices,
            x_indices=x_indices,
            zero_if_unused=zero_if_unused
        ).sum(dim=2)
        if laplacian.shape[1] == 1 and not keepdim:
            return laplacian[:, 0]
        return laplacian


class ModelHandler:
    def __init__(
            self,
            nn: AbstractNeuralNetwork,
            optimizer: AbstractOptimizer,
            data_fitting: DataFitting = None,
            regularization: Regularization = None,
            train_epochs: int = None,
            early_stop: int = 0,
            load_model: str = None,
            load_optimizer: bool = False,
            save_path: str = 'Model',
            train_record_path: str = 'train_record',
            recalculate_valid_loss: bool = True,
            model_name: str = None,
            callback_epoch: int = None,
            callbacks: tuple[object, ...] = None,
            random_seed: int = 2025,
            **kwargs
    ):
        self.locals = utils.get_local_dict(locals())
        utils.initialize_random_seed(random_seed)

        self.nn = nn
        self.optimizer = optimizer
        self.data_fitting = data_fitting
        self.regularization = regularization

        self.train_epochs = train_epochs
        self.early_stop = early_stop

        self.load_model = load_model
        self.load_optimizer = load_optimizer
        self.random_seed = random_seed

        self.save_path = save_path
        self.train_record_path = train_record_path
        self.recalculate_valid_loss = recalculate_valid_loss
        self.model_name = model_name
        self.kwargs = kwargs
        self.callback_epoch = callback_epoch
        self.callbacks = callbacks or []

        self._validate_inputs()
        self._prepare_model_folder()
        self._save_model_info()
        self._set_model()
        self._prepare_train_record()
        self._prepare_train_valuables()

    def _validate_inputs(self):
        if self.data_fitting is None and self.regularization is None:
            raise ValueError('At least one of data_handler_ or input_generators must be given.')

        if self.regularization is None:
            self.device = self.data_fitting.data_handler.device
            self.device_name = self.data_fitting.data_handler.device_name
            self.has_data = True
            self.has_reg = False
        else:
            self.device = self.regularization.device
            self.device_name = self.regularization.device_name
            self.has_reg = True
            if self.data_fitting is None:
                self.has_data = False
            else:
                self.has_data = True
            if self.has_data and self.data_fitting.data_handler.device != self.regularization.device:
                raise ValueError(f'Different device was given for data_fitting and regularization. '
                                 f'data_fitting: {self.data_fitting.data_handler.device}, '
                                 f'regularization: {self.regularization.device}')

        if self.model_name is None:
            self.model_name = utils.get_current_time(for_file_name=True)

        if (self.train_epochs is None or self.train_epochs == 0) and self.early_stop == 0:
            raise ValueError('At least of of train_epochs and early_stop must be give.')

    def _prepare_model_folder(self):
        folder_idx = len(glob.glob(f'{self.save_path}\\*'))
        self.model_name = f'{folder_idx + 1:0>5}_{self.model_name}'
        self.model_path = f'{self.save_path}\\{self.model_name}'
        os.makedirs(f'{self.model_path}')

    def _save_model_info(self):
        utils.save_txt(f'{self.model_path}\\nn_params', **self.nn.locals)
        utils.save_txt(f'{self.model_path}\\optimizer_params', **self.optimizer.locals)
        utils.save_txt(f'{self.model_path}\\model_handler_params', **self.locals)
        if self.has_data:
            utils.save_txt(f'{self.model_path}\\data_fitting_params', **self.data_fitting.locals)
            utils.save_txt(f'{self.model_path}\\data_handler_params', **self.data_fitting.data_handler.locals)
            torch.save(self.data_fitting.data_handler.normalizer_dict(), f'{self.model_path}\\normalizer.pth')
        if self.has_reg:
            utils.save_txt(f'{self.model_path}\\regularization_params', **self.regularization.locals)
            for idx, input_generator_ in enumerate(self.regularization.input_generators):
                utils.save_txt(f'{self.model_path}\\input_generator_{idx}_params', **input_generator_.locals)

    def _set_model(self):
        self.optimizer.set_parameters(self.nn.parameters())
        self.nn.to(self.device)

        if self.load_model is not None:
            self._load_state_dicts(from_outside=True, load_optimizer=self.load_optimizer)

    def _prepare_train_record(self):
        self.train_record_columns = ['Time', 'Epoch', 'Best loss', 'Test']
        if os.path.exists(f'{self.train_record_path}.csv'):
            self.train_record = pd.read_csv(f'{self.train_record_path}.csv', index_col=None, encoding='cp932')
        else:
            self.train_record = pd.DataFrame(
                np.empty([0, len(self.train_record_columns)]),
                columns=self.train_record_columns
            )

    def _prepare_train_valuables(self):
        self.epoch = 0
        self.best_loss = None
        self.best_updated = 0
        self.state_dicts = None
        self.dt_epoch = None
        self.t_init = time.perf_counter()

        self.evolution_col = ['epoch']

        if self.has_data:
            self.evolution_col += ['train', 'valid']
            self.has_valid = self.data_fitting.data_handler.n_data['valid'] > 0
            self.has_test = self.data_fitting.data_handler.n_data['test'] > 0
        else:
            self.has_valid = False
            self.has_test = False

        if self.has_reg:
            self.evolution_col += ['reg_total'] + self.regularization.reg_names

        if self.data_fitting is not None and self.regularization is not None:
            self.evolution_col = (
                    ['epoch', 'train', 'train_data', 'train_reg'] +
                    ['train_' + r for r in self.regularization.reg_names] +
                    ['valid', 'valid_data', 'valid_reg'] +
                    ['valid_' + r for r in self.regularization.reg_names]
            )

        self.evolution = []

    def _run_callbacks(self, method: str):
        for cb in self.callbacks:
            fn = getattr(cb, method, None)
            if fn is None:
                raise ValueError(f'No {method} exists in {cb.__name__}. Callbacks must inherit Callback class.')
            if callable(fn):
                fn(self)

    def __call__(self):
        self._run_callbacks('on_train_begin')
        while not self._is_training_finished():
            self._run_callbacks('on_epoch_begin')
            train_losses = self._get_loss('train')

            if self.data_fitting is not None:
                if self.has_valid:
                    valid_losses = self._get_loss('valid')
                else:
                    if self.recalculate_valid_loss:
                        valid_losses = self._get_loss('train_valid')
                    else:
                        valid_losses = train_losses
            else:
                valid_losses = train_losses

            self._update_best_loss(valid_losses)
            self._update_evolution(train_losses, valid_losses)
            self._finish_epoch()
            self._display_epoch_results()
            self._run_callbacks('on_epoch_end')

            self.epoch += 1

        self._post_train_treatments()
        self._run_callbacks('on_train_end')

    def _get_loss(self, phase: str):
        if phase == 'train':
            self.nn.train()
        else:
            self.nn.eval()

        if self.has_data:
            if self.has_reg:
                losses = []
                batch_sizes = []
                for x, y, _ in self.data_fitting.data_handler(phase):
                    losses.append(self._data_reg_step(x, y, phase=phase))
                    batch_sizes.append(len(x))
                return self._average_data_reg_losses(losses, batch_sizes)
            else:
                loss_sum = 0.0
                n_data = 0
                for x, y, _ in self.data_fitting.data_handler(phase):
                    loss_sum += self._data_step(x, y, phase=phase) * len(x)
                    n_data += len(x)
                return {'total': loss_sum / n_data}
        else:
            return self._reg_step()

    def _average_data_reg_losses(self, losses: list[dict], batch_sizes: list[int]) -> dict[str, object]:
        n_data = sum(batch_sizes)
        averaged = {
            'total': sum(loss['total'] * n for loss, n in zip(losses, batch_sizes)) / n_data,
            'data': sum(loss['data'] * n for loss, n in zip(losses, batch_sizes)) / n_data,
            'reg': float(np.mean([loss['reg'] for loss in losses])),
            'terms': {}
        }
        for name in self.regularization.reg_names:
            averaged['terms'][name] = float(np.mean([loss['terms'][name] for loss in losses]))
        return averaged

    def _data_reg_step(self, x: torch.Tensor, y: torch.Tensor, phase: str):
        if phase == 'train':
            self.optimizer.zero_grad()
            y_nn = self.nn(x=x)
            loss = self.data_fitting.loss_criteria(y, y_nn)
            data_loss = loss.item()

            reg_mean, reg_loss = self.regularization.get_regularization_value(
                nn=self.nn,
                data_handler=self.data_fitting.data_handler
            )
            if self.regularization.use_reg_prod:
                loss.mul_(reg_loss)
            else:
                loss.add_(reg_loss)
            reg_loss, reg_mean = reg_loss.item(), reg_mean.detach().cpu().numpy()

            loss.backward()
            self.optimizer.step(self.epoch)
            return {
                'total': loss.item(),
                'data': data_loss,
                'reg': reg_loss,
                'terms': self._regularization_terms_to_dict(reg_mean)
            }

        else:
            with torch.no_grad():
                y_nn = self.nn(x=x)
                data_loss = self.data_fitting.loss_criteria(y, y_nn).item()

            reg_mean, reg_loss = self.regularization.get_regularization_value(
                nn=self.nn,
                data_handler=self.data_fitting.data_handler
            )
            reg_loss, reg_mean = reg_loss.item(), reg_mean.detach().cpu().numpy()
            if self.regularization.use_reg_prod:
                loss = data_loss * reg_loss
            else:
                loss = data_loss + reg_loss
            return {
                'total': loss,
                'data': data_loss,
                'reg': reg_loss,
                'terms': self._regularization_terms_to_dict(reg_mean)
            }

    def _data_step(self, x: torch.Tensor, y: torch.Tensor, phase: str):
        if phase == 'train':
            self.optimizer.zero_grad()
            y_nn = self.nn(x=x)
            loss = self.data_fitting.loss_criteria(y, y_nn)
            loss.backward()
            self.optimizer.step(self.epoch)
            data_loss = loss.item()
            return data_loss

        else:
            with torch.no_grad():
                y_nn = self.nn(x=x)
                data_loss = self.data_fitting.loss_criteria(y, y_nn).item()
            return data_loss

    def _reg_step(self):
        self.optimizer.zero_grad()
        reg_mean, loss = self.regularization.get_regularization_value(nn=self.nn)
        loss.backward()
        self.optimizer.step(self.epoch)
        reg_loss, reg_mean = loss.item(), reg_mean.detach().cpu().numpy()
        return {
            'total': reg_loss,
            'terms': self._regularization_terms_to_dict(reg_mean)
        }

    def _regularization_terms_to_dict(self, reg_mean: np.ndarray) -> dict[str, float]:
        return {
            name: value.item() if hasattr(value, 'item') else float(value)
            for name, value in zip(self.regularization.reg_names, reg_mean)
        }

    def _update_best_loss(self, valid: dict[str, object]) -> None:
        current_loss = valid['total']
        if self.best_loss is None or self.best_loss > current_loss:
            self.best_loss = current_loss
            self.best_updated = 0
            self.state_dicts = self._get_state_dicts()
        else:
            self.best_updated += 1

    def _update_evolution(self, train: dict[str, object], valid: dict[str, object]) -> None:
        record = {'epoch': self.epoch}
        if self.has_data:
            if self.has_reg:
                record.update({
                    'train': train['total'],
                    'train_data': train['data'],
                    'train_reg': train['reg'],
                    'valid': valid['total'],
                    'valid_data': valid['data'],
                    'valid_reg': valid['reg']
                })
                for name in self.regularization.reg_names:
                    record[f'train_{name}'] = train['terms'][name]
                    record[f'valid_{name}'] = valid['terms'][name]
            else:
                record.update({'train': train['total'], 'valid': valid['total']})
        else:
            record['reg_total'] = train['total']
            for name in self.regularization.reg_names:
                record[name] = train['terms'][name]
        self.evolution.append(record)

    def _display_epoch_results(self):
        if self.epoch == 0:
            dt = self.dt_epoch
        else:
            dt = self.dt_epoch / self.epoch
        if dt < 1e-3:
            dt_str = f'({dt * 1e6:.1f} us)'
        elif dt < 1:
            dt_str = f'({dt * 1000:.1f} ms)'
        else:
            dt_str = f'({dt:.1f} s)'

        losses = self.evolution[-1]

        if self.has_data:
            if self.has_reg:
                valid_reg_str = ', '.join(
                    [f'{name}: {losses[f"valid_{name}"]:.3e}' for name in self.regularization.reg_names]
                )
                print(f'\r[{utils.get_current_time()}] '
                      f'{self.epoch + 1: >5} {dt_str} | '
                      f'Train {losses["train"]:.3e} ({losses["train_data"]:.3e} & {losses["train_reg"]:.3e}), '
                      f'Valid {losses["valid"]:.3e} ({losses["valid_data"]:.3e} & {losses["valid_reg"]:.3e}) | '
                      f'{valid_reg_str} | '
                      f'Best {self.best_loss:.3e} (no change for {self.best_updated: >4}) | '
                      f'lr {self.optimizer.current_lr():.2e}',
                      end='')
            else:
                print(f'\r[{utils.get_current_time()}] '
                      f'{self.epoch + 1: >5} {dt_str} | '
                      f'Train {losses["train"]:.3e}, Valid {losses["valid"]:.3e} | '
                      f'Best {self.best_loss:.3e} (no change for {self.best_updated: >4}) | '
                      f'lr {self.optimizer.current_lr():.2e}',
                      end='')
        else:
            reg_str = ', '.join(
                [f'{name}: {losses[name]:.3e}' for name in self.regularization.reg_names]
            )
            print(f'\r[{utils.get_current_time()}] '
                  f'{self.epoch + 1: >5} {dt_str} | '
                  f'Reg {losses["reg_total"]:.3e} | {reg_str} | '
                  f'Best {self.best_loss:.3e} (no change for {self.best_updated: >4}) | '
                  f'lr {self.optimizer.current_lr():.2e}',
                  end='')

    def _finish_epoch(self):
        self.dt_epoch = time.perf_counter() - self.t_init
        if self.epoch == 0:
            self.t_init = time.perf_counter()
        if self.callback_epoch is not None and self.epoch > 0 and self.epoch % self.callback_epoch == 0:
            self._run_callbacks('on_call')

    def _is_training_finished(self):
        epoch_limit_reached = (
                self.train_epochs is not None and
                self.train_epochs > 0 and
                self.train_epochs == self.epoch
        )
        return (0 < self.early_stop == self.best_updated) or epoch_limit_reached

    def _post_train_treatments(self):
        print('')
        self._load_state_dicts()
        self._save_model()
        self._save_train_record()

    def _load_state_dicts(self, from_outside: bool = False, load_optimizer: bool = False):
        if from_outside:
            state_dicts = torch.load(f'{self.load_model}\\state_dicts.pth', weights_only=False)
        else:
            state_dicts = copy.deepcopy(self.state_dicts)
        self.nn.load_state_dict(state_dicts['nn_state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(state_dicts['optimizer_state_dict'])

    def _save_model(self):
        pd.DataFrame(self.evolution, columns=self.evolution_col).to_csv(f'{self.model_path}\\evolution.csv')
        torch.save(self.state_dicts, f'{self.model_path}\\state_dicts.pth')

    def _save_train_record(self):
        if self.has_data and self.data_fitting.check_test:
            if self.has_reg:
                test_loss = self._get_loss('test')['data']
            else:
                test_loss = self._get_loss('test')['total']
            utils.logging(f'Test {test_loss:.3e}')
        else:
            test_loss = np.nan

        train_record = [utils.get_current_time(), self.epoch - self.best_updated, self.best_loss, test_loss]
        df = pd.DataFrame([train_record], columns=self.train_record_columns)
        df.to_csv(f'{self.model_path}\\train_record.csv', index=False)
        self.train_record = pd.concat([self.train_record, df], axis=0)
        self.train_record.to_csv(f'{self.train_record_path}.csv', index=False)

    def _get_state_dicts(self):
        return {
            'nn_state_dict': copy.deepcopy(self.nn.state_dict()),
            'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
        }

    def get_loss_info_fnc(self, need_data: bool = True, need_reg: bool = True):
        def get_xy_from_keys(evolution: list[dict], keys: list[str], labels: list[str]):
            if len(evolution) == 0:
                return np.empty([0]), [np.empty([0]) for _ in keys], labels
            df = pd.DataFrame(evolution)
            x = df['epoch'].to_numpy()
            ys = [df[key].to_numpy() for key in keys]
            return x, ys, labels

        if self.has_data:
            if self.has_reg:
                if need_data and need_reg:
                    n_data = 4 + self.regularization.n_reg
                    keys = (
                            ['train_data', 'valid', 'valid_data', 'valid_reg'] +
                            ['valid_' + r for r in self.regularization.reg_names]
                    )
                    labels = (['Train (data)', 'Valid (total)', 'Valid (data)', 'Valid (reg)'] +
                              self.regularization.reg_names)

                    def get_xy(evolution: list[dict], _: int):
                        return get_xy_from_keys(evolution, keys, labels)

                elif need_data:
                    n_data = 2
                    keys = ['train_data', 'valid_data']
                    labels = ['Train (data)', 'Valid (data)']

                    def get_xy(evolution: list[dict], _: int):
                        return get_xy_from_keys(evolution, keys, labels)

                elif need_reg:
                    n_data = self.regularization.n_reg
                    keys = ['valid_' + r for r in self.regularization.reg_names]
                    labels = self.regularization.reg_names

                    def get_xy(evolution: list[dict], _: int):
                        return get_xy_from_keys(evolution, keys, labels)

                else:
                    raise ValueError('At least one of need_loss or need_reg must be True.')
            else:
                n_data = 2
                keys = ['train', 'valid']
                labels = ['Train', 'Valid']

                def get_xy(evolution: list[dict], _: int):
                    return get_xy_from_keys(evolution, keys, labels)
        else:
            n_data = 1 + self.regularization.n_reg
            keys = ['reg_total'] + self.regularization.reg_names
            labels = ['Reg (all)'] + self.regularization.reg_names

            def get_xy(evolution: list[dict], _: int):
                return get_xy_from_keys(evolution, keys, labels)
        return n_data, get_xy
