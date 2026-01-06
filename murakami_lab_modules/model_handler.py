import time
import glob
import os
import pandas as pd
import numpy as np
import torch
import copy
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
            loss_criteria: callable = torch.nn.MSELoss(),
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
    def __init__(
            self,
            input_generators: (InputGenerator,),
            reg_weights: list,
            reg_func_name: str = 'regularization',
            reg_names: list = None,
            reg_criteria: callable = torch.square,
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

    def get_regularization_value(self, nn: AbstractNeuralNetwork, data_handler: DataHandler = None):
        full_reg = self.reg_criteria(torch.stack(self.reg_func(data_handler=data_handler, nn=nn)))

        is_finite = torch.isfinite(full_reg)
        if not is_finite.all():
            if not is_finite.any(dim=1).all():
                torch.save(full_reg, 'invalid_regularization.pth')
                raise ValueError(f'Too many invalid value was encountered during regularization.')
            utils.logging(f'Invalid value was encountered during regularization.')
            full_reg = torch.where(is_finite, full_reg, 0.0)
            count = is_finite.sum(dim=1)
            reg_mean = full_reg.sum(dim=1) / count
        else:
            reg_mean = full_reg.mean(dim=1)

        if self.use_reg_prod:
            reg_mean.add_(self.reg_min).pow_(self.reg_weights)
            if self.n_reg > 1:
                reg_value = torch.pow(reg_mean, self.reg_mean_pow).prod()
            else:
                reg_value = reg_mean.prod()
        else:
            reg_mean.mul_(self.reg_weights).add_(self.reg_min)
            reg_value = reg_mean.mean()

        return reg_mean, reg_value

    @staticmethod
    def grad(y: torch.Tensor, x: torch.Tensor, x_idx: int = None, y_idx: int = None):
        if y_idx is None:
            grad_outputs = torch.ones_like(y)
        else:
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, y_idx] = 1.0
        dy_dx = torch.autograd.grad(
            inputs=x,
            outputs=y,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True
        )[0]
        if x_idx is None:
            return dy_dx
        else:
            return dy_dx[:, x_idx]


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
            callbacks: tuple = None,
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

        self._prepare_model_folder()
        self._validate_inputs()
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

        evolution_col_count = 1
        self.evolution_col = ['epoch']

        if self.has_data:
            evolution_col_count += 2
            self.evolution_col += ['train', 'valid']
            self.has_valid = self.data_fitting.data_handler.n_data['valid'] > 0
            self.has_test = self.data_fitting.data_handler.n_data['test'] > 0
        else:
            self.has_valid = False
            self.has_test = False

        if self.has_reg:
            evolution_col_count += self.regularization.n_reg + 1
            self.evolution_col += ['reg_total'] + self.regularization.reg_names

        if self.data_fitting is not None and self.regularization is not None:
            evolution_col_count += self.regularization.n_reg + 3
            self.evolution_col = (
                    ['epoch', 'train', 'train_data', 'train_reg'] +
                    ['train_' + r for r in self.regularization.reg_names] +
                    ['valid', 'valid_data', 'valid_reg'] +
                    ['valid_' + r for r in self.regularization.reg_names]
            )

        self.evolution = np.empty([self.train_epochs, evolution_col_count])

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

        self._run_callbacks('on_train_end')
        self._post_train_treatments()

    def _get_loss(self, phase: str):
        if phase == 'train':
            self.nn.train()
        else:
            self.nn.eval()

        if self.has_data:
            if self.has_reg:
                losses = np.empty([self.data_fitting.data_handler.n_batch[phase], 3 + self.regularization.n_reg])
                for i, (x, y, _) in enumerate(self.data_fitting.data_handler(phase)):
                    losses[i] = self._data_reg_step(x, y, phase=phase)
            else:
                losses = np.empty([self.data_fitting.data_handler.n_batch[phase], 1])
                for i, (x, y, _) in enumerate(self.data_fitting.data_handler(phase)):
                    losses[i] = self._data_step(x, y, phase=phase)
            return losses.mean(axis=0)
        else:
            return self._reg_step()

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
            return np.hstack([[loss.item(), data_loss, reg_loss], reg_mean])

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
            return np.hstack([[loss, data_loss, reg_loss], reg_mean])

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
        return np.hstack([[reg_loss], reg_mean])

    def _update_best_loss(self, valid: np.ndarray):
        current_loss = valid[0].item()
        if self.best_loss is None or self.best_loss > current_loss:
            self.best_loss = current_loss
            self.best_updated = 0
            self.state_dicts = self._get_state_dicts()
        else:
            self.best_updated += 1

    def _update_evolution(self, train: np.ndarray, valid: np.ndarray):
        self.evolution[self.epoch, 0] = self.epoch
        if self.has_data:
            if self.has_reg:
                self.evolution[self.epoch, 1:self.regularization.n_reg + 4] = train
                self.evolution[self.epoch, self.regularization.n_reg + 4:] = valid
            else:
                self.evolution[self.epoch, 1] = train.item()
                self.evolution[self.epoch, 2] = valid.item()
        else:
            self.evolution[self.epoch, 1:] = train

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

        losses = self.evolution[self.epoch]

        if self.has_data:
            if self.has_reg:
                total_loss, train_data_loss, train_reg_loss = losses[1:4]
                valid_loss, valid_data_loss, valid_reg_loss \
                    = losses[self.regularization.n_reg + 4:self.regularization.n_reg + 7]
                valid_regs = losses[self.regularization.n_reg + 7:]
                valid_reg_str = ', '.join(
                    [f'{name}: {value:.3e}' for name, value in zip(self.regularization.reg_names, valid_regs)]
                )
                print(f'\r[{utils.get_current_time()}] '
                      f'{self.epoch + 1: >5} {dt_str} | '
                      f'Train {total_loss:.3e} ({train_data_loss:.3e} & {train_reg_loss:.3e}), '
                      f'Valid {valid_loss:.3e} ({valid_data_loss:.3e} & {valid_reg_loss:.3e}) | '
                      f'{valid_reg_str} | '
                      f'Best {self.best_loss:.3e} (no change for {self.best_updated: >4}) | '
                      f'lr {self.optimizer.current_lr():.2e}',
                      end='')
            else:
                total_loss, valid_loss = losses[1:3]
                print(f'\r[{utils.get_current_time()}] '
                      f'{self.epoch + 1: >5} {dt_str} | '
                      f'Train {total_loss:.3e}, Valid {valid_loss:.3e} | '
                      f'Best {self.best_loss:.3e} (no change for {self.best_updated: >4}) | '
                      f'lr {self.optimizer.current_lr():.2e}',
                      end='')
        else:
            reg_loss = losses[1]
            regs = losses[1:]
            reg_str = ', '.join(
                [f'{name}: {value:.3e}' for name, value in zip(self.regularization.reg_names, regs)]
            )
            print(f'\r[{utils.get_current_time()}] '
                  f'{self.epoch + 1: >5} {dt_str} | '
                  f'Reg {reg_loss:.3e} | {reg_str} | '
                  f'Best {self.best_loss:.3e} (no change for {self.best_updated: >4}) | '
                  f'lr {self.optimizer.current_lr():.2e}',
                  end='')

    def _finish_epoch(self):
        self.dt_epoch = time.perf_counter() - self.t_init
        if self.epoch == 0:
            self.t_init = time.perf_counter()
        if self.callback_epoch is not None and self.train_epochs > self.epoch > 0 == self.epoch % self.callback_epoch:
            self._run_callbacks('on_call')

    def _is_training_finished(self):
        return (0 < self.early_stop == self.best_updated) or self.train_epochs == self.epoch

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
        (pd.DataFrame(self.evolution[:min(self.epoch + 1, self.train_epochs + 1)], columns=self.evolution_col).
         to_csv(f'{self.model_path}\\evolution.csv'))
        torch.save(self.state_dicts, f'{self.model_path}\\state_dicts.pth')

    def _save_train_record(self):
        if self.has_data and self.data_fitting.check_test:
            if self.has_reg:
                test_loss = self._get_loss('test')[1].item()
            else:
                test_loss = self._get_loss('test')[0].item()
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
        if self.has_data:
            if self.has_reg:
                if need_data and need_reg:
                    n_data = 4 + self.regularization.n_reg

                    def get_xy(evolution: np.ndarray, epoch: int):
                        x = evolution[:epoch, 0]
                        ys = [
                                 evolution[:epoch, 2],
                                 evolution[:epoch, self.regularization.n_reg + 4],
                                 evolution[:epoch, self.regularization.n_reg + 5],
                                 evolution[:epoch, self.regularization.n_reg + 6]
                             ] + [
                                 evolution[:epoch, self.regularization.n_reg + 7 + i] for i in
                                 range(self.regularization.n_reg)
                             ]
                        labels = (['Train (data)', 'Valid (total)', 'Valid (data)', 'Valid (reg)'] +
                                  self.regularization.reg_names)
                        return x, ys, labels

                elif need_data:
                    n_data = 2

                    def get_xy(evolution: np.ndarray, epoch: int):
                        x = evolution[:epoch, 0]
                        ys = [
                            evolution[:epoch, 2],
                            evolution[:epoch, self.regularization.n_reg + 5]
                        ]
                        labels = ['Train (data)', 'Valid (data)']
                        return x, ys, labels

                elif need_reg:
                    n_data = self.regularization.n_reg

                    def get_xy(evolution: np.ndarray, epoch: int):
                        x = evolution[:epoch, 0]
                        ys = [
                            evolution[:epoch, self.regularization.n_reg + 7 + i] for i in
                            range(self.regularization.n_reg)
                        ]
                        labels = self.regularization.reg_names
                        return x, ys, labels

                else:
                    raise ValueError('At least one of need_loss or need_reg must be True.')
            else:
                n_data = 2

                def get_xy(evolution: np.ndarray, epoch: int):
                    x = evolution[:epoch, 0]
                    ys = [evolution[:epoch, 1], evolution[:epoch, 2]]
                    labels = ['Train', 'Valid']
                    return x, ys, labels
        else:
            n_data = 1 + self.regularization.n_reg

            def get_xy(evolution: np.ndarray, epoch: int):
                x = evolution[:epoch, 0]
                ys = [
                         evolution[:epoch, 1],
                     ] + [
                         evolution[:epoch, 2 + i] for i in range(self.regularization.n_reg)
                     ]
                labels = ['Reg (all)'] + self.regularization.reg_names
                return x, ys, labels
        return n_data, get_xy
