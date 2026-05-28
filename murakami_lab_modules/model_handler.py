import time
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from .neural_network import AbstractNeuralNetwork
from .optimizer import OptimizerBase
from .data_fitting import DataFitting as _DataFitting
from .regularization import Regularization as _Regularization
from .experiment import RunManager
from . import utils

__all__ = ['ModelHandler']


class ModelHandler:
    def __init__(
            self,
            nn: AbstractNeuralNetwork,
            optimizer: OptimizerBase,
            data_fitting: _DataFitting = None,
            regularization: _Regularization = None,
            train_epochs: int = None,
            load_model: str = None,
            load_optimizer: bool = False,
            save_path: str = 'Model',
            summary_path: str = 'run_summary',
            recalculate_valid_loss: bool = True,
            model_name: str = None,
            callbacks: tuple[object, ...] = None,
            random_seed: int = 2025,
            save_result: bool = True,
            save_model: bool = True,
            restore_best: bool = None,
            save_history: bool = True,
            history_policy: str = 'full',
            history_every: int = 1,
            keep_best_history: bool = True,
            keep_last_history: bool = True,
            verbose: bool = True,
            **kwargs
    ):
        self.locals = utils.get_local_dict(locals())
        utils.initialize_random_seed(random_seed)

        self.nn = nn
        self.optimizer = optimizer
        self.data_fitting = data_fitting
        self.regularization = regularization

        self.train_epochs = train_epochs

        self.load_model = load_model
        self.load_optimizer = load_optimizer
        self.random_seed = random_seed

        self.save_path = Path(save_path)
        self.summary_path = Path(summary_path)
        self.recalculate_valid_loss = recalculate_valid_loss
        self.run_manager = RunManager(save_path=save_path, model_name=model_name)
        self.model_name = self.run_manager.model_name
        self.original_model_name = self.run_manager.original_model_name
        self.kwargs = kwargs
        self.callbacks = callbacks or []
        self.save_result = save_result
        self.save_model = save_model
        self.restore_best = (save_result and save_model) if restore_best is None else bool(restore_best)
        if not self.save_result or not self.save_model:
            self.restore_best = False
        self.save_history = save_history
        self.history_policy = history_policy
        self.history_every = history_every
        self.keep_best_history = keep_best_history
        self.keep_last_history = keep_last_history
        self.verbose = verbose

        self._validate_inputs()
        if self.save_result:
            self.run_manager.prepare_model_folder()
            self.model_name = self.run_manager.model_name
            self.model_path = self.run_manager.model_path
            self.run_manager.save_metadata(self)
        else:
            self.model_path = None
        self._prepare_callbacks()
        self._set_model()
        self._prepare_train_valuables()

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'train_epochs': self.train_epochs,
            'load_model': self.load_model,
            'load_optimizer': self.load_optimizer,
            'save_path': self.save_path,
            'summary_path': self.summary_path,
            'recalculate_valid_loss': self.recalculate_valid_loss,
            'model_name': self.model_name,
            'callbacks': self.callbacks,
            'random_seed': self.random_seed,
            'save_result': self.save_result,
            'save_model': self.save_model,
            'restore_best': self.restore_best,
            'save_history': self.save_history,
            'history_policy': self.history_policy,
            'history_every': self.history_every,
            'keep_best_history': self.keep_best_history,
            'keep_last_history': self.keep_last_history,
            'verbose': self.verbose,
            **self.kwargs
        })

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

        if self.train_epochs is not None and self.train_epochs < 1:
            raise ValueError('train_epochs must be a positive int or None.')
        if self.history_policy not in {'full', 'sparse', 'last', 'none'}:
            raise ValueError("history_policy must be one of 'full', 'sparse', 'last', or 'none'.")
        if type(self.history_every) is not int or self.history_every <= 0:
            raise ValueError('history_every must be a positive int.')

    def _prepare_callbacks(self):
        from .callbacks import (
            BestModelTracker,
            ConsoleLogger,
            FinalStateDictSaver,
            HistoryLogger,
            HistoryRecorder,
            RegularizationReportSaver,
            RunSummaryLogger,
        )

        user_callbacks = list(self.callbacks)
        core_callbacks = [
            BestModelTracker(restore_best=self.restore_best),
            HistoryRecorder(
                policy=self.history_policy,
                every=self.history_every,
                keep_best=self.keep_best_history,
                keep_last=self.keep_last_history,
            ),
        ]
        if self.verbose:
            core_callbacks.append(ConsoleLogger())
        if self.save_result:
            if self.save_history and self.history_policy != 'none':
                core_callbacks.append(HistoryLogger(
                    path=self.model_path / 'evolution.csv',
                    every=None,
                    row_every=self.history_every if self.history_policy == 'sparse' else 1,
                    keep_best=self.keep_best_history,
                    keep_last=self.keep_last_history,
                ))
            if self.save_model:
                core_callbacks.append(FinalStateDictSaver())
            if self.has_reg:
                core_callbacks.append(RegularizationReportSaver())
            core_callbacks.append(RunSummaryLogger(
                summary_path=self.summary_path,
                show_test=self.verbose,
            ))
        self.callbacks = core_callbacks + user_callbacks
        self.callbacks.sort(key=lambda callback: callback.priority)

    def _set_model(self):
        self.optimizer.set_parameters(self.nn.parameters())
        self.nn.to(self.device)

        if self.load_model is not None:
            self._load_state_dicts(from_outside=True, load_optimizer=self.load_optimizer)

    def _prepare_train_valuables(self):
        self.epoch = 0
        self.best_loss = None
        self.best_epoch = None
        self.epochs_since_best = 0
        self.state_dicts = None
        self.stop_training = False
        self.stop_reason = None
        self.dt_epoch = None
        self.t_init = time.perf_counter()
        self.run_summary = None

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
        self.current_evolution = None
        self.current_train_losses = None
        self.current_valid_losses = None

    def _should_run_callback(self, cb, method: str, interval: bool) -> bool:
        if not interval:
            return True
        return cb.should_call(self)

    def _run_callbacks(self, method: str, interval: bool = False):
        for cb in self.callbacks:
            fn = getattr(cb, method, None)
            if fn is None:
                raise ValueError(
                    f'No {method} exists in {cb.__class__.__name__}. '
                    f'Callbacks must inherit Callback class.'
                )
            if not self._should_run_callback(cb, method=method, interval=interval):
                continue
            if callable(fn):
                fn(self)

    def __call__(self):
        self._run_callbacks('on_train_begin')
        while not self._is_training_finished():
            self._run_callbacks('on_epoch_begin')
            train_losses = self._get_loss('train')
            self.current_train_losses = train_losses
            self._run_callbacks('on_train_step_end', interval=True)

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
            self.current_valid_losses = valid_losses
            self._run_callbacks('on_validation_end', interval=True)

            self._update_evolution(train_losses, valid_losses)
            self._finish_epoch()
            self._run_callbacks('on_epoch_end', interval=True)

            self.epoch += 1

        self._run_callbacks('on_train_end')

    def request_stop(self, reason: str = None) -> None:
        self.stop_training = True
        self.stop_reason = reason

    def _get_loss(self, phase: str):
        if phase == 'train':
            self.nn.train()
        else:
            self.nn.eval()

        if self.has_data:
            if self.has_reg:
                return self._collect_data_reg_losses(phase)
            else:
                return self._collect_data_losses(phase)
        else:
            return self._reg_step()

    @staticmethod
    def _detach_loss(value) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach()
        return torch.as_tensor(value, dtype=torch.float32)

    @staticmethod
    def _to_float(value) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    @classmethod
    def _add_loss(cls, current: torch.Tensor | None, value) -> torch.Tensor:
        value = cls._detach_loss(value)
        if current is None:
            return value.clone()
        return current + value

    @classmethod
    def _add_weighted_loss(cls, current: torch.Tensor | None, value, weight: int) -> torch.Tensor:
        return cls._add_loss(current, cls._detach_loss(value) * weight)

    def _collect_data_losses(self, phase: str) -> dict[str, object]:
        total_sum = None
        data_sum = None
        n_data = 0
        for x, y, label in self.data_fitting.data_handler(phase):
            loss = self._data_step(x, y, label, phase=phase)
            batch_size = len(x)
            total_sum = self._add_weighted_loss(total_sum, loss['total'], batch_size)
            data_sum = self._add_weighted_loss(data_sum, loss['data'], batch_size)
            n_data += batch_size
        if n_data == 0:
            raise ValueError(f'Dataset for phase={phase} is empty.')
        return {
            'total': self._to_float(total_sum / n_data),
            'data': self._to_float(data_sum / n_data)
        }

    def _collect_data_reg_losses(self, phase: str) -> dict[str, object]:
        total_sum = None
        data_sum = None
        reg_sum = None
        term_sum = None
        n_data = 0
        n_batch = 0
        for x, y, label in self.data_fitting.data_handler(phase):
            loss = self._data_reg_step(x, y, label, phase=phase)
            batch_size = len(x)
            total_sum = self._add_weighted_loss(total_sum, loss['total'], batch_size)
            data_sum = self._add_weighted_loss(data_sum, loss['data'], batch_size)
            reg_sum = self._add_loss(reg_sum, loss['reg'])
            term_sum = self._add_loss(term_sum, loss['terms'])
            n_data += batch_size
            n_batch += 1
        if n_data == 0 or n_batch == 0:
            raise ValueError(f'Dataset for phase={phase} is empty.')

        terms = term_sum / n_batch
        return {
            'total': self._to_float(total_sum / n_data),
            'data': self._to_float(data_sum / n_data),
            'reg': self._to_float(reg_sum / n_batch),
            'terms': self._regularization_terms_to_dict(terms)
        }

    def _data_reg_step(self, x: torch.Tensor, y: torch.Tensor, label, phase: str):
        if phase == 'train':
            self.optimizer.zero_grad()
            data_loss_info = self.data_fitting.compute_loss(
                nn=self.nn,
                x=x,
                y=y,
                label=label,
                phase=phase,
                epoch=self.epoch
            )

            reg_mean, reg_loss = self.regularization.get_regularization_value(
                nn=self.nn,
                data_handler=self.data_fitting.data_handler,
                epoch=self.epoch,
                data_loss=data_loss_info['total'].detach()
            )
            if self.regularization.use_reg_prod:
                loss = data_loss_info['total'] * reg_loss
            else:
                loss = data_loss_info['total'] + reg_loss

            loss.backward()
            self.optimizer.step(self.epoch)
            return {
                'total': loss.detach(),
                'data': data_loss_info['total'].detach(),
                'reg': reg_loss.detach(),
                'terms': reg_mean.detach()
            }

        else:
            with torch.no_grad():
                data_loss_info = self.data_fitting.compute_loss(
                    nn=self.nn,
                    x=x,
                    y=y,
                    label=label,
                    phase=phase,
                    epoch=self.epoch
                )
                data_loss = data_loss_info['total'].detach()

            reg_mean, reg_loss = self.regularization.get_regularization_value(
                nn=self.nn,
                data_handler=self.data_fitting.data_handler,
                epoch=self.epoch,
                data_loss=data_loss
            )
            if self.regularization.use_reg_prod:
                loss = data_loss * reg_loss
            else:
                loss = data_loss + reg_loss
            return {
                'total': loss.detach(),
                'data': data_loss,
                'reg': reg_loss.detach(),
                'terms': reg_mean.detach()
            }

    def _data_step(self, x: torch.Tensor, y: torch.Tensor, label, phase: str):
        if phase == 'train':
            self.optimizer.zero_grad()
            loss_info = self.data_fitting.compute_loss(
                nn=self.nn,
                x=x,
                y=y,
                label=label,
                phase=phase,
                epoch=self.epoch
            )
            loss = loss_info['total']
            loss.backward()
            self.optimizer.step(self.epoch)
            return {
                'total': loss.detach(),
                'data': loss_info['terms']['data'].detach()
            }

        else:
            with torch.no_grad():
                loss_info = self.data_fitting.compute_loss(
                    nn=self.nn,
                    x=x,
                    y=y,
                    label=label,
                    phase=phase,
                    epoch=self.epoch
                )
            return {
                'total': loss_info['total'].detach(),
                'data': loss_info['terms']['data'].detach()
            }

    def _reg_step(self):
        self.optimizer.zero_grad()
        reg_mean, loss = self.regularization.get_regularization_value(nn=self.nn, epoch=self.epoch)
        loss.backward()
        self.optimizer.step(self.epoch)
        return {
            'total': self._to_float(loss),
            'terms': self._regularization_terms_to_dict(reg_mean)
        }

    def _regularization_terms_to_dict(self, reg_mean) -> dict[str, float]:
        if torch.is_tensor(reg_mean):
            reg_mean = reg_mean.detach().cpu()
        return {
            name: self._to_float(value)
            for name, value in zip(self.regularization.reg_names, reg_mean)
        }

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
        self.current_evolution = record

    def _finish_epoch(self):
        self.dt_epoch = time.perf_counter() - self.t_init
        if self.epoch == 0:
            self.t_init = time.perf_counter()

    def _is_training_finished(self):
        epoch_limit_reached = (
                self.train_epochs is not None and
                self.train_epochs > 0 and
                self.train_epochs == self.epoch
        )
        return self.stop_training or epoch_limit_reached

    def _load_state_dicts(self, from_outside: bool = False, load_optimizer: bool = False):
        if from_outside:
            state_dicts = torch.load(Path(self.load_model) / 'state_dicts.pth', weights_only=True)
        else:
            state_dicts = self.state_dicts
        self.nn.load_state_dict(state_dicts['nn_state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(state_dicts['optimizer_state_dict'])

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
