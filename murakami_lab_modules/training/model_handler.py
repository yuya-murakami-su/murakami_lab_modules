"""Main training loop for data fitting and PINN regularization."""

import time
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from ..models.neural_network import BaseNeuralNetwork
from .optimizer import Optimizer
from .data_fitting import DataFitting as _DataFitting
from ..pinn.regularization import Regularization as _Regularization
from ..experiment import RunManager
from .. import utils

__all__ = ['ModelHandler']

logger = utils.get_logger(__name__)


class ModelHandler:
    """Coordinate model training, callbacks, saving, and summaries.

    ``ModelHandler`` is intentionally a small orchestration layer. Data loss is
    delegated to ``DataFitting``, PINN/physics penalties are delegated to
    ``Regularization``, and side effects such as history logging, checkpoints,
    and early stopping are delegated to callbacks.

    Parameters
    ----------
    nn:
        PyTorch module to train.
    optimizer:
        ``murakami_lab_modules.training.Optimizer`` instance.
    data_fitting:
        Data-loss adapter. Required unless training is regularization-only.
    regularization:
        Optional PINN-style regularization object.
    save_result, save_model, save_history:
        Control how much training output is written to disk.
    history_policy:
        ``"full"``, ``"sparse"``, ``"last"``, or ``"none"``.
    verbose:
        Enables console progress through the default ``ConsoleLogger`` callback.
    """

    def __init__(
            self,
            nn: BaseNeuralNetwork,
            optimizer: Optimizer,
            data_fitting: _DataFitting = None,
            regularization: _Regularization = None,
            train_epochs: int = None,
            load_model: str = None,
            load_optimizer: bool = False,
            save_path: str = 'Model',
            summary_path: str = 'run_summary',
            recompute_validation_loss: bool = True,
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
            evaluate_test: bool = False,
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
        self.recompute_validation_loss = recompute_validation_loss
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
        self.evaluate_test = evaluate_test
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
        self._prepare_training_state()

    def config_dict(self) -> dict[str, object]:
        """Return serializable metadata for the training configuration."""

        return utils.make_object_config(self, {
            'train_epochs': self.train_epochs,
            'load_model': self.load_model,
            'load_optimizer': self.load_optimizer,
            'save_path': self.save_path,
            'summary_path': self.summary_path,
            'recompute_validation_loss': self.recompute_validation_loss,
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
            'evaluate_test': self.evaluate_test,
            'verbose': self.verbose,
            **self.kwargs
        })

    def _validate_inputs(self):
        if self.data_fitting is None and self.regularization is None:
            raise ValueError('At least one of data_fitting or regularization must be given.')

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
            FinalCheckpointSaver,
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
                core_callbacks.append(FinalCheckpointSaver())
            if self.has_reg:
                core_callbacks.append(RegularizationReportSaver())
            core_callbacks.append(RunSummaryLogger(
                summary_path=self.summary_path,
                evaluate_test=self.evaluate_test,
                show_test=self.verbose,
            ))
        self.callbacks = core_callbacks + user_callbacks
        self.callbacks.sort(key=lambda callback: callback.priority)

    def _set_model(self):
        self.optimizer.set_parameters(self.nn.parameters())
        self.nn.to(self.device)

        if self.load_model is not None:
            self._load_state_dicts(from_outside=True, load_optimizer=self.load_optimizer)

    def _prepare_training_state(self):
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
            self.evolution_col += ['train_loss', 'validation_loss']
            self.has_valid = self.data_fitting.data_handler.n_data['valid'] > 0
            self.has_test = self.data_fitting.data_handler.n_data['test'] > 0
        else:
            self.has_valid = False
            self.has_test = False

        if self.has_reg:
            self.evolution_col += ['regularization_loss'] + self.regularization.term_names

        if self.data_fitting is not None and self.regularization is not None:
            self.evolution_col = (
                    ['epoch', 'train_loss', 'train_data_loss', 'train_regularization_loss'] +
                    ['train_' + r for r in self.regularization.term_names] +
                    ['validation_loss', 'validation_data_loss', 'validation_regularization_loss'] +
                    ['validation_' + r for r in self.regularization.term_names]
            )

        self.evolution = []
        self.current_evolution = None
        self.current_train_losses = None
        self.current_validation_losses = None
        self.test_loss = np.nan
        self._test_loss_evaluated = False
        self._test_loss_shown = False

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
        """Run the training loop until ``train_epochs`` or a callback stops it."""

        self._run_callbacks('on_train_begin')
        while not self._is_training_finished():
            self._run_callbacks('on_epoch_begin')
            train_losses = self._get_loss('train')
            self.current_train_losses = train_losses
            self._run_callbacks('on_train_step_end', interval=True)

            if self.data_fitting is not None:
                if self.has_valid:
                    validation_losses = self._get_loss('valid')
                else:
                    if self.recompute_validation_loss:
                        validation_losses = self._get_loss('train_valid')
                    else:
                        validation_losses = train_losses
            else:
                validation_losses = train_losses
            self.current_validation_losses = validation_losses
            self._run_callbacks('on_validation_end', interval=True)

            self._update_evolution(train_losses, validation_losses)
            self._finish_epoch()
            self._run_callbacks('on_epoch_end', interval=True)

            self.epoch += 1

        self._run_callbacks('on_train_end')

    def request_stop(self, reason: str = None) -> None:
        """Ask the training loop to stop after the current callback phase."""

        self.stop_training = True
        self.stop_reason = reason

    def evaluate_test_loss(self, enabled: bool = None) -> float:
        """Return the cached test loss, evaluating it once when requested."""

        enabled = self.evaluate_test if enabled is None else bool(enabled)
        if not enabled:
            return np.nan
        if self._test_loss_evaluated:
            return self.test_loss
        self._test_loss_evaluated = True
        if not self.has_data:
            self.test_loss = np.nan
            return self.test_loss
        if self.data_fitting.data_handler.n_data['test'] == 0:
            logger.warning('evaluate_test=True, but no test data is available.')
            self.test_loss = np.nan
            return self.test_loss
        if self.has_reg:
            self.test_loss = self._get_loss('test')['data']
        else:
            self.test_loss = self._get_loss('test')['total']
        return self.test_loss

    def _get_loss(self, phase: str):
        if phase == 'train':
            self.nn.train()
        else:
            self.nn.eval()

        if self.has_data:
            if self.has_reg:
                return self._collect_data_regularization_losses(phase)
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
        return utils.to_float(value, reduce_non_scalar=False)

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

    def _collect_data_regularization_losses(self, phase: str) -> dict[str, object]:
        total_sum = None
        data_sum = None
        regularization_sum = None
        term_sum = None
        n_data = 0
        n_batch = 0
        for x, y, label in self.data_fitting.data_handler(phase):
            loss = self._data_reg_step(x, y, label, phase=phase)
            batch_size = len(x)
            total_sum = self._add_weighted_loss(total_sum, loss['total'], batch_size)
            data_sum = self._add_weighted_loss(data_sum, loss['data'], batch_size)
            regularization_sum = self._add_loss(regularization_sum, loss['regularization'])
            term_sum = self._add_loss(term_sum, loss['terms'])
            n_data += batch_size
            n_batch += 1
        if n_data == 0 or n_batch == 0:
            raise ValueError(f'Dataset for phase={phase} is empty.')

        terms = term_sum / n_batch
        return {
            'total': self._to_float(total_sum / n_data),
            'data': self._to_float(data_sum / n_data),
            'regularization': self._to_float(regularization_sum / n_batch),
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

            term_means, regularization_loss = self.regularization.get_regularization_value(
                nn=self.nn,
                data_handler=self.data_fitting.data_handler,
                epoch=self.epoch,
                data_loss=data_loss_info['total'].detach()
            )
            if self.regularization.combine_by_product:
                loss = data_loss_info['total'] * regularization_loss
            else:
                loss = data_loss_info['total'] + regularization_loss

            loss.backward()
            self.optimizer.step(self.epoch)
            return {
                'total': loss.detach(),
                'data': data_loss_info['total'].detach(),
                'regularization': regularization_loss.detach(),
                'terms': term_means.detach()
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

            term_means, regularization_loss = self.regularization.get_regularization_value(
                nn=self.nn,
                data_handler=self.data_fitting.data_handler,
                epoch=self.epoch,
                data_loss=data_loss
            )
            if self.regularization.combine_by_product:
                loss = data_loss * regularization_loss
            else:
                loss = data_loss + regularization_loss
            return {
                'total': loss.detach(),
                'data': data_loss,
                'regularization': regularization_loss.detach(),
                'terms': term_means.detach()
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
        term_means, loss = self.regularization.get_regularization_value(nn=self.nn, epoch=self.epoch)
        loss.backward()
        self.optimizer.step(self.epoch)
        return {
            'total': self._to_float(loss),
            'terms': self._regularization_terms_to_dict(term_means)
        }

    def _regularization_terms_to_dict(self, term_means) -> dict[str, float]:
        if torch.is_tensor(term_means):
            term_means = term_means.detach().cpu()
        return {
            name: self._to_float(value)
            for name, value in zip(self.regularization.term_names, term_means)
        }

    def _update_evolution(self, train: dict[str, object], valid: dict[str, object]) -> None:
        record = {'epoch': self.epoch}
        if self.has_data:
            if self.has_reg:
                record.update({
                    'train_loss': train['total'],
                    'train_data_loss': train['data'],
                    'train_regularization_loss': train['regularization'],
                    'validation_loss': valid['total'],
                    'validation_data_loss': valid['data'],
                    'validation_regularization_loss': valid['regularization']
                })
                for name in self.regularization.term_names:
                    record[f'train_{name}'] = train['terms'][name]
                    record[f'validation_{name}'] = valid['terms'][name]
            else:
                record.update({'train_loss': train['total'], 'validation_loss': valid['total']})
        else:
            record['regularization_loss'] = train['total']
            for name in self.regularization.term_names:
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

    def get_loss_series(self, need_data: bool = True, need_reg: bool = True):
        """Return a plotting adapter for the currently recorded loss history."""

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
                    n_data = 4 + self.regularization.n_terms
                    keys = (
                            ['train_data_loss', 'validation_loss', 'validation_data_loss',
                             'validation_regularization_loss'] +
                            ['validation_' + r for r in self.regularization.term_names]
                    )
                    labels = (['Train (data)', 'Validation (total)', 'Validation (data)', 'Validation (regularization)'] +
                              self.regularization.term_names)

                    def get_xy(evolution: list[dict], _: int):
                        return get_xy_from_keys(evolution, keys, labels)

                elif need_data:
                    n_data = 2
                    keys = ['train_data_loss', 'validation_data_loss']
                    labels = ['Train (data)', 'Validation (data)']

                    def get_xy(evolution: list[dict], _: int):
                        return get_xy_from_keys(evolution, keys, labels)

                elif need_reg:
                    n_data = self.regularization.n_terms
                    keys = ['validation_' + r for r in self.regularization.term_names]
                    labels = self.regularization.term_names

                    def get_xy(evolution: list[dict], _: int):
                        return get_xy_from_keys(evolution, keys, labels)

                else:
                    raise ValueError('At least one of need_data or need_reg must be True.')
            else:
                n_data = 2
                keys = ['train_loss', 'validation_loss']
                labels = ['Train', 'Validation']

                def get_xy(evolution: list[dict], _: int):
                    return get_xy_from_keys(evolution, keys, labels)
        else:
            n_data = 1 + self.regularization.n_terms
            keys = ['regularization_loss'] + self.regularization.term_names
            labels = ['Reg (all)'] + self.regularization.term_names

            def get_xy(evolution: list[dict], _: int):
                return get_xy_from_keys(evolution, keys, labels)
        return n_data, get_xy
