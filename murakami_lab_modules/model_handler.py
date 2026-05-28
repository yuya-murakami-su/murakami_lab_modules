import time
import os
import pandas as pd
import numpy as np
import torch
import copy
from datetime import datetime
from .neural_network import AbstractNeuralNetwork
from .optimizer import AbstractOptimizer
from .data_fitting import DataFitting as _DataFitting
from .regularization import Regularization as _Regularization
from . import utils

__all__ = ['ModelHandler']


class ModelHandler:
    def __init__(
            self,
            nn: AbstractNeuralNetwork,
            optimizer: AbstractOptimizer,
            data_fitting: _DataFitting = None,
            regularization: _Regularization = None,
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
            save_result: bool = True,
            save_model: bool = True,
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
        self.early_stop = early_stop

        self.load_model = load_model
        self.load_optimizer = load_optimizer
        self.random_seed = random_seed

        self.save_path = save_path
        self.train_record_path = train_record_path
        self.recalculate_valid_loss = recalculate_valid_loss
        self.model_name = model_name
        self.original_model_name = model_name
        self.kwargs = kwargs
        self.callback_epoch = callback_epoch
        self.callbacks = callbacks or []
        self.save_result = save_result
        self.save_model = save_model
        self.verbose = verbose

        self._validate_inputs()
        if self.save_result:
            self._prepare_model_folder()
            self._save_model_info()
        else:
            self.model_path = None
        self._set_model()
        self._prepare_train_record()
        self._prepare_train_valuables()

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'train_epochs': self.train_epochs,
            'early_stop': self.early_stop,
            'load_model': self.load_model,
            'load_optimizer': self.load_optimizer,
            'save_path': self.save_path,
            'train_record_path': self.train_record_path,
            'recalculate_valid_loss': self.recalculate_valid_loss,
            'model_name': self.model_name,
            'callback_epoch': self.callback_epoch,
            'callbacks': self.callbacks,
            'random_seed': self.random_seed,
            'save_result': self.save_result,
            'save_model': self.save_model,
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

        if (self.train_epochs is None or self.train_epochs == 0) and self.early_stop == 0:
            raise ValueError('At least of of train_epochs and early_stop must be give.')

    @staticmethod
    def _get_model_folder_timestamp() -> str:
        now = datetime.now()
        return f'{now:%y%m%d-%H%M%S}-{now.microsecond // 1000:03d}'

    def _prepare_model_folder(self):
        base_model_name = self.model_name
        for _ in range(1000):
            timestamp = self._get_model_folder_timestamp()
            if base_model_name is None:
                self.model_name = timestamp
            else:
                self.model_name = f'{timestamp}_{base_model_name}'
            self.model_path = os.path.join(self.save_path, self.model_name)
            try:
                os.makedirs(self.model_path, exist_ok=False)
                return
            except FileExistsError:
                time.sleep(0.001)
        raise RuntimeError(f'Failed to create a unique model folder under {self.save_path}.')

    def _save_model_info(self):
        config = {
            'format_version': 1,
            'nn': self.nn.config_dict(),
            'optimizer': self.optimizer.config_dict(),
            'model_handler': self.config_dict(),
            'data_fitting': self.data_fitting.config_dict() if self.has_data else None,
            'data_handler': self.data_fitting.data_handler.config_dict() if self.has_data else None,
            'regularization': self.regularization.config_dict() if self.has_reg else None
        }
        utils.save_json(f'{self.model_path}\\config.json', config)

        metadata_path = f'{self.model_path}\\metadata'
        utils.save_json(f'{metadata_path}\\nn.json', config['nn'])
        utils.save_json(f'{metadata_path}\\optimizer.json', config['optimizer'])
        utils.save_json(f'{metadata_path}\\model_handler.json', config['model_handler'])
        if self.has_data:
            utils.save_json(f'{metadata_path}\\data_fitting.json', config['data_fitting'])
            utils.save_json(f'{metadata_path}\\data_handler.json', config['data_handler'])
            self.data_fitting.data_handler.save_summary(f'{metadata_path}\\data_summary.json')
            self.data_fitting.data_handler.save_summary(f'{metadata_path}\\data_summary.csv')
            if self.save_model:
                torch.save(self.data_fitting.data_handler.normalizer_dict(), f'{self.model_path}\\normalizer.pth')
        if self.has_reg:
            utils.save_json(f'{metadata_path}\\regularization.json', config['regularization'])
            for idx, input_generator_ in enumerate(self.regularization.input_generators):
                utils.save_json(f'{metadata_path}\\input_generator_{idx}.json', input_generator_.config_dict())

    def _set_model(self):
        self.optimizer.set_parameters(self.nn.parameters())
        self.nn.to(self.device)

        if self.load_model is not None:
            self._load_state_dicts(from_outside=True, load_optimizer=self.load_optimizer)

    def _prepare_train_record(self):
        self.train_record_columns = ['Time', 'Epoch', 'Best loss', 'Test']
        if not self.save_result:
            self.train_record = pd.DataFrame(
                np.empty([0, len(self.train_record_columns)]),
                columns=self.train_record_columns
            )
        elif os.path.exists(f'{self.train_record_path}.csv'):
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
                for x, y, label in self.data_fitting.data_handler(phase):
                    losses.append(self._data_reg_step(x, y, label, phase=phase))
                    batch_sizes.append(len(x))
                return self._average_data_reg_losses(losses, batch_sizes)
            else:
                losses = []
                batch_sizes = []
                for x, y, label in self.data_fitting.data_handler(phase):
                    losses.append(self._data_step(x, y, label, phase=phase))
                    batch_sizes.append(len(x))
                return self._average_data_losses(losses, batch_sizes)
        else:
            return self._reg_step()

    def _average_data_losses(self, losses: list[dict], batch_sizes: list[int]) -> dict[str, object]:
        n_data = sum(batch_sizes)
        return {
            'total': sum(loss['total'] * n for loss, n in zip(losses, batch_sizes)) / n_data,
            'data': sum(loss['data'] * n for loss, n in zip(losses, batch_sizes)) / n_data
        }

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
            data_loss = data_loss_info['total'].item()

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
                data_loss_info = self.data_fitting.compute_loss(
                    nn=self.nn,
                    x=x,
                    y=y,
                    label=label,
                    phase=phase,
                    epoch=self.epoch
                )
                data_loss = data_loss_info['total'].item()

            reg_mean, reg_loss = self.regularization.get_regularization_value(
                nn=self.nn,
                data_handler=self.data_fitting.data_handler,
                epoch=self.epoch,
                data_loss=data_loss
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
            data_loss = loss_info['terms']['data'].item()
            return {
                'total': loss.item(),
                'data': data_loss
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
                data_loss = loss_info['terms']['data'].item()
            return {
                'total': loss_info['total'].item(),
                'data': data_loss
            }

    def _reg_step(self):
        self.optimizer.zero_grad()
        reg_mean, loss = self.regularization.get_regularization_value(nn=self.nn, epoch=self.epoch)
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
        if not self.verbose:
            return

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
        if self.verbose:
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
        if not self.save_result:
            return
        pd.DataFrame(self.evolution, columns=self.evolution_col).to_csv(f'{self.model_path}\\evolution.csv')
        if self.has_reg:
            self.regularization.save_weight_report(f'{self.model_path}\\regularization_weight_report.csv')
        if self.save_model:
            torch.save(self.state_dicts, f'{self.model_path}\\state_dicts.pth')

    def _save_train_record(self):
        if self.has_data and self.data_fitting.check_test:
            if self.has_reg:
                test_loss = self._get_loss('test')['data']
            else:
                test_loss = self._get_loss('test')['total']
            if self.verbose:
                utils.logging(f'Test {test_loss:.3e}')
        else:
            test_loss = np.nan

        if not self.save_result:
            return

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
