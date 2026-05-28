from collections.abc import Callable
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

__all__ = [
    'Callback',
    'CSVLogger',
    'CheckpointBest',
    'EarlyStopping',
    'GradientNormMonitor',
    'HistoryLogger',
    'LambdaCallback',
    'LearningRateLogger',
    'LossMonitor',
    'MaxTime',
    'SaveLossMonitor',
    'SaveParityPlot',
    'SavePredictionResults',
    'StateDictsSaver',
    'TargetLossReached',
    'TerminateOnNaN',
    'mse_error',
    'relative_error',
]


def mse_error(x_true: torch.Tensor, x_pred: torch.Tensor):
    return torch.square(x_true - x_pred).mean(dim=1, keepdim=True)


def relative_error(x_true: torch.Tensor, x_pred: torch.Tensor):
    return ((x_true - x_pred).abs() / (x_true.abs() + 1e-10)).mean(dim=1, keepdim=True)


def _labels_to_numpy(labels) -> np.ndarray:
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    return labels


def _get_label_columns(data_handler) -> list[str]:
    if data_handler.label_idx is None:
        return ['label']
    if len(data_handler.label_idx) == 1:
        return [str(data_handler.label_idx[0])]
    return [str(label) for label in data_handler.label_idx]


def _get_plotter_class():
    try:
        from murakami_lab_modules.plotter import Plotter
    except ImportError as e:
        raise ImportError(
            'Plot callbacks require matplotlib. '
            'Install it with `pip install murakami_lab_modules[plot]`.'
        ) from e
    return Plotter


def _require_saved_results(model_handler, callback_name: str) -> Path:
    if not getattr(model_handler, 'save_result', True) or model_handler.model_path is None:
        raise ValueError(f'{callback_name} requires ModelHandler(save_result=True).')
    return Path(model_handler.model_path)


def _current_epoch_number(model_handler) -> int:
    return model_handler.epoch + 1


def _ensure_evolution_column(model_handler, column: str) -> None:
    if column not in model_handler.evolution_col:
        model_handler.evolution_col.append(column)


def _latest_evolution_record(model_handler, callback_name: str) -> dict[str, object]:
    current_evolution = getattr(model_handler, 'current_evolution', None)
    if current_evolution is not None:
        return current_evolution
    if not model_handler.evolution:
        raise RuntimeError(f'{callback_name} requires at least one evolution record.')
    return model_handler.evolution[-1]


def _record_value(model_handler, monitor: str, callback_name: str) -> float:
    record = _latest_evolution_record(model_handler, callback_name)
    if monitor not in record:
        keys = ', '.join(record.keys())
        raise KeyError(f'{monitor} was not found in evolution record. Available keys: {keys}')
    return float(record[monitor])


def _is_improved(value: float, best_value: float | None, mode: str, min_delta: float) -> bool:
    if best_value is None:
        return True
    if mode == 'min':
        return value < best_value - min_delta
    return value > best_value + min_delta


def _state_dicts(model_handler, save_optimizer: bool) -> dict[str, object]:
    state_dicts = {'nn_state_dict': model_handler.nn.state_dict()}
    if save_optimizer:
        state_dicts['optimizer_state_dict'] = model_handler.optimizer.state_dict()
    return state_dicts


def _predict(model_handler, x: torch.Tensor, label=None, phase: str = None) -> torch.Tensor:
    if model_handler.data_fitting is not None:
        return model_handler.data_fitting.predict(
            nn=model_handler.nn,
            x=x,
            label=label,
            phase=phase,
            epoch=model_handler.epoch
        )
    try:
        return model_handler.nn(x=x)
    except TypeError as e:
        if "unexpected keyword argument 'x'" not in str(e):
            raise
        return model_handler.nn(x)


class Callback:
    def __init__(
            self,
            every: int = None,
            run_on_train_end: bool = True
    ):
        self.every = every
        self.run_on_train_end = run_on_train_end
        if every is not None and (type(every) is not int or every <= 0):
            raise ValueError('every must be a positive int or None.')

    def should_call(self, model_handler) -> bool:
        if self.every is None:
            return False
        return _current_epoch_number(model_handler) % self.every == 0

    def on_train_begin(self, model_handler):
        pass

    def on_epoch_begin(self, model_handler):
        pass

    def on_call(self, model_handler):
        pass

    def on_epoch_end(self, model_handler):
        pass

    def on_train_end(self, model_handler):
        pass


class EarlyStopping(Callback):
    def __init__(
            self,
            monitor: str = 'valid',
            patience: int = 100,
            mode: str = 'min',
            min_delta: float = 0.0,
            every: int = 1,
    ):
        super().__init__(every=every, run_on_train_end=False)
        if type(patience) is not int or patience < 0:
            raise ValueError('patience must be a non-negative int.')
        if mode not in {'min', 'max'}:
            raise ValueError("mode must be 'min' or 'max'.")
        if min_delta < 0:
            raise ValueError('min_delta must be non-negative.')
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value = None
        self.best_epoch = None
        self.wait = 0

    def _is_improved(self, value: float) -> bool:
        return _is_improved(value, self.best_value, self.mode, self.min_delta)

    def _current_value(self, model_handler) -> float:
        return _record_value(model_handler, self.monitor, self.__class__.__name__)

    def on_call(self, model_handler):
        value = self._current_value(model_handler)
        if self._is_improved(value):
            self.best_value = value
            self.best_epoch = model_handler.epoch
            self.wait = 0
            return

        self.wait += 1
        if self.wait >= self.patience:
            model_handler.request_stop(
                f'EarlyStopping(monitor={self.monitor}, patience={self.patience}, best_epoch={self.best_epoch + 1})'
            )


class TerminateOnNaN(Callback):
    def __init__(
            self,
            monitors: str | tuple[str, ...] | list[str] = None,
            every: int = 1
    ):
        super().__init__(every=every, run_on_train_end=False)
        if isinstance(monitors, str):
            monitors = (monitors,)
        self.monitors = None if monitors is None else tuple(monitors)

    def on_call(self, model_handler):
        record = _latest_evolution_record(model_handler, self.__class__.__name__)
        monitors = self.monitors or tuple(key for key in record if key != 'epoch')
        for monitor in monitors:
            if monitor not in record:
                keys = ', '.join(record.keys())
                raise KeyError(f'{monitor} was not found in evolution record. Available keys: {keys}')
            value = float(record[monitor])
            if not np.isfinite(value):
                model_handler.request_stop(f'TerminateOnNaN(monitor={monitor}, value={value})')
                return


class MaxTime(Callback):
    def __init__(
            self,
            seconds: float,
            every: int = 1
    ):
        super().__init__(every=every, run_on_train_end=False)
        if seconds < 0:
            raise ValueError('seconds must be non-negative.')
        self.seconds = float(seconds)
        self.start_time = None

    def on_train_begin(self, model_handler):
        self.start_time = time.perf_counter()

    def on_call(self, model_handler):
        if time.perf_counter() - self.start_time >= self.seconds:
            model_handler.request_stop(f'MaxTime(seconds={self.seconds:g})')


class TargetLossReached(Callback):
    def __init__(
            self,
            target: float,
            monitor: str = 'valid',
            mode: str = 'below',
            every: int = 1
    ):
        super().__init__(every=every, run_on_train_end=False)
        if mode not in {'below', 'above'}:
            raise ValueError("mode must be 'below' or 'above'.")
        self.target = float(target)
        self.monitor = monitor
        self.mode = mode

    def on_call(self, model_handler):
        value = _record_value(model_handler, self.monitor, self.__class__.__name__)
        if self.mode == 'below':
            reached = value <= self.target
        else:
            reached = value >= self.target
        if reached:
            model_handler.request_stop(
                f'TargetLossReached(monitor={self.monitor}, target={self.target:g}, value={value:g})'
            )


class LearningRateLogger(Callback):
    def __init__(
            self,
            column: str = 'lr',
            every: int = 1
    ):
        super().__init__(every=every, run_on_train_end=False)
        self.column = column

    def on_train_begin(self, model_handler):
        _ensure_evolution_column(model_handler, self.column)

    def on_call(self, model_handler):
        record = _latest_evolution_record(model_handler, self.__class__.__name__)
        record[self.column] = model_handler.optimizer.current_lr()


class GradientNormMonitor(Callback):
    def __init__(
            self,
            column: str = 'grad_norm',
            norm_type: float = 2.0,
            every: int = 1
    ):
        super().__init__(every=every, run_on_train_end=False)
        self.column = column
        self.norm_type = norm_type

    def on_train_begin(self, model_handler):
        _ensure_evolution_column(model_handler, self.column)

    def on_call(self, model_handler):
        record = _latest_evolution_record(model_handler, self.__class__.__name__)
        grads = [p.grad.detach() for p in model_handler.nn.parameters() if p.grad is not None]
        if not grads:
            record[self.column] = 0.0
            return
        if self.norm_type == float('inf'):
            value = max(float(g.abs().max().cpu().item()) for g in grads)
        else:
            norms = torch.stack([torch.linalg.vector_norm(g, ord=self.norm_type) for g in grads])
            value = float(torch.linalg.vector_norm(norms, ord=self.norm_type).cpu().item())
        record[self.column] = value


class HistoryLogger(Callback):
    def __init__(
            self,
            path: str | Path = None,
            every: int = None,
            index: bool = False,
            row_every: int = 1,
            keep_best: bool = True,
            keep_last: bool = True,
    ):
        super().__init__(every=every, run_on_train_end=True)
        if row_every is not None and (type(row_every) is not int or row_every <= 0):
            raise ValueError('row_every must be a positive int or None.')
        self.path = None if path is None else Path(path)
        self.index = index
        self.row_every = row_every
        self.keep_best = keep_best
        self.keep_last = keep_last

    def on_train_begin(self, model_handler):
        if self.path is None:
            model_path = _require_saved_results(model_handler, self.__class__.__name__)
            self.path = model_path / 'evolution.csv'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _should_save_record(self, model_handler, record: dict[str, object]) -> bool:
        epoch = record.get('epoch')
        if epoch is None:
            return True
        if self.row_every is not None and epoch % self.row_every == 0:
            return True
        if self.keep_best and epoch == getattr(model_handler, 'best_epoch', None):
            return True
        current = getattr(model_handler, 'current_evolution', None)
        if self.keep_last and current is not None and epoch == current.get('epoch'):
            return True
        return False

    def _selected_records(self, model_handler) -> list[dict[str, object]]:
        records_by_epoch = {}
        for record in model_handler.evolution:
            epoch = record.get('epoch')
            records_by_epoch[epoch] = record

        current = getattr(model_handler, 'current_evolution', None)
        if current is not None and self.keep_last:
            records_by_epoch[current.get('epoch')] = current

        records = sorted(
            records_by_epoch.values(),
            key=lambda record: -1 if record.get('epoch') is None else record.get('epoch')
        )
        return [
            record
            for record in records
            if self._should_save_record(model_handler, record)
        ]

    def _save(self, model_handler):
        pd.DataFrame(self._selected_records(model_handler), columns=model_handler.evolution_col).to_csv(
            self.path,
            index=self.index
        )

    def on_call(self, model_handler):
        self._save(model_handler)

    def on_train_end(self, model_handler):
        if self.run_on_train_end:
            self._save(model_handler)


class CSVLogger(HistoryLogger):
    pass


class CheckpointBest(Callback):
    def __init__(
            self,
            monitor: str = 'valid',
            mode: str = 'min',
            min_delta: float = 0.0,
            path: str | Path = None,
            save_optimizer: bool = True,
            every: int = 1
    ):
        super().__init__(every=every, run_on_train_end=False)
        if mode not in {'min', 'max'}:
            raise ValueError("mode must be 'min' or 'max'.")
        if min_delta < 0:
            raise ValueError('min_delta must be non-negative.')
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.path = None if path is None else Path(path)
        self.save_optimizer = save_optimizer
        self.best_value = None
        self.best_epoch = None

    def on_train_begin(self, model_handler):
        if self.path is None:
            model_path = _require_saved_results(model_handler, self.__class__.__name__)
            self.path = model_path / 'best_state_dicts.pth'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_call(self, model_handler):
        value = _record_value(model_handler, self.monitor, self.__class__.__name__)
        if not _is_improved(value, self.best_value, self.mode, self.min_delta):
            return
        self.best_value = value
        self.best_epoch = model_handler.epoch
        torch.save(_state_dicts(model_handler, self.save_optimizer), self.path)


class LambdaCallback(Callback):
    def __init__(
            self,
            on_train_begin: Callable[[object], None] = None,
            on_epoch_begin: Callable[[object], None] = None,
            on_call: Callable[[object], None] = None,
            on_epoch_end: Callable[[object], None] = None,
            on_train_end: Callable[[object], None] = None,
            every: int = None,
            run_on_train_end: bool = True,
    ):
        super().__init__(every=every, run_on_train_end=run_on_train_end)
        self._on_train_begin = on_train_begin
        self._on_epoch_begin = on_epoch_begin
        self._on_call = on_call
        self._on_epoch_end = on_epoch_end
        self._on_train_end = on_train_end

    def on_train_begin(self, model_handler):
        if self._on_train_begin is not None:
            self._on_train_begin(model_handler)

    def on_epoch_begin(self, model_handler):
        if self._on_epoch_begin is not None:
            self._on_epoch_begin(model_handler)

    def on_call(self, model_handler):
        if self._on_call is not None:
            self._on_call(model_handler)

    def on_epoch_end(self, model_handler):
        if self._on_epoch_end is not None:
            self._on_epoch_end(model_handler)

    def on_train_end(self, model_handler):
        if self.run_on_train_end and self._on_train_end is not None:
            self._on_train_end(model_handler)


class SaveLossMonitor(Callback):
    def __init__(
            self,
            need_data: bool = True,
            need_reg: bool = True,
            every: int = None,
            run_on_train_end: bool = True,
    ):
        super().__init__(
            every=every,
            run_on_train_end=run_on_train_end
        )
        self.need_data = need_data
        self.need_reg = need_reg
        self.n_data = None
        self.get_xy = None
        self.plotter = None
        self.output_dir = None

    def save_loss_monitor(self, model_handler):
        self.plotter.remove_plots()
        x, ys, labels = self.get_xy(model_handler.evolution, _current_epoch_number(model_handler))
        for y, label in zip(ys, labels):
            self.plotter.plot(x=x, y=y, label=label)

        epoch = _current_epoch_number(model_handler)
        self.plotter.add_details(
            x_lim=(0, epoch),
            legend_outside=True
        )
        self.plotter.save_fig(self.output_dir / f'{epoch:0>6}')

    def on_train_begin(self, model_handler):
        model_path = _require_saved_results(model_handler, self.__class__.__name__)
        self.output_dir = model_path / 'loss_evolution'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_data, self.get_xy = model_handler.get_loss_info_fnc(need_data=self.need_data, need_reg=self.need_reg)
        Plotter = _get_plotter_class()
        self.plotter = Plotter(
            window_name='',
            n_data=self.n_data
        )
        self.plotter.add_details(
            title='Loss evolution',
            x_label='Training epochs [-]',
            y_label='Loss [-]',
            y_log=True
        )

    def on_call(self, model_handler):
        self.save_loss_monitor(model_handler)

    def on_train_end(self, model_handler):
        if self.run_on_train_end:
            self.save_loss_monitor(model_handler)
        self.plotter.close()


class SavePredictionResults(Callback):
    def __init__(
            self,
            prediction_metrics: tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], ...] = (
                    mse_error,
                    relative_error
            ),
            normalized_metrics: tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], ...] = (),
            every: int = None,
            run_on_train_end: bool = True,
    ):
        super().__init__(
            every=every,
            run_on_train_end=run_on_train_end
        )
        self.prediction_metrics = prediction_metrics
        self.normalized_metrics = normalized_metrics
        self.output_dir = None

    def get_df(self, model_handler):
        model_handler.nn.eval()
        data_handler = model_handler.data_fitting.data_handler
        label_columns = _get_label_columns(data_handler)
        with torch.no_grad():
            prediction_results: list[pd.DataFrame] = []
            for key in ['train', 'valid', 'test']:
                if data_handler.n_data[key] == 0:
                    continue
                for x, y, label in data_handler(key):
                    y_pred = _predict(model_handler, x=x, label=label, phase=key)
                    label_np = _labels_to_numpy(label)

                    x_ = data_handler.undo_normalize_x(x)
                    y_ = data_handler.undo_normalize_y(y)
                    y_pred_ = data_handler.undo_normalize_y(y_pred)

                    batch = {}
                    for idx, column in enumerate(label_columns):
                        batch[column] = label_np[:, idx]
                    batch['key'] = np.full(label_np.shape[0], key, dtype=object)
                    for idx in range(model_handler.nn.n_input):
                        batch[f'x_{idx}'] = x_[:, idx].detach().cpu().numpy()
                    for idx in range(model_handler.nn.n_output):
                        batch[f'y_true_{idx}'] = y_[:, idx].detach().cpu().numpy()
                        batch[f'y_pred_{idx}'] = y_pred_[:, idx].detach().cpu().numpy()
                    for metric in self.prediction_metrics:
                        batch[f'{metric.__name__}_pred'] = metric(y_, y_pred_).detach().cpu().numpy().reshape(-1)
                    for metric in self.normalized_metrics:
                        batch[f'{metric.__name__}_norm'] = metric(y, y_pred).detach().cpu().numpy().reshape(-1)
                    prediction_results.append(pd.DataFrame(batch))
            if not prediction_results:
                return pd.DataFrame()

        columns = (
                label_columns + ['key'] +
                [f'x_{i}' for i in range(model_handler.nn.n_input)] +
                [f'y_true_{i}' for i in range(model_handler.nn.n_output)] +
                [f'y_pred_{i}' for i in range(model_handler.nn.n_output)] +
                [f'{metric.__name__}_pred' for metric in self.prediction_metrics] +
                [f'{metric.__name__}_norm' for metric in self.normalized_metrics]
        )

        return pd.concat(prediction_results, axis=0, ignore_index=True).loc[:, columns]

    def on_train_begin(self, model_handler):
        if not model_handler.has_data:
            raise ValueError('SavePredictionResults callback cannot be used if the model does not have data_fitting.')
        model_path = _require_saved_results(model_handler, self.__class__.__name__)
        self.output_dir = model_path / 'prediction_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_call(self, model_handler):
        df = self.get_df(model_handler)
        df.to_csv(self.output_dir / f'{_current_epoch_number(model_handler):0>6}.csv', index=False)

    def on_train_end(self, model_handler):
        if self.run_on_train_end:
            df = self.get_df(model_handler)
            df.to_csv(self.output_dir / f'{_current_epoch_number(model_handler):0>6}.csv', index=False)


class SaveParityPlot(Callback):
    def __init__(
            self,
            every: int = None,
            run_on_train_end: bool = True,
            fig_size: tuple[float, float] = (8, 8)
    ):
        super().__init__(
            every=every,
            run_on_train_end=run_on_train_end
        )
        self.fig_size = fig_size
        self.output_dir = None

    def save_parity_plot(self, model_handler, folder: Path):
        y_max = torch.full([1, model_handler.nn.n_output], -torch.inf).to(model_handler.device)
        y_min = torch.full([1, model_handler.nn.n_output], torch.inf).to(model_handler.device)
        model_handler.nn.eval()
        with torch.no_grad():
            results = {}
            for key in ['train', 'valid', 'test']:
                if model_handler.data_fitting.data_handler.n_data[key] == 0:
                    continue
                y_list, y_pred_list = [], []
                for x, y, label in model_handler.data_fitting.data_handler(key):
                    y_pred = _predict(model_handler, x=x, label=label, phase=key)
                    y_ = model_handler.data_fitting.data_handler.undo_normalize_y(y)
                    y_pred_ = model_handler.data_fitting.data_handler.undo_normalize_y(y_pred)
                    y_list.append(y_)
                    y_pred_list.append(y_pred_)

                    y_min, _ = torch.min(torch.vstack([y_, y_pred_, y_min]), dim=0, keepdim=True)
                    y_max, _ = torch.max(torch.vstack([y_, y_pred_, y_max]), dim=0, keepdim=True)

                results[key] = [torch.vstack(y_list), torch.vstack(y_pred_list)]

        for y_idx in range(model_handler.nn.n_output):
            Plotter = _get_plotter_class()
            y_max_, y_min_ = y_max[0, y_idx].cpu(), y_min[0, y_idx].cpu()
            dy = (y_max_ - y_min_) * 0.1
            total_plotter = Plotter(
                window_name='',
                n_data=3,
                fig_size=self.fig_size
            )
            total_plotter.plot(
                x=np.array([y_min_ - dy, y_max_ + dy]),
                y=np.array([y_min_ - dy, y_max_ + dy]),
                color='k',
                line_width=2
            )
            total_plotter.add_details(
                title=f'Parity plot ({y_idx=})',
                x_label=r'$y_{true}$',
                y_label=r'$y_{calc}$',
                x_lim=(y_min_ - dy, y_max_ + dy),
                y_lim=(y_min_ - dy, y_max_ + dy)
            )
            individual_plotter = Plotter(
                window_name='',
                n_data=3,
                fig_size=self.fig_size
            )
            individual_plotter.add_details(
                x_label=r'$y_{true}$',
                y_label=r'$y_{calc}$',
                x_lim=(y_min_ - dy, y_max_ + dy),
                y_lim=(y_min_ - dy, y_max_ + dy)
            )
            for key in ['train', 'valid', 'test']:
                if key not in results:
                    continue
                total_plotter.scatter(x=results[key][0][:, y_idx], y=results[key][1][:, y_idx], label=key)
                individual_plotter.plot(
                    x=np.array([y_min_ - dy, y_max_ + dy]),
                    y=np.array([y_min_ - dy, y_max_ + dy]),
                    color='k',
                    line_width=2
                )
                mse = np.square(results[key][0][:, y_idx] - results[key][1][:, y_idx]).mean()
                individual_plotter.scatter(x=results[key][0][:, y_idx], y=results[key][1][:, y_idx], label=key)
                individual_plotter.add_details(title=f'Parity plot ({y_idx=}, {key}) | MSE = {mse:.3e}')
                individual_plotter.save_fig(folder / f'parity_plot_y{y_idx}_{key}')
                individual_plotter.remove_plots(reset_idx=False)
            total_plotter.add_details(legend_inside=True)
            total_plotter.save_fig(folder / f'parity_plot_y{y_idx}')
            individual_plotter.close()
            total_plotter.close()

    def on_train_begin(self, model_handler):
        if not model_handler.has_data:
            raise ValueError('SaveParityPlot callback cannot be used if the model does not have data_fitting.')
        model_path = _require_saved_results(model_handler, self.__class__.__name__)
        self.output_dir = model_path / 'parity_plot'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_call(self, model_handler):
        folder = self.output_dir / f'{_current_epoch_number(model_handler):0>6}'
        folder.mkdir(parents=True, exist_ok=True)
        self.save_parity_plot(model_handler, folder)

    def on_train_end(self, model_handler):
        if self.run_on_train_end:
            folder = self.output_dir / f'{_current_epoch_number(model_handler):0>6}'
            folder.mkdir(parents=True, exist_ok=True)
            self.save_parity_plot(model_handler, folder)


class LossMonitor(Callback):
    def __init__(
            self,
            need_data: bool = True,
            need_reg: bool = True,
            every: int = None,
            show: bool = True,
            window_name: str = 'loss_monitor'
    ):
        super().__init__(every=every, run_on_train_end=False)
        self.need_data = need_data
        self.need_reg = need_reg
        self.show = show
        self.window_name = window_name
        self.n_data = None
        self.get_xy = None
        self.plotter = None

    def on_train_begin(self, model_handler):
        if not self.show:
            return
        self.n_data, self.get_xy = model_handler.get_loss_info_fnc(need_data=self.need_data, need_reg=self.need_reg)
        Plotter = _get_plotter_class()
        self.plotter = Plotter(
            window_name=self.window_name,
            n_data=self.n_data
        )
        self.plotter.add_details(
            title='Loss monitor',
            x_label='Training epochs [-]',
            y_label='Loss [-]',
            y_log=True
        )

    def on_call(self, model_handler):
        if not self.show:
            return
        self.plotter.remove_plots()
        x, ys, labels = self.get_xy(model_handler.evolution, _current_epoch_number(model_handler))
        for y, label in zip(ys, labels):
            self.plotter.plot(x=x, y=y, label=label)
        self.plotter.add_details(x_lim=(0, _current_epoch_number(model_handler)), legend_outside=True)
        self.plotter.update()

    def on_train_end(self, model_handler):
        if self.plotter is not None:
            self.plotter.close()


class StateDictsSaver(Callback):
    def __init__(
            self,
            every: int = None,
            save_optimizer: bool = True
    ):
        super().__init__(every=every, run_on_train_end=False)
        self.save_optimizer = save_optimizer
        self.output_dir = None

    def on_train_begin(self, model_handler):
        model_path = _require_saved_results(model_handler, self.__class__.__name__)
        self.output_dir = model_path / 'state_dicts'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_call(self, model_handler):
        torch.save(
            _state_dicts(model_handler, self.save_optimizer),
            self.output_dir / f'{_current_epoch_number(model_handler):0>6}.pth'
        )
