"""Callback system for :class:`murakami_lab_modules.training.ModelHandler`.

Callbacks are the supported extension point for side effects during training:
early stopping, logging, plotting, checkpointing, monitoring, and custom hooks.
Each callback can define an ``every`` interval and a ``priority``. Lower
priority values run earlier.
"""

from collections.abc import Callable
from pathlib import Path
import copy
import sys
import time

import numpy as np
import pandas as pd
import torch

from .. import utils
from ..evaluation.metrics import relative_error

logger = utils.get_logger(__name__)

__all__ = [
    'Callback',
    'BestModelTracker',
    'CSVLogger',
    'BestCheckpointSaver',
    'ConsoleLogger',
    'EarlyStopping',
    'FinalCheckpointSaver',
    'GradientNormMonitor',
    'HistoryRecorder',
    'HistoryLogger',
    'LambdaCallback',
    'LearningRateLogger',
    'LossMonitor',
    'MaxTime',
    'LossPlotSaver',
    'ParityPlotSaver',
    'PredictionResultSaver',
    'PeriodicCheckpointSaver',
    'TargetLossReached',
    'TerminateOnNaN',
    'RunSummaryLogger',
    'RegularizationReportSaver',
    'relative_error',
]


def prediction_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Return per-sample mean squared prediction error."""

    return torch.square(y_true - y_pred).mean(dim=1, keepdim=True)


def _get_label_columns(data_handler) -> list[str]:
    if data_handler.label_columns is None:
        return ['label']
    if len(data_handler.label_columns) == 1:
        return [str(data_handler.label_columns[0])]
    return [str(label) for label in data_handler.label_columns]


def _get_plotter_class():
    try:
        from murakami_lab_modules.visualization.plotter import Plotter
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


def _state_dicts(model_handler, save_optimizer: bool, copy_state: bool = False) -> dict[str, object]:
    state_dicts = {'nn_state_dict': model_handler.nn.state_dict()}
    if save_optimizer:
        state_dicts['optimizer_state_dict'] = model_handler.optimizer.state_dict()
    if copy_state:
        state_dicts = copy.deepcopy(state_dicts)
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
    """Base class for training callbacks.

    Subclasses can override any hook method. Hooks are called with the active
    ``ModelHandler`` instance.
    """

    def __init__(
            self,
            every: int = None,
            run_on_train_end: bool = True,
            priority: int = 100
    ):
        self.every = every
        self.run_on_train_end = run_on_train_end
        self.priority = priority
        if every is not None and (type(every) is not int or every <= 0):
            raise ValueError('every must be a positive int or None.')
        if type(priority) is not int:
            raise ValueError('priority must be an int.')

    def should_call(self, model_handler) -> bool:
        if self.every is None:
            return False
        return _current_epoch_number(model_handler) % self.every == 0

    def on_train_begin(self, model_handler):
        pass

    def on_epoch_begin(self, model_handler):
        pass

    def on_train_step_end(self, model_handler):
        pass

    def on_validation_end(self, model_handler):
        pass

    def on_epoch_end(self, model_handler):
        pass

    def on_train_end(self, model_handler):
        pass


class EarlyStopping(Callback):
    """Stop training when a monitored value stops improving."""

    def __init__(
            self,
            monitor: str = 'validation_loss',
            patience: int = 100,
            mode: str = 'min',
            min_delta: float = 0.0,
            every: int = 1,
            priority: int = 100,
    ):
        super().__init__(every=every, run_on_train_end=False, priority=priority)
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
        return utils.is_improved(value, self.best_value, self.mode, self.min_delta)

    def _current_value(self, model_handler) -> float:
        return _record_value(model_handler, self.monitor, self.__class__.__name__)

    def on_epoch_end(self, model_handler):
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
    """Stop training when a monitored history value becomes NaN or infinite."""

    def __init__(
            self,
            monitors: str | tuple[str, ...] | list[str] = None,
            every: int = 1,
            priority: int = 100,
    ):
        super().__init__(every=every, run_on_train_end=False, priority=priority)
        if isinstance(monitors, str):
            monitors = (monitors,)
        self.monitors = None if monitors is None else tuple(monitors)

    def on_epoch_end(self, model_handler):
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
    """Stop training after a wall-clock time limit."""

    def __init__(
            self,
            seconds: float,
            every: int = 1,
            priority: int = 100,
    ):
        super().__init__(every=every, run_on_train_end=False, priority=priority)
        if seconds < 0:
            raise ValueError('seconds must be non-negative.')
        self.seconds = float(seconds)
        self.start_time = None

    def on_train_begin(self, model_handler):
        self.start_time = time.perf_counter()

    def on_epoch_end(self, model_handler):
        if time.perf_counter() - self.start_time >= self.seconds:
            model_handler.request_stop(f'MaxTime(seconds={self.seconds:g})')


class TargetLossReached(Callback):
    """Stop training when a monitored value crosses a target threshold."""

    def __init__(
            self,
            target: float,
            monitor: str = 'validation_loss',
            mode: str = 'below',
            every: int = 1,
            priority: int = 100,
    ):
        super().__init__(every=every, run_on_train_end=False, priority=priority)
        if mode not in {'below', 'above'}:
            raise ValueError("mode must be 'below' or 'above'.")
        self.target = float(target)
        self.monitor = monitor
        self.mode = mode

    def on_epoch_end(self, model_handler):
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
    """Add the current learning rate to the epoch history."""

    def __init__(
            self,
            column: str = 'lr',
            every: int = 1,
            priority: int = 100,
    ):
        super().__init__(every=every, run_on_train_end=False, priority=priority)
        self.column = column

    def on_train_begin(self, model_handler):
        _ensure_evolution_column(model_handler, self.column)

    def on_epoch_end(self, model_handler):
        record = _latest_evolution_record(model_handler, self.__class__.__name__)
        record[self.column] = model_handler.optimizer.current_lr()


class GradientNormMonitor(Callback):
    """Add the gradient norm to the epoch history."""

    def __init__(
            self,
            column: str = 'grad_norm',
            norm_type: float = 2.0,
            every: int = 1,
            priority: int = 100,
    ):
        super().__init__(every=every, run_on_train_end=False, priority=priority)
        self.column = column
        self.norm_type = norm_type

    def on_train_begin(self, model_handler):
        _ensure_evolution_column(model_handler, self.column)

    def on_epoch_end(self, model_handler):
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
    """Save recorded training history to CSV.

    ``row_every`` can be used to write sparse history while optionally keeping
    the best and final rows.
    """

    def __init__(
            self,
            path: str | Path = None,
            every: int = None,
            index: bool = False,
            row_every: int = 1,
            keep_best: bool = True,
            keep_last: bool = True,
            priority: int = 300,
    ):
        super().__init__(every=every, run_on_train_end=True, priority=priority)
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

    def on_epoch_end(self, model_handler):
        self._save(model_handler)

    def on_train_end(self, model_handler):
        if self.run_on_train_end:
            self._save(model_handler)


class CSVLogger(HistoryLogger):
    """Alias for ``HistoryLogger`` kept for users who prefer callback-style naming."""

    pass


class BestCheckpointSaver(Callback):
    """Write a checkpoint whenever a monitored value improves."""

    def __init__(
            self,
            monitor: str = 'validation_loss',
            mode: str = 'min',
            min_delta: float = 0.0,
            path: str | Path = None,
            save_optimizer: bool = True,
            every: int = 1,
            priority: int = 100,
    ):
        super().__init__(every=every, run_on_train_end=False, priority=priority)
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

    def on_epoch_end(self, model_handler):
        value = _record_value(model_handler, self.monitor, self.__class__.__name__)
        if not utils.is_improved(value, self.best_value, self.mode, self.min_delta):
            return
        self.best_value = value
        self.best_epoch = model_handler.epoch
        torch.save(_state_dicts(model_handler, self.save_optimizer), self.path)


class LambdaCallback(Callback):
    """Callback that delegates hooks to user-provided callables."""

    def __init__(
            self,
            on_train_begin: Callable[[object], None] = None,
            on_epoch_begin: Callable[[object], None] = None,
            on_epoch_end: Callable[[object], None] = None,
            on_train_step_end: Callable[[object], None] = None,
            on_validation_end: Callable[[object], None] = None,
            on_train_end: Callable[[object], None] = None,
            every: int = None,
            run_on_train_end: bool = True,
            priority: int = 100,
    ):
        super().__init__(every=every, run_on_train_end=run_on_train_end, priority=priority)
        self._on_train_begin = on_train_begin
        self._on_epoch_begin = on_epoch_begin
        self._on_train_step_end = on_train_step_end
        self._on_validation_end = on_validation_end
        self._on_epoch_end = on_epoch_end
        self._on_train_end = on_train_end

    def on_train_begin(self, model_handler):
        if self._on_train_begin is not None:
            self._on_train_begin(model_handler)

    def on_epoch_begin(self, model_handler):
        if self._on_epoch_begin is not None:
            self._on_epoch_begin(model_handler)

    def on_train_step_end(self, model_handler):
        if self._on_train_step_end is not None:
            self._on_train_step_end(model_handler)

    def on_validation_end(self, model_handler):
        if self._on_validation_end is not None:
            self._on_validation_end(model_handler)

    def on_epoch_end(self, model_handler):
        if self._on_epoch_end is not None:
            self._on_epoch_end(model_handler)

    def on_train_end(self, model_handler):
        if self.run_on_train_end and self._on_train_end is not None:
            self._on_train_end(model_handler)


class BestModelTracker(Callback):
    """Track best loss and optionally restore the best in-memory state."""

    def __init__(
            self,
            monitor: str = None,
            mode: str = 'min',
            min_delta: float = 0.0,
            restore_best: bool = False,
            save_optimizer: bool = True,
            priority: int = 0,
    ):
        super().__init__(every=1, run_on_train_end=True, priority=priority)
        if mode not in {'min', 'max'}:
            raise ValueError("mode must be 'min' or 'max'.")
        if min_delta < 0:
            raise ValueError('min_delta must be non-negative.')
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.save_optimizer = save_optimizer
        self.best_value = None
        self.best_epoch = None
        self.best_state_dicts = None

    def _monitor(self, model_handler) -> str:
        if self.monitor is not None:
            return self.monitor
        return 'validation_loss' if model_handler.has_data else 'regularization_loss'

    def on_train_begin(self, model_handler):
        model_handler.best_loss = None
        model_handler.best_epoch = None
        model_handler.epochs_since_best = 0
        model_handler.state_dicts = None

    def on_epoch_end(self, model_handler):
        monitor = self._monitor(model_handler)
        value = _record_value(model_handler, monitor, self.__class__.__name__)
        if utils.is_improved(value, self.best_value, self.mode, self.min_delta):
            self.best_value = value
            self.best_epoch = model_handler.epoch
            model_handler.best_loss = value
            model_handler.best_epoch = model_handler.epoch
            model_handler.epochs_since_best = 0
            if self.restore_best:
                self.best_state_dicts = _state_dicts(
                    model_handler,
                    save_optimizer=self.save_optimizer,
                    copy_state=True
                )
                model_handler.state_dicts = self.best_state_dicts
            return

        model_handler.epochs_since_best += 1

    def on_train_end(self, model_handler):
        if not self.run_on_train_end or not self.restore_best or self.best_state_dicts is None:
            return
        model_handler.nn.load_state_dict(self.best_state_dicts['nn_state_dict'])
        if self.save_optimizer and 'optimizer_state_dict' in self.best_state_dicts:
            model_handler.optimizer.load_state_dict(self.best_state_dicts['optimizer_state_dict'])


class HistoryRecorder(Callback):
    """Store selected epoch records in ``model_handler.evolution``."""

    def __init__(
            self,
            policy: str = 'full',
            every: int = 1,
            keep_best: bool = True,
            keep_last: bool = True,
            priority: int = 10,
    ):
        super().__init__(every=1, run_on_train_end=True, priority=priority)
        if policy not in {'full', 'sparse', 'last', 'none'}:
            raise ValueError("policy must be one of 'full', 'sparse', 'last', or 'none'.")
        if type(every) is not int or every <= 0:
            raise ValueError('every must be a positive int.')
        self.policy = policy
        self.history_every = every
        self.keep_best = keep_best
        self.keep_last = keep_last

    def _should_keep(self, model_handler, record: dict[str, object]) -> bool:
        if self.policy == 'none':
            return False
        if self.policy in {'full', 'last'}:
            return True
        epoch = record['epoch']
        if epoch % self.history_every == 0:
            return True
        if self.keep_best and epoch == getattr(model_handler, 'best_epoch', None):
            return True
        return False

    def _store(self, model_handler, record: dict[str, object]) -> None:
        if self.policy == 'last':
            model_handler.evolution = [record]
            return
        if model_handler.evolution and model_handler.evolution[-1].get('epoch') == record.get('epoch'):
            model_handler.evolution[-1] = record
            return
        model_handler.evolution.append(record)

    def on_train_begin(self, model_handler):
        model_handler.evolution = []
        model_handler.current_evolution = None

    def on_epoch_end(self, model_handler):
        record = _latest_evolution_record(model_handler, self.__class__.__name__)
        if self._should_keep(model_handler, record):
            self._store(model_handler, record)

    def on_train_end(self, model_handler):
        if not self.run_on_train_end or not self.keep_last or self.policy == 'none':
            return
        current = getattr(model_handler, 'current_evolution', None)
        if current is not None:
            self._store(model_handler, current)


class ConsoleLogger(Callback):
    """Display throttled console progress and log a final summary."""

    def __init__(
            self,
            every: int = 1,
            priority: int = 200,
            progress: bool = True,
            log_summary: bool = True,
            min_interval: float = 1.0,
            leave_last_message: bool = True,
            stream=None
    ):
        super().__init__(every=every, run_on_train_end=True, priority=priority)
        if min_interval < 0:
            raise ValueError('min_interval must be non-negative.')
        self.progress = progress
        self.log_summary = log_summary
        self.min_interval = float(min_interval)
        self.leave_last_message = leave_last_message
        self.stream = stream if stream is not None else sys.stderr
        self._last_message_length = 0
        self._last_progress_time = None

    @staticmethod
    def _time_string(model_handler) -> str:
        if model_handler.epoch == 0:
            dt = model_handler.dt_epoch
        else:
            dt = model_handler.dt_epoch / model_handler.epoch
        if dt < 1e-3:
            return f'({dt * 1e6:.1f} us)'
        if dt < 1:
            return f'({dt * 1000:.1f} ms)'
        return f'({dt:.1f} s)'

    def on_epoch_end(self, model_handler):
        if not self.progress or not self._should_update_progress():
            return
        self._write_progress(self._format_progress_message(model_handler))

    def _format_progress_message(self, model_handler, test_loss: float = np.nan) -> str:
        losses = _latest_evolution_record(model_handler, self.__class__.__name__)
        display_epoch = losses['epoch'] + 1
        dt_str = self._time_string(model_handler)
        best_loss = getattr(model_handler, 'best_loss', np.nan)
        epochs_since_best = getattr(model_handler, 'epochs_since_best', 0)
        test_str = '' if np.isnan(test_loss) else f', Test {test_loss:.3e}'

        if model_handler.has_data:
            if model_handler.has_reg:
                validation_regularization = ', '.join(
                    [
                        f'{name}: {losses[f"validation_{name}"]:.3e}'
                        for name in model_handler.regularization.term_names
                    ]
                )
                message = (
                    f'[{pd.Timestamp.now():%H:%M:%S}] '
                    f'{display_epoch: >5} {dt_str} | '
                    f'Train {losses["train_loss"]:.3e} '
                    f'({losses["train_data_loss"]:.3e} & {losses["train_regularization_loss"]:.3e}), '
                    f'Validation {losses["validation_loss"]:.3e} '
                    f'({losses["validation_data_loss"]:.3e} & '
                    f'{losses["validation_regularization_loss"]:.3e})'
                    f'{test_str} | '
                    f'{validation_regularization} | '
                    f'Best {best_loss:.3e} (no change for {epochs_since_best: >4}) | '
                    f'lr {model_handler.optimizer.current_lr():.2e}'
                )
            else:
                message = (
                    f'[{pd.Timestamp.now():%H:%M:%S}] '
                    f'{display_epoch: >5} {dt_str} | '
                    f'Train {losses["train_loss"]:.3e}, '
                    f'Validation {losses["validation_loss"]:.3e}'
                    f'{test_str} | '
                    f'Best {best_loss:.3e} (no change for {epochs_since_best: >4}) | '
                    f'lr {model_handler.optimizer.current_lr():.2e}'
                )
        else:
            reg_str = ', '.join(
                [f'{name}: {losses[name]:.3e}' for name in model_handler.regularization.term_names]
            )
            message = (
                f'[{pd.Timestamp.now():%H:%M:%S}] '
                f'{display_epoch: >5} {dt_str} | '
                f'Reg {losses["regularization_loss"]:.3e} | {reg_str} | '
                f'Best {best_loss:.3e} (no change for {epochs_since_best: >4}) | '
                f'lr {model_handler.optimizer.current_lr():.2e}'
            )
        return message

    def _should_update_progress(self) -> bool:
        now = time.perf_counter()
        if self._last_progress_time is None or now - self._last_progress_time >= self.min_interval:
            self._last_progress_time = now
            return True
        return False

    def _write_progress(self, message: str) -> None:
        padding = max(self._last_message_length - len(message), 0)
        self.stream.write('\r' + message + ' ' * padding)
        self.stream.flush()
        self._last_message_length = len(message)

    def _clear_progress(self) -> None:
        if not self.progress or self._last_message_length == 0:
            return
        self.stream.write('\r' + ' ' * self._last_message_length + '\r')
        self.stream.flush()
        self._last_message_length = 0

    def _final_test_loss(self, model_handler) -> float:
        if not getattr(model_handler, 'evaluate_test', False):
            return np.nan
        test_loss = model_handler.evaluate_test_loss()
        if not np.isnan(test_loss):
            model_handler._test_loss_shown = True
        return test_loss

    def _leave_progress(self, model_handler, test_loss: float = np.nan) -> None:
        if not self.progress or getattr(model_handler, 'current_evolution', None) is None:
            return
        self._write_progress(self._format_progress_message(model_handler, test_loss=test_loss))
        self.stream.write('\n')
        self.stream.flush()
        self._last_message_length = 0

    def on_train_end(self, model_handler):
        if not self.run_on_train_end:
            return
        test_loss = self._final_test_loss(model_handler) if self.progress else np.nan
        if self.leave_last_message:
            self._leave_progress(model_handler, test_loss=test_loss)
        else:
            self._clear_progress()
            if not np.isnan(test_loss):
                logger.info('Test loss: %.3e', test_loss)
        if not self.log_summary:
            return
        message = (
            f'Training finished: epochs={model_handler.epoch}, '
            f'best_epoch={None if model_handler.best_epoch is None else model_handler.best_epoch + 1}, '
            f'best_loss={model_handler.best_loss}'
        )
        if getattr(model_handler, 'stop_reason', None) is not None:
            message += f', stop_reason={model_handler.stop_reason}'
        logger.info(message)


class FinalCheckpointSaver(Callback):
    """Save final model and optimizer state at training end."""

    def __init__(self, save_optimizer: bool = True, priority: int = 400):
        super().__init__(every=None, run_on_train_end=True, priority=priority)
        self.save_optimizer = save_optimizer

    def on_train_end(self, model_handler):
        if not self.run_on_train_end:
            return
        torch.save(
            _state_dicts(model_handler, self.save_optimizer),
            model_handler.model_path / 'state_dicts.pth'
        )


class RegularizationReportSaver(Callback):
    """Save regularization weights and calibrated term values at training end."""

    def __init__(self, filename: str = 'regularization_weight_report.csv', priority: int = 410):
        super().__init__(every=None, run_on_train_end=True, priority=priority)
        self.filename = filename

    def on_train_end(self, model_handler):
        if self.run_on_train_end and model_handler.has_reg:
            model_handler.regularization.save_weight_report(model_handler.model_path / self.filename)


class RunSummaryLogger(Callback):
    """Write per-run summary CSV files."""

    def __init__(
            self,
            summary_path: str | Path = 'run_summary',
            evaluate_test: bool = False,
            show_test: bool = True,
            priority: int = 500,
    ):
        super().__init__(every=None, run_on_train_end=True, priority=priority)
        self.summary_path = Path(summary_path)
        self.evaluate_test = evaluate_test
        self.show_test = show_test
        self.columns = [
            'time',
            'model_name',
            'model_path',
            'best_epoch',
            'best_loss',
            'train_loss',
            'validation_loss',
            'test_loss',
            'stop_reason',
            'n_epochs',
        ]
        self.run_summary = None

    def _summary_file(self) -> Path:
        if self.summary_path.suffix:
            return self.summary_path
        return self.summary_path.with_suffix('.csv')

    def on_train_begin(self, model_handler):
        if self._summary_file().exists():
            self.run_summary = pd.read_csv(self._summary_file(), index_col=None)
        else:
            self.run_summary = pd.DataFrame(np.empty([0, len(self.columns)]), columns=self.columns)
        model_handler.run_summary = self.run_summary

    def _test_loss(self, model_handler) -> float:
        return model_handler.evaluate_test_loss(enabled=self.evaluate_test)

    @staticmethod
    def _recorded_loss(model_handler, total_key: str, data_key: str) -> float:
        record = getattr(model_handler, 'current_evolution', None)
        if record is None:
            return np.nan
        if model_handler.has_reg and data_key in record:
            return record[data_key]
        return record.get(total_key, np.nan)

    def on_train_end(self, model_handler):
        if not self.run_on_train_end:
            return
        test_loss = self._test_loss(model_handler)
        if self.show_test and not np.isnan(test_loss) and not getattr(model_handler, '_test_loss_shown', False):
            logger.info('Test loss: %.3e', test_loss)
            model_handler._test_loss_shown = True

        row = [
            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            model_handler.model_name,
            str(model_handler.model_path),
            None if getattr(model_handler, 'best_epoch', None) is None else model_handler.best_epoch + 1,
            getattr(model_handler, 'best_loss', None),
            self._recorded_loss(model_handler, 'train_loss', 'train_data_loss'),
            self._recorded_loss(model_handler, 'validation_loss', 'validation_data_loss'),
            test_loss,
            getattr(model_handler, 'stop_reason', None),
            model_handler.epoch,
        ]
        df = pd.DataFrame([row], columns=self.columns)
        df.to_csv(model_handler.model_path / 'run_summary.csv', index=False)
        self.run_summary = pd.concat([self.run_summary, df], axis=0)
        self.run_summary.to_csv(self._summary_file(), index=False)
        model_handler.run_summary = self.run_summary


class LossPlotSaver(Callback):
    """Save loss-history plots at intervals and/or training end."""

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
        self.n_data, self.get_xy = model_handler.get_loss_series(need_data=self.need_data, need_reg=self.need_reg)
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

    def on_epoch_end(self, model_handler):
        self.save_loss_monitor(model_handler)

    def on_train_end(self, model_handler):
        if self.run_on_train_end:
            self.save_loss_monitor(model_handler)
        self.plotter.close()


class PredictionResultSaver(Callback):
    """Save per-sample predictions and prediction metrics as CSV files."""

    def __init__(
            self,
            prediction_metrics: tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], ...] = (
                    prediction_mse,
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
                    label_np = utils.labels_to_numpy(label)

                    x_ = data_handler.undo_normalize_x(x)
                    y_ = model_handler.data_fitting.to_observed_target(y)
                    y_pred_ = model_handler.data_fitting.to_observed_prediction(y_pred)

                    batch = {}
                    for idx, column in enumerate(label_columns):
                        batch[column] = label_np[:, idx]
                    batch['key'] = np.full(label_np.shape[0], key, dtype=object)
                    for idx in range(x_.shape[1]):
                        batch[f'x_{idx}'] = x_[:, idx].detach().cpu().numpy()
                    for idx in range(y_.shape[1]):
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
                [column for column in prediction_results[0].columns if column.startswith('x_')] +
                [column for column in prediction_results[0].columns if column.startswith('y_true_')] +
                [column for column in prediction_results[0].columns if column.startswith('y_pred_')] +
                [f'{metric.__name__}_pred' for metric in self.prediction_metrics] +
                [f'{metric.__name__}_norm' for metric in self.normalized_metrics]
        )

        return pd.concat(prediction_results, axis=0, ignore_index=True).loc[:, columns]

    def on_train_begin(self, model_handler):
        if not model_handler.has_data:
            raise ValueError('PredictionResultSaver callback cannot be used if the model does not have data_fitting.')
        model_path = _require_saved_results(model_handler, self.__class__.__name__)
        self.output_dir = model_path / 'prediction_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, model_handler):
        df = self.get_df(model_handler)
        df.to_csv(self.output_dir / f'{_current_epoch_number(model_handler):0>6}.csv', index=False)

    def on_train_end(self, model_handler):
        if self.run_on_train_end:
            df = self.get_df(model_handler)
            df.to_csv(self.output_dir / f'{_current_epoch_number(model_handler):0>6}.csv', index=False)


class ParityPlotSaver(Callback):
    """Save parity plots for train, validation, and test splits."""

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
        output_dim = model_handler.data_fitting.data_handler.outputs.shape[1]
        y_max = torch.full([1, output_dim], -torch.inf).to(model_handler.device)
        y_min = torch.full([1, output_dim], torch.inf).to(model_handler.device)
        model_handler.nn.eval()
        with torch.no_grad():
            results = {}
            for key in ['train', 'valid', 'test']:
                if model_handler.data_fitting.data_handler.n_data[key] == 0:
                    continue
                y_list, y_pred_list = [], []
                for x, y, label in model_handler.data_fitting.data_handler(key):
                    y_pred = _predict(model_handler, x=x, label=label, phase=key)
                    y_ = model_handler.data_fitting.to_observed_target(y)
                    y_pred_ = model_handler.data_fitting.to_observed_prediction(y_pred)
                    y_list.append(y_)
                    y_pred_list.append(y_pred_)

                    y_min, _ = torch.min(torch.vstack([y_, y_pred_, y_min]), dim=0, keepdim=True)
                    y_max, _ = torch.max(torch.vstack([y_, y_pred_, y_max]), dim=0, keepdim=True)

                results[key] = [torch.vstack(y_list), torch.vstack(y_pred_list)]

        for y_idx in range(output_dim):
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
                y_label=r'$y_{pred}$',
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
                y_label=r'$y_{pred}$',
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
            raise ValueError('ParityPlotSaver callback cannot be used if the model does not have data_fitting.')
        model_path = _require_saved_results(model_handler, self.__class__.__name__)
        self.output_dir = model_path / 'parity_plot'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, model_handler):
        folder = self.output_dir / f'{_current_epoch_number(model_handler):0>6}'
        folder.mkdir(parents=True, exist_ok=True)
        self.save_parity_plot(model_handler, folder)

    def on_train_end(self, model_handler):
        if self.run_on_train_end:
            folder = self.output_dir / f'{_current_epoch_number(model_handler):0>6}'
            folder.mkdir(parents=True, exist_ok=True)
            self.save_parity_plot(model_handler, folder)


class LossMonitor(Callback):
    """Show an interactive loss monitor window during training."""

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
        self.n_data, self.get_xy = model_handler.get_loss_series(need_data=self.need_data, need_reg=self.need_reg)
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

    def on_epoch_end(self, model_handler):
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


class PeriodicCheckpointSaver(Callback):
    """Save numbered state-dict checkpoints every ``every`` epochs."""

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

    def on_epoch_end(self, model_handler):
        torch.save(
            _state_dicts(model_handler, self.save_optimizer),
            self.output_dir / f'{_current_epoch_number(model_handler):0>6}.pth'
        )
