import pandas as pd
import pytest
import torch

from murakami_lab_modules.callbacks import (
    CSVLogger,
    Callback,
    CheckpointBest,
    EarlyStopping,
    GradientNormMonitor,
    HistoryLogger,
    LambdaCallback,
    LearningRateLogger,
    LossMonitor,
    MaxTime,
    SavePredictionResults,
    StateDictsSaver,
    TargetLossReached,
    TerminateOnNaN,
)
from murakami_lab_modules.data_fitting import DataFitting
from murakami_lab_modules.model_handler import ModelHandler
from murakami_lab_modules.neural_network import FeedForwardNeuralNetwork
from murakami_lab_modules.optimizer import ConstantLROptimizer

from tests.test_training_and_predictor import _make_data_handler


class CounterCallback(Callback):
    def __init__(self, every=None):
        super().__init__(every=every, run_on_train_end=False)
        self.calls = []

    def on_call(self, model_handler):
        self.calls.append(model_handler.epoch + 1)


def _make_model_handler(tmp_path, callbacks=(), train_epochs=3, save_result=True, **kwargs):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    return ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=train_epochs,
        save_path=str(tmp_path / 'Model'),
        train_record_path=str(tmp_path / 'train_record'),
        callbacks=callbacks,
        save_result=save_result,
        verbose=False,
        **kwargs,
    )


def test_callback_every_controls_on_call_interval(tmp_path):
    callback = CounterCallback(every=2)
    model_handler = _make_model_handler(tmp_path, callbacks=(callback,), train_epochs=5, save_result=False)

    model_handler()

    assert callback.calls == [2, 4]


def test_loss_monitor_can_be_disabled_for_headless_training(tmp_path):
    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(LossMonitor(show=False, every=1),),
        train_epochs=1,
        save_result=False,
    )

    model_handler()

    assert len(model_handler.evolution) == 1


def test_early_stopping_requests_stop_after_patience(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=0.0)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=10,
        callbacks=(EarlyStopping(monitor='valid', patience=2),),
        save_result=False,
        verbose=False,
    )

    model_handler()

    assert len(model_handler.evolution) == 3
    assert model_handler.stop_training
    assert model_handler.stop_reason.startswith('EarlyStopping')
    assert model_handler.best_epoch == 0
    assert model_handler.epochs_since_best == 2


def test_terminate_on_nan_requests_stop(tmp_path):
    def set_nan(model_handler):
        model_handler.evolution[-1]['valid'] = float('nan')

    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(
            LambdaCallback(on_call=set_nan, every=1),
            TerminateOnNaN(),
        ),
        train_epochs=5,
        save_result=False,
    )

    model_handler()

    assert len(model_handler.evolution) == 1
    assert model_handler.stop_training
    assert model_handler.stop_reason.startswith('TerminateOnNaN')


def test_max_time_requests_stop(tmp_path):
    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(MaxTime(seconds=0.0),),
        train_epochs=5,
        save_result=False,
    )

    model_handler()

    assert len(model_handler.evolution) == 1
    assert model_handler.stop_training
    assert model_handler.stop_reason.startswith('MaxTime')


def test_target_loss_reached_requests_stop(tmp_path):
    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(TargetLossReached(target=1e30, monitor='valid'),),
        train_epochs=5,
        save_result=False,
    )

    model_handler()

    assert len(model_handler.evolution) == 1
    assert model_handler.stop_training
    assert model_handler.stop_reason.startswith('TargetLossReached')


def test_learning_rate_and_gradient_norm_loggers_add_columns(tmp_path):
    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(
            LearningRateLogger(),
            GradientNormMonitor(),
        ),
        train_epochs=1,
        save_result=False,
    )

    model_handler()

    assert 'lr' in model_handler.evolution_col
    assert 'grad_norm' in model_handler.evolution_col
    assert model_handler.evolution[0]['lr'] == pytest.approx(1e-3)
    assert model_handler.evolution[0]['grad_norm'] >= 0.0


def test_csv_logger_writes_live_history(tmp_path):
    csv_path = tmp_path / 'history.csv'
    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(CSVLogger(path=csv_path),),
        train_epochs=2,
        save_result=False,
    )

    model_handler()

    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ['epoch', 'train', 'valid']
    assert len(df) == 2


def test_history_logger_can_save_sparse_rows(tmp_path):
    csv_path = tmp_path / 'history.csv'
    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(HistoryLogger(path=csv_path, row_every=2, keep_best=False, keep_last=True),),
        train_epochs=5,
        save_result=False,
    )

    model_handler()

    df = pd.read_csv(csv_path)
    assert df['epoch'].to_list() == [0, 2, 4]


def test_checkpoint_best_writes_state_dicts(tmp_path):
    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(CheckpointBest(every=1),),
        train_epochs=2,
    )

    model_handler()

    state_path = model_handler.model_path / 'best_state_dicts.pth'
    assert state_path.exists()
    state_dicts = torch.load(state_path, weights_only=False)
    assert set(state_dicts) == {'nn_state_dict', 'optimizer_state_dict'}


def test_lambda_callback_runs_hooks(tmp_path):
    calls = []
    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(LambdaCallback(
            on_train_begin=lambda _: calls.append('begin'),
            on_call=lambda handler: calls.append(handler.epoch + 1),
            on_train_end=lambda _: calls.append('end'),
            every=1,
        ),),
        train_epochs=2,
        save_result=False,
    )

    model_handler()

    assert calls == ['begin', 1, 2, 'end']


def test_save_prediction_results_requires_saved_results(tmp_path):
    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(SavePredictionResults(every=1),),
        train_epochs=1,
        save_result=False,
    )

    with pytest.raises(ValueError, match=r'SavePredictionResults requires ModelHandler\(save_result=True\)'):
        model_handler()


def test_save_prediction_results_writes_numeric_prediction_csv(tmp_path):
    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(SavePredictionResults(every=1, run_on_train_end=False),),
        train_epochs=1,
    )

    model_handler()

    csv_path = tmp_path / 'Model' / model_handler.model_name / 'prediction_results' / '000001.csv'
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert list(df.columns) == [
        'label',
        'key',
        'x_0',
        'y_true_0',
        'y_pred_0',
        'mse_error_pred',
        'relative_error_pred',
    ]
    assert pd.api.types.is_numeric_dtype(df['x_0'])
    assert pd.api.types.is_numeric_dtype(df['y_pred_0'])
    assert set(df['key']) == {'train'}


def test_state_dicts_saver_writes_interval_checkpoints(tmp_path):
    model_handler = _make_model_handler(
        tmp_path,
        callbacks=(StateDictsSaver(every=1),),
        train_epochs=2,
    )

    model_handler()

    state_dir = tmp_path / 'Model' / model_handler.model_name / 'state_dicts'
    assert (state_dir / '000001.pth').exists()
    assert (state_dir / '000002.pth').exists()
    state_dicts = torch.load(state_dir / '000002.pth', weights_only=False)
    assert set(state_dicts) == {'nn_state_dict', 'optimizer_state_dict'}
