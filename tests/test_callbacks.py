import pandas as pd
import pytest
import torch

from murakami_lab_modules.callbacks import Callback, LossMonitor, SavePredictionResults, StateDictsSaver
from murakami_lab_modules.data_fitting import DataFitting
from murakami_lab_modules.model_handler import ModelHandler
from murakami_lab_modules.neural_network import FeedForwardNeuralNetwork
from murakami_lab_modules.optimizer import Optimizer

from tests.test_training_and_predictor import _make_data_handler


class CounterCallback(Callback):
    def __init__(self, every=None):
        super().__init__(every=every, run_on_train_end=False)
        self.calls = []

    def on_call(self, model_handler):
        self.calls.append(model_handler.epoch + 1)


def _make_model_handler(tmp_path, callbacks=(), train_epochs=3, save_result=True):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = Optimizer(torch.optim.SGD, lr=1e-3)
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
