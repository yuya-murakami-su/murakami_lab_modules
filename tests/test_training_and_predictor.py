import numpy as np
import pandas as pd
import torch

from murakami_lab_modules.data_fitting import DataFitting
from murakami_lab_modules.data_handler import DataHandler
from murakami_lab_modules.model_handler import ModelHandler
from murakami_lab_modules.neural_network import FeedForwardNeuralNetwork
from murakami_lab_modules.optimizer import ConstantLROptimizer
from murakami_lab_modules.predictor import NNPredictor
from murakami_lab_modules.regularization import Regularization


class DummyInputGenerator:
    device = torch.device('cpu')
    device_name = 'cpu'

    def config_dict(self):
        return {'class': 'DummyInputGenerator', 'params': {}}


class OutputMagnitudeRegularization(Regularization):
    def regularization(self, data_handler, nn):
        x = torch.ones(2, 1)
        return [nn(x=x)]


class PositionalOnlyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, z):
        return self.linear(z)


def _make_data_handler(tmp_path):
    x_path = tmp_path / 'x.csv'
    y_path = tmp_path / 'y.csv'
    pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]}).to_csv(x_path, index=False)
    pd.DataFrame({'y': [0.0, 2.0, 4.0, 6.0]}).to_csv(y_path, index=False)

    return DataHandler(
        input_data_path=str(x_path),
        input_idx=['x'],
        output_data_path=str(y_path),
        output_idx=['y'],
        batch_size=2,
        split_ratio=(1.0,),
        use_train_as_valid=True,
        device_name='cpu',
    )


def test_model_handler_trains_one_epoch_and_saves_files(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=1,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
    )

    model_handler()

    assert len(model_handler.evolution) == 1
    assert (tmp_path / 'Model' / model_handler.model_name / 'config.json').exists()
    assert (tmp_path / 'Model' / model_handler.model_name / 'normalizer.pth').exists()
    assert (tmp_path / 'Model' / model_handler.model_name / 'state_dicts.pth').exists()
    assert (tmp_path / 'Model' / model_handler.model_name / 'evolution.csv').exists()
    assert (tmp_path / 'Model' / model_handler.model_name / 'run_summary.csv').exists()
    assert (tmp_path / 'run_summary.csv').exists()
    assert (tmp_path / 'Model' / model_handler.model_name / 'metadata' / 'data_summary.json').exists()
    assert (tmp_path / 'Model' / model_handler.model_name / 'metadata' / 'data_summary.csv').exists()
    summary = pd.read_csv(tmp_path / 'run_summary.csv')
    assert list(summary.columns) == [
        'time',
        'model_name',
        'model_path',
        'best_epoch',
        'best_loss',
        'train_loss',
        'valid_loss',
        'test_loss',
        'stop_reason',
        'n_epochs',
    ]
    assert pd.isna(summary.loc[0, 'test_loss'])


def test_model_handler_can_evaluate_test_loss_in_run_summary(tmp_path):
    x_path = tmp_path / 'x.csv'
    y_path = tmp_path / 'y.csv'
    pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]}).to_csv(x_path, index=False)
    pd.DataFrame({'y': [0.0, 2.0, 4.0, 6.0]}).to_csv(y_path, index=False)
    data_handler = DataHandler(
        input_data_path=str(x_path),
        input_idx=['x'],
        output_data_path=str(y_path),
        output_idx=['y'],
        batch_size=2,
        split_type='index_split',
        train_indices=[0, 1],
        test_indices=[2, 3],
        use_train_as_valid=True,
        device_name='cpu',
    )
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=1,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
        evaluate_test=True,
        verbose=False,
    )

    model_handler()

    summary = pd.read_csv(tmp_path / 'run_summary.csv')
    assert pd.notna(summary.loc[0, 'test_loss'])


def test_model_handler_can_skip_heavy_model_files(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=1,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
        save_model=False,
    )

    model_handler()
    model_path = tmp_path / 'Model' / model_handler.model_name

    assert model_path.exists()
    assert (model_path / 'config.json').exists()
    assert (model_path / 'evolution.csv').exists()
    assert (model_path / 'run_summary.csv').exists()
    assert (model_path / 'metadata' / 'data_summary.json').exists()
    assert not (model_path / 'state_dicts.pth').exists()
    assert not (model_path / 'normalizer.pth').exists()


def test_model_handler_can_disable_history_file(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=1,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
        save_history=False,
    )

    model_handler()
    model_path = tmp_path / 'Model' / model_handler.model_name

    assert model_path.exists()
    assert not (model_path / 'evolution.csv').exists()


def test_model_handler_can_keep_sparse_history(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=0.0)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=5,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
        history_policy='sparse',
        history_every=2,
        keep_best_history=False,
        verbose=False,
    )

    model_handler()
    model_path = tmp_path / 'Model' / model_handler.model_name

    assert [record['epoch'] for record in model_handler.evolution] == [0, 2, 4]
    evolution = pd.read_csv(model_path / 'evolution.csv')
    assert evolution['epoch'].to_list() == [0, 2, 4]


def test_model_handler_can_skip_in_memory_history(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=2,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
        history_policy='none',
        verbose=False,
    )

    model_handler()
    model_path = tmp_path / 'Model' / model_handler.model_name

    assert model_handler.evolution == []
    assert model_handler.current_evolution['epoch'] == 1
    assert not (model_path / 'evolution.csv').exists()


def test_save_model_false_skips_best_state_snapshot(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=1,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
        save_model=False,
    )

    def fail_get_state_dicts():
        raise AssertionError('_get_state_dicts should not be called when save_model=False.')

    model_handler._get_state_dicts = fail_get_state_dicts
    model_handler()

    assert model_handler.state_dicts is None


def test_model_handler_can_disable_all_file_outputs(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=1,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
        save_result=False,
    )

    model_handler()

    assert model_handler.model_path is None
    assert len(model_handler.evolution) == 1
    assert not (tmp_path / 'Model').exists()
    assert not (tmp_path / 'run_summary.csv').exists()


def test_save_result_false_skips_best_state_snapshot(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=1,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
        save_result=False,
    )

    def fail_get_state_dicts():
        raise AssertionError('_get_state_dicts should not be called when save_result=False.')

    model_handler._get_state_dicts = fail_get_state_dicts
    model_handler()

    assert model_handler.state_dicts is None


def test_model_handler_verbose_false_still_saves_results(tmp_path, capsys):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=1,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
        verbose=False,
    )

    model_handler()

    captured = capsys.readouterr()
    model_path = tmp_path / 'Model' / model_handler.model_name
    assert captured.out == ''
    assert len(model_handler.evolution) == 1
    assert (model_path / 'evolution.csv').exists()
    assert (model_path / 'state_dicts.pth').exists()


def test_model_handler_with_regularization_saves_weight_report(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    regularization = OutputMagnitudeRegularization(
        input_generators=[DummyInputGenerator()],
        reg_weights=[0.1],
        reg_names=['output_magnitude'],
    )
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        regularization=regularization,
        train_epochs=1,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
        verbose=False,
    )

    model_handler()

    report_path = tmp_path / 'Model' / model_handler.model_name / 'regularization_weight_report.csv'
    assert report_path.exists()
    report = pd.read_csv(report_path)
    assert list(report['name']) == ['output_magnitude']
    assert np.allclose(report['weight'].to_numpy(), [0.1])


def test_data_fitting_caches_positional_nn_call_style(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = PositionalOnlyNetwork()
    x, y, label = next(data_handler('train'))

    data_fitting.compute_loss(nn=nn, x=x, y=y, label=label)
    data_fitting.compute_loss(nn=nn, x=x, y=y, label=label)

    assert data_fitting._nn_call_styles[id(nn)] == 'positional'


def test_nn_predictor_restores_saved_model_and_predicts_numpy_and_torch(tmp_path):
    data_handler = _make_data_handler(tmp_path)
    data_fitting = DataFitting(data_handler, loss_criteria=torch.nn.MSELoss())
    nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=1)
    optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=1,
        save_path=str(tmp_path / 'Model'),
        summary_path=str(tmp_path / 'run_summary'),
    )
    model_handler()

    predictor = NNPredictor(model_path=model_handler.model_path)
    np_output = predictor(np.asarray([[1.0], [2.0]], dtype=np.float32))
    torch_output = predictor(torch.tensor([[1.0], [2.0]], dtype=torch.float32))

    assert np_output.shape == (2, 1)
    assert torch_output.shape == (2, 1)
    assert np.allclose(np_output, torch_output.detach().cpu().numpy())
