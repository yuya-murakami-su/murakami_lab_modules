import numpy as np
import pandas as pd
import torch

from murakami_lab_modules.data_fitting import DataFitting
from murakami_lab_modules.data_handler import DataHandler
from murakami_lab_modules.model_handler import ModelHandler
from murakami_lab_modules.model_selection import (
    CrossValidator,
    GridSearch,
    KFoldSplitter,
    Metric,
    NestedCrossValidator,
    RandomSearch,
    iter_parameter_grid,
    sample_parameter_space,
)
from murakami_lab_modules.neural_network import FeedForwardNeuralNetwork
from murakami_lab_modules.optimizer import ConstantLROptimizer


class CountingDataFitting(DataFitting):
    def __init__(self, *args, counter: list[int], **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = counter

    def compute_loss(self, *args, **kwargs):
        self.counter[0] += 1
        return super().compute_loss(*args, **kwargs)


def _write_linear_data(tmp_path, n_data=8):
    tmp_path.mkdir(parents=True, exist_ok=True)
    x_path = tmp_path / 'x.csv'
    y_path = tmp_path / 'y.csv'
    x = np.linspace(0.0, 1.0, n_data, dtype=np.float32)
    y = 2.0 * x + 1.0
    pd.DataFrame({'x': x}).to_csv(x_path, index=False)
    pd.DataFrame({'y': y}).to_csv(y_path, index=False)
    return x_path, y_path


def _make_factory(tmp_path, n_data=8):
    x_path, y_path = _write_linear_data(tmp_path, n_data=n_data)

    def factory(params, split, context):
        data_handler = DataHandler(
            input_data_path=str(x_path),
            input_idx=['x'],
            output_data_path=str(y_path),
            output_idx=['y'],
            batch_size=2,
            split_type='index_split',
            train_indices=split.train_indices,
            valid_indices=split.valid_indices,
            test_indices=split.test_indices,
            use_train_as_valid=split.valid_indices is None,
            device_name='cpu',
        )
        data_fitting = DataFitting(data_handler=data_handler, loss_criteria=torch.nn.MSELoss())
        nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=context.seed)
        optimizer = ConstantLROptimizer(torch.optim.SGD, lr=params.get('lr', 1e-3))
        return ModelHandler(
            nn=nn,
            optimizer=optimizer,
            data_fitting=data_fitting,
            train_epochs=1,
            save_result=False,
            verbose=False,
        )

    return factory


def test_kfold_splitter_covers_indices_without_overlap():
    splitter = KFoldSplitter(n_splits=3, shuffle=False)
    splits = splitter.split(7)

    assert len(splits) == 3
    valid_indices = np.concatenate([split.valid_indices for split in splits])
    assert sorted(valid_indices.tolist()) == list(range(7))
    for split in splits:
        assert not set(split.train_indices.tolist()) & set(split.valid_indices.tolist())


def test_parameter_grid_and_sampler_are_reproducible():
    grid = iter_parameter_grid({'lr': [1e-3, 1e-2], 'batch': [4, 8]})
    samples_1 = sample_parameter_space({'lr': [1e-3, 1e-2], 'flag': True}, n_iter=3, random_seed=1)
    samples_2 = sample_parameter_space({'lr': [1e-3, 1e-2], 'flag': True}, n_iter=3, random_seed=1)

    assert grid == [
        {'lr': 1e-3, 'batch': 4},
        {'lr': 1e-3, 'batch': 8},
        {'lr': 1e-2, 'batch': 4},
        {'lr': 1e-2, 'batch': 8},
    ]
    assert samples_1 == samples_2
    assert all(sample['flag'] is True for sample in samples_1)


def test_cross_validator_runs_model_factory_for_each_fold(tmp_path):
    splitter = KFoldSplitter(n_splits=2, shuffle=False)
    cv = CrossValidator(
        model_factory=_make_factory(tmp_path),
        splitter=splitter,
        indices=6,
        metrics=(Metric('mae', lambda y, y_pred: torch.abs(y - y_pred).mean()),),
    )

    results = cv.run(params={'lr': 1e-3})

    assert len(results) == 2
    assert all(result.status == 'ok' for result in results)
    assert all(result.params == {'lr': 1e-3} for result in results)
    assert all(result.valid_loss is not None for result in results)
    assert all('valid_mae' in result.metrics for result in results)


def test_cross_validator_uses_recorded_valid_loss_without_extra_forward(tmp_path):
    x_path, y_path = _write_linear_data(tmp_path, n_data=6)
    counters = []

    def factory(params, split, context):
        counter = [0]
        counters.append(counter)
        data_handler = DataHandler(
            input_data_path=str(x_path),
            input_idx=['x'],
            output_data_path=str(y_path),
            output_idx=['y'],
            batch_size=2,
            split_type='index_split',
            train_indices=split.train_indices,
            valid_indices=split.valid_indices,
            device_name='cpu',
        )
        data_fitting = CountingDataFitting(
            data_handler=data_handler,
            loss_criteria=torch.nn.MSELoss(),
            counter=counter,
        )
        nn = FeedForwardNeuralNetwork(n_input=1, n_output=1, n_layer=0, random_seed=context.seed)
        optimizer = ConstantLROptimizer(torch.optim.SGD, lr=1e-3)
        return ModelHandler(
            nn=nn,
            optimizer=optimizer,
            data_fitting=data_fitting,
            train_epochs=1,
            save_result=False,
            verbose=False,
        )

    cv = CrossValidator(
        model_factory=factory,
        splitter=KFoldSplitter(n_splits=2, shuffle=False),
        indices=6,
    )

    results = cv.run()

    assert all(result.valid_loss is not None for result in results)
    assert [counter[0] for counter in counters] == [3, 3]


def test_grid_and_random_search_rank_results(tmp_path):
    splitter = KFoldSplitter(n_splits=2, shuffle=False)
    grid_search = GridSearch(
        model_factory=_make_factory(tmp_path / 'grid'),
        splitter=splitter,
        indices=6,
        param_grid={'lr': [1e-3, 1e-2]},
    )
    random_search = RandomSearch(
        model_factory=_make_factory(tmp_path / 'random'),
        splitter=splitter,
        indices=6,
        param_space={'lr': [1e-3, 1e-2]},
        n_iter=2,
        random_seed=1,
    )

    grid_results = grid_search.run()
    random_results = random_search.run()

    assert len(grid_results) == 2
    assert len(grid_search.cv_results_) == 4
    assert grid_search.best_result_ is not None
    assert sorted(result.rank for result in grid_results) == [1, 2]
    assert len(random_results) == 2
    assert len(random_search.cv_results_) == 4


def test_nested_cross_validator_selects_params_and_evaluates_outer_test(tmp_path):
    nested = NestedCrossValidator(
        model_factory=_make_factory(tmp_path, n_data=8),
        outer_splitter=KFoldSplitter(n_splits=2, shuffle=False),
        inner_splitter=KFoldSplitter(n_splits=2, shuffle=False),
        indices=8,
        param_grid={'lr': [1e-3, 1e-2]},
        metrics=(Metric('mae', lambda y, y_pred: torch.abs(y - y_pred).mean()),),
    )

    outer_results = nested.run()

    assert len(outer_results) == 2
    assert all(result.status == 'ok' for result in outer_results)
    assert all(result.test_loss is not None for result in outer_results)
    assert all('test_mae' in result.metrics for result in outer_results)
    assert len(nested.inner_search_results_) == 4
    assert len(nested.inner_cv_results_) == 8
    assert sorted({result.outer_fold for result in nested.inner_cv_results_}) == [0, 1]
