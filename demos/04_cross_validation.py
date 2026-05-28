"""Cross-validation and grid-search demo.

This script demonstrates the model-selection API. The key idea is that a
model_factory receives split indices and returns a fresh ModelHandler for each
fold and parameter setting.

Run from the project root:

    python demos/04_cross_validation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from murakami_lab_modules.data import DataHandler
from murakami_lab_modules.evaluation import GridSearch, KFoldSplitter, Metric, results_to_dataframe
from murakami_lab_modules.models import FeedForwardNeuralNetwork
from murakami_lab_modules.training import DataFitting, ModelHandler, Optimizer


def target_function(x: np.ndarray) -> np.ndarray:
    """Return a smooth one-dimensional regression target."""

    return np.cos(3.0 * x[:, 0]) + 0.5 * x[:, 0]


def main() -> None:
    output_dir = Path(__file__).resolve().parent / 'demo_outputs' / '04_cross_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2025)
    x = rng.uniform(-1.0, 1.0, size=(90, 1)).astype(np.float32)
    y = (target_function(x) + rng.normal(0.0, 0.04, size=x.shape[0])).astype(np.float32).reshape(-1, 1)
    pd.DataFrame({'x': x[:, 0], 'y': y[:, 0]}).to_csv(output_dir / 'cross_validation_data.csv', index=False)

    def model_factory(params, split, context):
        data_handler = DataHandler.from_tensors(
            inputs=x,
            outputs=y,
            split_type='index_split',
            train_indices=split.train_indices,
            valid_indices=split.valid_indices,
            batch_size=params['batch_size'],
            random_seed=context.seed,
        )
        nn = FeedForwardNeuralNetwork(
            input_dim=1,
            output_dim=1,
            n_hidden_layers=1,
            hidden_dim=params['hidden_dim'],
            activation=torch.nn.Tanh(),
            random_seed=context.seed,
        )
        return ModelHandler(
            nn=nn,
            optimizer=Optimizer(torch.optim.Adam, lr=params['lr']),
            data_fitting=DataFitting(data_handler=data_handler, loss_fn=torch.nn.MSELoss()),
            train_epochs=120,
            save_result=False,
            verbose=False,
        )

    grid_search = GridSearch(
        model_factory=model_factory,
        splitter=KFoldSplitter(n_splits=3, shuffle=True, random_seed=2025),
        indices=x.shape[0],
        param_grid={
            'lr': [1e-3, 3e-3],
            'hidden_dim': [16, 32],
            'batch_size': [24],
        },
        metrics=(Metric('mae', lambda y_true, y_pred: torch.abs(y_true - y_pred).mean()),),
        score_key='validation_loss',
        greater_is_better=False,
        random_seed=2025,
    )
    search_results = grid_search.run()

    results_df = results_to_dataframe(search_results)
    results_df.to_csv(output_dir / 'grid_search_summary.csv', index=False)

    if grid_search.best_result_ is not None:
        print(f'Best params: {grid_search.best_result_.params}')
        print(f'Best mean validation loss: {grid_search.best_result_.mean_score:.4e}')
    print(f'Grid-search summary saved to: {output_dir / "grid_search_summary.csv"}')


if __name__ == '__main__':
    main()
