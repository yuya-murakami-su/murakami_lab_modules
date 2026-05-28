"""Basic regression demo.

This script trains a small feed-forward neural network on synthetic tabular
data, saves the model, reloads it with NeuralNetworkPredictor, and saves a plot
of the fitted curve.

Run from the project root:

    python demos/01_basic_regression.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib
except ImportError as e:
    raise SystemExit(
        'This demo requires matplotlib. Install plotting dependencies with `pip install -e ".[plot]"`.'
    ) from e

matplotlib.use('Agg')

from murakami_lab_modules.data import DataHandler
from murakami_lab_modules.models import FeedForwardNeuralNetwork, NeuralNetworkPredictor
from murakami_lab_modules.training import DataFitting, EarlyStopping, ModelHandler, Optimizer
from murakami_lab_modules.visualization import Plotter


def target_function(x: np.ndarray) -> np.ndarray:
    """Return a smooth nonlinear target used to generate synthetic data."""

    return np.sin(2.0 * np.pi * x[:, 0]) + 0.3 * x[:, 1] ** 2


def main() -> None:
    output_dir = Path(__file__).resolve().parent / 'demo_outputs' / '01_basic_regression'
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2025)
    n_samples = 120
    x = rng.uniform(0.0, 1.0, size=(n_samples, 2)).astype(np.float32)
    y = (target_function(x) + rng.normal(0.0, 0.05, size=n_samples)).astype(np.float32).reshape(-1, 1)

    data_path = output_dir / 'synthetic_regression.csv'
    pd.DataFrame({'x0': x[:, 0], 'x1': x[:, 1], 'y': y[:, 0]}).to_csv(data_path, index=False)

    data_handler = DataHandler(
        input_data_path=str(data_path),
        input_columns=['x0', 'x1'],
        output_columns=['y'],
        split_ratio=(0.8, 0.2),
        batch_size=32,
        random_seed=2025,
    )
    data_fitting = DataFitting(data_handler=data_handler, loss_fn=torch.nn.MSELoss())

    nn = FeedForwardNeuralNetwork(
        input_dim=2,
        output_dim=1,
        n_hidden_layers=2,
        hidden_dim=32,
        activation=torch.nn.Tanh(),
        random_seed=2025,
    )
    optimizer = Optimizer(torch.optim.Adam, lr=2e-3)

    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=600,
        callbacks=(EarlyStopping(monitor='validation_loss', patience=100),),
        save_path=str(output_dir / 'models'),
        summary_path=str(output_dir / 'run_summary'),
        verbose=False,
    )
    model_handler()

    predictor = NeuralNetworkPredictor(model_path=model_handler.model_path)
    x_line = np.stack([np.linspace(0.0, 1.0, 300), np.full(300, 0.5)], axis=1).astype(np.float32)
    y_line_true = target_function(x_line)
    y_line_pred = predictor(x_line).reshape(-1)

    plotter = Plotter(n_data=2)
    plotter.scatter(x=x[:, 0], y=y[:, 0], label='Noisy data', marker_size=4, alpha=0.35, series=0)
    plotter.plot(x=x_line[:, 0], y=y_line_true, label='True function at x1=0.5', series=0)
    plotter.plot(x=x_line[:, 0], y=y_line_pred, label='Neural network', series=1)
    plotter.add_details(
        title='Basic regression',
        x_label='x0',
        y_label='y',
        x_lim=(0.0, 1.0),
        legend_outside=True,
    )
    plotter.save_fig(output_dir / 'basic_regression.png')
    plotter.close()

    print(f'Model saved to: {model_handler.model_path}')
    print(f'Figure saved to: {output_dir / "basic_regression.png"}')


if __name__ == '__main__':
    main()
