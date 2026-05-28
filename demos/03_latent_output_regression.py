"""Latent-output regression demo.

This script demonstrates a common scientific modeling pattern:

    y = x1 * x2 * N(x)

The neural network learns the latent correction term N(x), not the observed
output y directly. ``LatentOutputFitting`` converts between the observed output
space and the latent output space while keeping the training loss in the
observed y space.

Run from the project root:

    python demos/03_latent_output_regression.py
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
from murakami_lab_modules.training import (
    EarlyStopping,
    InputProductOutputTransform,
    LatentOutputFitting,
    ModelHandler,
    Optimizer,
)
from murakami_lab_modules.visualization import Plotter


def latent_function(x: np.ndarray) -> np.ndarray:
    """Return a smooth latent correction term."""

    return 1.0 + 0.35 * np.sin(2.5 * x[:, 0]) - 0.20 * np.cos(2.0 * x[:, 1])


def make_data(random_seed: int = 2025) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic data with a known input-product structure."""

    rng = np.random.default_rng(random_seed)
    x = rng.uniform(0.3, 2.0, size=(180, 2)).astype(np.float32)
    latent = latent_function(x).astype(np.float32).reshape(-1, 1)
    y_clean = x[:, 0:1] * x[:, 1:2] * latent
    y = y_clean + rng.normal(0.0, 0.02, size=y_clean.shape).astype(np.float32)
    return x, y, latent


def main() -> None:
    output_dir = Path(__file__).resolve().parent / 'demo_outputs' / '03_latent_output_regression'
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, latent_true = make_data()
    pd.DataFrame({
        'x1': x[:, 0],
        'x2': x[:, 1],
        'latent_true': latent_true[:, 0],
        'y': y[:, 0],
    }).to_csv(output_dir / 'latent_output_data.csv', index=False)

    data_handler = DataHandler.from_tensors(
        inputs=x,
        outputs=y,
        split_ratio=(0.8, 0.2),
        batch_size=32,
        random_seed=2025,
    )
    data_fitting = LatentOutputFitting(
        data_handler=data_handler,
        output_transform=InputProductOutputTransform(input_indices=[0, 1]),
        loss_fn=torch.nn.MSELoss(),
    )

    nn = FeedForwardNeuralNetwork(
        input_dim=2,
        output_dim=1,
        n_hidden_layers=2,
        hidden_dim=32,
        activation=torch.nn.Tanh(),
        random_seed=2025,
    )
    model_handler = ModelHandler(
        nn=nn,
        optimizer=Optimizer(torch.optim.Adam, lr=3e-3),
        data_fitting=data_fitting,
        train_epochs=800,
        callbacks=(EarlyStopping(monitor='validation_loss', patience=120),),
        save_path=str(output_dir / 'models'),
        summary_path=str(output_dir / 'run_summary'),
        verbose=False,
    )
    model_handler()

    predictor = NeuralNetworkPredictor(model_path=model_handler.model_path)
    y_pred = predictor(x)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    x_norm = data_handler.normalize_x(x_tensor)
    with torch.no_grad():
        latent_pred = data_fitting.predict_latent(model_handler.nn, x_norm).detach().cpu().numpy()

    prediction_df = pd.DataFrame({
        'x1': x[:, 0],
        'x2': x[:, 1],
        'latent_true': latent_true[:, 0],
        'latent_pred': latent_pred[:, 0],
        'y_true': y[:, 0],
        'y_pred': y_pred[:, 0],
    })
    prediction_df.to_csv(output_dir / 'latent_output_predictions.csv', index=False)

    parity_plot = Plotter(n_data=1)
    parity_plot.scatter(x=y[:, 0], y=y_pred[:, 0], label='Prediction', marker_size=4, alpha=0.7)
    y_min = min(float(y.min()), float(y_pred.min()))
    y_max = max(float(y.max()), float(y_pred.max()))
    parity_plot.plot(x=np.array([y_min, y_max]), y=np.array([y_min, y_max]), color='k', line_width=2)
    parity_plot.add_details(
        title='Observed output prediction',
        x_label='Observed y',
        y_label='Predicted y',
        legend_inside=True,
    )
    parity_plot.save_fig(output_dir / 'observed_parity.png')
    parity_plot.close()

    latent_plot = Plotter(n_data=2)
    latent_plot.scatter(
        x=x[:, 0],
        y=latent_true[:, 0],
        label='True latent',
        marker_size=4,
        alpha=0.55,
        series=0,
    )
    latent_plot.scatter(
        x=x[:, 0],
        y=latent_pred[:, 0],
        label='Predicted latent',
        marker_size=4,
        alpha=0.55,
        series=1,
    )
    latent_plot.add_details(
        title='Latent correction term',
        x_label='x1',
        y_label='N(x)',
        legend_inside=True,
    )
    latent_plot.save_fig(output_dir / 'latent_correction.png')
    latent_plot.close()

    rmse = float(np.sqrt(np.mean(np.square(y[:, 0] - y_pred[:, 0]))))
    latent_rmse = float(np.sqrt(np.mean(np.square(latent_true[:, 0] - latent_pred[:, 0]))))
    print(f'Model saved to: {model_handler.model_path}')
    print(f'Prediction CSV saved to: {output_dir / "latent_output_predictions.csv"}')
    print(f'Observed RMSE: {rmse:.4e}')
    print(f'Latent RMSE: {latent_rmse:.4e}')


if __name__ == '__main__':
    main()
