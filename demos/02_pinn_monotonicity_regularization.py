"""PINN-style monotonicity and curvature regularization demo.

This script fits sparse noisy data from a monotone concave function. A custom
Regularization subclass samples collocation points and penalizes violations of
dy/dx >= 0 and d2y/dx2 <= 0.

Run from the project root:

    python demos/02_pinn_monotonicity_regularization.py
"""

from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
except ImportError as e:
    raise SystemExit(
        'This demo requires matplotlib. Install plotting dependencies with `pip install -e ".[plot]"`.'
    ) from e

matplotlib.use('Agg')

from murakami_lab_modules.data import DataHandler
from murakami_lab_modules.models import BaseNeuralNetwork, FeedForwardNeuralNetwork, NeuralNetworkPredictor
from murakami_lab_modules.pinn import InputGenerator, Regularization
from murakami_lab_modules.training import DataFitting, EarlyStopping, ModelHandler, Optimizer
from murakami_lab_modules.visualization import Plotter


def target_function(x: np.ndarray) -> np.ndarray:
    """Return a monotone concave target."""

    return np.log1p(6.0 * x[:, 0])


class MonotoneConcaveRegularization(Regularization):
    """Penalize monotonicity and curvature violations on generated inputs."""

    def regularization(self, data_handler: DataHandler, nn: BaseNeuralNetwork):
        x = self.input_generators[0]()
        y = nn(x=data_handler.normalize_x(x))
        dy_dx, d2y_dx2 = self.partial2(y=y, x=x, x_idx=0, y_idx=0)
        monotonicity_violation = (-dy_dx).clamp_min(0.0)
        concavity_violation = d2y_dx2.clamp_min(0.0)
        return [monotonicity_violation, concavity_violation]


def main() -> None:
    output_dir = Path(__file__).resolve().parent / 'demo_outputs' / '02_pinn_monotonicity'
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2025)
    x_train = np.sort(rng.uniform(0.0, 1.0, size=28)).astype(np.float32).reshape(-1, 1)
    y_train = (target_function(x_train) + rng.normal(0.0, 0.04, size=x_train.shape[0])).astype(np.float32).reshape(-1, 1)

    data_handler = DataHandler.from_tensors(
        inputs=x_train,
        outputs=y_train,
        split_ratio=(0.8, 0.2),
        batch_size=16,
        random_seed=2025,
    )
    data_fitting = DataFitting(data_handler=data_handler, loss_fn=torch.nn.MSELoss())

    nn = FeedForwardNeuralNetwork(
        input_dim=1,
        output_dim=1,
        n_hidden_layers=2,
        hidden_dim=32,
        activation=torch.nn.Tanh(),
        random_seed=2025,
    )
    optimizer = Optimizer(torch.optim.Adam, lr=2e-3)

    input_generator = InputGenerator(
        n_samples=96,
        input_range=((0.0, 1.0),),
        sampling='sobol',
        resample=True,
        requires_grad=True,
        random_seed=2025,
    )
    regularization = MonotoneConcaveRegularization(
        input_generators=[input_generator],
        weights=[0.05, 0.05],
        term_names=['monotonicity_violation', 'concavity_violation'],
    )

    model_handler = ModelHandler(
        nn=nn,
        optimizer=optimizer,
        data_fitting=data_fitting,
        regularization=regularization,
        train_epochs=800,
        callbacks=(EarlyStopping(monitor='validation_loss', patience=150),),
        save_path=str(output_dir / 'models'),
        summary_path=str(output_dir / 'run_summary'),
        verbose=False,
    )
    model_handler()

    predictor = NeuralNetworkPredictor(model_path=model_handler.model_path)
    x_line = np.linspace(0.0, 1.0, 300, dtype=np.float32).reshape(-1, 1)
    y_line_true = target_function(x_line)
    y_line_pred = predictor(x_line).reshape(-1)

    plotter = Plotter(n_data=2)
    plotter.scatter(x=x_train[:, 0], y=y_train[:, 0], label='Noisy data', marker_size=4, alpha=0.5, series=0)
    plotter.plot(x=x_line[:, 0], y=y_line_true, label='True function', series=0)
    plotter.plot(x=x_line[:, 0], y=y_line_pred, label='Regularized network', series=1)
    plotter.add_details(
        title='PINN-style monotonicity regularization',
        x_label='x',
        y_label='y',
        x_lim=(0.0, 1.0),
        legend_outside=True,
    )
    plotter.save_fig(output_dir / 'pinn_monotonicity_regularization.png')
    plotter.close()

    print(f'Model saved to: {model_handler.model_path}')
    print(f'Figure saved to: {output_dir / "pinn_monotonicity_regularization.png"}')


if __name__ == '__main__':
    main()
