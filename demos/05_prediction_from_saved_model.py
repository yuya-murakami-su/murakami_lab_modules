"""Prediction from a saved model demo.

This script trains a tiny regression model, reloads it with
NeuralNetworkPredictor, and compares NumPy and torch prediction calls.

Run from the project root:

    python demos/05_prediction_from_saved_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from murakami_lab_modules.data import DataHandler
from murakami_lab_modules.models import FeedForwardNeuralNetwork, NeuralNetworkPredictor
from murakami_lab_modules.training import DataFitting, ModelHandler, Optimizer


def main() -> None:
    output_dir = Path(__file__).resolve().parent / 'demo_outputs' / '05_prediction_from_saved_model'
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-1.0, 1.0, 80, dtype=np.float32).reshape(-1, 1)
    y = (0.5 * x[:, 0] ** 3 - 0.2 * x[:, 0] + 0.1).astype(np.float32).reshape(-1, 1)

    data_handler = DataHandler.from_tensors(
        inputs=x,
        outputs=y,
        split_ratio=(0.8, 0.2),
        batch_size=20,
        random_seed=2025,
    )
    model_handler = ModelHandler(
        nn=FeedForwardNeuralNetwork(
            input_dim=1,
            output_dim=1,
            n_hidden_layers=2,
            hidden_dim=24,
            activation=torch.nn.Tanh(),
            random_seed=2025,
        ),
        optimizer=Optimizer(torch.optim.Adam, lr=2e-3),
        data_fitting=DataFitting(data_handler=data_handler, loss_fn=torch.nn.MSELoss()),
        train_epochs=400,
        save_path=str(output_dir / 'models'),
        summary_path=str(output_dir / 'run_summary'),
        verbose=False,
    )
    model_handler()

    predictor = NeuralNetworkPredictor(model_path=model_handler.model_path)
    x_new_np = np.asarray([[-0.75], [0.0], [0.75]], dtype=np.float32)
    x_new_torch = torch.tensor(x_new_np)

    y_pred_np = predictor(x_new_np)
    y_pred_torch = predictor(x_new_torch).detach().cpu().numpy()

    pd.DataFrame({
        'x': x_new_np[:, 0],
        'prediction_from_numpy': y_pred_np[:, 0],
        'prediction_from_torch': y_pred_torch[:, 0],
    }).to_csv(output_dir / 'saved_model_predictions.csv', index=False)

    print(f'Model saved to: {model_handler.model_path}')
    print(f'Prediction CSV saved to: {output_dir / "saved_model_predictions.csv"}')
    print(f'Max NumPy/Torch prediction difference: {np.max(np.abs(y_pred_np - y_pred_torch)):.3e}')


if __name__ == '__main__':
    main()
