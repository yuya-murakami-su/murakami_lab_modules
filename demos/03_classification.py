"""Multi-class classification demo.

This script trains a small classifier on three synthetic Gaussian clusters and
uses NeuralNetworkPredictor to return class probabilities and predicted labels.

Run from the project root:

    python demos/03_classification.py
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
from murakami_lab_modules.evaluation import multiclass_accuracy_from_logits
from murakami_lab_modules.models import FeedForwardNeuralNetwork, NeuralNetworkPredictor
from murakami_lab_modules.training import EarlyStopping, ModelHandler, MultiClassClassificationFitting, Optimizer
from murakami_lab_modules.visualization import Plotter


def make_classification_data(random_seed: int = 2025) -> tuple[np.ndarray, np.ndarray]:
    """Create a compact three-class dataset."""

    rng = np.random.default_rng(random_seed)
    centers = np.asarray([[-1.0, -0.2], [1.0, -0.2], [0.0, 1.0]], dtype=np.float32)
    inputs, targets = [], []
    for class_idx, center in enumerate(centers):
        points = center + rng.normal(0.0, 0.25, size=(80, 2)).astype(np.float32)
        inputs.append(points)
        targets.append(np.full(points.shape[0], class_idx, dtype=np.int64))
    return np.vstack(inputs), np.concatenate(targets)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / 'demo_outputs' / '03_classification'
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y = make_classification_data()
    data_handler = DataHandler.from_tensors(
        inputs=x,
        outputs=y,
        output_dtype=torch.long,
        split_ratio=(0.8, 0.2),
        batch_size=32,
        random_seed=2025,
    )
    data_fitting = MultiClassClassificationFitting(data_handler=data_handler)

    nn = FeedForwardNeuralNetwork(
        input_dim=2,
        output_dim=3,
        n_hidden_layers=2,
        hidden_dim=24,
        activation=torch.nn.ReLU(),
        random_seed=2025,
    )
    model_handler = ModelHandler(
        nn=nn,
        optimizer=Optimizer(torch.optim.Adam, lr=5e-3),
        data_fitting=data_fitting,
        train_epochs=300,
        callbacks=(EarlyStopping(monitor='validation_loss', patience=80),),
        save_path=str(output_dir / 'models'),
        summary_path=str(output_dir / 'run_summary'),
        verbose=False,
    )
    model_handler()

    x_all, y_all, _ = data_handler.datasets['all'](shuffle=False)
    with torch.no_grad():
        logits = data_fitting.predict(model_handler.nn, x_all)
        accuracy = multiclass_accuracy_from_logits(y_all, logits).item()

    class_predictor = NeuralNetworkPredictor(model_path=model_handler.model_path, postprocess='class')
    probability_predictor = NeuralNetworkPredictor(model_path=model_handler.model_path, postprocess='probability')
    predicted_class = class_predictor(x)
    predicted_probability = probability_predictor(x)

    pd.DataFrame({
        'x0': x[:, 0],
        'x1': x[:, 1],
        'target': y,
        'predicted_class': predicted_class,
        'p0': predicted_probability[:, 0],
        'p1': predicted_probability[:, 1],
        'p2': predicted_probability[:, 2],
    }).to_csv(output_dir / 'classification_predictions.csv', index=False)

    plotter = Plotter(n_data=3)
    for class_idx in range(3):
        mask = y == class_idx
        plotter.scatter(
            x=x[mask, 0],
            y=x[mask, 1],
            label=f'Class {class_idx}',
            marker_size=4,
            alpha=0.65,
            series=class_idx,
        )
    plotter.add_details(
        title=f'Multi-class classification (accuracy={accuracy:.3f})',
        x_label='x0',
        y_label='x1',
        legend_inside=True,
    )
    plotter.save_fig(output_dir / 'classification_data.png')
    plotter.close()

    print(f'Model saved to: {model_handler.model_path}')
    print(f'Prediction CSV saved to: {output_dir / "classification_predictions.csv"}')
    print(f'Accuracy on all normalized data: {accuracy:.3f}')


if __name__ == '__main__':
    main()
