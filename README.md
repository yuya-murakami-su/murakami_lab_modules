# murakami_lab_modules

`murakami_lab_modules` is a PyTorch-based toolkit for small-data neural-network fitting and
PINN-style regularization.

The package is intentionally lightweight. It provides the training, data handling, automatic
differentiation, regularization, logging, plotting, and model-selection utilities that are commonly
needed in laboratory-scale regression and physics-informed modeling workflows, without wrapping the
entire PyTorch ecosystem.

## Features

- Tabular and tensor data loading from CSV, NumPy, and PyTorch files
- Train, validation, and test split management with reproducible indexing
- Input and output normalization through replaceable normalizer classes
- Simple feed-forward neural networks for regression, classification, and ODE-style call signatures
- A compact training loop with callbacks, checkpointing, summaries, and optional history saving
- PINN-oriented derivative helpers such as `partial`, `partial2`, `jacobian`, `hessian_diag`, and
  `laplacian`
- Regularization terms with static or automatically calibrated weights
- Native lightweight batching by default, with optional `torch.utils.data.DataLoader`
- Cross-validation, nested cross-validation, grid search, and random search
- Plotting helpers for line, scatter, parity, loss, and contour plots

## Installation

Python 3.11 or newer is required.

```bash
pip install git+https://github.com/yuya-murakami-su/murakami_lab_modules.git
```

PyTorch is a required dependency. If your environment needs a specific CUDA build, install the
PyTorch wheel recommended for that machine first, then install this package.

Optional plotting dependencies:

```bash
pip install "murakami_lab_modules[plot] @ git+https://github.com/yuya-murakami-su/murakami_lab_modules.git"
```

For local development:

```bash
git clone https://github.com/yuya-murakami-su/murakami_lab_modules.git
cd murakami_lab_modules
pip install -e ".[dev]"
python -m pytest
```

## Basic Training

The core workflow is:

1. Load and split data with `DataHandler`.
2. Wrap the data loss with `DataFitting`.
3. Define a neural network and optimizer.
4. Run training with `ModelHandler`.

```python
import torch

from murakami_lab_modules.data import DataHandler
from murakami_lab_modules.models import FeedForwardNeuralNetwork
from murakami_lab_modules.training import DataFitting, ModelHandler, Optimizer

data_handler = DataHandler(
    input_data_path="data.csv",
    input_columns=["x1", "x2"],
    output_columns=["y"],
    split_ratio=(0.8, 0.2),
    batch_size=32,
    device_name="cpu",
)

nn = FeedForwardNeuralNetwork(
    input_dim=2,
    output_dim=1,
    n_hidden_layers=2,
    hidden_dim=64,
    activation=torch.nn.Tanh(),
)

model_handler = ModelHandler(
    nn=nn,
    optimizer=Optimizer(torch.optim.Adam, lr=1e-3),
    data_fitting=DataFitting(data_handler, loss_fn=torch.nn.MSELoss()),
    train_epochs=1000,
)

model_handler()
```

Training creates a timestamped folder under `Model/` by default. The folder contains configuration
metadata, data summaries, training history, and model state dicts when saving is enabled.

## Core Concepts

### Data

`DataHandler` owns data loading, split creation, normalization, and batching.

Supported file formats:

- `.csv`
- `.npy`
- `.npz`
- `.pt`
- `.pth`

CSV encoding is detected from common encodings by default. Use `csv_encoding=` when the file has a
known encoding.

```python
from murakami_lab_modules.data import DataHandler

data_handler = DataHandler(
    input_data_path="measurements.csv",
    input_columns=["temperature", "pressure"],
    output_columns=["rate"],
    label_columns=["sample_id"],
    split_type="index_split",
    train_indices=[0, 1, 2, 3],
    valid_indices=[4, 5],
    test_indices=[6, 7],
)
```

For in-memory arrays or tensors:

```python
data_handler = DataHandler.from_tensors(
    inputs=x,
    outputs=y,
    labels=sample_ids,
    split_ratio=(0.8, 0.2),
    batch_size=64,
)
```

`DataHandler` targets homogeneous tensors with the sample axis first, for example `[N, features]`,
`[N, channels, length]`, or `[N, channels, height, width]`. For variable-shaped samples or multiple
heterogeneous inputs, use `StructuredDataset` directly or subclass it.

### Normalization

Normalization is implemented through normalizer classes rather than through hard-coded index flags.
The default is standardization for floating-point inputs and outputs.

```python
from murakami_lab_modules.data import IdentityNormalizer, LogStandardNormalizer

data_handler = DataHandler.from_tensors(
    inputs=x,
    outputs=y,
    input_normalizer=LogStandardNormalizer(epsilon=1e-8),
    output_normalizer=IdentityNormalizer(),
)
```

Available normalizers:

- `IdentityNormalizer`
- `StandardNormalizer`
- `LogStandardNormalizer`

Custom normalizers can subclass `BaseNormalizer`.

### Models

`FeedForwardNeuralNetwork` is a small fully connected network suitable for most tabular regression
and PINN examples.

```python
from murakami_lab_modules.models import FeedForwardNeuralNetwork

nn = FeedForwardNeuralNetwork(
    input_dim=3,
    output_dim=2,
    n_hidden_layers=3,
    hidden_dim=128,
    dropout=0.1,
    batch_norm=False,
)
```

The network is a regular `torch.nn.Module`, so users can also provide their own PyTorch modules to
`ModelHandler` as long as the module can be called with `nn(x=x)` or `nn(x)`.

### Data Fitting

`DataFitting` computes the data loss. It can be subclassed when prediction requires labels,
time integration, a custom forward path, or multiple loss terms.

```python
from murakami_lab_modules.training import DataFitting

data_fitting = DataFitting(data_handler, loss_fn=torch.nn.MSELoss())
```

Classification is supported through explicit fitting classes:

```python
from murakami_lab_modules.training import BinaryClassificationFitting, MultiClassClassificationFitting

multi_class_fitting = MultiClassClassificationFitting(data_handler)
binary_fitting = BinaryClassificationFitting(data_handler)
```

For multi-class classification, use integer class-index targets and set `output_dtype=torch.long`.
For binary classification with logits, use float targets and disable output normalization.

#### Latent Output Models

Some scientific models should not predict the observed output directly. For example, reaction-rate
or mixture-property models often use a known physical factor and a learned correction:

```text
y = x1 * x2 * N(x)
```

Use `LatentOutputFitting` for this pattern. The neural network predicts the normalized latent
quantity `N(x)`, while the loss is computed in the observed `y` space.

```python
from murakami_lab_modules.training import (
    InputProductOutputTransform,
    LatentOutputFitting,
)

data_fitting = LatentOutputFitting(
    data_handler=data_handler,
    output_transform=InputProductOutputTransform(input_indices=[0, 1]),
)
```

The transform is responsible for:

- `to_latent(x, y)`: build latent training targets from raw inputs and outputs
- `to_observed(x, z)`: map latent predictions back to observed outputs

`BaseNormalizer` is still used for scale conversion. In this workflow it normalizes the latent
quantity `N(x)`, not the full physical expression. Subclass `BaseOutputTransform` when the physical
relationship is more complex than an input product.

### Optimizers

`Optimizer` combines a PyTorch optimizer class with optional learning-rate schedules.

```python
from murakami_lab_modules.training import Optimizer, cosine_annealing_lr, step_decay_lr

optimizer = Optimizer(torch.optim.Adam, lr=1e-3, weight_decay=1e-6)

scheduled_optimizer = Optimizer(
    torch.optim.Adam,
    lr_schedule=cosine_annealing_lr(initial_lr=1e-3, total_epochs=10_000),
)
```

Available schedule factories:

- `constant_lr`
- `linear_warmup_lr`
- `warmup_decay_lr`
- `inverse_time_decay_lr`
- `exponential_decay_lr`
- `step_decay_lr`
- `cosine_annealing_lr`
- `polynomial_decay_lr`

### Training

`ModelHandler` coordinates the model, data fitting, optional regularization, callbacks, summaries,
and saved outputs.

Important saving options:

- `save_result=False`: do not create result folders or summary files
- `save_model=False`: save metadata and history, but skip model state dicts
- `save_history=False`: skip `evolution.csv`
- `history_policy="sparse"` with `history_every=N`: keep only sparse history rows
- `verbose=False`: disable progress output

```python
model_handler = ModelHandler(
    nn=nn,
    optimizer=optimizer,
    data_fitting=data_fitting,
    train_epochs=500,
    save_model=False,
    history_policy="sparse",
    history_every=10,
    verbose=False,
)
model_handler()
```

## PINN Regularization

PINN-style training is built from three pieces:

1. `InputGenerator` creates collocation points.
2. Derivative helpers compute required differential terms.
3. `Regularization` combines one or more residual tensors into a scalar loss.

```python
import torch

from murakami_lab_modules.pinn import InputGenerator, Regularization, TargetTotalRegularizationWeight


class HeatEquationRegularization(Regularization):
    def regularization(self, data_handler, nn):
        x = self.input_generators[0]()
        y = nn(x=x)
        _, d2y_dx2 = self.partial2(y, x, x_idx=0, y_idx=0)
        return [d2y_dx2]


input_generator = InputGenerator(
    n_samples=256,
    input_range=((0.0, 1.0),),
    sampling="sobol",
    requires_grad=True,
)

regularization = HeatEquationRegularization(
    input_generators=[input_generator],
    term_names=["heat_residual"],
    weight_policy=TargetTotalRegularizationWeight(target_total=1.0),
)

model_handler = ModelHandler(
    nn=nn,
    optimizer=optimizer,
    data_fitting=data_fitting,
    regularization=regularization,
    train_epochs=1000,
)
```

Derivative helpers are available from `murakami_lab_modules.pinn`:

- `grad`
- `partial`
- `partial2`
- `jacobian`
- `hessian_diag`
- `laplacian`

By default, unused gradients raise an error. Use `zero_if_unused=True` only when independence from
the differentiated input is intentional.

## Callbacks

The training loop uses callbacks for monitoring, stopping, logging, plotting, and checkpointing.
Each callback owns its execution interval through `every`. Lower `priority` values run earlier.

```python
from murakami_lab_modules.training import (
    EarlyStopping,
    GradientNormMonitor,
    LearningRateLogger,
    LossPlotSaver,
    PredictionResultSaver,
)

callbacks = (
    EarlyStopping(monitor="validation_loss", patience=100),
    LearningRateLogger(every=1),
    GradientNormMonitor(every=10),
    LossPlotSaver(every=100),
    PredictionResultSaver(every=500),
)
```

Core callbacks for best-model tracking, history recording, final checkpoints, regularization
reports, run summaries, and console progress are installed by `ModelHandler` automatically when the
corresponding `ModelHandler` options are enabled.

## Saved Outputs and Prediction

A saved model directory typically contains:

- `config.json`
- `metadata/`
- `metadata/data_summary.json`
- `metadata/data_summary.csv`
- `evolution.csv`
- `run_summary.csv`
- `regularization_weight_report.csv` when regularization is used
- `state_dicts.pth` and `normalizer.pth` when `save_model=True`

Use `NeuralNetworkPredictor` for inference from a saved model directory:

```python
from murakami_lab_modules.models import NeuralNetworkPredictor

predictor = NeuralNetworkPredictor(model_path="Model/260528-120000-123")
y_pred = predictor(x_new)
```

For classifiers:

```python
probability_predictor = NeuralNetworkPredictor(model_path=model_path, postprocess="probability")
class_predictor = NeuralNetworkPredictor(model_path=model_path, postprocess="class")
```

`.pth` data and model files should be loaded only from trusted sources.

## Cross-Validation and Search

Model selection utilities use factory functions so each fold or trial receives a fresh model,
optimizer, and data handler.

```python
from murakami_lab_modules.evaluation import CrossValidator, KFoldSplitter


def model_factory(params, split, context):
    data_handler = DataHandler(
        input_data_path="data.csv",
        input_columns=["x1", "x2"],
        output_columns=["y"],
        split_type="index_split",
        train_indices=split.train_indices,
        valid_indices=split.valid_indices,
        batch_size=params["batch_size"],
    )
    nn = FeedForwardNeuralNetwork(
        input_dim=2,
        output_dim=1,
        n_hidden_layers=2,
        hidden_dim=params["hidden_dim"],
    )
    return ModelHandler(
        nn=nn,
        optimizer=Optimizer(torch.optim.Adam, lr=params["lr"]),
        data_fitting=DataFitting(data_handler),
        train_epochs=200,
        save_result=False,
        verbose=False,
    )


cv = CrossValidator(
    model_factory=model_factory,
    splitter=KFoldSplitter(n_splits=5),
    indices=100,
)

results = cv.run(params={"lr": 1e-3, "batch_size": 32, "hidden_dim": 64})
```

`GridSearch`, `RandomSearch`, and `NestedCrossValidator` are available from
`murakami_lab_modules.evaluation`.

## Visualization

Plotting is optional and requires the `plot` extra.

```python
from murakami_lab_modules.visualization import Plotter

plotter = Plotter()
plotter.scatter(x=x, y=y, label="data")
plotter.plot(x=x_line, y=y_line, label="model")
plotter.add_details(x_label="x", y_label="y", legend_inside=True)
plotter.save_fig("fit.png")
```

Contour plots use restrained colormap aliases intended for research figures:

```python
plotter = Plotter()
plotter.contourf(x=x_grid, y=y_grid, z=z_grid, cmap="blue_white_red", colorbar_label="value")
plotter.contour(x=x_grid, y=y_grid, z=z_grid, levels=10, label=True)
plotter.save_fig("contour.png")
```

Built-in aliases:

- `blue_white_red`
- `red_white_blue`
- `white_orange`
- `white_blue`

## Package Layout

The public API is grouped by purpose:

- `murakami_lab_modules.data`: datasets, data loading, data loaders, normalizers
- `murakami_lab_modules.models`: neural networks and saved-model prediction
- `murakami_lab_modules.training`: fitting, optimization, training loop, callbacks
- `murakami_lab_modules.pinn`: automatic differentiation, input generation, regularization
- `murakami_lab_modules.evaluation`: losses, metrics, cross-validation, search
- `murakami_lab_modules.visualization`: plotting helpers
- `murakami_lab_modules.utils`: serialization, logging, device, and small utility functions

Common classes are also exported lazily from the top-level package:

```python
from murakami_lab_modules import DataHandler, ModelHandler, Regularization
```

## Logging

The package uses Python's standard `logging` module. Console progress is controlled by
`ModelHandler(verbose=...)`. Persistent logs are not created unless configured by the user.

```python
from murakami_lab_modules import utils

utils.configure_logging(log_file="training.log")
```

## Development

Run the test suite before publishing changes:

```bash
python -m pytest
```

The project favors small, explicit abstractions over large framework wrappers. When extending the
library, prefer adding focused modules around stable concepts such as data, training, PINN,
evaluation, visualization, or future analysis utilities.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).

## Author

Yuya Murakami, Shizuoka University, Japan
