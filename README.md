# murakami_lab_modules

`murakami_lab_modules` is a small PyTorch-based toolkit for neural-network fitting and PINN-style regularization.
It includes utilities for data loading, normalization, training loops, regularization terms, automatic differentiation,
plotting callbacks, prediction, and cross-validation / hyper-parameter search.

## Installation

For a standard CPU-oriented Python environment:

```bash
pip install git+https://github.com/yuya-murakami-su/murakami_lab_modules.git
```

PyTorch is a core dependency. If you need a specific CUDA build, install the PyTorch wheel recommended for your
environment first, then install this package. Pip will keep the already installed compatible PyTorch package.

Optional plotting and statistics dependencies:

```bash
pip install "murakami_lab_modules[plot]"
pip install "murakami_lab_modules[statistics]"
pip install "murakami_lab_modules[all]"
```

For local development:

```bash
pip install -e ".[test,plot,statistics]"
python -m pytest
```

## Minimal Training Example

```python
import torch

from murakami_lab_modules.data_fitting import DataFitting
from murakami_lab_modules.data_handler import DataHandler
from murakami_lab_modules.model_handler import ModelHandler
from murakami_lab_modules.neural_network import FeedForwardNeuralNetwork
from murakami_lab_modules.optimizer import ConstantLROptimizer

data_handler = DataHandler(
    input_data_path="x.csv",
    input_idx=["x1", "x2"],
    output_data_path="y.csv",
    output_idx=["y"],
    split_ratio=(0.8, 0.2),
    batch_size=32,
)

model_handler = ModelHandler(
    nn=FeedForwardNeuralNetwork(n_input=2, n_output=1, n_layer=2, n_node=64),
    optimizer=ConstantLROptimizer(torch.optim.Adam, lr=1e-3),
    data_fitting=DataFitting(data_handler, loss_criteria=torch.nn.MSELoss()),
    train_epochs=1000,
)

model_handler()
```

## Data Shapes

`DataHandler` is designed for tabular data and homogeneous tensors with a sample axis first. Tensor data can be
2D or higher dimensional, for example `[N, features]`, `[N, channels, length]`, or `[N, channels, height, width]`.
Standard normalization computes statistics over the sample axis and preserves the remaining shape.

For variable-shaped samples or multiple heterogeneous inputs, use `StructuredDataset` directly or subclass it. In that
case batches are list-backed when tensors cannot be stacked, and the model/loss code should define how those structures
are consumed.

## PINN-Style Regularization

Define a subclass of `Regularization` and return one tensor per regularization term.

```python
from murakami_lab_modules.input_generator import InputGenerator
from murakami_lab_modules.regularization import Regularization, TargetTotalRegWeight


class MyRegularization(Regularization):
    def regularization(self, data_handler, nn):
        x = self.input_generators[0]()
        y = nn(x=x)
        return [y]


regularization = MyRegularization(
    input_generators=[
        InputGenerator(
            size_of_generated_inputs=256,
            input_range=((0.0, 1.0), (0.0, 1.0)),
            sampling="sobol",
        )
    ],
    reg_names=["output_penalty"],
    reg_weight_policy=TargetTotalRegWeight(target_total=1.0),
)
```

## Callbacks

Callbacks own their own execution interval through `every`.

```python
from murakami_lab_modules.callbacks import LossMonitor, SavePredictionResults, StateDictsSaver

callbacks = (
    LossMonitor(every=10),
    SavePredictionResults(every=100),
    StateDictsSaver(every=1000),
)
```

## Optimizers and Learning-Rate Schedules

Optimizer wrappers combine a PyTorch optimizer class with a learning-rate schedule.

```python
from murakami_lab_modules.optimizer import (
    ConstantLROptimizer,
    ExponentialDecayOptimizer,
    StepDecayOptimizer,
    CosineAnnealingOptimizer,
    WarmupDecayOptimizer,
)

optimizer = ConstantLROptimizer(torch.optim.Adam, lr=1e-3)
optimizer = StepDecayOptimizer(torch.optim.Adam, initial_lr=1e-3, step_size=1000, gamma=0.5)
optimizer = CosineAnnealingOptimizer(torch.optim.Adam, initial_lr=1e-3, total_epochs=10_000, min_lr=1e-5)
```

Available schedules:

- `ConstantLROptimizer`
- `WarmupOptimizer`
- `WarmupDecayOptimizer`
- `InverseTimeDecayOptimizer`
- `ExponentialDecayOptimizer`
- `StepDecayOptimizer`
- `CosineAnnealingOptimizer`
- `PolynomialDecayOptimizer`

## Cross-Validation and Hyper-Parameter Search

`model_selection` intentionally uses a factory function so every fold/trial receives a fresh model, optimizer, and
data handler.

```python
from murakami_lab_modules.model_selection import CrossValidator, KFoldSplitter


def model_factory(params, split, context):
    data_handler = DataHandler(
        input_data_path="x.csv",
        input_idx=["x1", "x2"],
        output_data_path="y.csv",
        output_idx=["y"],
        split_type="index_split",
        train_indices=split.train_indices,
        valid_indices=split.valid_indices,
        batch_size=params["batch_size"],
    )
    return ModelHandler(
        nn=FeedForwardNeuralNetwork(n_input=2, n_output=1, n_layer=1, n_node=params["n_node"]),
        optimizer=ConstantLROptimizer(torch.optim.Adam, lr=params["lr"]),
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
results = cv.run(params={"lr": 1e-3, "batch_size": 32, "n_node": 64})
```

## Notes on Saved Models

Saved model folders contain:

- `config.json` and per-component metadata JSON files
- `run_summary.csv`
- `evolution.csv` when history saving is enabled
- lightweight data summaries
- `state_dicts.pth` and `normalizer.pth` when `save_model=True`

History storage can be reduced for long runs or cross-validation with `history_policy='sparse'`, `'last'`, or `'none'`.
Use `save_history=False` to skip `evolution.csv`, or add `HistoryLogger`/`CSVLogger` callbacks for custom CSV output.
Set `evaluate_test=True` on `ModelHandler` when `run_summary.csv` should include a final test loss.

The library uses Python's standard `logging` package. By default it does not create persistent log files. Configure
logging in your script when you want important messages to remain in the console or in a file:

```python
from murakami_lab_modules import utils

utils.configure_logging(log_file="training.log")
```

Configuration files are intended as audit metadata. Reproducible training should be based on your script plus the saved
state dicts and metadata.

`.pth` data loading should be used only with files you trust. Model state dicts and normalizer files written by this
library are loaded with PyTorch's restricted weights-only loader where possible.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).

## Author

Yuya Murakami, Shizuoka University, Japan
