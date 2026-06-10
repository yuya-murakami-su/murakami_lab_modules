"""Public package interface for :mod:`murakami_lab_modules`.

The package is organized by workflow area:

``data``
    Data loading, splitting, batching, labels, and normalization.
``models``
    Built-in neural networks and saved-model prediction.
``training``
    Data fitting, optimizers, callbacks, and the main training loop.
``pinn``
    PINN-oriented input generation, automatic differentiation, and regularization.
``evaluation``
    Losses, metrics, cross-validation, and hyper-parameter search.
``visualization``
    Matplotlib-based plotting helpers.

Common classes are lazily re-exported at the top level so simple scripts can use
``from murakami_lab_modules import DataHandler, ModelHandler`` without importing
matplotlib or other optional modules until they are actually needed.
"""

from importlib import import_module

__version__ = '1.1.3'

_SUBMODULES = {
    'data',
    'evaluation',
    'experiment',
    'models',
    'pinn',
    'training',
    'utils',
    'visualization',
}

_OBJECTS = {
    'DataHandler': 'murakami_lab_modules.data',
    'Dataset': 'murakami_lab_modules.data',
    'StructuredDataset': 'murakami_lab_modules.data',
    'DataLoader': 'murakami_lab_modules.data',
    'TorchDataLoader': 'murakami_lab_modules.data',
    'create_data_loader': 'murakami_lab_modules.data',
    'BaseNormalizer': 'murakami_lab_modules.data',
    'IdentityNormalizer': 'murakami_lab_modules.data',
    'StandardNormalizer': 'murakami_lab_modules.data',
    'LogStandardNormalizer': 'murakami_lab_modules.data',
    'DataFitting': 'murakami_lab_modules.training',
    'LatentOutputFitting': 'murakami_lab_modules.training',
    'MultiClassClassificationFitting': 'murakami_lab_modules.training',
    'BinaryClassificationFitting': 'murakami_lab_modules.training',
    'BaseOutputTransform': 'murakami_lab_modules.training',
    'IdentityOutputTransform': 'murakami_lab_modules.training',
    'InputProductOutputTransform': 'murakami_lab_modules.training',
    'ModelHandler': 'murakami_lab_modules.training',
    'Optimizer': 'murakami_lab_modules.training',
    'constant_lr': 'murakami_lab_modules.training',
    'linear_warmup_lr': 'murakami_lab_modules.training',
    'warmup_decay_lr': 'murakami_lab_modules.training',
    'inverse_time_decay_lr': 'murakami_lab_modules.training',
    'exponential_decay_lr': 'murakami_lab_modules.training',
    'step_decay_lr': 'murakami_lab_modules.training',
    'cosine_annealing_lr': 'murakami_lab_modules.training',
    'polynomial_decay_lr': 'murakami_lab_modules.training',
    'Callback': 'murakami_lab_modules.training',
    'EarlyStopping': 'murakami_lab_modules.training',
    'LossMonitor': 'murakami_lab_modules.training',
    'PredictionResultSaver': 'murakami_lab_modules.training',
    'PeriodicCheckpointSaver': 'murakami_lab_modules.training',
    'BaseNeuralNetwork': 'murakami_lab_modules.models',
    'FeedForwardNeuralNetwork': 'murakami_lab_modules.models',
    'ODEFeedForwardNeuralNetwork': 'murakami_lab_modules.models',
    'NeuralNetworkPredictor': 'murakami_lab_modules.models',
    'InputGenerator': 'murakami_lab_modules.pinn',
    'Regularization': 'murakami_lab_modules.pinn',
    'RegularizationWeightPolicy': 'murakami_lab_modules.pinn',
    'StaticRegularizationWeights': 'murakami_lab_modules.pinn',
    'TargetTotalRegularizationWeight': 'murakami_lab_modules.pinn',
    'MatchDataLossRegularizationWeight': 'murakami_lab_modules.pinn',
    'grad': 'murakami_lab_modules.pinn',
    'partial': 'murakami_lab_modules.pinn',
    'partial2': 'murakami_lab_modules.pinn',
    'jacobian': 'murakami_lab_modules.pinn',
    'hessian_diag': 'murakami_lab_modules.pinn',
    'laplacian': 'murakami_lab_modules.pinn',
    'relative_mse_loss': 'murakami_lab_modules.evaluation',
    'component_weighted_mse_loss': 'murakami_lab_modules.evaluation',
    'relative_error': 'murakami_lab_modules.evaluation',
    'multiclass_accuracy_from_logits': 'murakami_lab_modules.evaluation',
    'binary_accuracy_from_logits': 'murakami_lab_modules.evaluation',
    'Metric': 'murakami_lab_modules.evaluation',
    'KFoldSplitter': 'murakami_lab_modules.evaluation',
    'CrossValidator': 'murakami_lab_modules.evaluation',
    'GridSearch': 'murakami_lab_modules.evaluation',
    'RandomSearch': 'murakami_lab_modules.evaluation',
    'NestedCrossValidator': 'murakami_lab_modules.evaluation',
    'Plotter': 'murakami_lab_modules.visualization',
    'plot_histogram': 'murakami_lab_modules.visualization',
}

_OPTIONAL_EXPORTS = {
    'visualization',
    'Plotter',
    'plot_histogram',
}

__all__ = sorted((_SUBMODULES | set(_OBJECTS)) - _OPTIONAL_EXPORTS)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    if name in _OBJECTS:
        module = import_module(_OBJECTS[name])
        obj = getattr(module, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
