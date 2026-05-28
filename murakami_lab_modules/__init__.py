from importlib import import_module

__version__ = '1.1.0'

__all__ = [
    'callbacks',
    'data_handler',
    'dataset',
    'data_fitting',
    'differential',
    'experiment',
    'input_generator',
    'losses',
    'model_handler',
    'model_selection',
    'neural_network',
    'normalizer',
    'optimizer',
    'plotter',
    'predictor',
    'statistics',
    'utils',
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
