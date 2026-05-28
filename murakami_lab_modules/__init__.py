from importlib import import_module

__version__ = '0.2.3'

__all__ = [
    'callbacks',
    'data_handler',
    'data_fitting',
    'differential',
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
