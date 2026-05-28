"""Shared utility functions for configuration, logging, devices, and JSON IO."""

import random
import torch
import numpy as np
import datetime
import importlib
import inspect
import json
import logging as py_logging
from pathlib import Path
import sys

logger = py_logging.getLogger('murakami_lab_modules')
logger.addHandler(py_logging.NullHandler())


def get_logger(name: str = None) -> py_logging.Logger:
    """Return the package logger or a child logger."""

    if name is None:
        return logger
    return py_logging.getLogger(f'murakami_lab_modules.{name}')


def configure_logging(
        level: int = py_logging.INFO,
        stream: bool = True,
        log_file: str | Path = None,
        fmt: str = '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt: str = '%Y-%m-%d %H:%M:%S'
) -> py_logging.Logger:
    """Configure package logging.

    By default the package installs only a ``NullHandler``. Call this from an
    application script when console or file logs should be emitted.
    """

    logger.setLevel(level)
    formatter = py_logging.Formatter(fmt=fmt, datefmt=datefmt)
    for handler in list(logger.handlers):
        if not isinstance(handler, py_logging.NullHandler):
            logger.removeHandler(handler)
    if stream:
        stream_handler = py_logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = py_logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def initialize_random_seed(seed: int) -> None:
    """Initialize Python, NumPy, and torch random seeds."""

    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


def get_local_dict(locals_: dict[str, object]) -> dict[str, object]:
    """Return constructor locals without ``self`` and with ``kwargs`` flattened."""

    local_dict = {key: value for key, value in locals_.items() if key != 'self'}
    if 'kwargs' in local_dict.keys():
        kwargs: dict[str, object] = local_dict.pop('kwargs')
        return local_dict | kwargs
    return local_dict


def get_current_time(for_file_name: bool = False) -> str:
    """Return the current local time as a display or filename string."""

    if for_file_name:
        return datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S')
    else:
        return datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')


def logging(log: str, level: int = py_logging.INFO) -> None:
    """Log a message with the package logger.

    This compatibility helper delegates to the standard logging API.
    """

    logger.log(level, log)


def to_float(value, reduce_non_scalar: bool = True) -> float | None:
    """Convert scalars, NumPy scalars, and scalar tensors to Python floats."""

    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            if not reduce_non_scalar:
                raise ValueError(f'Expected a scalar tensor, but shape={tuple(value.shape)} was given.')
            value = value.mean()
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def is_improved(value: float, best_value: float | None, mode: str, min_delta: float) -> bool:
    """Return whether ``value`` improves on ``best_value`` under ``mode``."""

    if best_value is None:
        return True
    if mode == 'min':
        return value < best_value - min_delta
    if mode == 'max':
        return value > best_value + min_delta
    raise ValueError("mode must be 'min' or 'max'.")


def labels_to_numpy(labels) -> np.ndarray:
    """Convert labels to a two-dimensional NumPy array without forcing numeric dtype."""

    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    return labels


def resolve_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Resolve a torch dtype object or dtype name string."""

    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype_name = dtype.replace('torch.', '')
        if hasattr(torch, dtype_name):
            resolved = getattr(torch, dtype_name)
            if isinstance(resolved, torch.dtype):
                return resolved
    raise TypeError(f'dtype must be torch.dtype or torch dtype name. {dtype!r} was given.')


def is_floating_dtype(dtype: torch.dtype) -> bool:
    """Return whether a torch dtype is floating point."""

    return torch.empty((), dtype=dtype).is_floating_point()


_device_alert = True
def get_device(device_name: str) -> torch.device:
    """Return a torch device, falling back to CPU when CUDA is unavailable."""

    global _device_alert
    if 'cuda' in device_name:
        if torch.cuda.is_available():
            device = torch.device(device_name)
            if _device_alert:
                logger.info('CUDA was found.')
                _device_alert = False
        else:
            device = torch.device('cpu')
            if _device_alert:
                logger.warning('CUDA was not found. CPU will be used.')
                _device_alert = False
    else:
        device = torch.device('cpu')
        if _device_alert:
            logger.info('CPU will be used.')
            _device_alert = False
    return device


def get_object_path(obj: object) -> str:
    """Return ``module.qualname`` for an importable object."""

    return f'{obj.__module__}.{obj.__qualname__}'


def import_object(target: str) -> object:
    """Import an object from a ``module.qualname`` string."""

    module_name, object_name = target.rsplit('.', 1)
    obj = importlib.import_module(module_name)
    for attr in object_name.split('.'):
        obj = getattr(obj, attr)
    return obj


def _serialize_torch_module(module: torch.nn.Module) -> dict[str, object]:
    params = {}
    try:
        signature = inspect.signature(module.__class__.__init__)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        for name, param in signature.parameters.items():
            if name == 'self' or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if hasattr(module, name):
                params[name] = serialize_config_value(getattr(module, name))

    return {
        '__type__': 'object',
        'target': get_object_path(module.__class__),
        'params': params
    }


def serialize_config_value(value: object) -> object:
    """Convert common Python, NumPy, torch, and callable values to JSON-safe data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return {
            '__type__': 'ndarray',
            'dtype': str(value.dtype),
            'data': value.tolist()
        }
    if isinstance(value, Path):
        return {'__type__': 'path', 'value': str(value)}
    if isinstance(value, torch.device):
        return {'__type__': 'torch.device', 'value': str(value)}
    if isinstance(value, torch.dtype):
        return {'__type__': 'torch.dtype', 'value': str(value).replace('torch.', '')}
    if isinstance(value, torch.nn.Module):
        return _serialize_torch_module(value)
    if isinstance(value, tuple):
        return {'__type__': 'tuple', 'items': [serialize_config_value(item) for item in value]}
    if isinstance(value, list):
        return [serialize_config_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): serialize_config_value(item)
            for key, item in value.items()
        }
    if inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return {'__type__': 'reference', 'target': get_object_path(value)}
    if callable(value) and hasattr(value, '__class__'):
        return _serialize_torch_module(value) if isinstance(value, torch.nn.Module) else {
            '__type__': 'unserializable',
            'target': get_object_path(value.__class__),
            'repr': repr(value)
        }
    if torch.is_tensor(value):
        return {
            '__type__': 'tensor',
            'shape': list(value.shape),
            'dtype': str(value.dtype).replace('torch.', ''),
            'device': str(value.device),
            'data': value.detach().cpu().tolist()
        }
    return {
        '__type__': 'unserializable',
        'target': get_object_path(value.__class__),
        'repr': repr(value)
    }


def deserialize_config_value(value: object) -> object:
    """Reconstruct values produced by ``serialize_config_value`` when possible."""

    if isinstance(value, list):
        return [deserialize_config_value(item) for item in value]
    if not isinstance(value, dict) or '__type__' not in value:
        if isinstance(value, dict):
            return {key: deserialize_config_value(item) for key, item in value.items()}
        return value

    value_type = value['__type__']
    if value_type == 'tuple':
        return tuple(deserialize_config_value(item) for item in value['items'])
    if value_type == 'path':
        return Path(value['value'])
    if value_type == 'torch.device':
        return torch.device(value['value'])
    if value_type == 'torch.dtype':
        return getattr(torch, value['value'])
    if value_type == 'ndarray':
        return np.asarray(value['data'], dtype=value['dtype'])
    if value_type == 'tensor':
        return torch.tensor(value['data'], dtype=getattr(torch, value['dtype']))
    if value_type == 'reference':
        return import_object(value['target'])
    if value_type == 'object':
        cls = import_object(value['target'])
        params = {
            key: deserialize_config_value(item)
            for key, item in value.get('params', {}).items()
        }
        return cls(**params)
    if value_type == 'unserializable':
        return value.get('repr')
    raise TypeError(f'Cannot deserialize config value of type {value_type}.')


def make_object_config(obj: object, params: dict[str, object]) -> dict[str, object]:
    """Return a serializable object configuration with class path and params."""

    return {
        'class': get_object_path(obj.__class__),
        'params': serialize_config_value(params)
    }


def deserialize_params(params: dict[str, object]) -> dict[str, object]:
    """Deserialize every value in a parameter dictionary."""

    return {
        key: deserialize_config_value(value)
        for key, value in params.items()
    }


def save_json(path: str | Path, data: dict[str, object]) -> None:
    """Save a dictionary as UTF-8 JSON, creating parent directories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, object]:
    """Load a UTF-8 JSON file."""

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
