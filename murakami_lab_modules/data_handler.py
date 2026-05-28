import json
from collections import Counter
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from . import utils
from . import dataset as _dataset
from .normalizer import AbstractNormalizer, StandardNormalizer

IndexLike = torch.Tensor | np.ndarray

__all__ = [
    'DataHandler',
]


class DataHandler:
    def __init__(
            self,
            input_data_path: str,
            input_idx: list[int | str] = None,
            output_idx: list[int | str] = None,
            batch_size: int = None,
            device_name: str = 'cpu',
            label_data_path: str = None,
            label_idx: list[int | str] = None,
            output_data_path: str = None,
            input_key: str = None,
            output_key: str = None,
            label_key: str = None,
            input_normalizer: AbstractNormalizer = None,
            output_normalizer: AbstractNormalizer = None,
            split_type: str = 'random_split',
            is_validation_data_batched: bool = False,
            use_train_as_valid: bool = False,
            random_seed: int = 2025,
            csv_encoding: str = None,
            dataloader_type: str = 'native',
            dataloader_kwargs: dict[str, object] = None,
            **kwargs
    ):
        self.locals = utils.get_local_dict(locals())
        self.device = utils.get_device(device_name)
        utils.initialize_random_seed(random_seed)

        self.input_data_path = input_data_path
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.label_idx = label_idx
        self.input_key = input_key
        self.output_key = output_key
        self.label_key = label_key
        self.batch_size = batch_size
        self.device_name = device_name
        self.input_normalizer = input_normalizer or StandardNormalizer()
        self.output_normalizer = output_normalizer or StandardNormalizer()
        self.split_type = split_type
        self.is_validation_data_batched = is_validation_data_batched
        self.use_train_as_valid = use_train_as_valid
        self.random_seed = random_seed
        self.csv_encoding = csv_encoding
        self.dataloader_type = dataloader_type
        self.dataloader_kwargs = dict(dataloader_kwargs or {})
        self.kwargs = kwargs

        if label_data_path is None:
            self.label_data_path = input_data_path
        else:
            self.label_data_path = label_data_path
        if output_data_path is None:
            self.output_data_path = input_data_path
        else:
            self.output_data_path = output_data_path

        self._load_datafiles()
        self._setup_datasets(split_type=split_type)

    @classmethod
    def from_tensors(
            cls,
            inputs,
            outputs,
            labels=None,
            batch_size: int = None,
            device_name: str = 'cpu',
            input_normalizer: AbstractNormalizer = None,
            output_normalizer: AbstractNormalizer = None,
            split_type: str = 'random_split',
            is_validation_data_batched: bool = False,
            use_train_as_valid: bool = False,
            random_seed: int = 2025,
            dataloader_type: str = 'native',
            dataloader_kwargs: dict[str, object] = None,
            **kwargs
    ) -> 'DataHandler':
        obj = cls.__new__(cls)
        local_vars = {key: value for key, value in locals().items() if key != 'cls'}
        obj.locals = utils.get_local_dict(local_vars)
        obj.device = utils.get_device(device_name)
        utils.initialize_random_seed(random_seed)

        obj.input_data_path = None
        obj.output_data_path = None
        obj.label_data_path = None
        obj.input_idx = None
        obj.output_idx = None
        obj.label_idx = None
        obj.input_key = None
        obj.output_key = None
        obj.label_key = None
        obj.batch_size = batch_size
        obj.device_name = device_name
        obj.input_normalizer = input_normalizer or StandardNormalizer()
        obj.output_normalizer = output_normalizer or StandardNormalizer()
        obj.split_type = split_type
        obj.is_validation_data_batched = is_validation_data_batched
        obj.use_train_as_valid = use_train_as_valid
        obj.random_seed = random_seed
        obj.csv_encoding = None
        obj.dataloader_type = dataloader_type
        obj.dataloader_kwargs = dict(dataloader_kwargs or {})
        obj.kwargs = kwargs

        obj.inputs = cls._to_feature_tensor(inputs)
        obj.outputs = cls._to_feature_tensor(outputs)
        if labels is None:
            obj.labels = np.arange(obj.inputs.shape[0]).reshape(-1, 1)
        else:
            obj.labels = cls._to_label_array(labels)
        obj._validate_data_lengths()
        obj._setup_datasets(split_type=split_type)
        return obj

    def _setup_datasets(self, split_type: str) -> None:
        self._send_to_device()
        self._get_default_dataset()
        getattr(self, f'_{split_type}')(**self.kwargs)
        self._normalize_split_data()
        self._update_datasets()
        self._get_data_loader()

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'input_data_path': self.input_data_path,
            'input_idx': self.input_idx,
            'output_idx': self.output_idx,
            'batch_size': self.batch_size,
            'device_name': self.device_name,
            'label_data_path': self.label_data_path,
            'label_idx': self.label_idx,
            'output_data_path': self.output_data_path,
            'input_key': self.input_key,
            'output_key': self.output_key,
            'label_key': self.label_key,
            'input_normalizer': self.input_normalizer.config_dict(),
            'output_normalizer': self.output_normalizer.config_dict(),
            'split_type': self.split_type,
            'is_validation_data_batched': self.is_validation_data_batched,
            'use_train_as_valid': self.use_train_as_valid,
            'random_seed': self.random_seed,
            'csv_encoding': self.csv_encoding,
            'dataloader_type': self.dataloader_type,
            'dataloader_kwargs': self.dataloader_kwargs,
            **self.kwargs
        })

    def summary_dict(self, max_label_counts: int = 20) -> dict[str, object]:
        return {
            'paths': {
                'input_data_path': self.input_data_path,
                'output_data_path': self.output_data_path,
                'label_data_path': self.label_data_path,
            },
            'columns': {
                'input_idx': self.input_idx,
                'output_idx': self.output_idx,
                'label_idx': self.label_idx,
                'input_key': self.input_key,
                'output_key': self.output_key,
                'label_key': self.label_key,
            },
            'shapes': {
                'inputs': list(self.inputs.shape),
                'outputs': list(self.outputs.shape),
                'labels': list(self.labels.shape),
            },
            'n_data': dict(self.n_data),
            'n_batch': dict(self.n_batch),
            'split': {
                'split_type': self.split_type,
                'split_params': self._json_safe(self.kwargs),
                'use_train_as_valid': self.use_train_as_valid,
                'is_validation_data_batched': self.is_validation_data_batched,
            },
            'loading': {
                'csv_encoding': self.csv_encoding,
                'device_name': self.device_name,
                'batch_size': self.batch_size,
                'random_seed': self.random_seed,
                'dataloader_type': self.dataloader_type,
                'dataloader_kwargs': self._json_safe(self.dataloader_kwargs),
            },
            'normalizers': {
                'input': self._normalizer_summary(self.input_normalizer),
                'output': self._normalizer_summary(self.output_normalizer),
            },
            'labels': self._label_summary(max_label_counts=max_label_counts),
        }

    def save_summary(self, path: str | Path, max_label_counts: int = 20) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.summary_dict(max_label_counts=max_label_counts)
        if path.suffix.lower() == '.csv':
            pd.DataFrame(self._flatten_summary(summary), columns=['key', 'value']).to_csv(path, index=False)
        else:
            utils.save_json(path, summary)

    @classmethod
    def _flatten_summary(cls, data: dict[str, object], prefix: str = '') -> list[dict[str, object]]:
        rows = []
        for key, value in data.items():
            flat_key = f'{prefix}.{key}' if prefix else str(key)
            if isinstance(value, dict):
                rows.extend(cls._flatten_summary(value, flat_key))
            else:
                rows.append({'key': flat_key, 'value': cls._summary_value_to_string(value)})
        return rows

    @staticmethod
    def _summary_value_to_string(value: object) -> object:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return json.dumps(DataHandler._json_safe(value), ensure_ascii=False)

    @staticmethod
    def _json_safe(value: object) -> object:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if torch.is_tensor(value):
            return DataHandler._tensor_summary(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return [DataHandler._json_safe(item) for item in value]
        if isinstance(value, list):
            return [DataHandler._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): DataHandler._json_safe(item) for key, item in value.items()}
        return repr(value)

    @staticmethod
    def _tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
        tensor_cpu = tensor.detach().cpu()
        summary = {
            'shape': list(tensor_cpu.shape),
            'dtype': str(tensor_cpu.dtype).replace('torch.', ''),
            'device': str(tensor.device),
        }
        if tensor_cpu.numel() == 0:
            return summary | {'empty': True}
        if not torch.is_floating_point(tensor_cpu):
            tensor_cpu = tensor_cpu.to(dtype=torch.float32)
        return summary | {
            'empty': False,
            'min': tensor_cpu.min().item(),
            'max': tensor_cpu.max().item(),
            'mean': tensor_cpu.mean().item(),
        }

    @staticmethod
    def _normalizer_summary(normalizer: AbstractNormalizer) -> dict[str, object]:
        state_summary = {}
        for key, value in normalizer.state_dict().items():
            if torch.is_tensor(value):
                state_summary[key] = DataHandler._tensor_summary(value)
            else:
                state_summary[key] = DataHandler._json_safe(value)
        return {
            'class': utils.get_object_path(normalizer.__class__),
            'state': state_summary,
        }

    def _label_summary(self, max_label_counts: int = 20) -> dict[str, object]:
        labels = np.asarray(self.labels)
        if labels.ndim == 1:
            values = labels.tolist()
        else:
            values = [
                row[0] if len(row) == 1 else tuple(row.tolist())
                for row in labels
            ]
        counter = Counter(values)
        summary = {
            'shape': list(labels.shape),
            'dtype': str(labels.dtype),
            'n_unique': len(counter),
            'counts_included': len(counter) <= max_label_counts,
            'max_label_counts': max_label_counts,
        }
        if len(counter) <= max_label_counts:
            summary['counts'] = {str(key): int(value) for key, value in counter.items()}
        return summary

    def _load_datafiles(self):
        self.inputs = self._load_datafile(self.input_data_path, self.input_idx, self.input_key, self.csv_encoding)
        self.outputs = self._load_datafile(self.output_data_path, self.output_idx, self.output_key, self.csv_encoding)
        if self.label_idx is None and self.label_key is None:
            self.labels = np.arange(self.inputs.shape[0]).reshape(-1, 1)
        else:
            self.labels = self._load_label_file(self.label_data_path, self.label_idx, self.label_key, self.csv_encoding)
        self._validate_data_lengths()

    def _validate_data_lengths(self):
        n_inputs = self.inputs.shape[0]
        n_outputs = self.outputs.shape[0]
        n_labels = len(self.labels)
        if n_inputs == n_outputs == n_labels:
            return

        raise ValueError(
            f'Inconsistent number of rows among input, output, and label data. '
            f'input_data_path={self.input_data_path}: {n_inputs}, '
            f'output_data_path={self.output_data_path}: {n_outputs}, '
            f'label_data_path={self.label_data_path}: {n_labels}.'
        )

    def _send_to_device(self):
        self.inputs = self.inputs.to(self.device)
        self.outputs = self.outputs.to(self.device)

    @staticmethod
    def _read_csv(data_path: str | Path, encoding: str = None) -> pd.DataFrame:
        if encoding is not None:
            return pd.read_csv(data_path, encoding=encoding, index_col=None)

        last_error = None
        for encoding_candidate in ('utf-8-sig', 'utf-8', 'cp932'):
            try:
                return pd.read_csv(data_path, encoding=encoding_candidate, index_col=None)
            except UnicodeDecodeError as e:
                last_error = e
        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            f'Failed to decode {data_path} as utf-8-sig, utf-8, or cp932.'
        )

    @staticmethod
    def _load_raw_file(data_path: str, csv_encoding: str = None):
        data_path = Path(data_path)
        if data_path.suffix == '.csv':
            return DataHandler._read_csv(data_path, encoding=csv_encoding)
        elif data_path.suffix in ('.pth', '.pt'):
            return torch.load(data_path, weights_only=False)
        elif data_path.suffix == '.npy':
            return np.load(data_path, allow_pickle=True)
        elif data_path.suffix == '.npz':
            return dict(np.load(data_path, allow_pickle=True))
        else:
            raise NotImplementedError(f'Unsupported data file format: {data_path}')

    @staticmethod
    def _select_key(raw_data, key: str, data_path: str):
        if isinstance(raw_data, pd.DataFrame):
            return raw_data

        if isinstance(raw_data, np.ndarray) and raw_data.shape == () and isinstance(raw_data.item(), dict):
            raw_data = raw_data.item()

        if isinstance(raw_data, dict):
            if key is None:
                if len(raw_data) == 1:
                    return next(iter(raw_data.values()))
                available_keys = ', '.join(map(str, raw_data.keys()))
                raise ValueError(f'key must be specified for dict-like data in {data_path}. Available keys: {available_keys}')
            if key not in raw_data:
                available_keys = ', '.join(map(str, raw_data.keys()))
                raise KeyError(f'{key} was not found in {data_path}. Available keys: {available_keys}')
            return raw_data[key]

        if key is not None:
            raise ValueError(f'key={key} was given, but {data_path} does not contain dict-like data.')
        return raw_data

    @staticmethod
    def _as_2d_array(data):
        if torch.is_tensor(data):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return data
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data

    @staticmethod
    def _select_columns(data, indices: list[int | str] = None):
        if isinstance(data, pd.DataFrame):
            if indices is None:
                return data.to_numpy()
            if len(indices) == 0:
                raise ValueError('indices must not be empty.')
            if type(indices[0]) is str:
                return data.loc[:, indices].to_numpy()
            return data.iloc[:, indices].to_numpy()

        data = DataHandler._as_2d_array(data)
        if indices is None:
            return data
        if len(indices) == 0:
            raise ValueError('indices must not be empty.')
        if type(indices[0]) is str:
            raise TypeError('String column indices are only supported for CSV data. Use key for dict-like data.')
        return data[:, indices]

    @staticmethod
    def _to_feature_tensor(data) -> torch.Tensor:
        if torch.is_tensor(data):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return data.to(dtype=torch.float32)
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return torch.tensor(data, dtype=torch.float32)

    @staticmethod
    def _to_label_array(data) -> np.ndarray:
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data

    @staticmethod
    def _load_datafile(
            data_path: str,
            indices: list[int | str] = None,
            key: str = None,
            csv_encoding: str = None
    ) -> torch.Tensor:
        raw_data = DataHandler._load_raw_file(data_path, csv_encoding=csv_encoding)
        selected = DataHandler._select_key(raw_data, key=key, data_path=data_path)
        selected = DataHandler._select_columns(selected, indices=indices)
        return DataHandler._to_feature_tensor(selected)

    @staticmethod
    def _load_label_file(
            data_path: str,
            indices: list[int | str] = None,
            key: str = None,
            csv_encoding: str = None
    ) -> np.ndarray:
        raw_data = DataHandler._load_raw_file(data_path, csv_encoding=csv_encoding)
        selected = DataHandler._select_key(raw_data, key=key, data_path=data_path)
        selected = DataHandler._select_columns(selected, indices=indices)
        return DataHandler._to_label_array(selected)

    def _normalize_split_data(self):
        self.input_normalizer.fit(self.train.inputs)
        self.output_normalizer.fit(self.train.outputs)

        normed_train_inputs = self.input_normalizer.transform(self.train.inputs)
        normed_train_outputs = self.output_normalizer.transform(self.train.outputs)

        self.normed_inputs = self.input_normalizer.transform(self.dataset.inputs)
        self.normed_outputs = self.output_normalizer.transform(self.dataset.outputs)

        self.dataset = _dataset.Dataset(self.normed_inputs, self.normed_outputs, self.dataset.labels)
        self.train = _dataset.Dataset(normed_train_inputs, normed_train_outputs, self.train.labels)
        self.valid = _dataset.Dataset(
            self.input_normalizer.transform(self.valid.inputs),
            self.output_normalizer.transform(self.valid.outputs),
            self.valid.labels
        )
        self.test = _dataset.Dataset(
            self.input_normalizer.transform(self.test.inputs),
            self.output_normalizer.transform(self.test.outputs),
            self.test.labels
        )
        self.datasets = {'all': self.dataset}

    def _get_default_dataset(self):
        self.dataset = _dataset.Dataset(self.inputs, self.outputs, self.labels)
        self.n_data: dict[str, int] = {'all': self.dataset.n_data}
        self.datasets: dict[str, _dataset.Dataset] = {'all': self.dataset}

    def _random_split(self, split_ratio: tuple[float, ...] = None, **_: object):
        if split_ratio is None:
            raise ValueError('split_ratio must be specified for _random_split')

        if split_ratio[0] > 0.999 and not self.use_train_as_valid:
            raise ValueError('[Warning] use_train_as_valid was set to False while no valid dataset was given.')

        self.train, self.valid, self.test = self.dataset.random_split(split_ratio)

    def _index_split(
            self,
            train_indices: IndexLike = None,
            valid_indices: IndexLike = None,
            test_indices: IndexLike = None,
            **_: object
    ):
        if train_indices is None:
            raise ValueError('indices must be specified for _index_split')

        self.train = self.dataset.index_split(train_indices)
        if valid_indices is not None:
            self.valid = self.dataset.index_split(valid_indices)
        else:
            self.valid = _dataset.Dataset.empty_dataset()
        if test_indices is not None:
            self.test = self.dataset.index_split(test_indices)
        else:
            self.test = _dataset.Dataset.empty_dataset()

    def _update_datasets(self):
        self.n_data = self.n_data | {'train': self.train.n_data, 'valid': self.valid.n_data, 'test': self.test.n_data}
        self.datasets = self.datasets | {'train': self.train, 'valid': self.valid, 'test': self.test}
        if self.use_train_as_valid:
            self.n_data['train_valid'] = self.n_data['train']

    def _get_data_loader(self):
        if self.is_validation_data_batched:
            self.data_loader = {
                'all': self._make_data_loader('all', batch_size=self.batch_size, shuffle=False),
                'train': self._make_data_loader('train', batch_size=self.batch_size, shuffle=True),
                'valid': self._make_data_loader('valid', batch_size=self.batch_size, shuffle=False),
                'test': self._make_data_loader('test', batch_size=self.batch_size, shuffle=False)
            }
        else:
            self.data_loader = {
                'all': self._make_data_loader('all', shuffle=False),
                'train': self._make_data_loader('train', batch_size=self.batch_size, shuffle=True),
                'valid': self._make_data_loader('valid', shuffle=False),
                'test': self._make_data_loader('test', shuffle=False)
            }
        if self.use_train_as_valid:
            if self.is_validation_data_batched:
                utils.logging('[Warning] Both of use_train_as_valid and is_validation_data_batched are True.'
                              'Please consider to prepare valid dataset.')
            if self.n_data['valid'] > 0:
                raise ValueError('use_train_as_valid was set to True, while valid dataset is not empty.')
            self.data_loader['train_valid'] = self._make_data_loader('train', shuffle=False)
        self.n_batch = {key: self.data_loader[key].n_batch for key in self.data_loader.keys()}

    def _make_data_loader(self, dataset_name: str, batch_size: int = None, shuffle: bool = False):
        return _dataset.create_data_loader(
            dataset=self.datasets[dataset_name],
            batch_size=batch_size,
            shuffle=shuffle,
            dataloader_type=self.dataloader_type,
            **self.dataloader_kwargs
        )

    def normalize_x(self, x: torch.Tensor):
        return self.input_normalizer.transform(x)

    def normalize_y(self, y: torch.Tensor):
        return self.output_normalizer.transform(y)

    def undo_normalize_x(self, x: torch.Tensor):
        return self.input_normalizer.inverse_transform(x)

    def undo_normalize_y(self, y: torch.Tensor):
        return self.output_normalizer.inverse_transform(y)

    def normalizer_dict(self):
        return {
            'input_normalizer': self.input_normalizer.config_dict(),
            'output_normalizer': self.output_normalizer.config_dict()
        }

    def __call__(self, dataset_name: str):
        for x, y, label in self.data_loader[dataset_name]():
            yield x, y, label

