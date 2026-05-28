import torch
import pandas as pd
import numpy as np
from . import utils
from .normalizer import AbstractNormalizer, StandardNormalizer

IndexLike = torch.Tensor | np.ndarray


class DataHandler:
    _device_warned = False
    _std_warned = False
    _new_normalizer_warned = False

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
            classic_normalizer: bool = False,
            random_seed: int = 2025,
            csv_encoding: str = None,
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
        self.classic_normalizer = classic_normalizer
        self.random_seed = random_seed
        self.csv_encoding = csv_encoding
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
        self._send_to_device()
        if self.classic_normalizer:
            self._normalize_data()
            self._get_default_dataset()
            getattr(self, f'_{split_type}')(**self.kwargs)
        else:
            self._warn_new_normalizer()
            self._get_default_dataset(normalized=False)
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
            'classic_normalizer': self.classic_normalizer,
            'random_seed': self.random_seed,
            'csv_encoding': self.csv_encoding,
            **self.kwargs
        })

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
    def _read_csv(data_path: str, encoding: str = None) -> pd.DataFrame:
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
        if data_path.endswith('.csv'):
            return DataHandler._read_csv(data_path, encoding=csv_encoding)
        elif data_path.endswith(('.pth', '.pt')):
            return torch.load(data_path, weights_only=False)
        elif data_path.endswith('.npy'):
            return np.load(data_path, allow_pickle=True)
        elif data_path.endswith('.npz'):
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

    def _normalize_data(self):
        self.input_normalizer.fit(self.inputs)
        self.output_normalizer.fit(self.outputs)
        self.normed_inputs = self.input_normalizer.transform(self.inputs)
        self.normed_outputs = self.output_normalizer.transform(self.outputs)

    def _warn_new_normalizer(self):
        if not self.__class__._new_normalizer_warned:
            utils.logging(
                '[Warning] classic_normalizer=False is used. Normalization statistics are calculated from train data '
                'only. Set classic_normalizer=True to use the previous behavior.'
            )
            self.__class__._new_normalizer_warned = True

    def _normalize_split_data(self):
        self.input_normalizer.fit(self.train.inputs)
        self.output_normalizer.fit(self.train.outputs)

        normed_train_inputs = self.input_normalizer.transform(self.train.inputs)
        normed_train_outputs = self.output_normalizer.transform(self.train.outputs)

        self.normed_inputs = self.input_normalizer.transform(self.dataset.inputs)
        self.normed_outputs = self.output_normalizer.transform(self.dataset.outputs)

        self.dataset = Dataset(self.normed_inputs, self.normed_outputs, self.dataset.labels)
        self.train = Dataset(normed_train_inputs, normed_train_outputs, self.train.labels)
        self.valid = Dataset(
            self.input_normalizer.transform(self.valid.inputs),
            self.output_normalizer.transform(self.valid.outputs),
            self.valid.labels
        )
        self.test = Dataset(
            self.input_normalizer.transform(self.test.inputs),
            self.output_normalizer.transform(self.test.outputs),
            self.test.labels
        )
        self.datasets = {'all': self.dataset}

    def _get_default_dataset(self, normalized: bool = True):
        if normalized:
            inputs = self.normed_inputs
            outputs = self.normed_outputs
        else:
            inputs = self.inputs
            outputs = self.outputs
        self.dataset = Dataset(inputs, outputs, self.labels)
        self.n_data: dict[str, int] = {'all': self.dataset.n_data}
        self.datasets: dict[str, Dataset] = {'all': self.dataset}

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
            self.valid = Dataset.empty_dataset()
        if test_indices is not None:
            self.test = self.dataset.index_split(test_indices)
        else:
            self.test = Dataset.empty_dataset()

    def _update_datasets(self):
        self.n_data = self.n_data | {'train': self.train.n_data, 'valid': self.valid.n_data, 'test': self.test.n_data}
        self.datasets = self.datasets | {'train': self.train, 'valid': self.valid, 'test': self.test}
        if self.use_train_as_valid:
            self.n_data['train_valid'] = self.n_data['train']

    def _get_data_loader(self):
        if self.is_validation_data_batched:
            self.data_loader = {
                'all': DataLoader(self.datasets['all'], batch_size=self.batch_size, shuffle=False),
                'train': DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True),
                'valid': DataLoader(self.datasets['valid'], batch_size=self.batch_size, shuffle=False),
                'test': DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False)
            }
        else:
            self.data_loader = {
                'all': DataLoader(self.datasets['all'], shuffle=False),
                'train': DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True),
                'valid': DataLoader(self.datasets['valid'], shuffle=False),
                'test': DataLoader(self.datasets['test'], shuffle=False)
            }
        if self.use_train_as_valid:
            if self.is_validation_data_batched:
                utils.logging('[Warning] Both of use_train_as_valid and is_validation_data_batched are True.'
                              'Please consider to prepare valid dataset.')
            if self.n_data['valid'] > 0:
                raise ValueError('use_train_as_valid was set to True, while valid dataset is not empty.')
            self.data_loader['train_valid'] = DataLoader(self.datasets['train'], shuffle=False)
        self.n_batch = {key: self.data_loader[key].n_batch for key in self.data_loader.keys()}

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


class Dataset:
    def __init__(
            self,
            inputs: torch.Tensor | list,
            outputs: torch.Tensor | list,
            labels
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        if type(inputs) is torch.Tensor:
            self.n_data = inputs.shape[0]
        else:
            self.n_data = len(inputs)

    @staticmethod
    def _get_tensor_indices(indices: IndexLike | list[int] | tuple[int, ...], device: torch.device) -> torch.Tensor:
        if torch.is_tensor(indices):
            return indices.to(device=device)
        return torch.as_tensor(indices, device=device)

    def _get_permutation(self):
        if type(self.inputs) is torch.Tensor:
            return torch.randperm(self.n_data, device=self.inputs.device)
        return torch.randperm(self.n_data).tolist()

    @staticmethod
    def _indices_for_metadata(indices):
        if torch.is_tensor(indices):
            return indices.detach().cpu().numpy()
        return indices

    @staticmethod
    def _select_metadata(metadata, indices):
        indices = Dataset._indices_for_metadata(indices)
        if isinstance(metadata, np.ndarray):
            return metadata[indices]
        if torch.is_tensor(metadata):
            return metadata[indices]
        if isinstance(indices, slice):
            return metadata[indices]
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if len(indices) > 0 and type(indices[0]) is bool:
            assert len(indices) == len(metadata)
            indices = [idx for idx, use in enumerate(indices) if use]
        return [metadata[idx] for idx in indices]

    def _select(self, indices) -> 'Dataset':
        if type(self.inputs) is torch.Tensor:
            if isinstance(indices, slice):
                return Dataset(self.inputs[indices], self.outputs[indices], self._select_metadata(self.labels, indices))
            tensor_indices = self._get_tensor_indices(indices, self.inputs.device)
            return Dataset(
                self.inputs[tensor_indices],
                self.outputs[tensor_indices],
                self._select_metadata(self.labels, tensor_indices)
            )

        elif type(self.inputs) is list:
            if isinstance(indices, slice):
                indices = range(*indices.indices(self.n_data))
            elif torch.is_tensor(indices):
                indices = indices.detach().cpu().tolist()
            elif isinstance(indices, np.ndarray):
                indices = indices.tolist()

            if len(indices) > 0 and type(indices[0]) is bool:
                assert len(indices) == self.n_data
                indices = [idx for idx, use in enumerate(indices) if use]

            return Dataset(
                [self.inputs[idx] for idx in indices],
                [self.outputs[idx] for idx in indices],
                self._select_metadata(self.labels, indices)
            )

        else:
            raise ValueError(f'self.inputs must be either of torch.Tensor or list: {type(self.inputs)} was given.')

    def random_split(self, split_ratio: tuple[float, ...]):
        if len(split_ratio) == 1:
            if split_ratio[0] > 0.999:
                n_train = int(self.n_data * split_ratio[0])
                n_valid, n_test = 0, 0
            else:
                n_train = int(self.n_data * split_ratio[0])
                n_valid = self.n_data - n_train
                n_test = 0
        elif len(split_ratio) == 2:
            if split_ratio[0] + split_ratio[1] > 0.999:
                n_train = int(self.n_data * split_ratio[0])
                n_valid = self.n_data - n_train
                n_test = 0
            else:
                n_train = int(self.n_data * split_ratio[0])
                n_valid = int(self.n_data * split_ratio[1])
                n_test = self.n_data - n_train - n_valid
        else:
            n_train = int(self.n_data * split_ratio[0])
            n_valid = int(self.n_data * split_ratio[1])
            n_test = self.n_data - n_train - n_valid

        shuffled = self._select(self._get_permutation())

        train = shuffled._select(slice(None, n_train))
        if n_valid > 0:
            idx = slice(n_train, n_train + n_valid)
            valid = shuffled._select(idx)
        else:
            valid = self.empty_dataset()
        if n_test > 0:
            idx = slice(n_train + n_valid, None)
            test = shuffled._select(idx)
        else:
            test = self.empty_dataset()
        return train, valid, test

    def index_split(self, indices: IndexLike):
        return self._select(indices)

    def __call__(self, shuffle: bool = False):
        if shuffle:
            return self._select(self._get_permutation())()
        else:
            return self.inputs, self.outputs, self.labels

    @staticmethod
    def empty_dataset() -> 'Dataset':
        return Dataset(torch.empty([0]), torch.empty([0]), np.empty([0, 1], dtype=object))


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = None, shuffle: bool = False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.n_data = dataset.n_data
        if batch_size is None:
            self.batch_size = self.n_data
        else:
            self.batch_size = batch_size
        if self.n_data > 0:
            self.n_batch = (self.n_data + self.batch_size - 1) // self.batch_size
        else:
            self.n_batch = 0

    def __call__(self):
        inputs, outputs, labels = self.dataset(self.shuffle)
        for i in range(self.n_batch):
            idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
            yield inputs[idx], outputs[idx], labels[idx]
