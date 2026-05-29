"""Dataset and data-loader primitives used by :class:`DataHandler`.

The default loader is intentionally small and keeps tensors on their current
device. This is often faster for small scientific datasets than repeatedly
moving CPU batches through ``torch.utils.data.DataLoader``. A torch-backed
loader is still available for users who need its ecosystem features.
"""

import math
from collections.abc import Iterator

import numpy as np
import torch

from .. import utils

IndexLike = torch.Tensor | np.ndarray

__all__ = [
    'Dataset',
    'StructuredDataset',
    'DataLoader',
    'TorchDataLoader',
    'create_data_loader',
]


class Dataset:
    """Container for homogeneous inputs, outputs, and labels.

    Parameters
    ----------
    inputs, outputs:
        Tensors or lists with the sample axis first.
    labels:
        Metadata labels. Labels are kept as metadata and are not converted to a
        training tensor unless a downstream user chooses to do so.

    Notes
    -----
    ``to(device)`` moves tensor inputs and outputs in-place and returns this
    dataset. Labels are metadata and are intentionally left unchanged.
    """

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
    def _move_tensors_to_device(value, device: torch.device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, list):
            return [Dataset._move_tensors_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(Dataset._move_tensors_to_device(item, device) for item in value)
        if isinstance(value, dict):
            return {key: Dataset._move_tensors_to_device(item, device) for key, item in value.items()}
        return value

    def to(self, device: torch.device | str) -> 'Dataset':
        """Move tensor inputs and outputs to ``device`` in-place.

        Nested tensors inside list, tuple, or dict containers are moved
        recursively. Labels are metadata and are not moved or converted.
        """

        device = torch.device(device)
        self.inputs = self._move_tensors_to_device(self.inputs, device)
        self.outputs = self._move_tensors_to_device(self.outputs, device)
        return self

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
            indices = indices.detach().cpu()
            if indices.ndim == 0:
                return int(indices.item())
            return indices.numpy()
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
        if isinstance(indices, (int, np.integer)):
            return metadata[int(indices)]
        if isinstance(indices, np.ndarray):
            if indices.ndim == 0:
                return metadata[int(indices.item())]
            indices = indices.tolist()
        if len(indices) > 0 and type(indices[0]) is bool:
            assert len(indices) == len(metadata)
            indices = [idx for idx, use in enumerate(indices) if use]
        return [metadata[idx] for idx in indices]

    def _select(self, indices) -> 'Dataset':
        inputs, outputs, labels = self._take(indices)
        return Dataset(inputs, outputs, labels)

    def _take(self, indices):
        if type(self.inputs) is torch.Tensor:
            if isinstance(indices, slice):
                return self.inputs[indices], self.outputs[indices], self._select_metadata(self.labels, indices)
            tensor_indices = self._get_tensor_indices(indices, self.inputs.device)
            return (
                self.inputs[tensor_indices],
                self.outputs[tensor_indices],
                self._select_metadata(self.labels, tensor_indices)
            )

        if type(self.inputs) is list:
            if isinstance(indices, slice):
                indices = range(*indices.indices(self.n_data))
            elif torch.is_tensor(indices):
                indices = indices.detach().cpu().tolist()
            elif isinstance(indices, np.ndarray):
                indices = indices.tolist()
            elif isinstance(indices, (int, np.integer)):
                indices = [int(indices)]

            if len(indices) > 0 and type(indices[0]) is bool:
                assert len(indices) == self.n_data
                indices = [idx for idx, use in enumerate(indices) if use]

            return (
                [self.inputs[idx] for idx in indices],
                [self.outputs[idx] for idx in indices],
                self._select_metadata(self.labels, indices)
            )

        raise ValueError(f'self.inputs must be either of torch.Tensor or list: {type(self.inputs)} was given.')

    def _take_item(self, index: int):
        if type(self.inputs) is torch.Tensor:
            return self.inputs[index], self.outputs[index], self._select_metadata(self.labels, index)
        if type(self.inputs) is list:
            return self.inputs[index], self.outputs[index], self._select_metadata(self.labels, index)
        raise ValueError(f'self.inputs must be either of torch.Tensor or list: {type(self.inputs)} was given.')

    def random_split(self, split_ratio: tuple[float, ...]):
        """Return train, validation, and test subsets without mutating this dataset."""

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
            valid = self._empty_like()
        if n_test > 0:
            idx = slice(n_train + n_valid, None)
            test = shuffled._select(idx)
        else:
            test = self._empty_like()
        return train, valid, test

    def index_split(self, indices: IndexLike):
        """Return a subset selected by integer, boolean, NumPy, or torch indices."""

        return self._select(indices)

    def __call__(self, shuffle: bool = False):
        """Return the full dataset, optionally shuffled."""

        if shuffle:
            return self._select(self._get_permutation())()
        return self.inputs, self.outputs, self.labels

    def iter_batches(self, batch_size: int, shuffle: bool = False):
        """Yield mini-batches from this dataset."""

        if self.n_data == 0:
            return

        indices = self._get_permutation() if shuffle else None
        n_batch = math.ceil(self.n_data / batch_size)
        for i in range(n_batch):
            idx = slice(i * batch_size, (i + 1) * batch_size)
            if indices is None:
                yield self.inputs[idx], self.outputs[idx], self._select_metadata(self.labels, idx)
            else:
                yield self._take(indices[idx])

    @staticmethod
    def empty_dataset() -> 'Dataset':
        return Dataset(torch.empty([0]), torch.empty([0]), np.empty([0, 1], dtype=object))

    def _empty_like(self) -> 'Dataset':
        return self._select(slice(0, 0))


class StructuredDataset(Dataset):
    """List-backed dataset for user-defined structured samples.

    `DataHandler` intentionally targets homogeneous tensors. Use this class
    directly, or subclass it, when each sample is a custom structure or tensors
    have different shapes. Native batching returns lists; torch batching falls
    back to lists when tensors cannot be stacked.
    """

    def __init__(
            self,
            inputs: list,
            outputs: list,
            labels=None
    ):
        if type(inputs) is not list or type(outputs) is not list:
            raise TypeError('StructuredDataset expects list-backed inputs and outputs.')
        if len(inputs) != len(outputs):
            raise ValueError(f'inputs and outputs must have the same length: {len(inputs)} != {len(outputs)}.')
        if labels is None:
            labels = np.arange(len(inputs)).reshape(-1, 1)
        labels = self._normalize_labels(labels)
        if len(labels) != len(inputs):
            raise ValueError(f'labels must have the same length as inputs: {len(labels)} != {len(inputs)}.')
        super().__init__(inputs, outputs, labels)

    @staticmethod
    def _normalize_labels(labels):
        return utils.labels_to_numpy(labels)

    def _select(self, indices) -> 'StructuredDataset':
        inputs, outputs, labels = self._take(indices)
        return StructuredDataset(inputs, outputs, labels)


class DataLoader:
    """Lightweight in-memory loader for small datasets.

    The loader slices the underlying tensors directly. It does not spawn worker
    processes and does not move data between devices.
    """

    def __init__(self, dataset: Dataset, batch_size: int = None, shuffle: bool = False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.n_data = dataset.n_data
        self.batch_size = self.n_data if batch_size is None else batch_size
        self.n_batch = math.ceil(self.n_data / self.batch_size) if self.n_data > 0 else 0

    def __call__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, object]]:
        yield from self.dataset.iter_batches(batch_size=self.batch_size, shuffle=self.shuffle)


class _TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return self.dataset.n_data

    def __getitem__(self, index: int):
        return self.dataset._take_item(index)


class TorchDataLoader:
    """Adapter around ``torch.utils.data.DataLoader``.

    This is useful when a project benefits from PyTorch's standard data-loader
    behavior. Variable-shaped tensor samples are returned as lists instead of
    forcing an invalid stack operation.
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = None,
            shuffle: bool = False,
            **dataloader_kwargs
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.n_data = dataset.n_data
        self.batch_size = self.n_data if batch_size is None else batch_size
        self.n_batch = math.ceil(self.n_data / self.batch_size) if self.n_data > 0 else 0
        if self.n_data == 0:
            self.loader = None
            return
        self.loader = torch.utils.data.DataLoader(
            _TorchDataset(dataset),
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            **dataloader_kwargs
        )

    @staticmethod
    def _stack_or_list(values):
        first = values[0]
        if torch.is_tensor(first):
            try:
                return torch.stack(values, dim=0)
            except RuntimeError:
                return list(values)
        return list(values)

    @staticmethod
    def _collate(batch):
        inputs, outputs, labels = zip(*batch)
        labels = np.stack([np.asarray(label) for label in labels], axis=0)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        return (
            TorchDataLoader._stack_or_list(inputs),
            TorchDataLoader._stack_or_list(outputs),
            labels
        )

    def __call__(self):
        if self.loader is None:
            return
        yield from self.loader


def create_data_loader(
        dataset: Dataset,
        batch_size: int = None,
        shuffle: bool = False,
        dataloader_type: str = 'native',
        **dataloader_kwargs
):
    """Create either the native lightweight loader or a torch DataLoader adapter."""

    if dataloader_type == 'native':
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    if dataloader_type == 'torch':
        return TorchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **dataloader_kwargs
        )
    raise ValueError("dataloader_type must be 'native' or 'torch'.")
