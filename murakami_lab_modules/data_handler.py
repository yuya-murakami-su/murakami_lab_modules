import torch
import pandas as pd
import numpy as np
from . import utils

class DataHandler:
    _device_warned = False
    _std_warned = False

    def __init__(
            self,
            input_data_path: str,
            input_idx: list,
            output_idx: list,
            batch_size: int,
            device_name: str,
            label_data_path: str = None,
            label_idx: list = None,
            output_data_path: str = None,
            unnormalized_input_idx: list = None,
            unnormalized_output_idx: list = None,
            split_type: str = 'random_split',
            is_validation_data_batched: bool = False,
            use_train_as_valid: bool = False,
            random_seed: int = 2025,
            **kwargs
    ):
        self.locals = utils.get_local_dict(locals())
        self.device = utils.get_device(device_name)
        utils.initialize_random_seed(random_seed)

        self.input_data_path = input_data_path
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.label_idx = label_idx
        self.batch_size = batch_size
        self.device_name = device_name
        self.unnormalized_input_idx = unnormalized_input_idx
        self.unnormalized_output_idx = unnormalized_output_idx
        self.split_type = split_type
        self.is_validation_data_batched = is_validation_data_batched
        self.use_train_as_valid = use_train_as_valid
        self.random_seed = random_seed
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
        self._normalize_data()
        self._get_default_dataset()
        exec(f'self._{split_type}(**self.kwargs)')
        self._update_datasets()
        self._get_data_loader()

    def _load_datafiles(self):
        self.inputs = self._load_datafile(self.input_data_path, self.input_idx)
        self.outputs = self._load_datafile(self.output_data_path, self.output_idx)
        if self.label_idx is None:
            self.labels = torch.arange(self.inputs.shape[0]).view([-1, 1])
        else:
            self.labels = self._load_datafile(self.label_data_path, self.label_idx)

    def _send_to_device(self):
        self.inputs = self.inputs.to(self.device)
        self.outputs = self.outputs.to(self.device)
        self.labels = self.labels.to(self.device)

    @staticmethod
    def _load_datafile(data_path: str, indices: list) -> torch.Tensor:
        if data_path[-4:] == '.csv':
            data_df = pd.read_csv(data_path, encoding='cp932', index_col=None)
            if type(indices[0]) is str:
                loaded = torch.tensor(data_df.loc[:, indices].to_numpy(), dtype=torch.float32)
            else:
                loaded = torch.tensor(data_df.iloc[:, indices].to_numpy(), dtype=torch.float32)
        elif data_path[-4:] == '.pth':
            loaded = torch.load(data_path, weights_only=True).to(torch.float32)[:, indices]
        else:
            raise NotImplementedError
        return loaded

    def _normalize_data(self):
        self._input_normalizer()
        self._output_normalizer()

    def _input_normalizer(self):
        self.normed_inputs, self.input_ave, self.input_std = (
            self._default_normalizer(self.inputs, self.unnormalized_input_idx))

    def _output_normalizer(self):
        self.normed_outputs, self.output_ave, self.output_std = (
            self._default_normalizer(self.outputs, self.unnormalized_output_idx))

    def _default_normalizer(self, data: torch.Tensor, avoid_indices: list) -> (torch.Tensor, torch.Tensor):
        ave = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)

        if torch.eq(std, 0).any():
            std = torch.where(torch.eq(std, 0), 1, std)
            if not self.__class__._std_warned:
                utils.logging('STD = 0 was found!')
                self.__class__._std_warned = True

        if avoid_indices is not None:
            ave[:, avoid_indices] = 0
            std[:, avoid_indices] = 1

        normalized = (data - ave) / std

        return normalized, ave, std

    def _get_default_dataset(self):
        self.dataset = Dataset(self.normed_inputs, self.normed_outputs, self.labels)
        self.n_data: dict = {'all': self.dataset.n_data}
        self.datasets: dict = {'all': self.dataset}

    def _random_split(self, split_ratio: tuple = None, **_: dict):
        if split_ratio is None:
            raise ValueError('split_ratio must be specified for _random_split')

        self.train, self.valid, self.test = self.dataset.random_split(split_ratio)

    def _index_split(
            self,
            train_indices: torch.Tensor | np.ndarray = None,
            valid_indices: torch.Tensor | np.ndarray = None,
            test_indices: torch.Tensor | np.ndarray = None,
            **_: dict
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
        return (x - self.inputs_ave) / self.inputs_std

    def normalize_y(self, y: torch.Tensor):
        return (y - self.outputs_ave) / self.outputs_std

    def undo_normalize_x(self, x: torch.Tensor):
        return x * self.inputs_std + self.inputs_ave

    def undo_normalize_y(self, y: torch.Tensor):
        return y * self.outputs_std + self.outputs_ave

    def normalizer_dict(self):
        return {
            'inputs_ave': self.inputs_ave,
            'inputs_std': self.inputs_std,
            'outputs_ave': self.outputs_ave,
            'outputs_std': self.outputs_std
        }

    def __call__(self, dataset_name: str):
        for x, y, label in self.data_loader[dataset_name]():
            yield x, y, label


class Dataset:
    def __init__(
            self,
            inputs: torch.Tensor | list,
            outputs: torch.Tensor | list,
            labels: torch.Tensor | list
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        if type(inputs) is torch.Tensor:
            self.n_data = inputs.shape[0]
        else:
            self.n_data = len(inputs)

    def random_split(self, split_ratio: tuple):
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

        idx = np.arange(self.n_data)
        np.random.shuffle(idx)
        self.inputs = self.inputs[idx]
        self.outputs = self.outputs[idx]
        self.labels = self.labels[idx]

        train = Dataset(self.inputs[:n_train], self.outputs[:n_train], self.labels[:n_train])
        if n_valid > 0:
            idx = slice(n_train, n_train + n_valid)
            valid = Dataset(self.inputs[idx], self.outputs[idx], self.labels[idx])
        else:
            valid = self.empty_dataset()
        if n_test > 0:
            idx = slice(n_train + n_valid, None)
            test = Dataset(self.inputs[idx], self.outputs[idx], self.labels[idx])
        else:
            test = self.empty_dataset()
        return train, valid, test

    def index_split(self, indices: np.ndarray | torch.Tensor):
        if type(self.inputs) is torch.Tensor:
            return Dataset(self.inputs[indices], self.outputs[indices], self.labels[indices])

        elif type(self.inputs) is list:
            inputs, outputs, labels = [], [], []
            if type(indices[0]) is bool:
                assert len(indices) == self.n_data
                for idx in range(self.n_data):
                    if indices[idx]:
                        inputs.append(self.inputs[idx])
                        outputs.append(self.outputs[idx])
                        labels.append(self.labels[idx])
            else:
                for idx in indices:
                    inputs.append(self.inputs[idx])
                    outputs.append(self.outputs[idx])
                    labels.append(self.labels[idx])
            return Dataset(inputs, outputs, labels)

        else:
            raise ValueError(f'self.inputs must be either of torch.Tensor or list: {type(self.inputs)} was given.')

    def __call__(self, shuffle: bool = False):
        if shuffle:
            idx = np.arange(self.n_data)
            np.random.shuffle(idx)
            if type(self.inputs) is torch.Tensor:
                return self.inputs[idx], self.outputs[idx], self.labels[idx]
            else:
                inputs = [self.inputs[i] for i in idx]
                outputs = [self.outputs[i] for i in idx]
                labels = [self.labels[i] for i in idx]
                return inputs, outputs, labels
        else:
            return self.inputs, self.outputs, self.labels

    @staticmethod
    def empty_dataset():
        return Dataset(torch.empty([0]), torch.empty([0]), torch.empty([0]))


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