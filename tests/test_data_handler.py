import numpy as np
import pandas as pd
import torch

from murakami_lab_modules.data_handler import DataHandler, Dataset
from murakami_lab_modules.normalizer import LogStandardNormalizer, StandardNormalizer


def test_dataset_random_split_does_not_mutate_original():
    inputs = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    outputs = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    labels = np.asarray(['a', 'b', 'c', 'd', 'e', 'f']).reshape(-1, 1)
    dataset = Dataset(inputs, outputs, labels)

    original_inputs = dataset.inputs.clone()
    train, valid, test = dataset.random_split((0.5, 0.25, 0.25))

    assert torch.equal(dataset.inputs, original_inputs)
    assert train.n_data == 3
    assert valid.n_data == 1
    assert test.n_data == 2
    assert train.labels.dtype.kind in {'U', 'S', 'O'}


def test_data_handler_keeps_string_labels_and_normalizes_with_exclusions(tmp_path):
    x_path = tmp_path / 'x.csv'
    y_path = tmp_path / 'y.csv'
    label_path = tmp_path / 'label.csv'
    pd.DataFrame({'id': [10.0, 20.0, 30.0, 40.0], 'value': [1.0, 2.0, 3.0, 4.0]}).to_csv(
        x_path, index=False
    )
    pd.DataFrame({'target': [2.0, 4.0, 6.0, 8.0]}).to_csv(y_path, index=False)
    pd.DataFrame({'label': ['bc', 'ic', 'data', 'data']}).to_csv(label_path, index=False)

    data_handler = DataHandler(
        input_data_path=str(x_path),
        input_idx=['id', 'value'],
        output_data_path=str(y_path),
        output_idx=['target'],
        label_data_path=str(label_path),
        label_idx=['label'],
        input_normalizer=StandardNormalizer(exclude_indices=[0]),
        batch_size=2,
        split_ratio=(0.5, 0.25, 0.25),
        device_name='cpu',
    )

    assert data_handler.labels.shape == (4, 1)
    assert data_handler.labels[0, 0] == 'bc'
    assert data_handler.n_data['train'] == 2
    assert data_handler.n_data['valid'] == 1
    assert data_handler.n_data['test'] == 1

    x = torch.tensor([[50.0, 5.0]])
    normalized = data_handler.normalize_x(x)
    restored = data_handler.undo_normalize_x(normalized)
    assert normalized[0, 0].item() == 50.0
    assert torch.allclose(restored, x)

    summary = data_handler.summary_dict()
    assert summary['shapes']['inputs'] == [4, 2]
    assert summary['n_data']['train'] == 2
    assert summary['labels']['counts']['bc'] == 1
    assert summary['labels']['counts']['data'] == 2
    assert summary['normalizers']['input']['state']['ave']['shape'] == [1, 2]

    json_path = tmp_path / 'summary.json'
    csv_path = tmp_path / 'summary.csv'
    data_handler.save_summary(json_path)
    data_handler.save_summary(csv_path)
    assert json_path.exists()
    assert csv_path.exists()
    summary_csv = pd.read_csv(csv_path)
    assert {'key', 'value'} == set(summary_csv.columns)
    assert 'n_data.train' in set(summary_csv['key'])


def test_normalizers_roundtrip():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]])
    normalizer = StandardNormalizer(exclude_indices=[0]).fit(data)
    transformed = normalizer.transform(data)
    assert torch.allclose(normalizer.inverse_transform(transformed), data)
    assert torch.allclose(transformed[:, 0], data[:, 0])

    log_data = torch.tensor([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 4.0]])
    log_normalizer = LogStandardNormalizer(exclude_indices=[0]).fit(log_data)
    log_transformed = log_normalizer.transform(log_data)
    assert torch.allclose(log_normalizer.inverse_transform(log_transformed), log_data, atol=1e-6)
    assert torch.allclose(log_transformed[:, 0], log_data[:, 0])
