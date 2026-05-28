import pandas as pd
import torch

from murakami_lab_modules.regularization import (
    MatchDataLossRegWeight,
    Regularization,
    TargetTotalRegWeight,
)


class DummyInputGenerator:
    device = torch.device('cpu')
    device_name = 'cpu'

    def config_dict(self):
        return {'class': 'DummyInputGenerator', 'params': {}}


class ConstantRegularization(Regularization):
    def regularization(self, data_handler, nn):
        return [
            torch.full((2, 1), 2.0),
            torch.full((2, 1), 4.0),
        ]


def test_target_total_reg_weight_calibrates_to_requested_total():
    regularization = ConstantRegularization(
        input_generators=[DummyInputGenerator()],
        reg_weight_policy=TargetTotalRegWeight(target_total=10.0),
        reg_names=['small', 'large'],
    )

    report = regularization.calibrate_weights(nn=None)
    weighted_terms, total = regularization.get_regularization_value(nn=None)

    assert torch.allclose(regularization.reg_weights, torch.tensor([1.25, 0.3125]))
    assert torch.allclose(weighted_terms, torch.tensor([5.0, 5.0]))
    assert torch.allclose(total, torch.tensor(10.0))
    assert list(report['name']) == ['small', 'large']
    assert list(report['raw_mean']) == [4.0, 16.0]


def test_target_total_reg_weight_supports_relative_factors():
    regularization = ConstantRegularization(
        input_generators=[DummyInputGenerator()],
        reg_weight_policy=TargetTotalRegWeight(target_total=10.0, factors=[1.0, 3.0]),
        reg_names=['small', 'large'],
    )

    regularization.calibrate_weights(nn=None)
    weighted_terms, total = regularization.get_regularization_value(nn=None)

    assert torch.allclose(weighted_terms, torch.tensor([2.5, 7.5]))
    assert torch.allclose(total, torch.tensor(10.0))


def test_match_data_loss_reg_weight_uses_data_loss_times_alpha():
    regularization = ConstantRegularization(
        input_generators=[DummyInputGenerator()],
        reg_weight_policy=MatchDataLossRegWeight(alpha=2.0),
        reg_names=['small', 'large'],
    )

    report = regularization.calibrate_weights(nn=None, data_loss=3.0)
    weighted_terms, total = regularization.get_regularization_value(nn=None)

    assert torch.allclose(weighted_terms, torch.tensor([3.0, 3.0]))
    assert torch.allclose(total, torch.tensor(6.0))
    assert list(report['data_loss']) == [3.0, 3.0]


def test_match_data_loss_reg_weight_can_auto_calibrate_on_first_value_call():
    regularization = ConstantRegularization(
        input_generators=[DummyInputGenerator()],
        reg_weight_policy=MatchDataLossRegWeight(alpha=2.0),
        reg_names=['small', 'large'],
    )

    weighted_terms, total = regularization.get_regularization_value(nn=None, data_loss=3.0)

    assert torch.allclose(weighted_terms, torch.tensor([3.0, 3.0]))
    assert torch.allclose(total, torch.tensor(6.0))
    assert regularization.is_weight_calibrated


def test_regularization_weight_report_can_be_saved(tmp_path):
    regularization = ConstantRegularization(
        input_generators=[DummyInputGenerator()],
        reg_weight_policy=TargetTotalRegWeight(target_total=10.0),
        reg_names=['small', 'large'],
    )
    regularization.calibrate_weights(nn=None)
    report_path = tmp_path / 'regularization_weight_report.csv'

    regularization.save_weight_report(str(report_path))

    report = pd.read_csv(report_path)
    assert list(report['name']) == ['small', 'large']
    assert list(report['weight_policy']) == ['TargetTotalRegWeight', 'TargetTotalRegWeight']


def test_regularization_validation_once_validates_only_first_call():
    regularization = ConstantRegularization(
        input_generators=[DummyInputGenerator()],
        reg_weights=[1.0, 1.0],
        reg_names=['small', 'large'],
        validation='once',
    )
    calls = []
    original_validate = regularization._validate_regularization_outputs

    def validate_with_count(regs):
        calls.append(1)
        return original_validate(regs)

    regularization._validate_regularization_outputs = validate_with_count

    regularization.get_regularization_value(nn=None)
    regularization.get_regularization_value(nn=None)

    assert len(calls) == 1


def test_regularization_validation_modes_control_validation_frequency():
    always = ConstantRegularization(
        input_generators=[DummyInputGenerator()],
        reg_weights=[1.0, 1.0],
        reg_names=['small', 'large'],
        validation='always',
    )
    never = ConstantRegularization(
        input_generators=[DummyInputGenerator()],
        reg_weights=[1.0, 1.0],
        reg_names=['small', 'large'],
        validation='never',
    )
    always_calls = []
    never_calls = []

    always._validate_regularization_outputs = lambda regs: always_calls.append(1) or regs
    never._validate_regularization_outputs = lambda regs: never_calls.append(1) or regs

    always.get_regularization_value(nn=None)
    always.get_regularization_value(nn=None)
    never.get_regularization_value(nn=None)

    assert len(always_calls) == 2
    assert len(never_calls) == 0
