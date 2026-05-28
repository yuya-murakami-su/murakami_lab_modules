import torch

from murakami_lab_modules.evaluation import (
    component_weighted_mse_loss,
    relative_mse_loss,
)
from murakami_lab_modules.evaluation import (
    binary_accuracy_from_logits,
    multiclass_accuracy_from_logits,
    relative_error,
)


def test_relative_mse_loss_is_scalar_by_default():
    y_true = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    y_pred = torch.tensor([[2.0, 1.0], [2.0, 4.0]])

    assert relative_mse_loss(y_true, y_pred).ndim == 0
    assert relative_mse_loss(y_true, y_pred, reduction='none').shape == y_true.shape


def test_component_weighted_mse_loss():
    y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_pred = torch.zeros_like(y_true)

    component_loss = component_weighted_mse_loss(
        y_true,
        y_pred,
        component_weight=[1.0, 0.5],
        reduction='sum'
    )
    assert torch.allclose(component_loss, torch.tensor(1.0 + 2.0 + 9.0 + 8.0))


def test_regression_metrics_return_per_sample_values():
    y_true = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    y_pred = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    assert relative_error(y_true, y_pred).shape == (2, 1)


def test_classification_accuracy_metrics():
    y_class = torch.tensor([[1], [0], [1]])
    logits = torch.tensor([[0.0, 2.0], [3.0, 1.0], [1.0, 4.0]])
    assert torch.allclose(multiclass_accuracy_from_logits(y_class, logits), torch.tensor(1.0))

    y_binary = torch.tensor([[0.0], [1.0], [1.0]])
    binary_logits = torch.tensor([[-2.0], [2.0], [1.0]])
    assert torch.allclose(binary_accuracy_from_logits(y_binary, binary_logits), torch.tensor(1.0))
