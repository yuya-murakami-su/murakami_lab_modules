"""Metric helpers for regression and classification evaluation."""

import torch

__all__ = [
    'relative_error',
    'multiclass_accuracy_from_logits',
    'binary_accuracy_from_logits',
]


def relative_error(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        epsilon: float = 1e-10
) -> torch.Tensor:
    """Return per-sample mean absolute relative error."""

    return ((y_true - y_pred).abs() / (y_true.abs() + epsilon)).mean(dim=1, keepdim=True)


def multiclass_accuracy_from_logits(y_true: torch.Tensor, y_logits: torch.Tensor) -> torch.Tensor:
    """Return accuracy for class-index targets and multi-class logits."""

    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true[:, 0]
    elif y_true.ndim != 1:
        raise ValueError(
            'multiclass_accuracy_from_logits expects class-index targets with shape [N] or [N, 1]. '
            f'y_true.shape={tuple(y_true.shape)} was given.'
        )
    y_pred = torch.argmax(y_logits, dim=-1)
    return (y_pred == y_true.long()).to(dtype=torch.float32).mean()


def binary_accuracy_from_logits(
        y_true: torch.Tensor,
        y_logits: torch.Tensor,
        threshold: float = 0.5
) -> torch.Tensor:
    """Return binary accuracy from logits using a sigmoid threshold."""

    y_prob = torch.sigmoid(y_logits)
    y_pred = (y_prob >= threshold).to(dtype=torch.float32)
    y_true = y_true.to(dtype=torch.float32)
    if y_true.shape != y_pred.shape:
        if y_pred.ndim == 2 and y_pred.shape[1] == 1 and y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        elif y_pred.ndim == 1 and y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = y_true[:, 0]
        else:
            raise ValueError(
                'binary_accuracy_from_logits target shape must match logits, allowing [N] <-> [N, 1]. '
                f'y_true.shape={tuple(y_true.shape)}, y_logits.shape={tuple(y_logits.shape)} were given.'
            )
    return (y_pred == y_true).to(dtype=torch.float32).mean()
