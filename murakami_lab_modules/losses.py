import torch

__all__ = [
    'relative_error',
    'mse_error',
    'get_relative_error',
    'get_mean_squared_error',
    'get_absolute_error',
    'multiclass_accuracy_from_logits',
    'binary_accuracy_from_logits',
]


def relative_error(y_true: torch.Tensor, y_calc: torch.Tensor):
    return ((y_true - y_calc).abs() / (y_true.abs() + 1e-10)).mean(dim=1, keepdim=True)


def mse_error(y_true: torch.Tensor, y_calc: torch.Tensor):
    return torch.square(y_true - y_calc).mean(dim=1, keepdim=True)


def get_relative_error(epsilon: float = 1e-10, as_loss_function: bool = False):
    if as_loss_function:
        def relative_error(y_true: torch.Tensor, y_calc: torch.Tensor):
            return ((y_true - y_calc).abs() / (y_true.abs() + epsilon)).mean()
    else:
        def relative_error(y_true: torch.Tensor, y_calc: torch.Tensor):
            return ((y_true - y_calc).abs() / (y_true.abs() + epsilon)).mean(dim=1, keepdim=True)
    return relative_error


def get_mean_squared_error(as_loss_function: bool = False):
    if as_loss_function:
        mse_func = torch.nn.MSELoss()

        def mse(y_true: torch.Tensor, y_calc: torch.Tensor):
            return mse_func(y_true, y_calc).mean()
    else:
        mse_func = torch.nn.MSELoss(reduction='none')

        def mse(y_true: torch.Tensor, y_calc: torch.Tensor):
            return mse_func(y_true, y_calc).mean(dim=1, keepdim=True)
    return mse


def get_absolute_error(as_loss_function: bool = False):
    if as_loss_function:
        def absolute_error(y_true: torch.Tensor, y_calc: torch.Tensor):
            return (y_true - y_calc).abs().mean()
    else:
        def absolute_error(y_true: torch.Tensor, y_calc: torch.Tensor):
            return (y_true - y_calc).abs().mean(dim=1, keepdim=True)
    return absolute_error


def multiclass_accuracy_from_logits(y_true: torch.Tensor, y_logits: torch.Tensor) -> torch.Tensor:
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
