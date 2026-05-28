import torch

__all__ = [
    'get_relative_error',
    'get_mean_squared_error',
    'get_absolute_error',
]


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
