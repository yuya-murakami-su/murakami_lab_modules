import torch

__all__ = [
    'relative_mse_loss',
    'component_weighted_mse_loss',
]


def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    raise ValueError("reduction must be one of 'none', 'mean', or 'sum'.")


def relative_mse_loss(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        epsilon: float = 1e-10,
        reduction: str = 'mean'
) -> torch.Tensor:
    loss = torch.square((y_true - y_pred) / (y_true.abs() + epsilon))
    return _reduce_loss(loss, reduction=reduction)


def _weighted_mse_loss(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        weight: torch.Tensor | float,
        reduction: str = 'mean'
) -> torch.Tensor:
    weight = torch.as_tensor(weight, dtype=y_pred.dtype, device=y_pred.device)
    loss = torch.square(y_true - y_pred) * weight
    return _reduce_loss(loss, reduction=reduction)


def component_weighted_mse_loss(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        component_weight: torch.Tensor | list[float] | tuple[float, ...],
        reduction: str = 'mean'
) -> torch.Tensor:
    component_weight = torch.as_tensor(component_weight, dtype=y_pred.dtype, device=y_pred.device)
    if component_weight.ndim != 1:
        raise ValueError('component_weight must be a 1D tensor or sequence.')
    if y_pred.shape[-1] != component_weight.numel():
        raise ValueError(
            'The last dimension of y_pred must match component_weight length. '
            f'y_pred.shape={tuple(y_pred.shape)}, component_weight.shape={tuple(component_weight.shape)}.'
        )
    view_shape = [1] * y_pred.ndim
    view_shape[-1] = component_weight.numel()
    return _weighted_mse_loss(
        y_true=y_true,
        y_pred=y_pred,
        weight=component_weight.reshape(view_shape),
        reduction=reduction
    )
