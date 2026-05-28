"""Automatic-differentiation helpers for PINN residuals.

PyTorch's ``autograd.grad`` computes vector-Jacobian products. These helpers
wrap that behavior into common PINN operations such as selected partial
derivatives, Jacobians, diagonal Hessian terms, and Laplacians.
"""

import torch

__all__ = [
    'grad',
    'partial',
    'partial2',
    'jacobian',
    'hessian_diag',
    'laplacian',
]


def _normalize_indices(indices: int | list[int] | tuple[int, ...] | None, size: int, name: str) -> list[int]:
    if indices is None:
        return list(range(size))
    if type(indices) is int:
        indices = (indices,)
    elif not isinstance(indices, (list, tuple)):
        raise TypeError(f'{name} must be int, list[int], tuple[int, ...], or None. {type(indices)} was given.')

    normalized = []
    for idx in indices:
        if type(idx) is not int:
            raise TypeError(f'All elements of {name} must be int. {type(idx)} was given.')
        if not -size <= idx < size:
            raise IndexError(f'{name} contains {idx}, which is out of range for size {size}.')
        normalized.append(idx % size)
    return normalized


def _normalize_y_indices(y: torch.Tensor, y_indices: int | list[int] | tuple[int, ...] | None) -> list[int | None]:
    if y.ndim == 1:
        if y_indices is None:
            return [None]
        normalized = _normalize_indices(y_indices, 1, 'y_indices')
        return [None for _ in normalized]
    if y.ndim < 1:
        raise ValueError(f'y must have at least 1 dimension. y.shape={tuple(y.shape)}.')
    return _normalize_indices(y_indices, y.shape[1], 'y_indices')


def _normalize_x_indices(x: torch.Tensor, x_indices: int | list[int] | tuple[int, ...] | None) -> list[int]:
    if x.ndim < 2:
        raise ValueError(f'x_indices requires x.ndim >= 2. x.shape={tuple(x.shape)}.')
    return _normalize_indices(x_indices, x.shape[1], 'x_indices')


def grad(
        y: torch.Tensor,
        x: torch.Tensor,
        x_idx: int = None,
        y_idx: int = None,
        zero_if_unused: bool = False,
        keepdim: bool = False
):
    """Differentiate ``y`` with respect to ``x``.

    When ``y`` has multiple output components, ``y_idx`` selects one component.
    When ``x`` has multiple input components, ``x_idx`` selects one component.
    By default an unused graph connection raises a clear error; set
    ``zero_if_unused=True`` only when a zero derivative is mathematically
    expected.
    """

    if not torch.is_tensor(y):
        raise TypeError(f'y must be torch.Tensor. {type(y)} was given.')
    if not torch.is_tensor(x):
        raise TypeError(f'x must be torch.Tensor. {type(x)} was given.')
    if not x.requires_grad:
        raise ValueError(f'x must require grad. x.requires_grad={x.requires_grad}.')

    if y_idx is not None:
        if y.ndim < 2:
            raise ValueError(f'y_idx requires y.ndim >= 2. y.shape={tuple(y.shape)}.')
        if not -y.shape[1] <= y_idx < y.shape[1]:
            raise IndexError(f'y_idx={y_idx} is out of range for y.shape={tuple(y.shape)}.')

    if x_idx is not None:
        if x.ndim < 2:
            raise ValueError(f'x_idx requires x.ndim >= 2. x.shape={tuple(x.shape)}.')
        if not -x.shape[1] <= x_idx < x.shape[1]:
            raise IndexError(f'x_idx={x_idx} is out of range for x.shape={tuple(x.shape)}.')
        normalized_x_idx = x_idx % x.shape[1]

    def select_x_idx(dy_dx: torch.Tensor):
        if x_idx is None:
            return dy_dx
        elif keepdim:
            return dy_dx[:, normalized_x_idx:normalized_x_idx + 1]
        else:
            return dy_dx[:, x_idx]

    if not y.requires_grad:
        if zero_if_unused:
            return select_x_idx(x * 0.0)
        raise ValueError(f'y must require grad. y.requires_grad={y.requires_grad}.')

    if y_idx is None:
        grad_outputs = torch.ones_like(y)
    else:
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, y_idx] = 1.0

    try:
        dy_dx = torch.autograd.grad(
            inputs=x,
            outputs=y,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True,
            allow_unused=zero_if_unused
        )[0]
    except RuntimeError as e:
        if 'not have been used in the graph' not in str(e):
            raise
        raise RuntimeError(
            f'Failed to compute grad(y, x) because y does not depend on x. '
            f'y.shape={tuple(y.shape)}, x.shape={tuple(x.shape)}, '
            f'x_idx={x_idx}, y_idx={y_idx}. If y is intentionally independent of x, '
            f'set zero_if_unused=True.'
        ) from e

    if dy_dx is None:
        dy_dx = x * 0.0
    elif zero_if_unused and not dy_dx.requires_grad:
        dy_dx = dy_dx + x * 0.0

    return select_x_idx(dy_dx)


def partial(
        y: torch.Tensor,
        x: torch.Tensor,
        x_idx: int,
        y_idx: int = None,
        zero_if_unused: bool = False,
        keepdim: bool = False
) -> torch.Tensor:
    """Return ``d y[..., y_idx] / d x[..., x_idx]`` for batched tensors."""

    if x_idx is None:
        raise ValueError('x_idx must be given for partial().')
    return grad(
        y=y,
        x=x,
        x_idx=x_idx,
        y_idx=y_idx,
        zero_if_unused=zero_if_unused,
        keepdim=keepdim
    )


def partial2(
        y: torch.Tensor,
        x: torch.Tensor,
        x_idx: int,
        y_idx: int = None,
        zero_if_unused: bool = False,
        keepdim: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first and unmixed second partial derivatives for one input axis."""

    dy_dx = partial(
        y=y,
        x=x,
        x_idx=x_idx,
        y_idx=y_idx,
        zero_if_unused=zero_if_unused,
        keepdim=keepdim
    )
    d2y_dx2 = partial(
        y=dy_dx,
        x=x,
        x_idx=x_idx,
        zero_if_unused=zero_if_unused or not dy_dx.requires_grad,
        keepdim=keepdim
    )
    return dy_dx, d2y_dx2


def jacobian(
        y: torch.Tensor,
        x: torch.Tensor,
        y_indices: int | list[int] | tuple[int, ...] = None,
        x_indices: int | list[int] | tuple[int, ...] = None,
        zero_if_unused: bool = False
) -> torch.Tensor:
    """Return selected Jacobian entries with shape ``[N, n_y, n_x]``."""

    y_indices = _normalize_y_indices(y, y_indices)
    x_indices = _normalize_x_indices(x, x_indices)
    jacobian_ = []
    for y_idx in y_indices:
        row = [
            partial(
                y=y,
                x=x,
                x_idx=x_idx,
                y_idx=y_idx,
                zero_if_unused=zero_if_unused,
                keepdim=True
            )
            for x_idx in x_indices
        ]
        jacobian_.append(torch.cat(row, dim=1))
    return torch.stack(jacobian_, dim=1)


def hessian_diag(
        y: torch.Tensor,
        x: torch.Tensor,
        y_indices: int | list[int] | tuple[int, ...] = None,
        x_indices: int | list[int] | tuple[int, ...] = None,
        zero_if_unused: bool = False,
        return_first: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return selected diagonal Hessian entries with shape ``[N, n_y, n_x]``.

    Mixed second derivatives are intentionally not computed. Set
    ``return_first=True`` to also receive the corresponding Jacobian terms.
    """

    y_indices = _normalize_y_indices(y, y_indices)
    x_indices = _normalize_x_indices(x, x_indices)
    jacobian_ = []
    hessian_diag_ = []
    for y_idx in y_indices:
        jacobian_row = []
        hessian_diag_row = []
        for x_idx in x_indices:
            dy_dx, d2y_dx2 = partial2(
                y=y,
                x=x,
                x_idx=x_idx,
                y_idx=y_idx,
                zero_if_unused=zero_if_unused,
                keepdim=True
            )
            jacobian_row.append(dy_dx)
            hessian_diag_row.append(d2y_dx2)
        jacobian_.append(torch.cat(jacobian_row, dim=1))
        hessian_diag_.append(torch.cat(hessian_diag_row, dim=1))

    jacobian_ = torch.stack(jacobian_, dim=1)
    hessian_diag_ = torch.stack(hessian_diag_, dim=1)
    if return_first:
        return jacobian_, hessian_diag_
    return hessian_diag_


def laplacian(
        y: torch.Tensor,
        x: torch.Tensor,
        y_indices: int | list[int] | tuple[int, ...] = None,
        x_indices: int | list[int] | tuple[int, ...] = None,
        zero_if_unused: bool = False,
        keepdim: bool = False
) -> torch.Tensor:
    """Return the sum of selected diagonal Hessian entries."""

    laplacian_ = hessian_diag(
        y=y,
        x=x,
        y_indices=y_indices,
        x_indices=x_indices,
        zero_if_unused=zero_if_unused
    ).sum(dim=2)
    if laplacian_.shape[1] == 1 and not keepdim:
        return laplacian_[:, 0]
    return laplacian_
