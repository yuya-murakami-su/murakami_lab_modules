import pytest
import torch

from murakami_lab_modules.differential import grad, hessian_diag, jacobian, laplacian, partial, partial2


def test_partial_jacobian_hessian_and_laplacian_for_known_function():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = torch.stack([x[:, 0] ** 2 + x[:, 1], x[:, 0] * x[:, 1]], dim=1)

    assert torch.allclose(partial(y, x, x_idx=0, y_idx=0), torch.tensor([2.0, 6.0]))
    dy_dx0, d2y_dx02 = partial2(y, x, x_idx=0, y_idx=0)
    assert torch.allclose(dy_dx0, torch.tensor([2.0, 6.0]))
    assert torch.allclose(d2y_dx02, torch.tensor([2.0, 2.0]))

    expected_jacobian = torch.tensor([
        [[2.0, 1.0], [2.0, 1.0]],
        [[6.0, 1.0], [4.0, 3.0]],
    ])
    assert torch.allclose(jacobian(y, x), expected_jacobian)

    expected_hessian_diag = torch.tensor([
        [[2.0, 0.0], [0.0, 0.0]],
        [[2.0, 0.0], [0.0, 0.0]],
    ])
    assert torch.allclose(hessian_diag(y, x), expected_hessian_diag)
    assert torch.allclose(laplacian(y, x), torch.tensor([[2.0, 0.0], [2.0, 0.0]]))


def test_unused_grad_errors_by_default_and_can_return_zero():
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    unrelated = torch.tensor([[3.0], [4.0]], requires_grad=True)
    y = unrelated ** 2

    with pytest.raises(RuntimeError, match='does not depend on x'):
        grad(y, x)

    zero = grad(y, x, zero_if_unused=True)
    assert torch.allclose(zero, torch.zeros_like(x))
    assert zero.requires_grad
