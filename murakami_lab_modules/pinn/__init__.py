"""PINN-oriented automatic differentiation, input generation, and regularization."""

from .differential import grad, hessian_diag, jacobian, laplacian, partial, partial2
from .input_generator import InputGenerator
from .regularization import (
    MatchDataLossRegularizationWeight,
    Regularization,
    RegularizationWeightPolicy,
    StaticRegularizationWeights,
    TargetTotalRegularizationWeight,
)

__all__ = [
    'InputGenerator',
    'MatchDataLossRegularizationWeight',
    'Regularization',
    'RegularizationWeightPolicy',
    'StaticRegularizationWeights',
    'TargetTotalRegularizationWeight',
    'grad',
    'hessian_diag',
    'jacobian',
    'laplacian',
    'partial',
    'partial2',
]
