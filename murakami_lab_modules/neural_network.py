import torch
import copy
from collections.abc import Callable
from . import utils


class BaseNeuralNetwork(torch.nn.Module):
    def __init__(
            self,
            input_dim: int = 1,
            output_dim: int = 1,
            n_hidden_layers: int = 2,
            hidden_dim: int = 100,
            activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.Tanh(),
            dropout: float = 0.0,
            batch_norm: bool = False,
            random_seed: int = 2025,
            **kwargs
    ):
        self.locals: dict[str, object] = utils.get_local_dict(locals())
        utils.initialize_random_seed(random_seed)
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.random_seed = random_seed
        self.kwargs = kwargs

        self.nn = None

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'n_hidden_layers': self.n_hidden_layers,
            'hidden_dim': self.hidden_dim,
            'activation': self.activation,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'random_seed': self.random_seed,
            **self.kwargs
        })

    @staticmethod
    def _copy_activation(activation: Callable[[torch.Tensor], torch.Tensor]):
        if isinstance(activation, torch.nn.Module):
            return copy.deepcopy(activation)
        return activation

    @staticmethod
    def _hidden_block(
            input_dim: int,
            output_dim: int,
            activation: Callable[[torch.Tensor], torch.Tensor],
            dropout: float,
            batch_norm: bool
    ) -> list[torch.nn.Module]:
        modules = [torch.nn.Linear(input_dim, output_dim)]
        if batch_norm:
            modules.append(torch.nn.BatchNorm1d(output_dim))
        if activation is not None:
            modules.append(BaseNeuralNetwork._copy_activation(activation))
        if dropout > 0.0:
            modules.append(torch.nn.Dropout(p=dropout))
        return modules

    @staticmethod
    def get_neural_network_model(
            input_dim: int,
            output_dim: int,
            n_hidden_layers: int,
            hidden_dim: int,
            activation: Callable[[torch.Tensor], torch.Tensor],
            dropout: float = 0.0,
            batch_norm: bool = False,
            **kwargs
    ) -> torch.nn.Sequential:
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must satisfy 0.0 <= dropout < 1.0.')
        if n_hidden_layers == 0:
            modules = [torch.nn.Linear(input_dim, output_dim)]

        else:
            modules = BaseNeuralNetwork._hidden_block(
                input_dim=input_dim,
                output_dim=hidden_dim,
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm
            )

            for _ in range(n_hidden_layers - 1):
                modules += BaseNeuralNetwork._hidden_block(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    activation=activation,
                    dropout=dropout,
                    batch_norm=batch_norm
                )

            modules += [torch.nn.Linear(hidden_dim, output_dim)]
        return torch.nn.Sequential(*modules)


class FeedForwardNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nn = self.get_neural_network_model(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_hidden_layers=self.n_hidden_layers,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            **self.kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


class ODEFeedForwardNeuralNetwork(BaseNeuralNetwork):
    def __init__(
            self,
            input_dim: int = 1,
            output_dim: int = 1,
            n_hidden_layers: int = 2,
            hidden_dim: int = 100,
            activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.Tanh(),
            dropout: float = 0.0,
            batch_norm: bool = False,
            random_seed: int = 2025,
            use_time_as_input: bool = False,
            **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            random_seed=random_seed,
            use_time_as_input=use_time_as_input,
            **kwargs
        )
        self.use_time_as_input = use_time_as_input
        model_input_dim = input_dim + 1 if use_time_as_input else input_dim
        self.nn = self.get_neural_network_model(
            input_dim=model_input_dim,
            output_dim=output_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            **kwargs
        )

    @staticmethod
    def _expand_time_input(t: torch.Tensor | float | int, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        else:
            t = t.to(dtype=x.dtype, device=x.device)

        if x.ndim == 1:
            if t.ndim == 0:
                return t.reshape(1)
            if t.numel() == 1:
                return t.reshape(1)
            raise ValueError(f't must be scalar when x is 1D: t.shape = {tuple(t.shape)}')

        target_shape = x.shape[:-1] + (1,)
        if t.ndim == 0:
            return t.expand(target_shape)
        if t.shape == target_shape:
            return t
        if t.shape == x.shape[:-1]:
            return t.unsqueeze(-1)
        if t.numel() == 1:
            return t.reshape(1).expand(target_shape)
        raise ValueError(f't cannot be broadcast to x batch shape: t.shape = {tuple(t.shape)}, x.shape = {tuple(x.shape)}')

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.use_time_as_input:
            t_ = self._expand_time_input(t, x)
            return self.nn(torch.cat([t_, x], dim=-1))
        else:
            return self.nn(x)
