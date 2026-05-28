import torch
from collections.abc import Callable
from . import utils


class AbstractNeuralNetwork(torch.nn.Module):
    def __init__(
            self,
            n_input: int = 1,
            n_output: int = 1,
            n_layer: int = 2,
            n_node: int = 100,
            activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.Tanh(),
            random_seed: int = 2025,
            **kwargs
    ):
        self.locals: dict[str, object] = utils.get_local_dict(locals())
        utils.initialize_random_seed(random_seed)
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.n_layer = n_layer
        self.n_node = n_node
        self.activation = activation
        self.random_seed = random_seed
        self.kwargs = kwargs

        self.nn = None

    def config_dict(self) -> dict[str, object]:
        return utils.make_object_config(self, {
            'n_input': self.n_input,
            'n_output': self.n_output,
            'n_layer': self.n_layer,
            'n_node': self.n_node,
            'activation': self.activation,
            'random_seed': self.random_seed,
            **self.kwargs
        })

    @ staticmethod
    def get_neural_network_model(
            n_input: int,
            n_output: int,
            n_layer: int,
            n_node: int,
            activation: Callable[[torch.Tensor], torch.Tensor],
            dropout: float = 0.0,
            **kwargs
    ) -> torch.nn.Sequential:
        if n_layer == 0:
            modules = [torch.nn.Linear(n_input, n_output)]

        else:
            modules = [torch.nn.Linear(n_input, n_node)]
            if activation is not None:
                modules += [activation]
            if dropout > 0.0:
                modules += [torch.nn.Dropout(p=dropout)]

            for _ in range(n_layer - 1):
                modules += [torch.nn.Linear(n_node, n_node)]
                if activation is not None:
                    modules += [activation]
                if dropout > 0.0:
                    modules += [torch.nn.Dropout(p=dropout)]

            modules += [torch.nn.Linear(n_node, n_output)]
        return torch.nn.Sequential(*modules)


class FeedForwardNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nn = self.get_neural_network_model(
            n_input=self.n_input,
            n_output=self.n_output,
            n_layer=self.n_layer,
            n_node=self.n_node,
            activation=self.activation,
            **self.kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


class NeuralNetworkForODE(AbstractNeuralNetwork):
    def __init__(
            self,
            n_input: int = 1,
            n_output: int = 1,
            n_layer: int = 2,
            n_node: int = 100,
            activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.Tanh(),
            random_seed: int = 2025,
            use_time_as_input: bool = False,
            **kwargs
    ):
        super().__init__(
            n_input=n_input,
            n_output=n_output,
            n_layer=n_layer,
            n_node=n_node,
            activation=activation,
            random_seed=random_seed,
            use_time_as_input=use_time_as_input,
            **kwargs
        )
        self.use_time_as_input = use_time_as_input
        nn_input = n_input + 1 if use_time_as_input else n_input
        self.nn = self.get_neural_network_model(
            n_input=nn_input,
            n_output=n_output,
            n_layer=n_layer,
            n_node=n_node,
            activation=activation,
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
