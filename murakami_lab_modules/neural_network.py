import torch
from . import utils


class AbstractNeuralNetwork(torch.nn.Module):
    def __init__(
            self,
            n_input: int = 1,
            n_output: int = 1,
            n_layer: int = 2,
            n_node: int = 100,
            activation: callable = torch.nn.Tanh(),
            random_seed: int = 2025,
            **kwargs
    ):
        self.locals: dict = utils.get_local_dict(locals())
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

    @ staticmethod
    def get_neural_network_model(
            n_input: int,
            n_output: int,
            n_layer: int,
            n_node: int,
            activation: callable
    ) -> torch.nn.Sequential:
        if n_layer == 0:
            modules = [torch.nn.Linear(n_input, n_output)]

        else:
            modules = [torch.nn.Linear(n_input, n_node)]
            if activation is not None:
                modules += [activation]

            for _ in range(n_layer - 1):
                modules += [torch.nn.Linear(n_node, n_node)]
                if activation is not None:
                    modules += [activation]

            modules += [torch.nn.Linear(n_node, n_output)]
        return torch.nn.Sequential(*modules)


class FeedForwardNeuralNetwork(AbstractNeuralNetwork):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


class NeuralNetworkForODE(AbstractNeuralNetwork):
    def __init__(
            self,
            n_input: int = 1,
            n_output: int = 1,
            n_layer: int = 2,
            n_node: int = 100,
            activation: callable = torch.nn.Tanh(),
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

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.use_time_as_input:
            t_: torch.Tensor = torch.zeros_like(x[:, :1]) + t
            return self.nn(torch.hstack([t_, x]))
        else:
            return self.nn(x)