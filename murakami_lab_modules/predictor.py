import torch
import numpy as np
import os
from .neural_network import AbstractNeuralNetwork
from . import utils


class AbstractPredictor:
    def __init__(self):
        self.model = None

    def predict_with_1_variable(self, variables: tuple, n_step: int = 100, return_torch: bool = False):
        n_total_variable = len(variables)
        n_variable = 0
        variable_idx = None
        for i, variable in enumerate(variables):
            if type(variable) is list or type(variable) is tuple:
                n_variable += 1
                variable_idx = i
                if len(variable) != 2:
                    raise ValueError(
                        f'Range of variable must be given by the iterable with 2 elements. ({variable} was given)'
                    )

        if n_variable != 1:
            raise ValueError(
                f'predict_with_1_variable accept tuple in which only one element is tuple and others are single value. '
                f'(Number of given iterables: {n_variable})'
            )

        if return_torch:
            model_inputs = torch.array([n_total_variable, n_step])
            for i in range(n_total_variable):
                if i == variable_idx:
                    model_inputs[:, i] = torch.linspace(*variables[variable_idx], steps=n_step)
                else:
                    model_inputs[:, i] = variables[i]
        else:
            model_inputs = np.array([n_total_variable, n_step])
            for i in range(n_total_variable):
                if i == variable_idx:
                    model_inputs[:, i] = np.linspace(*variables[variable_idx], num=n_step)
                else:
                    model_inputs[:, i] = variables[i]

        return self.model(model_inputs)

    def predict_with_2_variable(self):
        raise NotImplementedError('This method will be implemented in the future version.')


class NNPredictor(AbstractPredictor):
    def __init__(
            self,
            model_path: str,
            nn_class: type(AbstractNeuralNetwork),
            load_normalizer: bool = True,
            device_name: str = 'cpu',
    ):
        super().__init__()
        self.model_path = model_path
        self.nn_class = nn_class
        self.load_normalizer = load_normalizer
        self.device_name = device_name

        self.device = utils.get_device(device_name)
        self.model = self._load_nn_model()

    def _load_nn_model(self):
        self._prepare_nn()
        self._send_to_device()

        if self.load_normalizer:
            def nn_function(x: torch.Tensor | np.ndarray):
                with torch.no_grad():
                    if type(x) is np.ndarray:
                        x = torch.tensor(x, dtype=torch.float32).to(self.device)
                        output_np = True
                    else:
                        output_np = False
                    nn_inputs = (x - self.normalizer['inputs_ave']) / self.normalizer['inputs_std']
                    nn_outputs = self.nn(nn_inputs)
                    outputs = nn_outputs * self.normalizer['outputs_std'] + self.normalizer['outputs_ave']
                    if output_np:
                        return outputs.cpu().numpy()
                    else:
                        return outputs

        else:
            def nn_function(x: torch.Tensor | np.ndarray):
                with torch.no_grad():
                    if type(x) is np.ndarray:
                        x = torch.tensor(x, dtype=torch.float32).to(self.device)
                        output_np = True
                    else:
                        output_np = False
                    nn_outputs = self.nn(x)
                    if output_np:
                        return nn_outputs.cpu().numpy()
                    else:
                        return nn_outputs

        return nn_function

    def _prepare_nn(self):
        nn_locals = utils.load_txt(f'{self.model_path}\\nn_params')
        activation = getattr(torch.nn, nn_locals['activation'].replace('(', '').replace(')', ''))()

        self.nn = self.nn_class(
            n_input=nn_locals['n_input'],
            n_output=nn_locals['n_output'],
            n_layer=nn_locals['n_layer'],
            n_node=nn_locals['n_node'],
            activation=activation,
            random_seed=nn_locals['random_seed']
        )

    def _send_to_device(self):
        state_dicts = torch.load(f'{self.model_path}\\state_dicts.pth', weights_only=False, map_location='cpu')
        self.nn.load_state_dict(state_dicts['nn_state_dict'])
        self.nn.to(self.device)
        if self.load_normalizer:
            if not os.path.exists(f'{self.model_path}\\normalizer.pth'):
                raise ValueError('Normalizer is not found. Please set to load_normalizer = False.')
            self.normalizer = torch.load(f'{self.model_path}\\normalizer.pth', weights_only=False, map_location='cpu')
            for key in self.normalizer.keys():
                self.normalizer[key].to(self.device)
        else:
            self.normalizer = None

    def __call__(self, x: torch.Tensor | np.ndarray):
        return self.model(x)