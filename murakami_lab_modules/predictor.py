import torch
import numpy as np
import os
from .neural_network import AbstractNeuralNetwork
from . import utils

VariableSpec = int | float | tuple[float, float] | list[float]

class AbstractPredictor:
    def __init__(self):
        self.model = None

    def predict_with_1_variable(
            self,
            variables: tuple[VariableSpec, ...],
            n_step: int = 100,
            return_torch: bool = False
    ):
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
            model_inputs = torch.empty([n_step, n_total_variable], dtype=torch.float32)
            for i in range(n_total_variable):
                if i == variable_idx:
                    model_inputs[:, i] = torch.linspace(*variables[variable_idx], steps=n_step)
                else:
                    model_inputs[:, i] = float(variables[i])
        else:
            model_inputs = np.empty([n_step, n_total_variable], dtype=np.float32)
            for i in range(n_total_variable):
                if i == variable_idx:
                    model_inputs[:, i] = np.linspace(*variables[variable_idx], num=n_step)
                else:
                    model_inputs[:, i] = float(variables[i])

        return self.model(model_inputs)

    def predict_with_2_variable(
            self,
            variables: tuple[VariableSpec, ...],
            n_step: int | tuple[int, int] | list[int] = 100,
            return_torch: bool = False,
            return_grid: bool = True,
            squeeze_output: bool = True
    ):
        n_total_variable = len(variables)
        variable_indices = []
        for i, variable in enumerate(variables):
            if type(variable) is list or type(variable) is tuple:
                variable_indices.append(i)
                if len(variable) != 2:
                    raise ValueError(
                        f'Range of variable must be given by the iterable with 2 elements. ({variable} was given)'
                    )

        if len(variable_indices) != 2:
            raise ValueError(
                f'predict_with_2_variable accept tuple in which two elements are tuple and others are single value. '
                f'(Number of given iterables: {len(variable_indices)})'
            )

        if type(n_step) is int:
            n_steps = (n_step, n_step)
        elif type(n_step) is tuple or type(n_step) is list:
            if len(n_step) != 2:
                raise ValueError(f'n_step must be int or iterable with 2 elements. ({n_step} was given)')
            n_steps = tuple(n_step)
        else:
            raise ValueError(f'n_step must be int, tuple, or list. ({type(n_step)} was given)')

        if return_torch:
            variable_0 = torch.linspace(*variables[variable_indices[0]], steps=n_steps[0])
            variable_1 = torch.linspace(*variables[variable_indices[1]], steps=n_steps[1])
            grid_0, grid_1 = torch.meshgrid(variable_0, variable_1, indexing='ij')
            model_inputs = torch.empty([n_steps[0] * n_steps[1], n_total_variable], dtype=torch.float32)
        else:
            variable_0 = np.linspace(*variables[variable_indices[0]], num=n_steps[0])
            variable_1 = np.linspace(*variables[variable_indices[1]], num=n_steps[1])
            grid_0, grid_1 = np.meshgrid(variable_0, variable_1, indexing='ij')
            model_inputs = np.empty([n_steps[0] * n_steps[1], n_total_variable], dtype=np.float32)

        for i in range(n_total_variable):
            if i == variable_indices[0]:
                model_inputs[:, i] = grid_0.reshape(-1)
            elif i == variable_indices[1]:
                model_inputs[:, i] = grid_1.reshape(-1)
            else:
                model_inputs[:, i] = float(variables[i])

        outputs = self.model(model_inputs)
        if not return_grid:
            return outputs

        output_shape = outputs.shape[1:]
        grid_outputs = outputs.reshape(n_steps[0], n_steps[1], *output_shape)
        if squeeze_output and len(output_shape) == 1 and output_shape[0] == 1:
            grid_outputs = grid_outputs.reshape(n_steps[0], n_steps[1])

        return grid_0, grid_1, grid_outputs


class NNPredictor(AbstractPredictor):
    def __init__(
            self,
            model_path: str,
            nn_class: type[AbstractNeuralNetwork] = None,
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
                    nn_inputs = (x - self.normalizer['input_ave']) / self.normalizer['input_std']
                    nn_outputs = self.nn(nn_inputs)
                    outputs = nn_outputs * self.normalizer['output_std'] + self.normalizer['output_ave']
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
        config = utils.load_json(f'{self.model_path}\\config.json')
        nn_config = config['nn']
        nn_class = self.nn_class or utils.import_object(nn_config['class'])
        self.nn = nn_class(**utils.deserialize_params(nn_config['params']))

    def _send_to_device(self):
        state_dicts = torch.load(f'{self.model_path}\\state_dicts.pth', weights_only=False, map_location='cpu')
        self.nn.load_state_dict(state_dicts['nn_state_dict'])
        self.nn.to(self.device)
        if self.load_normalizer:
            if not os.path.exists(f'{self.model_path}\\normalizer.pth'):
                raise ValueError('Normalizer is not found. Please set to load_normalizer = False.')
            self.normalizer = torch.load(f'{self.model_path}\\normalizer.pth', weights_only=False,
                                         map_location=self.device_name)
        else:
            self.normalizer = None

    def __call__(self, x: torch.Tensor | np.ndarray):
        return self.model(x)
