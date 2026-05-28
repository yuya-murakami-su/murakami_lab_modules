"""Prediction helper for models saved by :class:`ModelHandler`."""

from pathlib import Path

import numpy as np
import torch

from .neural_network import BaseNeuralNetwork
from .. import utils
from ..data.normalizer import BaseNormalizer

__all__ = ['NeuralNetworkPredictor']


class NeuralNetworkPredictor:
    """Load a saved neural network and expose a callable prediction interface.

    Parameters
    ----------
    model_path:
        Directory created by ``ModelHandler``. It must contain ``config.json``
        and ``state_dicts.pth``.
    network_class:
        Optional class override. By default the class stored in ``config.json``
        is imported.
    load_normalizer:
        When true, ``normalizer.pth`` is loaded and inputs/outputs are
        transformed in the same way as during training.
    postprocess:
        ``"raw"`` returns network outputs, ``"probability"`` applies
        sigmoid/softmax, and ``"class"`` returns predicted classes.

    Notes
    -----
    Saved PyTorch files should only be loaded from trusted sources.
    """

    def __init__(
            self,
            model_path: str,
            network_class: type[BaseNeuralNetwork] = None,
            load_normalizer: bool = True,
            device_name: str = 'cpu',
            postprocess: str = 'raw',
            binary_threshold: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.network_class = network_class
        self.load_normalizer = load_normalizer
        self.device_name = device_name
        self.postprocess = postprocess
        self.binary_threshold = binary_threshold

        self.device = utils.get_device(device_name)
        self._validate_postprocess()
        self.model = self._load_network()

    def _load_network(self):
        self._prepare_network()
        self._send_to_device()

        if self.load_normalizer:
            def predict_fn(x: torch.Tensor | np.ndarray):
                with torch.no_grad():
                    if type(x) is np.ndarray:
                        x = torch.tensor(x, dtype=torch.float32).to(self.device)
                        output_np = True
                    else:
                        x = x.to(self.device)
                        output_np = False
                    model_inputs = self.input_normalizer.transform(x)
                    network_outputs = self.network(model_inputs)
                    outputs = self.output_normalizer.inverse_transform(network_outputs)
                    outputs = self._postprocess_outputs(outputs)
                    if output_np:
                        return outputs.cpu().numpy()
                    return outputs

        else:
            def predict_fn(x: torch.Tensor | np.ndarray):
                with torch.no_grad():
                    if type(x) is np.ndarray:
                        x = torch.tensor(x, dtype=torch.float32).to(self.device)
                        output_np = True
                    else:
                        x = x.to(self.device)
                        output_np = False
                    network_outputs = self.network(x)
                    network_outputs = self._postprocess_outputs(network_outputs)
                    if output_np:
                        return network_outputs.cpu().numpy()
                    return network_outputs

        return predict_fn

    def _validate_postprocess(self) -> None:
        if self.postprocess not in {'raw', 'probability', 'class'}:
            raise ValueError("postprocess must be one of 'raw', 'probability', or 'class'.")
        if not 0.0 <= self.binary_threshold <= 1.0:
            raise ValueError('binary_threshold must satisfy 0.0 <= binary_threshold <= 1.0.')

    def _postprocess_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.postprocess == 'raw':
            return outputs
        if self.postprocess == 'probability':
            if outputs.shape[-1] == 1:
                return torch.sigmoid(outputs)
            return torch.softmax(outputs, dim=-1)
        if outputs.shape[-1] == 1:
            return (torch.sigmoid(outputs) >= self.binary_threshold).to(dtype=torch.long)
        return torch.argmax(outputs, dim=-1)

    def _prepare_network(self):
        config = utils.load_json(self.model_path / 'config.json')
        network_config = config['nn']
        network_class = self.network_class or utils.import_object(network_config['class'])
        self.network = network_class(**utils.deserialize_params(network_config['params']))

    def _send_to_device(self):
        state_dicts = torch.load(self.model_path / 'state_dicts.pth', weights_only=True, map_location='cpu')
        self.network.load_state_dict(state_dicts['nn_state_dict'])
        self.network.to(self.device)
        if self.load_normalizer:
            if not (self.model_path / 'normalizer.pth').exists():
                raise ValueError('Normalizer is not found. Please set to load_normalizer = False.')
            self.normalizer = torch.load(
                self.model_path / 'normalizer.pth',
                weights_only=True,
                map_location=self.device_name
            )
            self.input_normalizer = BaseNormalizer.from_config_dict(self.normalizer['input_normalizer']).to(self.device)
            self.output_normalizer = BaseNormalizer.from_config_dict(
                self.normalizer['output_normalizer']
            ).to(self.device)
        else:
            self.normalizer = None
            self.input_normalizer = None
            self.output_normalizer = None

    def __call__(self, x: torch.Tensor | np.ndarray):
        """Predict from a NumPy array or torch tensor."""

        return self.model(x)
